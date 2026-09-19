"""Batched greedy Jacobi decoding.

All prompts in a batch advance in lockstep through the same block index. The KV
cache holds only the clean prefix (prompt plus accepted fixed points), which is
uniform across the batch because prompts are left padded. The noisy block slots
are re-fed every iteration and the cache is cropped back to the prefix, so no
per-sequence cache surgery is needed and the whole batch shares one forward pass.
"""

from typing import List, Optional

import torch
from torch import nn

from .decoder import BlockTrajectory, PromptTrajectory


class BatchedJacobiDecoder:
    """Greedy Jacobi decoder that decodes a batch of prompts concurrently."""

    def __init__(
        self,
        model: nn.Module,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        draft_source: str = "context",
        seed: int = 0,
    ) -> None:
        self.model = model
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0 if pad_token_id is None else int(pad_token_id)
        self.draft_source = draft_source
        self.seed = seed
        self.vocab_size = int(getattr(model.config, "vocab_size", 0))
        self._generator: Optional[torch.Generator] = None

    def reseed(self, seed: int) -> None:
        self.seed = seed
        self._generator = None

    def _generator_for(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator.device != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generator = generator
        return self._generator

    def _sample_tail(self, prompts: List[torch.Tensor], count: int, device: torch.device) -> torch.Tensor:
        generator = self._generator_for(device)
        tail = torch.randint(0, self.vocab_size, (len(prompts), count), device=device, generator=generator)
        if self.draft_source == "context":
            for index, prompt in enumerate(prompts):
                prompt = prompt.to(device)
                if prompt.numel() > 0:
                    positions = torch.randint(0, prompt.numel(), (count,), device=device, generator=generator)
                    tail[index] = prompt[positions]
        return tail

    @torch.inference_mode()
    def decode(
        self,
        prompts: List[torch.Tensor],
        block_size: int,
        max_new_tokens: int,
    ) -> List[PromptTrajectory]:
        """Decode a batch and return one trajectory record per prompt."""
        from transformers.cache_utils import DynamicCache

        device = next(self.model.parameters()).device
        batch = len(prompts)
        if batch == 0:
            return []

        lengths = torch.tensor([prompt.numel() for prompt in prompts], dtype=torch.long, device=device)
        prompt_width = int(lengths.max().item())
        input_ids = torch.full((batch, prompt_width), self.pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros((batch, prompt_width), dtype=torch.long, device=device)
        for index, prompt in enumerate(prompts):
            length = prompt.numel()
            if length:
                input_ids[index, prompt_width - length :] = prompt.to(device)
                attention[index, prompt_width - length :] = 1
        position_ids = (attention.cumsum(dim=-1) - 1).clamp(min=0)

        cache = DynamicCache()
        prefill = self.model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(prompt_width, device=device),
        )
        next_tokens = torch.argmax(prefill.logits[:, -1, :], dim=-1)

        prefix_length = prompt_width
        active = torch.ones(batch, dtype=torch.bool, device=device)
        fixed_chunks: List[List[torch.Tensor]] = [[] for _ in range(batch)]
        block_records: List[List[BlockTrajectory]] = [[] for _ in range(batch)]
        relative_positions = torch.arange(block_size, device=device)

        max_blocks = max(1, max_new_tokens // block_size)
        for block_index in range(max_blocks):
            if not bool(active.any()):
                break

            draft = torch.full((batch, block_size), self.pad_token_id, dtype=torch.long, device=device)
            draft[:, 0] = next_tokens
            if block_size > 1:
                draft[:, 1:] = self._sample_tail(prompts, block_size - 1, device)

            fixed = torch.full((batch, block_size), self.pad_token_id, dtype=torch.long, device=device)
            total_accepted = torch.zeros(batch, dtype=torch.long, device=device)
            done = ~active
            block_next = torch.zeros(batch, dtype=torch.long, device=device)
            fixed_length = torch.full((batch,), block_size, dtype=torch.long, device=device)
            states: List[List[torch.Tensor]] = [[] for _ in range(batch)]
            iterations = [0] * batch
            slots_position = lengths[:, None] + block_index * block_size + relative_positions[None, :]

            for _ in range(block_size):
                for index in range(batch):
                    if bool(active[index]) and not bool(done[index]):
                        states[index].append(draft[index].clone())

                mask = torch.cat(
                    [attention, torch.ones((batch, block_size), dtype=torch.long, device=device)],
                    dim=-1,
                )
                outputs = self.model(
                    input_ids=draft,
                    attention_mask=mask,
                    position_ids=slots_position,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.arange(prefix_length, prefix_length + block_size, device=device),
                )
                cache.crop(prefix_length)
                predictions = torch.argmax(outputs.logits, dim=-1)

                mismatch = draft[:, 1:] != predictions[:, :-1]
                accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1) + 1
                accepted = torch.minimum(accepted, torch.full_like(accepted, block_size))

                for index in range(batch):
                    if not bool(active[index]) or bool(done[index]):
                        continue
                    iterations[index] += 1
                    count = max(int(accepted[index]), int(total_accepted[index]))
                    fixed[index, :count] = draft[index, :count]
                    total_accepted[index] = count

                    if self.eos_token_id is not None:
                        eos_positions = (fixed[index, :count] == self.eos_token_id).nonzero(as_tuple=False)
                        if eos_positions.numel() > 0:
                            stop = int(eos_positions[0].item()) + 1
                            total_accepted[index] = stop
                            fixed_length[index] = stop
                            done[index] = True
                            active[index] = False
                            continue

                    if count >= block_size:
                        block_next[index] = int(predictions[index, block_size - 1].item())
                        done[index] = True
                        continue

                    corrected = predictions[index, count - 1]
                    remainder = predictions[index, count : block_size - 1]
                    draft[index, :count] = fixed[index, :count]
                    draft[index, count] = corrected
                    if remainder.numel() > 0:
                        draft[index, count + 1 : block_size] = remainder

                if bool(done.all()):
                    break

            fixed_block = torch.full((batch, block_size), self.pad_token_id, dtype=torch.long, device=device)
            for index in range(batch):
                if not bool(active[index]) and fixed_length[index] < block_size:
                    fixed_block[index, : int(fixed_length[index])] = fixed[index, : int(fixed_length[index])]
                else:
                    fixed_block[index] = fixed[index]

            append_mask = torch.cat(
                [attention, torch.ones((batch, block_size), dtype=torch.long, device=device)],
                dim=-1,
            )
            self.model(
                input_ids=fixed_block,
                attention_mask=append_mask,
                position_ids=slots_position,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(prefix_length, prefix_length + block_size, device=device),
            )
            attention = append_mask
            prefix_length += block_size

            for index in range(batch):
                if not states[index]:
                    continue
                length = int(fixed_length[index]) if int(total_accepted[index]) >= block_size else int(total_accepted[index])
                fixed_point = fixed[index, :length].clone()
                block_records[index].append(
                    BlockTrajectory(
                        fixed_point=fixed_point,
                        states=states[index],
                        next_token=block_next[index].clone() if bool(active[index]) else None,
                        iterations=iterations[index],
                    )
                )
                fixed_chunks[index].append(fixed_point)

            for index in range(batch):
                if bool(active[index]):
                    next_tokens[index] = block_next[index]

        results: List[PromptTrajectory] = []
        for index in range(batch):
            generated = torch.cat(fixed_chunks[index]) if fixed_chunks[index] else torch.empty(0, dtype=torch.long, device=device)
            results.append(
                PromptTrajectory(
                    prompt_ids=prompts[index].cpu(),
                    generated_ids=generated.cpu(),
                    blocks=block_records[index],
                    block_size=block_size,
                )
            )
        return results
