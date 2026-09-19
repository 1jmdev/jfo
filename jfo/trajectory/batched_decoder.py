"""Batched greedy Jacobi decoding.

All prompts in a batch advance in lockstep through the same block index. The KV
cache holds only the clean prefix (prompt plus accepted fixed points), which is
uniform across the batch because prompts are left padded. The noisy block slots
are re-fed every iteration and the cache is cropped back to the prefix, so no
per-sequence cache surgery is needed and the whole batch shares one forward pass.

The generation loop is free of host synchronization. Acceptance, rebuild and
end-of-sequence bookkeeping are vectorized, no early break is taken, and every
``.tolist()`` is deferred until all blocks have been queued. That keeps the
accelerator saturated instead of stalling on scalar reads between blocks.
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

        positions = torch.arange(block_size, device=device)[None, :]
        pad_block = torch.full((batch, block_size), self.pad_token_id, dtype=torch.long, device=device)
        prefix_length = prompt_width
        active = torch.ones(batch, dtype=torch.bool, device=device)
        stored: List[dict] = []

        max_blocks = max(1, max_new_tokens // block_size)
        for block_index in range(max_blocks):
            draft = pad_block.clone()
            draft[:, 0] = next_tokens
            if block_size > 1:
                draft[:, 1:] = self._sample_tail(prompts, block_size - 1, device)

            fixed = pad_block.clone()
            total_accepted = torch.zeros(batch, dtype=torch.long, device=device)
            done = ~active
            block_next = torch.zeros(batch, dtype=torch.long, device=device)
            state_stack: List[torch.Tensor] = []
            counts = torch.zeros(batch, dtype=torch.long, device=device)
            slots_position = lengths[:, None] + block_index * block_size + positions

            for _ in range(block_size):
                updating = active & ~done
                counts += updating.long()
                state_stack.append(draft.clone())

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
                accepted = torch.maximum(accepted, total_accepted)
                accepted = torch.where(updating, accepted, total_accepted)

                fixed = torch.where(positions < accepted[:, None], draft, fixed)
                total_accepted = accepted

                if self.eos_token_id is not None:
                    eos_hit = fixed == self.eos_token_id
                    first_eos = (eos_hit.cumsum(dim=-1) == 0).sum(dim=-1)
                    newly_eos = updating & (first_eos < block_size)
                    total_accepted = torch.where(newly_eos, first_eos + 1, total_accepted)
                    done = done | newly_eos
                    active = active & ~newly_eos

                complete = (total_accepted >= block_size) & ~done
                block_next = torch.where(complete, predictions[:, block_size - 1], block_next)
                done = done | complete

                shifted = torch.gather(predictions, 1, (positions - 1).clamp(min=0).expand(batch, block_size))
                draft = torch.where(positions < total_accepted[:, None], fixed, shifted)

            fixed_block = torch.where(positions < total_accepted[:, None], fixed, pad_block)
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

            stored.append(
                {
                    "states": torch.stack(state_stack),
                    "counts": counts,
                    "fixed": fixed,
                    "lengths": total_accepted,
                    "next": block_next,
                    "active": active,
                }
            )
            next_tokens = torch.where(active, block_next, next_tokens)

        cpu_entries = [
            {
                "states": entry["states"].cpu(),
                "counts": entry["counts"].tolist(),
                "fixed": entry["fixed"].cpu(),
                "lengths": entry["lengths"].tolist(),
                "next": entry["next"].cpu(),
                "active": entry["active"].tolist(),
            }
            for entry in stored
        ]

        results: List[PromptTrajectory] = []
        for index in range(batch):
            blocks: List[BlockTrajectory] = []
            fixed_points: List[torch.Tensor] = []
            for entry in cpu_entries:
                count = entry["counts"][index]
                if count == 0:
                    continue
                length = min(int(entry["lengths"][index]), block_size)
                fixed_point = entry["fixed"][index, :length].clone()
                states = entry["states"][:count, index].clone()
                next_token = entry["next"][index].clone() if entry["active"][index] else None
                blocks.append(
                    BlockTrajectory(
                        fixed_point=fixed_point,
                        states=states,
                        next_token=next_token,
                        iterations=count,
                    )
                )
                fixed_points.append(fixed_point)

            generated = (
                torch.cat(fixed_points)
                if fixed_points
                else torch.empty(0, dtype=torch.long, device=prompts[index].device)
            )
            results.append(
                PromptTrajectory(
                    prompt_ids=prompts[index].cpu(),
                    generated_ids=generated,
                    blocks=blocks,
                    block_size=block_size,
                )
            )
        return results
