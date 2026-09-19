"""Greedy Jacobi block decoding with full trajectory recording.

The decoder runs Jacobi fixed-point iteration inside a block of ``block_size``
tokens while the causal KV cache holds the accepted prefix. Every intermediate
draft is recorded so that packing can later pair a noisy block with its
converged counterpart.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn


@dataclass
class BlockTrajectory:
    """Outcome of decoding one block.

    ``fixed_point`` holds the converged tokens, shortened when an end-of-
    sequence token appears. ``states`` holds every recorded draft of length
    ``block_size`` in visit order.
    """

    fixed_point: torch.Tensor
    states: List[torch.Tensor] = field(default_factory=list)
    next_token: Optional[torch.Tensor] = None
    iterations: int = 0


@dataclass
class PromptTrajectory:
    """Decoding record for one prompt at one block size."""

    prompt_ids: torch.Tensor
    generated_ids: torch.Tensor
    blocks: List[BlockTrajectory]
    block_size: int

    @property
    def fixed_points(self) -> torch.Tensor:
        return torch.cat([block.fixed_point for block in self.blocks], dim=0)


class JacobiDecoder:
    """Greedy Jacobi decoder that records each block trajectory."""

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
        self.pad_token_id = pad_token_id
        self.draft_source = draft_source
        self.seed = seed
        self.vocab_size = int(getattr(model.config, "vocab_size", 0))
        self._generators: Dict[str, torch.Generator] = {}

    def reseed(self, seed: int) -> None:
        """Reset the sampling generator for a reproducible prompt run."""
        self.seed = seed
        self._generators.clear()

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[key] = generator
        return self._generators[key]

    @staticmethod
    def _truncate_cache(cache, target_length: int) -> None:
        if target_length >= cache.get_seq_length():
            return
        if hasattr(cache, "crop"):
            cache.crop(target_length)
            return
        for layer in range(len(cache.key_cache)):
            cache.key_cache[layer] = cache.key_cache[layer][..., :target_length, :]
            cache.value_cache[layer] = cache.value_cache[layer][..., :target_length, :]

    def _forward(self, token_ids: torch.Tensor, cache) -> torch.Tensor:
        past_length = cache.get_seq_length()
        cache_position = torch.arange(past_length, past_length + token_ids.shape[1], device=token_ids.device)
        outputs = self.model(
            input_ids=token_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        return outputs.logits

    def _sample_draft(self, count: int, context_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        if count <= 0:
            return torch.empty(1, 0, dtype=torch.long, device=device)
        generator = self._generator(device)
        if self.draft_source == "context" and context_ids.numel() > 0:
            pool = context_ids.reshape(-1)
            positions = torch.randint(0, pool.numel(), (count,), device=device, generator=generator)
            return pool[positions].reshape(1, -1)
        return torch.randint(0, self.vocab_size, (1, count), device=device, generator=generator)

    def prefill(self, prompt_ids: torch.Tensor, cache) -> torch.Tensor:
        logits = self._forward(prompt_ids, cache)
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    def decode_block(
        self,
        cache,
        first_token: torch.Tensor,
        block_size: int,
        context_ids: torch.Tensor,
    ) -> BlockTrajectory:
        device = first_token.device
        draft = torch.cat([first_token, self._sample_draft(block_size - 1, context_ids, device)], dim=-1)
        accepted = draft.clone()
        states: List[torch.Tensor] = [draft.clone()]
        total_accepted = 0
        iterations = 0
        next_token: Optional[torch.Tensor] = None

        while total_accepted < block_size:
            iterations += 1
            cache_length_before = cache.get_seq_length()
            logits = self._forward(draft, cache)
            probabilities = torch.softmax(logits.float(), dim=-1)

            candidate = torch.argmax(probabilities[:, :-1, :], dim=-1)
            mismatch = draft[:, 1:] != candidate
            num_accepted = int((mismatch.cumsum(dim=-1) == 0).sum(dim=-1).item()) + 1
            num_accepted = min(num_accepted, draft.shape[1])

            accepted[:, total_accepted : total_accepted + num_accepted] = draft[:, :num_accepted]
            total_accepted += num_accepted

            if self.eos_token_id is not None:
                eos_positions = (accepted[0, :total_accepted] == self.eos_token_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    eos_index = int(eos_positions[0].item())
                    self._truncate_cache(cache, cache_length_before + eos_index + 1)
                    return BlockTrajectory(
                        fixed_point=accepted[0, : eos_index + 1].clone(),
                        states=states,
                        next_token=None,
                        iterations=iterations,
                    )

            if num_accepted < draft.shape[1]:
                self._truncate_cache(cache, cache_length_before + num_accepted)
                next_token = torch.argmax(probabilities[:, num_accepted - 1, :], dim=-1, keepdim=True)
                remainder_logits = probabilities[:, num_accepted:-1, :]
                if remainder_logits.shape[1] > 0:
                    remainder = torch.argmax(remainder_logits, dim=-1)
                else:
                    remainder = torch.empty(1, 0, dtype=torch.long, device=device)
                draft = torch.cat([next_token, remainder], dim=-1)
                states.append(torch.cat([accepted[:, :total_accepted], draft], dim=-1))
            else:
                next_token = torch.argmax(probabilities[:, -1, :], dim=-1, keepdim=True)
                break

        return BlockTrajectory(
            fixed_point=accepted[0, :total_accepted].clone(),
            states=states,
            next_token=next_token,
            iterations=iterations,
        )

    def decode_prompt(
        self,
        prompt_ids: torch.Tensor,
        block_size: int,
        max_new_tokens: int,
        max_blocks: Optional[int] = None,
    ) -> PromptTrajectory:
        from transformers.cache_utils import DynamicCache

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        cache = DynamicCache()
        first_token = self.prefill(prompt_ids, cache)

        chunks: List[torch.Tensor] = []
        blocks: List[BlockTrajectory] = []
        limit = max_blocks if max_blocks is not None else max(1, max_new_tokens // block_size)

        for _ in range(limit):
            context = torch.cat([prompt_ids] + chunks, dim=-1) if chunks else prompt_ids
            block = self.decode_block(cache, first_token, block_size, context)
            blocks.append(block)
            chunks.append(block.fixed_point.unsqueeze(0))
            if block.next_token is None:
                break
            first_token = block.next_token

        generated = (
            torch.cat(chunks, dim=-1)
            if chunks
            else torch.empty(1, 0, dtype=torch.long, device=prompt_ids.device)
        )
        return PromptTrajectory(
            prompt_ids=prompt_ids[0],
            generated_ids=generated[0],
            blocks=blocks,
            block_size=block_size,
        )
