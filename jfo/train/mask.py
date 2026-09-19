"""Noise-aware causal attention masking.

Queries in a noisy block attend to the prompt, every previous clean block, and
their own causal noisy prefix. Queries in a clean block attend to the prompt,
every previous clean block, and their own causal clean prefix. This lets a
single forward pass produce both teacher logits on clean blocks and student
logits on noisy blocks.
"""

from typing import Optional

import torch

from .layout import PackedSequenceLayout


def _predicate(layout: PackedSequenceLayout, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Boolean visibility predicate shared by the FlexAttention and dense paths."""
    block_size = layout.block_size
    prompt_length = layout.prompt_length
    pair_count = layout.pair_count

    block_q = torch.div(q - prompt_length, block_size, rounding_mode="floor")
    block_k = torch.div(k - prompt_length, block_size, rounding_mode="floor")
    prompt_q = q < prompt_length
    prompt_k = k < prompt_length

    noisy_q = (~prompt_q) & (block_q % 2 == 0)
    clean_q = (~prompt_q) & (block_q % 2 == 1)
    clean_k = (~prompt_k) & (block_k % 2 == 1)

    pair_q = torch.clamp(block_q // 2, min=0, max=pair_count - 1)
    previous_clean = clean_k & (block_k < 2 * pair_q)

    noisy_context = noisy_q & (prompt_k | previous_clean)
    clean_context = clean_q & (prompt_k | previous_clean)
    causal_prompt = prompt_q & (k <= q)

    within_noisy = noisy_q & (block_q == block_k) & (k >= layout.noisy_starts[pair_q]) & (k <= q)
    within_clean = clean_q & (block_q == block_k) & (k >= layout.clean_starts[pair_q]) & (k <= q)

    return causal_prompt | noisy_context | clean_context | within_noisy | within_clean


def build_block_mask(
    layout: PackedSequenceLayout,
    num_attention_heads: int,
    compile_block_mask: bool = True,
):
    """Build a FlexAttention :class:`BlockMask` for a packed sequence."""
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(batch, head, q, k):
        return _predicate(layout, q, k)

    return create_block_mask(
        mask_mod,
        B=1,
        H=num_attention_heads,
        Q_LEN=layout.sequence_length,
        KV_LEN=layout.sequence_length,
        device=layout.device,
        _compile=compile_block_mask,
    )


def dense_mask(layout: PackedSequenceLayout) -> torch.Tensor:
    """Materialize the visibility predicate as a dense ``[L, L]`` boolean matrix.

    Intended for inspection and unit-level reasoning rather than training.
    """
    indices = torch.arange(layout.sequence_length, device=layout.device)
    q = indices[:, None]
    k = indices[None, :]
    return _predicate(layout, q, k)
