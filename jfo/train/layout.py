"""Index arithmetic for packed training sequences.

A packed sequence is ``prompt | noisy_0 | clean_0 | noisy_1 | clean_1 | ...``.
Noisy and clean units share the same RoPE positions so that the teacher and
student distributions are compared at identical offsets.
"""

from typing import Dict

import torch


class PackedSequenceLayout:
    """Precomputed index tensors for one packed sequence."""

    def __init__(self, prompt_length: int, pair_count: int, block_size: int, device: torch.device) -> None:
        if prompt_length < 1:
            raise ValueError("prompt_length must be positive")
        if pair_count < 1:
            raise ValueError("pair_count must be positive")
        self.prompt_length = prompt_length
        self.pair_count = pair_count
        self.block_size = block_size
        self.device = device
        self.sequence_length = prompt_length + 2 * pair_count * block_size

        pairs = torch.arange(pair_count, device=device)
        self.noisy_starts = prompt_length + 2 * pairs * block_size
        self.clean_starts = prompt_length + (2 * pairs + 1) * block_size
        self.position_ids = self._build_position_ids()

    @classmethod
    def from_sample(cls, sample: Dict, device: torch.device) -> "PackedSequenceLayout":
        """Construct a layout from a packed dataset record."""
        input_ids = sample["input_ids"]
        layout = cls(
            prompt_length=int(sample["prompt_len"]),
            pair_count=int(sample["pair_count"]),
            block_size=int(sample["block_size"]),
            device=device,
        )
        if input_ids.numel() != layout.sequence_length:
            raise ValueError(
                f"packed length {input_ids.numel()} does not match layout {layout.sequence_length}"
            )
        return layout

    def _build_position_ids(self) -> torch.Tensor:
        positions = torch.empty(self.sequence_length, dtype=torch.long, device=self.device)
        positions[: self.prompt_length] = torch.arange(self.prompt_length, device=self.device)
        relative = torch.arange(self.block_size, device=self.device)
        for pair in range(self.pair_count):
            base = self.prompt_length + pair * self.block_size
            noisy_start = int(self.noisy_starts[pair])
            clean_start = int(self.clean_starts[pair])
            positions[noisy_start : noisy_start + self.block_size] = base + relative
            positions[clean_start : clean_start + self.block_size] = base + relative
        return positions

    def final_clean_start(self) -> int:
        return int(self.clean_starts[-1])
