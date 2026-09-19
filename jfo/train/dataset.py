"""Static packed dataset and mixed block-size sampling.

Records are read lazily by byte offset so a large packed corpus never has to be
held in memory. The sampler interleaves block sizes round-robin so that every
gradient accumulation window sees the whole curriculum.
"""

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..config import TrainConfig


class PackedDataset(Dataset):
    """Offset-indexed reader for noise-schedule packed sequences."""

    def __init__(self, path: str, max_sequence_length: int = 0) -> None:
        self.path = str(Path(path).expanduser())
        self.max_sequence_length = max_sequence_length
        self.offsets: List[int] = []
        self.block_sizes: List[int] = []

        offset = 0
        with open(self.path, "rb") as handle:
            for raw in handle:
                if raw.strip():
                    record = json.loads(raw)
                    self.offsets.append(offset)
                    self.block_sizes.append(int(record["block_size"]))
                offset += len(raw)

        if not self.offsets:
            raise ValueError(f"no packed sequences found in {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Dict:
        with open(self.path, "rb") as handle:
            handle.seek(self.offsets[index])
            record = json.loads(handle.readline())

        input_ids = record["input_ids"]
        prompt_length = int(record["prompt_len"])
        block_size = int(record["block_size"])
        pair_count = int(record["pair_count"])

        if self.max_sequence_length and len(input_ids) > self.max_sequence_length:
            pair_count = (self.max_sequence_length - prompt_length) // (2 * block_size)
            pair_count = max(pair_count, 1)
            input_ids = input_ids[: prompt_length + 2 * pair_count * block_size]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_len": prompt_length,
            "pair_count": pair_count,
            "block_size": block_size,
        }


class MixedBlockSampler(Sampler):
    """Round-robin sampler across block-size groups."""

    def __init__(self, block_sizes: Sequence[int], seed: int = 0, length: int = 0) -> None:
        self.groups: Dict[int, List[int]] = {}
        for index, size in enumerate(block_sizes):
            self.groups.setdefault(int(size), []).append(index)
        self.order = sorted(self.groups)
        self.length = length if length > 0 else len(block_sizes)
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shuffled = {size: list(indices) for size, indices in self.groups.items()}
        for indices in shuffled.values():
            rng.shuffle(indices)

        cursors = {size: 0 for size in self.order}
        produced = 0
        while produced < self.length:
            for size in self.order:
                bucket = shuffled[size]
                yield bucket[cursors[size] % len(bucket)]
                cursors[size] += 1
                produced += 1
                if produced >= self.length:
                    break

    def __len__(self) -> int:
        return self.length


def _collate_single(batch: List[Dict]) -> Dict:
    return batch[0]


def build_loader(config: TrainConfig):
    """Construct the packed dataset and its mixed block-size loader."""
    dataset = PackedDataset(config.data_path, max_sequence_length=config.max_sequence_length)
    sampler = MixedBlockSampler(
        dataset.block_sizes,
        seed=config.seed,
        length=config.max_steps * config.accumulation,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=_collate_single,
        pin_memory=True,
    )
    return dataset, loader
