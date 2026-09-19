"""Phase 2: map the progressive noise schedule onto static packed sequences.

A packed sequence is

    prompt | noisy_0 | clean_0 | noisy_1 | clean_1 | ...

where each clean block is a partition of the model's fixed point and each noisy
block is a trajectory state whose fraction of unconverged tokens is closest to
the scheduled noise ratio. Sub-blocks are derived from the reference trajectory,
so a single collection run serves every smaller training block size.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from ..config import PackConfig
from ..progress import progress

try:  # optional fast serializer
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def noise_schedule(window_size: int, low: float, high: float, strategy: str) -> np.ndarray:
    """Return the window of noise ratios used by the schedule."""
    if window_size < 1:
        raise ValueError("window_size must be positive")
    linear = np.linspace(low, high, window_size)
    if strategy == "linear_progressive":
        return linear
    if strategy == "reverse_progressive":
        return linear[::-1].copy()
    if strategy == "random":
        return linear
    raise ValueError(f"unknown schedule strategy: {strategy}")


def scheduled_ratio(schedule: np.ndarray, block_index: int, strategy: str, rng: np.random.Generator) -> float:
    """Return the target noise ratio for a block index."""
    if strategy == "random":
        return float(schedule[rng.integers(0, schedule.shape[0])])
    return float(schedule[block_index % schedule.shape[0]])


def unconverged_ratio(candidate: Sequence[int], clean: Sequence[int]) -> float:
    """Fraction of positions where a draft differs from the fixed point."""
    length = len(clean)
    if length == 0:
        return 0.0
    differences = sum(1 for left, right in zip(candidate, clean) if left != right)
    return differences / length


def read_records(path: str) -> Iterator[Dict]:
    """Yield trajectory records from a JSONL file."""
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _padded_fixed_points(blocks: List[Dict], reference: int):
    """Concatenate fixed points, padding the final short block."""
    padded: List[List[int]] = []
    valid_length = 0
    for block in blocks:
        fixed_point = list(block["fixed_point"])
        valid_length += len(fixed_point)
        padded.append(fixed_point + [0] * (reference - len(fixed_point)))
        if len(fixed_point) < reference:
            break
    answer: List[int] = [token for block in padded for token in block]
    return padded, answer, valid_length


def pack_record(
    record: Dict,
    block_size: int,
    config: PackConfig,
    rng: np.random.Generator,
    schedule: np.ndarray,
) -> Optional[Dict]:
    """Build one packed training sequence from one trajectory record."""
    reference = int(record["block_size"])
    if block_size > reference or reference % block_size != 0:
        raise ValueError(f"block size {block_size} must divide reference {reference}")

    reference_blocks = record["blocks"]
    if not reference_blocks:
        return None
    padded_fixed_points, answer, valid_length = _padded_fixed_points(reference_blocks, reference)

    prompt_ids = list(record["prompt_ids"])
    budget = config.max_sequence_length - len(prompt_ids)
    max_pairs = min(valid_length // block_size, max(0, budget) // (2 * block_size))
    if max_pairs < config.min_pairs:
        return None

    pairs: List[int] = []
    for block_index in range(max_pairs):
        offset = block_index * block_size
        reference_index = offset // reference
        inner_offset = offset % reference
        clean = answer[offset : offset + block_size]

        reference_block = reference_blocks[reference_index]
        states = reference_block["states"]
        candidates: List[List[int]] = [
            list(state[inner_offset : inner_offset + block_size]) for state in states
        ]
        candidates.append(list(padded_fixed_points[reference_index][inner_offset : inner_offset + block_size]))

        target = scheduled_ratio(schedule, block_index, config.schedule, rng)
        ratios = [abs(unconverged_ratio(candidate, clean) - target) for candidate in candidates]
        selected = candidates[int(np.argmin(ratios))]
        pairs.extend(selected)
        pairs.extend(clean)

    return {
        "input_ids": prompt_ids + pairs,
        "prompt_len": len(prompt_ids),
        "pair_count": max_pairs,
        "block_size": block_size,
    }


def run_packing(config: PackConfig) -> int:
    """Execute phase 2 and return the number of written packed sequences."""
    if not config.trajectory_path or not config.output_path:
        raise ValueError("trajectory_path and output_path are required")

    schedule = noise_schedule(config.window_size, config.min_noise_ratio, config.max_noise_ratio, config.schedule)
    rng = np.random.default_rng(config.seed)
    output_path = Path(config.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("wb") as handle:
        for record in progress(read_records(config.trajectory_path), desc="pack", unit="record"):
            payload = bytearray()
            for block_size in config.block_sizes:
                packed = pack_record(record, int(block_size), config, rng, schedule)
                if packed is None:
                    continue
                if orjson is not None:
                    payload.extend(orjson.dumps(packed) + b"\n")
                else:
                    payload.extend((json.dumps(packed, ensure_ascii=False) + "\n").encode("utf-8"))
                written += 1
            handle.write(payload)

    return written
