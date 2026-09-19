"""Low-precision numerics.

The subspace residual matrices are tiny. When stored in bfloat16 an ordinary
optimizer update can fall below one unit in the last place and vanish.
Stochastic rounding preserves the expected update by snapping to a neighbouring
representable value with probability proportional to the residual distance.
"""

from typing import Optional

import math

import torch

_NAMES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def mantissa_bits(dtype: torch.dtype) -> int:
    """Number of stored mantissa bits for a floating-point dtype."""
    return int(round(-math.log2(float(torch.finfo(dtype).eps))))


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve a dtype name into a ``torch.dtype``."""
    try:
        return _NAMES[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def unit_in_last_place(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Spacing between adjacent representable ``dtype`` values near ``tensor``.

    ``tensor`` is interpreted in float32. The spacing follows from the binary
    exponent, so it is exact across the normal range.
    """
    _, exponent = torch.frexp(tensor)
    return torch.ldexp(torch.ones_like(tensor), exponent - mantissa_bits(dtype) - 1)


def stochastic_round(
    tensor: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Round to ``dtype`` while preserving sub-ulp information in expectation."""
    if tensor.dtype == dtype:
        return tensor
    source = tensor.float()
    spacing = unit_in_last_place(source, dtype)
    uniform = torch.rand(source.shape, generator=generator, device=source.device, dtype=source.dtype)
    noise = (uniform - 0.5) * spacing
    return (source + noise).to(dtype)


def maybe_round(tensor: torch.Tensor, dtype: torch.dtype, stochastic: bool) -> torch.Tensor:
    """Round with stochastic or nearest rounding depending on ``stochastic``."""
    if stochastic:
        return stochastic_round(tensor, dtype)
    return tensor.to(dtype)
