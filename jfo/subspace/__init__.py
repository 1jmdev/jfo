"""SVD-aligned low-rank subspace adaptation."""

from .build import (
    adapt_model,
    adapt_model_from_bases,
    compute_bases,
    extract_bases,
    freeze_except_residual,
    load_subspace_state,
    residual_parameters,
    subspace_state,
    trainable_parameter_count,
)
from .layer import SubspaceLinear
from .merge import merge_subspace

__all__ = [
    "SubspaceLinear",
    "adapt_model",
    "adapt_model_from_bases",
    "compute_bases",
    "extract_bases",
    "freeze_except_residual",
    "load_subspace_state",
    "merge_subspace",
    "residual_parameters",
    "subspace_state",
    "trainable_parameter_count",
]
