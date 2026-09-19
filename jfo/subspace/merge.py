"""Materialize the low-rank residual into dense backbone weights."""

from typing import List, Tuple

import torch
from torch import nn

from .build import resolve_parent
from .layer import SubspaceLinear


def merge_subspace(model: nn.Module) -> int:
    """Fold ``U R V^T`` into every base weight and drop the residual modules.

    The merged model has no runtime overhead: each adapted layer becomes a
    single dense projection, so causal attention and KV-cache reuse are intact.
    """
    adapted: List[Tuple[str, SubspaceLinear]] = [
        (name, module) for name, module in model.named_modules() if isinstance(module, SubspaceLinear)
    ]

    for name, module in adapted:
        base_weight = module.base.weight
        merged_weight = module.effective_weight().to(device=base_weight.device, dtype=base_weight.dtype)
        base_bias = module.base.bias
        linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=base_bias is not None,
            device=base_weight.device,
            dtype=base_weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(merged_weight)
            if base_bias is not None:
                linear.bias.copy_(base_bias)
        parent, child_name = resolve_parent(model, name)
        setattr(parent, child_name, linear)

    return len(adapted)
