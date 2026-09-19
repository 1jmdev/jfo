"""SVD basis construction, installation, and state serialization."""

from typing import Dict, Iterator, List, Tuple

import torch
from torch import nn

from ..config import SubspaceConfig
from ..precision import resolve_dtype
from .layer import SubspaceLinear

Basis = Tuple[torch.Tensor, torch.Tensor]


def compute_bases(
    weight: torch.Tensor,
    rank: int,
    niter: int = 4,
    oversampling: int = 0,
) -> Basis:
    """Return the leading ``rank`` singular directions of ``weight``.

    Randomized low-rank SVD avoids factorizing the full matrix, which matters
    for projections with tens of thousands of rows.
    """
    matrix = weight.detach().float()
    out_features, in_features = matrix.shape
    width = min(rank + oversampling, out_features, in_features)
    left, _, right = torch.svd_lowrank(matrix, q=width, niter=niter)
    return left[:, :rank].contiguous(), right[:, :rank].contiguous()


def resolve_parent(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    """Return the parent module and attribute name for ``qualified_name``."""
    parent_name, _, child_name = qualified_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, child_name


def _target_linears(model: nn.Module, config: SubspaceConfig) -> List[Tuple[str, nn.Linear]]:
    targets = tuple(config.target_modules)
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(targets)
    ]


def _install(model: nn.Module, config: SubspaceConfig, bases: Dict[str, Basis]) -> int:
    residual_dtype = resolve_dtype(config.dtype)
    candidates = _target_linears(model, config)

    for name, module in candidates:
        if name not in bases:
            raise KeyError(f"missing SVD bases for {name}")
        basis_out, basis_in = bases[name]
        adapted = SubspaceLinear(
            base_weight=module.weight.detach(),
            base_bias=module.bias.detach() if module.bias is not None else None,
            basis_out=basis_out.to(device=module.weight.device, dtype=residual_dtype),
            basis_in=basis_in.to(device=module.weight.device, dtype=residual_dtype),
            residual_dtype=residual_dtype,
        ).to(device=module.weight.device)
        parent, child_name = resolve_parent(model, name)
        setattr(parent, child_name, adapted)

    freeze_except_residual(model)
    return len(candidates)


def adapt_model(model: nn.Module, config: SubspaceConfig) -> int:
    """Compute SVD bases for every target projection and install adaptation."""
    bases: Dict[str, Basis] = {}
    for name, module in _target_linears(model, config):
        bases[name] = compute_bases(
            module.weight,
            rank=config.rank,
            niter=config.svd_niter,
            oversampling=config.svd_oversampling,
        )
    return _install(model, config, bases)


def adapt_model_from_bases(model: nn.Module, config: SubspaceConfig, bases: Dict[str, Basis]) -> int:
    """Install adaptation using previously saved SVD bases."""
    return _install(model, config, bases)


def freeze_except_residual(model: nn.Module) -> None:
    """Disable gradients everywhere except subspace residuals."""
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, SubspaceLinear):
            module.residual.requires_grad_(True)


def residual_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield trainable residual parameters in module order."""
    for module in model.modules():
        if isinstance(module, SubspaceLinear):
            yield module.residual


def trainable_parameter_count(model: nn.Module) -> int:
    """Count parameters that require gradients."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def subspace_state(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """Serialize SVD bases and residuals keyed by module name."""
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, SubspaceLinear):
            state[name] = {
                "basis_out": module.basis_out.detach().cpu(),
                "basis_in": module.basis_in.detach().cpu(),
                "residual": module.residual.detach().cpu(),
            }
    return state


def extract_bases(state: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Basis]:
    """Recover the frozen SVD bases from a serialized subspace state."""
    return {
        name: (entry["basis_out"], entry["basis_in"])
        for name, entry in state.items()
    }


def load_subspace_state(model: nn.Module, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
    """Restore SVD bases and residuals produced by :func:`subspace_state`."""
    remaining = set(state)
    for name, module in model.named_modules():
        if isinstance(module, SubspaceLinear):
            if name not in state:
                raise KeyError(f"state is missing bases for {name}")
            entry = state[name]
            with torch.no_grad():
                module.basis_out.copy_(entry["basis_out"].to(module.basis_out.device, module.basis_out.dtype))
                module.basis_in.copy_(entry["basis_in"].to(module.basis_in.device, module.basis_in.dtype))
                module.residual.copy_(entry["residual"].to(module.residual.device, module.residual.dtype))
            remaining.discard(name)
    if remaining:
        raise KeyError(f"unexpected subspace entries: {sorted(remaining)}")
