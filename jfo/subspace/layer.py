"""Frozen base projection with a trainable low-rank subspace residual."""

from typing import Optional

import torch
from torch import nn


class SubspaceLinear(nn.Module):
    """Linear layer with weight ``W0 + U R V^T``.

    ``W0`` and the SVD bases ``U`` and ``V`` are frozen. Only ``R`` of shape
    ``[rank, rank]`` receives gradients, so each adapted projection costs
    ``rank ** 2`` trainable parameters instead of ``in * out``.
    """

    def __init__(
        self,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor],
        basis_out: torch.Tensor,
        basis_in: torch.Tensor,
        residual_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        out_features, in_features = base_weight.shape
        rank = basis_out.shape[1]
        if basis_in.shape[1] != rank:
            raise ValueError("basis_out and basis_in must share the same rank")

        self.base = nn.Linear(in_features, out_features, bias=base_bias is not None)
        with torch.no_grad():
            self.base.weight.copy_(base_weight)
            if base_bias is not None:
                self.base.bias.copy_(base_bias)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.register_buffer("basis_out", basis_out.detach().clone())
        self.register_buffer("basis_in", basis_in.detach().clone())
        self.residual = nn.Parameter(torch.zeros(rank, rank, dtype=residual_dtype))

    @property
    def rank(self) -> int:
        return self.residual.shape[0]

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def effective_weight(self) -> torch.Tensor:
        """Materialize ``W0 + U R V^T`` in the base weight dtype."""
        residual = self.residual.to(self.base.weight.dtype)
        correction = (self.basis_out.to(residual.dtype) @ residual) @ self.basis_in.to(residual.dtype).transpose(0, 1)
        return self.base.weight + correction

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.base(hidden_states)
        compute_dtype = self.residual.dtype

        projected = hidden_states.to(compute_dtype) @ self.basis_in.to(compute_dtype)
        projected = projected @ self.residual.transpose(0, 1)
        correction = projected @ self.basis_out.to(compute_dtype).transpose(0, 1)
        return output + correction.to(output.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"
