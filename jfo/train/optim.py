"""AdamW variant that writes low-precision parameters with stochastic rounding."""

import math
from typing import Dict, Iterable, List, Optional

import torch

from ..precision import maybe_round


def learning_rate_at(
    step: int,
    max_steps: int,
    base_rate: float,
    warmup_steps: int,
    schedule: str,
) -> float:
    """Warmup followed by the configured decay schedule."""
    if step < warmup_steps:
        return base_rate * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "cosine":
        return base_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "linear":
        return base_rate * (1.0 - progress)
    if schedule == "constant":
        return base_rate
    raise ValueError(f"unknown schedule: {schedule}")


class SubspaceAdamW:
    """AdamW over subspace residuals with optional bfloat16 stochastic rounding.

    Moments are maintained in float32 regardless of parameter dtype. After the
    update is computed in float32 the parameter is written back through
    :func:`~jfo.precision.stochastic_round`, so updates smaller than one
    representable step still accumulate in expectation.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        learning_rate: float,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        stochastic_rounding: bool = True,
        max_grad_norm: Optional[float] = 1.0,
    ) -> None:
        self.parameters: List[torch.nn.Parameter] = [parameter for parameter in parameters if parameter.requires_grad]
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.stochastic_rounding = stochastic_rounding
        self.max_grad_norm = max_grad_norm
        self.state: List[Dict[str, torch.Tensor]] = [{} for _ in self.parameters]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def set_learning_rate(self, rate: float) -> None:
        self.learning_rate = rate

    @torch.no_grad()
    def step(self) -> None:
        if self.max_grad_norm is not None and self.parameters:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)

        beta1, beta2 = self.betas
        for parameter, state in zip(self.parameters, self.state):
            gradient = parameter.grad
            if gradient is None:
                continue
            gradient = gradient.float()

            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(gradient)
                state["exp_avg_sq"] = torch.zeros_like(gradient)
                state["step"] = torch.zeros((), dtype=torch.float32)

            state["step"] += 1
            state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

            bias1 = 1.0 - beta1 ** int(state["step"].item())
            bias2 = 1.0 - beta2 ** int(state["step"].item())
            denominator = (state["exp_avg_sq"].sqrt() / math.sqrt(bias2)).add_(self.eps)
            update = (state["exp_avg"] / bias1) / denominator
            if self.weight_decay:
                update = update.add(parameter.float(), alpha=self.weight_decay)

            updated = parameter.float() - self.learning_rate * update
            parameter.copy_(maybe_round(updated, parameter.dtype, self.stochastic_rounding))

    def state_dict(self) -> Dict:
        return {
            "learning_rate": self.learning_rate,
            "state": [
                {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in entry.items()}
                for entry in self.state
            ],
        }

    def load_state_dict(self, snapshot: Dict) -> None:
        self.learning_rate = snapshot.get("learning_rate", self.learning_rate)
        restored = snapshot.get("state", [])
        if len(restored) != len(self.state):
            raise ValueError("optimizer state size does not match the parameter set")
        self.state = [
            {key: (value.clone() if isinstance(value, torch.Tensor) else value) for key, value in entry.items()}
            for entry in restored
        ]
