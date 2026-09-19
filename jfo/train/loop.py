"""Phase 3: single-round mixed block-size subspace training loop."""

import dataclasses
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from ..config import TrainConfig
from ..precision import resolve_dtype
from ..subspace import (
    adapt_model,
    residual_parameters,
    subspace_state,
    trainable_parameter_count,
)
from .dataset import build_loader
from .layout import PackedSequenceLayout
from .loss import total_loss
from .mask import build_block_mask
from .optim import SubspaceAdamW, learning_rate_at


def set_seed(seed: int) -> None:
    """Seed every random source used by the pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_backbone(model_path: str, dtype: torch.dtype, device: torch.device):
    """Load a backbone configured for noise-aware FlexAttention training."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="flex_attention",
        device_map={"": device},
    )
    model.config.use_cache = False
    return model


def configure_model(model, config: TrainConfig) -> int:
    """Install subspace adaptation and training-time memory optimizations."""
    adapted = adapt_model(model, config.adaptation)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return adapted


def compute_loss(model, sample: Dict, config: TrainConfig, device: torch.device, tokenizer) -> Tuple[torch.Tensor, ...]:
    """Forward one packed sequence and return the combined loss and its parts."""
    input_ids = sample["input_ids"].to(device, non_blocking=True)
    layout = PackedSequenceLayout.from_sample(sample, device)
    block_mask = build_block_mask(
        layout,
        model.config.num_attention_heads,
        compile_block_mask=torch.cuda.is_available(),
    )
    outputs = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=block_mask,
        position_ids=layout.position_ids.unsqueeze(0),
        use_cache=False,
    )
    loss, consistency, autoregressive = total_loss(
        outputs.logits,
        input_ids,
        layout,
        temperature=config.temperature,
        ar_weight=config.ar_weight,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return loss, consistency.detach(), autoregressive.detach()


def save_checkpoint(output_dir: Path, model, optimizer: SubspaceAdamW, step: int, config: TrainConfig) -> None:
    """Persist residuals, optimizer state, and the step counter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "subspace": subspace_state(model),
            "optimizer": optimizer.state_dict(),
            "train_config": dataclasses.asdict(config),
        },
        output_dir / "checkpoint.pt",
    )


def load_checkpoint(path: Path, model) -> Tuple[int, Dict]:
    """Restore the subspace state and return the saved step and optimizer state."""
    from ..subspace import load_subspace_state

    payload = torch.load(path, map_location="cpu", weights_only=False)
    load_subspace_state(model, payload["subspace"])
    return int(payload.get("step", 0)), payload.get("optimizer", {})


def train(config: TrainConfig, device: torch.device = None) -> Path:
    """Execute phase 3 and return the directory holding the final residuals."""
    if not config.model_path or not config.data_path:
        raise ValueError("model_path and data_path are required")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    tokenizer = load_tokenizer(config.model_path)
    model = load_backbone(config.model_path, resolve_dtype(config.dtype), device)
    adapted = configure_model(model, config)
    model.train()

    print(f"adapted projections: {adapted}")
    print(f"trainable parameters: {trainable_parameter_count(model)}")

    _, loader = build_loader(config)
    optimizer = SubspaceAdamW(
        residual_parameters(model),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        stochastic_rounding=config.stochastic_rounding,
        max_grad_norm=config.max_grad_norm,
    )

    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    checkpoint_path = output_dir / "checkpoint.pt"
    if config.resume and checkpoint_path.exists():
        start_step, optimizer_state = load_checkpoint(checkpoint_path, model)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        print(f"resumed from step {start_step}")

    iterator = iter(loader)
    for step in range(start_step, config.max_steps):
        learning_rate = learning_rate_at(
            step, config.max_steps, config.learning_rate, config.warmup_steps, config.schedule
        )
        optimizer.set_learning_rate(learning_rate)
        optimizer.zero_grad()

        accumulated = 0.0
        consistency_total = 0.0
        autoregressive_total = 0.0
        started = time.time()

        for _ in range(config.accumulation):
            try:
                sample = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                sample = next(iterator)
            loss, consistency, autoregressive = compute_loss(model, sample, config, device, tokenizer)
            (loss / config.accumulation).backward()
            accumulated += float(loss.detach())
            consistency_total += float(consistency)
            autoregressive_total += float(autoregressive)

        optimizer.step()
        completed = step + 1

        if completed % config.log_every == 0 or completed == config.max_steps:
            elapsed = time.time() - started
            print(
                f"step {completed}/{config.max_steps} "
                f"loss {accumulated / config.accumulation:.4f} "
                f"consistency {consistency_total / config.accumulation:.4f} "
                f"ar {autoregressive_total / config.accumulation:.4f} "
                f"lr {learning_rate:.3e} "
                f"step_time {elapsed:.2f}s"
            )
        if completed % config.save_every == 0 or completed == config.max_steps:
            save_checkpoint(output_dir, model, optimizer, completed, config)

    torch.save(
        {
            "subspace": subspace_state(model),
            "adaptation": dataclasses.asdict(config.adaptation),
        },
        output_dir / "subspace.pt",
    )
    print(f"final residuals written to {output_dir / 'subspace.pt'}")
    return output_dir
