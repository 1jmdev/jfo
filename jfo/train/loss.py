"""Training objectives for mixed block-size subspace distillation.

The progressive consistency loss matches the student distribution on noisy
blocks to the frozen teacher distribution on the corresponding clean block. The
autoregressive loss keeps the clean chain intact and follows the fixed point.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .layout import PackedSequenceLayout


def soft_cross_entropy(
    predict_logits: torch.Tensor,
    target_logits: torch.Tensor,
    ignore_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross entropy against a soft target distribution."""
    log_probability = F.log_softmax(predict_logits, dim=-1)
    target_probability = F.softmax(target_logits, dim=-1)
    loss = -(target_probability * log_probability).sum(dim=-1)
    if ignore_mask is not None:
        valid = int((~ignore_mask).sum().item())
        if valid == 0:
            return predict_logits.new_zeros(())
        loss = loss.masked_fill(ignore_mask, 0.0)
        return loss.sum() / valid
    return loss.mean()


def autoregressive_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    layout: PackedSequenceLayout,
    ignore_index: int = -100,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Next-token loss over the prompt and the clean blocks only."""
    device = input_ids.device
    block_size = layout.block_size
    prompt_length = layout.prompt_length

    logit_positions = []
    target_positions = []

    if prompt_length > 1:
        positions = torch.arange(0, prompt_length - 1, device=device)
        logit_positions.append(positions)
        target_positions.append(positions + 1)

    for pair in range(layout.pair_count):
        clean_start = int(layout.clean_starts[pair])
        clean_block = input_ids[clean_start : clean_start + block_size]

        end = block_size
        if eos_token_id is not None:
            eos = torch.nonzero(clean_block == eos_token_id, as_tuple=False)
            if eos.numel() > 0:
                end = int(eos[0].item()) + 1

        bridge_logit = prompt_length - 1 if pair == 0 else int(layout.clean_starts[pair - 1]) + block_size - 1
        if pad_token_id is not None and int(input_ids[bridge_logit].item()) == pad_token_id:
            continue

        logit_positions.append(torch.tensor([bridge_logit], device=device))
        target_positions.append(torch.tensor([clean_start], device=device))

        if end > 1:
            positions = torch.arange(clean_start, clean_start + end - 1, device=device)
            logit_positions.append(positions)
            target_positions.append(positions + 1)

    if not logit_positions:
        return logits.new_zeros(())

    flat_logits = torch.cat(logit_positions)
    flat_targets = torch.cat(target_positions)
    selected = logits[0].index_select(0, flat_logits).float()
    targets = input_ids.index_select(0, flat_targets)
    if pad_token_id is not None:
        targets = targets.masked_fill(targets == pad_token_id, ignore_index)
    if int((targets != ignore_index).sum().item()) == 0:
        return logits.new_zeros(())
    return F.cross_entropy(selected, targets, ignore_index=ignore_index, reduction="mean")


def progressive_consistency_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    layout: PackedSequenceLayout,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Distribution matching on the diverged portion of each noisy block."""
    device = input_ids.device
    block_size = layout.block_size

    student_positions = []
    teacher_positions = []
    for pair in range(layout.pair_count):
        noisy_start = int(layout.noisy_starts[pair])
        clean_start = int(layout.clean_starts[pair])
        noisy = input_ids[noisy_start : noisy_start + block_size]
        clean = input_ids[clean_start : clean_start + block_size]

        differences = noisy != clean
        first_difference = (
            int(torch.nonzero(differences, as_tuple=False)[0].item()) if bool(differences.any()) else block_size
        )
        offsets = torch.arange(block_size - 1, device=device)
        keep = offsets >= first_difference
        if bool(keep.any()):
            student_positions.append(noisy_start + offsets[keep])
            teacher_positions.append(clean_start + offsets[keep])

    if not student_positions:
        return logits.new_zeros(())

    flat_student = torch.cat(student_positions)
    flat_teacher = torch.cat(teacher_positions)
    student = logits[0].index_select(0, flat_student).float()
    teacher = logits[0].index_select(0, flat_teacher).float().detach()

    per_token = soft_cross_entropy(student / temperature, teacher / temperature)
    return per_token * (temperature ** 2) / layout.pair_count


def total_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    layout: PackedSequenceLayout,
    temperature: float = 1.0,
    ar_weight: float = 10.0,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
):
    """Combine the progressive consistency and autoregressive objectives."""
    consistency = progressive_consistency_loss(logits, input_ids, layout, temperature=temperature)
    autoregressive = autoregressive_loss(
        logits,
        input_ids,
        layout,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    return consistency + ar_weight * autoregressive, consistency, autoregressive
