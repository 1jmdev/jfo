"""Accelerator setup and attention backend resolution.

The pipeline targets a single modern NVIDIA GPU. This module enables the
fast-path math modes once per process and selects FlashAttention-2 when the
kernel is installed, falling back to PyTorch SDPA otherwise.
"""

from functools import lru_cache

import torch

_FA2_NAMES = {"flash_attention_2", "fa2", "flash-attn", "flashattention2"}


@lru_cache(maxsize=1)
def flash_attention_available() -> bool:
    """Whether the ``flash_attn`` kernel can be used on this device."""
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def resolve_attention(name: str) -> str:
    """Return a usable attention implementation for the requested ``name``."""
    if name in _FA2_NAMES:
        return "flash_attention_2" if flash_attention_available() else "sdpa"
    return name


def configure_accelerator() -> None:
    """Enable the fast math and memory paths for training and decoding."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
