"""Mixed block-size subspace training."""

from .dataset import MixedBlockSampler, PackedDataset, build_loader
from .layout import PackedSequenceLayout
from .loss import autoregressive_loss, progressive_consistency_loss, soft_cross_entropy, total_loss
from .loop import compute_loss, configure_model, load_backbone, load_tokenizer, set_seed, train
from .mask import build_block_mask, dense_mask
from .optim import SubspaceAdamW, learning_rate_at

__all__ = [
    "MixedBlockSampler",
    "PackedDataset",
    "PackedSequenceLayout",
    "SubspaceAdamW",
    "autoregressive_loss",
    "build_block_mask",
    "build_loader",
    "compute_loss",
    "configure_model",
    "dense_mask",
    "learning_rate_at",
    "load_backbone",
    "load_tokenizer",
    "progressive_consistency_loss",
    "set_seed",
    "soft_cross_entropy",
    "total_loss",
    "train",
]
