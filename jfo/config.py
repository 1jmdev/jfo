"""Configuration objects and YAML loading for every pipeline phase."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml


def as_tuple(value: Any) -> Tuple:
    """Coerce a scalar or delimited string into a tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(value.replace(",", " ").split())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _select(cls: type, mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only the keys declared by the dataclass ``cls``."""
    names = {item.name for item in dataclasses.fields(cls)}
    return {key: value for key, value in mapping.items() if key in names}


@dataclass
class CollectConfig:
    """Phase 1: collect Jacobi trajectories from the frozen base model."""

    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt_path: str = ""
    output_path: str = "data/trajectories.jsonl"
    dataset: str = "ise-uiuc/Magicoder-OSS-Instruct-75K"
    dataset_split: str = "train"
    prompt_field: str = "problem"
    max_prompts: int = 50000
    block_sizes: Sequence[int] = (32,)
    max_new_tokens: int = 512
    max_prompt_tokens: int = 1024
    batch_size: int = 32
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    draft_source: str = "context"
    chat_template: bool = False
    seed: int = 0
    shard_index: int = 0
    shard_count: int = 1

    def __post_init__(self) -> None:
        self.block_sizes = tuple(int(size) for size in as_tuple(self.block_sizes))


@dataclass
class PackConfig:
    """Phase 2: map the noise schedule onto static packed sequences."""

    trajectory_path: str = "data/trajectories.jsonl"
    output_path: str = "data/packed.jsonl"
    block_sizes: Sequence[int] = (4, 8, 16, 32)
    window_size: int = 16
    schedule: str = "linear_progressive"
    min_noise_ratio: float = 0.0
    max_noise_ratio: float = 1.0
    max_sequence_length: int = 2048
    min_pairs: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        self.block_sizes = tuple(int(size) for size in as_tuple(self.block_sizes))


@dataclass
class SubspaceConfig:
    """SVD-aligned low-rank subspace adaptation parameters."""

    rank: int = 8
    target_modules: Sequence[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    svd_niter: int = 4
    svd_oversampling: int = 0
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        self.target_modules = tuple(str(name) for name in as_tuple(self.target_modules))


@dataclass
class TrainConfig:
    """Phase 3: single-round mixed block-size subspace training."""

    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    data_path: str = "data/packed.jsonl"
    output_dir: str = "runs/jfo_qwen2_5_0_5b"
    block_sizes: Sequence[int] = (4, 8, 16, 32)
    max_steps: int = 3000
    learning_rate: float = 2.0e-4
    accumulation: int = 16
    max_sequence_length: int = 2048
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    stochastic_rounding: bool = True
    temperature: float = 1.0
    ar_weight: float = 10.0
    schedule: str = "cosine"
    warmup_steps: int = 50
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every: int = 500
    resume: bool = False
    seed: int = 0
    num_workers: int = 4
    adaptation: SubspaceConfig = field(default_factory=SubspaceConfig)

    def __post_init__(self) -> None:
        self.block_sizes = tuple(int(size) for size in as_tuple(self.block_sizes))
        if isinstance(self.adaptation, Mapping):
            self.adaptation = SubspaceConfig(**_select(SubspaceConfig, self.adaptation))


@dataclass
class MergeConfig:
    """Phase 4: fold the subspace residual into the dense backbone."""

    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path: str = "runs/jfo_qwen2_5_0_5b/subspace.pt"
    output_path: str = "runs/jfo_qwen2_5_0_5b/merged"
    dtype: str = "bfloat16"
    attention: str = "sdpa"


@dataclass
class ValidateConfig:
    """Phase 4: Jacobi decoding sanity check on the merged model."""

    model_path: str = "runs/jfo_qwen2_5_0_5b/merged"
    prompt_path: str = ""
    dataset: str = "ise-uiuc/Magicoder-OSS-Instruct-75K"
    dataset_split: str = "train"
    prompt_field: str = "problem"
    block_size: int = 32
    max_new_tokens: int = 256
    max_prompt_tokens: int = 512
    num_prompts: int = 8
    dtype: str = "bfloat16"
    attention: str = "sdpa"


@dataclass
class PipelineConfig:
    """Aggregate view over a pipeline YAML document."""

    collect: CollectConfig = field(default_factory=CollectConfig)
    pack: PackConfig = field(default_factory=PackConfig)
    subspace: SubspaceConfig = field(default_factory=SubspaceConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    validate: ValidateConfig = field(default_factory=ValidateConfig)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PipelineConfig":
        subspace_mapping = dict(mapping.get("subspace", {}))
        train_mapping = dict(mapping.get("train", {}))
        train_mapping.setdefault("adaptation", subspace_mapping)
        return cls(
            collect=CollectConfig(**_select(CollectConfig, mapping.get("collect", {}))),
            pack=PackConfig(**_select(PackConfig, mapping.get("pack", {}))),
            subspace=SubspaceConfig(**_select(SubspaceConfig, subspace_mapping)),
            train=TrainConfig(**_select(TrainConfig, train_mapping)),
            merge=MergeConfig(**_select(MergeConfig, mapping.get("merge", {}))),
            validate=ValidateConfig(**_select(ValidateConfig, mapping.get("validate", {}))),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        mapping = load_yaml(path)
        if not isinstance(mapping, Mapping):
            raise ValueError(f"expected a mapping at the root of {path}")
        return cls.from_mapping(mapping)


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML mapping, returning an empty mapping when no path is given."""
    if not path:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle) or {}
    if not isinstance(mapping, Mapping):
        raise ValueError(f"expected a mapping at the root of {path}")
    return dict(mapping)
