"""Command line entry point for the four-phase JFO pipeline."""

import argparse
import dataclasses
from typing import Any, Dict, List

from .config import (
    CollectConfig,
    MergeConfig,
    PackConfig,
    TrainConfig,
    ValidateConfig,
    load_yaml,
)


def _section(config_path: str, name: str) -> Dict[str, Any]:
    mapping = load_yaml(config_path)
    section = mapping.get(name, {})
    return dict(section) if isinstance(section, dict) else {}


def _overrides(namespace: argparse.Namespace, keys: List[str]) -> Dict[str, Any]:
    return {key: getattr(namespace, key) for key in keys if getattr(namespace, key, None) is not None}


def _instantiate(cls: type, mapping: Dict[str, Any]):
    names = {item.name for item in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in mapping.items() if key in names})


def _add_common(parser: argparse.ArgumentParser, include_config: bool = True) -> None:
    if include_config:
        parser.add_argument("--config", default=None, help="YAML pipeline configuration")


def command_collect(args: argparse.Namespace) -> None:
    from .trajectory import run_collection

    mapping = _section(args.config, "collect")
    mapping.update(
        _overrides(
            args,
            [
                "model_path",
                "prompt_path",
                "output_path",
                "dataset",
                "dataset_split",
                "prompt_field",
                "max_prompts",
                "block_sizes",
                "max_new_tokens",
                "max_prompt_tokens",
                "batch_size",
                "draft_source",
                "shard_index",
                "shard_count",
                "dtype",
                "attention",
            ],
        )
    )
    if args.chat_template:
        mapping["chat_template"] = True
    written = run_collection(_instantiate(CollectConfig, mapping))
    print(f"collected {written} trajectory records")


def command_pack(args: argparse.Namespace) -> None:
    from .trajectory import run_packing

    mapping = _section(args.config, "pack")
    mapping.update(
        _overrides(
            args,
            [
                "trajectory_path",
                "output_path",
                "block_sizes",
                "window_size",
                "schedule",
                "min_noise_ratio",
                "max_noise_ratio",
                "max_sequence_length",
            ],
        )
    )
    written = run_packing(_instantiate(PackConfig, mapping))
    print(f"packed {written} training sequences")


def command_train(args: argparse.Namespace) -> None:
    from .train import train

    mapping = _section(args.config, "train")
    adaptation = _section(args.config, "subspace")
    if args.rank is not None:
        adaptation["rank"] = args.rank
    if args.target_modules is not None:
        adaptation["target_modules"] = args.target_modules
    if args.no_stochastic_rounding:
        mapping["stochastic_rounding"] = False
    if args.no_gradient_checkpointing:
        mapping["gradient_checkpointing"] = False
    if args.resume:
        mapping["resume"] = True
    mapping["adaptation"] = adaptation
    mapping.update(
        _overrides(
            args,
            [
                "model_path",
                "data_path",
                "output_dir",
                "block_sizes",
                "max_steps",
                "learning_rate",
                "accumulation",
                "max_sequence_length",
                "dtype",
                "log_every",
                "save_every",
            ],
        )
    )
    train(_instantiate(TrainConfig, mapping))


def command_merge(args: argparse.Namespace) -> None:
    from .merge import run_merge

    mapping = _section(args.config, "merge")
    mapping.update(_overrides(args, ["model_path", "adapter_path", "output_path", "dtype", "attention"]))
    run_merge(_instantiate(MergeConfig, mapping))


def command_validate(args: argparse.Namespace) -> None:
    from .validate import run_validation

    mapping = _section(args.config, "validate")
    mapping.update(
        _overrides(
            args,
            [
                "model_path",
                "prompt_path",
                "dataset",
                "dataset_split",
                "prompt_field",
                "block_size",
                "max_new_tokens",
                "max_prompt_tokens",
                "num_prompts",
                "dtype",
                "attention",
            ],
        )
    )
    run_validation(_instantiate(ValidateConfig, mapping))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jfo", description="Optimized Jacobi forcing conversion pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect Jacobi trajectories")
    _add_common(collect)
    collect.add_argument("--model-path")
    collect.add_argument("--prompt-path")
    collect.add_argument("--output-path")
    collect.add_argument("--dataset")
    collect.add_argument("--dataset-split")
    collect.add_argument("--prompt-field")
    collect.add_argument("--max-prompts", type=int)
    collect.add_argument("--block-sizes")
    collect.add_argument("--max-new-tokens", type=int)
    collect.add_argument("--max-prompt-tokens", type=int)
    collect.add_argument("--batch-size", type=int)
    collect.add_argument("--draft-source", choices=["context", "vocabulary"])
    collect.add_argument("--shard-index", type=int)
    collect.add_argument("--shard-count", type=int)
    collect.add_argument("--dtype")
    collect.add_argument("--attention")
    collect.add_argument("--chat-template", action="store_true")
    collect.set_defaults(handler=command_collect)

    pack = subparsers.add_parser("pack", help="pack trajectories with the noise schedule")
    _add_common(pack)
    pack.add_argument("--trajectory-path")
    pack.add_argument("--output-path")
    pack.add_argument("--block-sizes")
    pack.add_argument("--window-size", type=int)
    pack.add_argument("--schedule", choices=["linear_progressive", "reverse_progressive", "random"])
    pack.add_argument("--min-noise-ratio", type=float)
    pack.add_argument("--max-noise-ratio", type=float)
    pack.add_argument("--max-sequence-length", type=int)
    pack.set_defaults(handler=command_pack)

    train = subparsers.add_parser("train", help="run mixed block-size subspace training")
    _add_common(train)
    train.add_argument("--model-path")
    train.add_argument("--data-path")
    train.add_argument("--output-dir")
    train.add_argument("--block-sizes")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--accumulation", type=int)
    train.add_argument("--max-sequence-length", type=int)
    train.add_argument("--rank", type=int)
    train.add_argument("--target-modules")
    train.add_argument("--dtype")
    train.add_argument("--log-every", type=int)
    train.add_argument("--save-every", type=int)
    train.add_argument("--no-stochastic-rounding", action="store_true")
    train.add_argument("--no-gradient-checkpointing", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.set_defaults(handler=command_train)

    merge = subparsers.add_parser("merge", help="merge residuals into a dense backbone")
    _add_common(merge)
    merge.add_argument("--model-path")
    merge.add_argument("--adapter-path")
    merge.add_argument("--output-path")
    merge.add_argument("--dtype")
    merge.add_argument("--attention")
    merge.set_defaults(handler=command_merge)

    validate = subparsers.add_parser("validate", help="run a Jacobi decoding sanity check")
    _add_common(validate)
    validate.add_argument("--model-path")
    validate.add_argument("--prompt-path")
    validate.add_argument("--dataset")
    validate.add_argument("--dataset-split")
    validate.add_argument("--prompt-field")
    validate.add_argument("--block-size", type=int)
    validate.add_argument("--max-new-tokens", type=int)
    validate.add_argument("--max-prompt-tokens", type=int)
    validate.add_argument("--num-prompts", type=int)
    validate.add_argument("--dtype")
    validate.add_argument("--attention")
    validate.set_defaults(handler=command_validate)

    return parser


def main(argv: List[str] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
