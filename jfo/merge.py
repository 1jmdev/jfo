"""Phase 4: merge trained subspace residuals into the dense backbone."""

from pathlib import Path

import torch

from .config import MergeConfig, SubspaceConfig
from .precision import resolve_dtype
from .subspace import adapt_model_from_bases, extract_bases, load_subspace_state, merge_subspace


def run_merge(config: MergeConfig, device: torch.device = None) -> Path:
    """Fold the residual adaptation into the base model and save the result."""
    if not config.model_path or not config.adapter_path or not config.output_path:
        raise ValueError("model_path, adapter_path and output_path are required")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(config.adapter_path).expanduser(), map_location="cpu", weights_only=False)
    adaptation = SubspaceConfig(**payload["adaptation"])
    state = payload["subspace"]

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=resolve_dtype(config.dtype),
        attn_implementation=config.attention,
        device_map={"": device},
    )
    adapt_model_from_bases(model, adaptation, extract_bases(state))
    load_subspace_state(model, state)
    merged = merge_subspace(model)

    output_path = Path(config.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    AutoTokenizer.from_pretrained(config.model_path).save_pretrained(output_path)
    print(f"merged {merged} projections into {output_path}")
    return output_path
