"""Phase 4: Jacobi decoding sanity check on a merged model."""

import time
from typing import Dict, List

import torch

from .config import ValidateConfig
from .precision import resolve_dtype
from .trajectory import JacobiDecoder, load_model, load_prompt_stream


def _decode_all(config: ValidateConfig, device: torch.device) -> Dict[str, float]:
    model, tokenizer = load_model(config.model_path, resolve_dtype(config.dtype), config.attention, device)
    decoder = JacobiDecoder(
        model,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        seed=0,
    )
    stream = load_prompt_stream(
        config.dataset,
        config.dataset_split,
        config.prompt_field,
        config.prompt_path,
        tokenizer,
        False,
        config.max_prompt_tokens,
    )
    prompts: List[List[int]] = [ids for _, ids in zip(range(config.num_prompts), stream)]

    total_tokens = 0
    total_iterations = 0
    total_seconds = 0.0
    for index, prompt_ids in enumerate(prompts):
        decoder.reseed(index)
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        started = time.time()
        with torch.inference_mode():
            trajectory = decoder.decode_prompt(prompt, config.block_size, config.max_new_tokens)
        total_seconds += time.time() - started
        total_tokens += int(trajectory.generated_ids.numel())
        total_iterations += sum(block.iterations for block in trajectory.blocks)

    tokens_per_forward = total_tokens / max(1, total_iterations)
    return {
        "prompts": len(prompts),
        "generated_tokens": total_tokens,
        "decode_iterations": total_iterations,
        "tokens_per_forward": tokens_per_forward,
        "wall_seconds": total_seconds,
        "tokens_per_second": total_tokens / total_seconds if total_seconds > 0 else 0.0,
        "wall_clock_speedup": tokens_per_forward,
    }


def run_validation(config: ValidateConfig, device: torch.device = None) -> Dict[str, float]:
    """Decode a handful of prompts and report Jacobi throughput statistics."""
    if not config.model_path or not config.prompt_path:
        raise ValueError("model_path and prompt_path are required")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = _decode_all(config, device)
    print(
        f"prompts {metrics['prompts']} "
        f"tokens {metrics['generated_tokens']} "
        f"iterations {metrics['decode_iterations']} "
        f"tokens/forward {metrics['tokens_per_forward']:.2f} "
        f"speedup {metrics['wall_clock_speedup']:.2f}x "
        f"tokens/second {metrics['tokens_per_second']:.1f}"
    )
    return metrics
