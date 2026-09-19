"""Phase 1: collect Jacobi trajectories from the frozen base model.

Prompts are streamed from a HuggingFace dataset by default and decoded in
length-sorted batches. A single reference block size is sufficient, because
packing derives sub-blocks for smaller training block sizes from the same
reference trajectory.
"""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from ..config import CollectConfig
from ..precision import resolve_dtype
from ..progress import progress
from ..runtime import configure_accelerator, resolve_attention
from .batched_decoder import BatchedJacobiDecoder
from .decoder import BlockTrajectory

try:  # optional fast serializer
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

_PROMPT_FIELDS = ("prompt", "problem", "question", "input", "instruction", "query", "text")


def load_model(model_path: str, dtype: torch.dtype, attention: str, device: torch.device):
    """Load a causal language model and tokenizer for inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    configure_accelerator()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=resolve_attention(attention),
        device_map={"": device},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _encode_prompt(tokenizer, text: str, chat_template: bool) -> List[int]:
    if chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    return tokenizer(text).input_ids


def _extract_prompt(record: Dict, field: str) -> str:
    """Read the prompt text from a dataset record using the configured field."""
    if field and record.get(field):
        return str(record[field])
    for candidate in _PROMPT_FIELDS:
        value = record.get(candidate)
        if value:
            return str(value)
    raise KeyError(f"no prompt field found in record with keys {sorted(record)}")


def load_prompt_stream(
    dataset: str,
    dataset_split: str,
    prompt_field: str,
    prompt_path: str,
    tokenizer,
    chat_template: bool,
    max_prompt_tokens: int,
) -> Iterator[List[int]]:
    """Yield tokenized prompts, downloading them when a dataset is configured."""
    if dataset:
        from datasets import load_dataset

        stream = load_dataset(dataset, split=dataset_split, streaming=True)
        for record in stream:
            text = _extract_prompt(record, prompt_field)
            yield _encode_prompt(tokenizer, text, chat_template)[:max_prompt_tokens]
        return

    if not prompt_path:
        raise ValueError("either dataset or prompt_path must be set")

    suffix = Path(prompt_path).suffix.lower()
    with Path(prompt_path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if suffix == ".jsonl":
                record = json.loads(line)
                if "prompt_ids" in record:
                    ids = list(record["prompt_ids"])
                else:
                    ids = _encode_prompt(tokenizer, _extract_prompt(record, prompt_field), chat_template)
            else:
                ids = _encode_prompt(tokenizer, line, chat_template)
            yield ids[:max_prompt_tokens]


def block_record(block: BlockTrajectory) -> Dict:
    """Serialize a block trajectory into JSON-friendly primitives."""
    states = block.states
    if isinstance(states, torch.Tensor):
        state_list = states.reshape(states.shape[0], -1).tolist()
    else:
        state_list = [state.reshape(-1).tolist() for state in states]
    return {
        "states": state_list,
        "fixed_point": block.fixed_point.reshape(-1).tolist(),
        "iterations": block.iterations,
    }


def _encode_record(record: Dict) -> bytes:
    if orjson is not None:
        return orjson.dumps(record) + b"\n"
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


def _flush_batch(
    decoder: BatchedJacobiDecoder,
    buffer: List[Tuple[int, torch.Tensor]],
    block_size: int,
    config: CollectConfig,
    handle,
    bar,
) -> int:
    """Decode a length-sorted buffer in batches and write one record per prompt."""
    batch_size = max(1, config.batch_size)
    buffer.sort(key=lambda item: item[1].numel())
    device = next(decoder.model.parameters()).device
    written = 0
    for start in range(0, len(buffer), batch_size):
        chunk = buffer[start : start + batch_size]
        decoder.reseed(config.seed + chunk[0][0])
        trajectories = decoder.decode(
            [item[1].to(device, non_blocking=True) for item in chunk],
            block_size=block_size,
            max_new_tokens=config.max_new_tokens,
        )
        payload = bytearray()
        for (prompt_id, prompt_ids), trajectory in zip(chunk, trajectories):
            record = {
                "prompt_id": prompt_id,
                "block_size": block_size,
                "prompt_ids": prompt_ids.tolist(),
                "blocks": [block_record(block) for block in trajectory.blocks],
            }
            payload.extend(_encode_record(record))
            written += 1
        handle.write(payload)
        handle.flush()
        bar.update(len(chunk))
    return written


def run_collection(config: CollectConfig, device: Optional[torch.device] = None) -> int:
    """Execute phase 1 and return the number of written records."""
    if not config.model_path or not config.output_path:
        raise ValueError("model_path and output_path are required")
    if not config.dataset and not config.prompt_path:
        raise ValueError("either dataset or prompt_path must be set")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(config.model_path, resolve_dtype(config.dtype), config.attention, device)
    decoder = BatchedJacobiDecoder(
        model,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        draft_source=config.draft_source,
        seed=config.seed,
    )

    output_path = Path(config.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buffer_limit = max(1, config.batch_size) * 8
    written = 0
    with output_path.open("wb") as handle:
        for block_size in config.block_sizes:
            stream = load_prompt_stream(
                config.dataset,
                config.dataset_split,
                config.prompt_field,
                config.prompt_path,
                tokenizer,
                config.chat_template,
                config.max_prompt_tokens,
            )
            bar = progress(
                total=config.max_prompts if config.max_prompts else None,
                desc=f"collect n={block_size}",
                unit="prompt",
            )
            buffer: List[Tuple[int, torch.Tensor]] = []
            for index, prompt_ids in enumerate(stream):
                if config.max_prompts and index >= config.max_prompts:
                    break
                if config.shard_count > 1 and index % config.shard_count != config.shard_index:
                    continue
                buffer.append((index, torch.tensor(prompt_ids, dtype=torch.long)))
                if len(buffer) >= buffer_limit:
                    written += _flush_batch(decoder, buffer, int(block_size), config, handle, bar)
                    buffer = []
            if buffer:
                written += _flush_batch(decoder, buffer, int(block_size), config, handle, bar)
            bar.close()

    return written
