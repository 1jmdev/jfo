# JFO — Optimized Jacobi Forcing

`jfo` converts a pretrained autoregressive model into a causal parallel decoder.
It replaces the original Jacobi Forcing two-round distillation schedule with
SVD-aligned subspace training: the backbone is frozen and only a tiny `r x r`
residual is trained inside the leading singular subspace of each projection.

Reference paper: [Fast and Accurate Causal Parallel Decoding using Jacobi Forcing](https://arxiv.org/abs/2512.14681) (arXiv:2512.14681).

## Requirements

- Python 3.10+
- One CUDA GPU with bfloat16 support
- Internet access on the first run, to download the base model and the prompt dataset

## Install

```bash
pip install .
```

All dependencies are installed by this command. Nothing else needs to be set up.

## Run

Run the phases in order from the project root. Every value has a built-in
default, so no arguments are required. The first phase downloads the base model
`Qwen/Qwen2.5-0.5B-Instruct` and 50,000 prompts from
`ise-uiuc/Magicoder-OSS-Instruct-75K` automatically.

```bash
# 1. Download prompts and decode Jacobi trajectories. Writes data/trajectories.jsonl
jfo collect

# 2. Map the progressive noise schedule onto packed training sequences. Writes data/packed.jsonl
jfo pack

# 3. Train the subspace residuals with mixed block sizes. Writes runs/jfo_qwen2_5_0_5b/subspace.pt
jfo train

# 4. Fold the residuals into dense weights. Writes runs/jfo_qwen2_5_0_5b/merged
jfo merge

# 5. Measure Jacobi decoding throughput on the merged model.
jfo validate
```

`jfo collect` is the slow phase. If you want a quick end-to-end smoke test
first, limit it to a few hundred prompts:

```bash
jfo collect --max-prompts 500
```

## Configuration

All defaults live in `configs/default.yaml`. Pass the file to keep one source of
truth, or override single values directly:

```bash
jfo train --config configs/default.yaml --rank 16 --max-steps 2000
```

To use a different prompt source, change the dataset or point at a local file:

```bash
jfo collect --dataset nvidia/OpenCodeInstruct --prompt-field input --max-prompts 20000
jfo collect --prompt-path /path/to/prompts.jsonl --dataset ""
```

## How it works

- **One trajectory pass.** Jacobi decoding runs once at a reference block size
  and records every noisy draft next to the converged fixed point.
- **Static mixed-block packing.** Fixed points are re-partitioned into training
  blocks, and smaller block sizes are derived from the reference trajectory, so
  `{4, 8, 16, 32}` share a single collection run.
- **SVD-aligned subspace training.** Each projection becomes
  `W = W0 + U_r R V_r^T` with `W0`, `U_r` and `V_r` frozen, so only `r^2` values
  per layer are trained and the optimizer state is negligible.
- **Cheap steps.** One forward pass computes both the progressive consistency
  loss and the autoregressive loss through a noise-aware attention mask, with
  gradient checkpointing and bfloat16 stochastic rounding.

## Citation

```bibtex
@misc{hu2025fastaccuratecausalparallel,
  title={Fast and Accurate Causal Parallel Decoding using Jacobi Forcing},
  author={Lanxiang Hu and Siqi Kou and Yichao Fu and Samyam Rajbhandari and Tajana Rosing and Yuxiong He and Zhijie Deng and Hao Zhang},
  year={2025},
  eprint={2512.14681},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2512.14681},
}
```
