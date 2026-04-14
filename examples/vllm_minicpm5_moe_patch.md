# MiniCPM5 MoE on vLLM — patch notes (A1: `sitecustomize`)

This repo uses an **A1 patch** to make the MiniCPM5 MoE checkpoint work with **vLLM 0.19** running via the **Transformers fallback** (`TransformersMoEForCausalLM`).

## Why a patch is needed

When vLLM runs MoE through the Transformers backend, it may replace the model’s expert container with a fused module:

- Model code expects: `self.experts` is an `nn.ModuleList` (supports `len(...)`, indexing, loops).
- vLLM provides: `self.experts` becomes a fused module (e.g. `TransformersFusedMoE`) that **does not implement list semantics**.

MiniCPM5’s remote `modeling_minicpm.py` calls:

- `len(self.experts)` for `one_hot(..., num_classes=len(self.experts))`
- loops over `range(len(self.experts))`

This crashes with:

- `TypeError: object of type 'TransformersFusedMoE' has no len()`

## What the patch does

We **monkeypatch** the remote module **at import time** so that `MiniCPM5MoEMoE.moe()` becomes:

- If `self.experts` is **not** an `nn.ModuleList`, call the fused module directly:
  - `self.experts(hidden_states, topk_indices, topk_weights)`
- Otherwise, fall back to the original Python loop implementation.

This keeps the change minimal and localized (Option A spirit), but makes it robust against the remote-code cache being regenerated.

## Where it lives

- `InfiniLM/examples/vllm_patches/sitecustomize.py`

Python auto-imports `sitecustomize` **if it is discoverable on `PYTHONPATH`**.

## How to enable it (required for MiniCPM5 MoE on vLLM 0.19)

Inside the container (or any environment where you run vLLM), activate the vLLM venv and add the patch dir to `PYTHONPATH`:

```bash
REPO=/home/zenghua/workspace/minicpm5-moe-support
source "$REPO/.venv-vllm/bin/activate"
export PYTHONPATH="$REPO/InfiniLM/examples/vllm_patches:${PYTHONPATH:-}"
```

**Important:** vLLM uses multiprocessing workers (`spawn`). The patch must be enabled in the **parent environment** so workers inherit the same `PYTHONPATH`.

## How to disable it

Just remove that path from `PYTHONPATH` (or run in a clean shell without it). Nothing is installed into the environment.

## What this patch does NOT solve

- It does **not** make MiniCPM5 “native” in vLLM; you are still using `TransformersMoEForCausalLM` fallback.
- Performance may be sub-optimal vs a dedicated vLLM model implementation.
- You may still need `--enforce-eager` to avoid `torch.compile`/cudagraph issues with remote code.

## Suggested run command (single prompt metrics)

```bash
REPO=/home/zenghua/workspace/minicpm5-moe-support
source "$REPO/.venv-vllm/bin/activate"
export PYTHONPATH="$REPO/InfiniLM/examples/vllm_patches:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

python -u "$REPO/InfiniLM/examples/vllm_bench_match_jiuge.py" \
  --model-path /data-aisoft/zenghua/models/minicpm5.16a3.v0314 \
  --prompt "Hi" --max-new-tokens 16 \
  --max-model-len 512 --gpu-memory-utilization 0.85 \
  --enforce-eager --json
```

