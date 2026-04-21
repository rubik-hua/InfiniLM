# Profiling progress log (InfiniLM vs vLLM)

This document records **ongoing profiling progress** and decisions. It complements `minicpm5_moe_inference_profiling.md` (which is more “how to run”).

---

## Environment (fill per session)

- **date**:
- **host/container**:
- **GPU**: (model, SM, driver)
- **CUDA_VISIBLE_DEVICES**:
- **model**:
- **transformers**:
- **infinicore / infinilm build**: (commit/tag, build flags if relevant)
- **attention backend**: `static-attn` | `paged-attn` | `flash-attn`
- **paged KV**: (enabled? block_size?)
- **workload**: prompt≈256, max_new_tokens≈256, greedy (top_k=1, top_p=1, temp=1) unless noted

---

## Current status (executive summary)

- **InfiniLM bottleneck hypothesis (current)**:
  - Python-side per-token metadata construction is allocation-heavy (`from_list()` → NumPy → CPU tensor → H2D).
  - Per-token D2H + `syncStream()` after sampling adds hard synchronization.
  - MiniCPM5 MoE block is still correctness-first CPU-style routing/dispatch (dominates runtime).
- **Top 3 expected wins**:
  - Replace MiniCPM5 MoE block with on-device routing + batched experts.
  - Remove per-token `from_list()` from the decode loop (build metadata on-device / in C++).
  - Remove unconditional per-token `syncStream()` and avoid D2H copies on the critical path.

---

## Runs (append-only)

Add one row per run (or link to JSON artifacts if produced by `bench_balanced.py`).


| date | engine | model | attn | paged | prompt | new | TTFT ms | decode ITL ms | notes |
| ---- | ------ | ----- | ---- | ----- | ------ | --- | ------- | ------------- | ----- |
| 2026-04-15 | infinilm | minicpm5.16a3.v0314 | paged-attn | yes (block=256) | 256 | 256 | 9651.83 | 48.72 | rerun `bench_balanced.py`; artifact `InfiniLM/examples/profiles/20260415_093442_rerun/bench_infinilm.json` |


---

## Findings (by category)

### CPU prep / Python overhead

- **Observed**:
  - `InferEngine.generate()` creates `position_ids`, `past_kv_lengths`, `total_kv_lengths`, `cu_seqlens`, `input_offsets`, and (paged) `slot_mapping` every token via `infinicore.from_list(...)`.
- **Why it matters**:
  - `from_list()` always builds a NumPy array on CPU and then copies (CPU → GPU). This adds per-token Python+NumPy overhead and host↔device traffic.
- **Patch candidates**:
  - Move these metadata tensors to C++ and update in-place on device.
  - Provide device-side constructors (e.g., `arange` / `full`) to avoid NumPy.

### Sync points / host↔device copies

- **Observed**:
  - `RankWorker` converts sampled `output_ids` to CPU and then calls `infinicore::context::syncStream()` every token.
- **Patch candidates**:
  - Keep sampled ids on device and only synchronize when the host must consume them.
  - Reduce lock scope so compute doesn’t run under the worker mutex.

### Attention kernel selection

- **Observed**:
  - `AttentionLayer` dispatches explicitly by `attention_backend` (STATIC/PAGED/FLASH).
  - Flash backend uses `mha_varlen`_ for prefill and `mha_kvcache` for decode in paged mode.
- **Next validation**:
  - Confirm the benchmark run is actually using `flash-attn` (or `paged-attn`) as intended (log once at init and/or add NVTX ranges later).

### MoE routing / experts

- **Observed**:
  - `MiniCPM5MoeSparseMoeBlock` is correctness-first and CPU-heavy:
    - router logits copied to CPU
    - routing/topk/grouping on CPU
    - expert outputs copied back to CPU and accumulated in scalar loops
- **Patch candidates**:
  - Use `infinicore::op::topkrouter` to compute top-k routing on device.
  - Batch tokens by expert, run expert MLPs on packed batches, scatter-add back on device.

---

## Patch backlog (live)

Keep this list short; link to PRs/branches when they exist.

- **P0**: Fast MiniCPM5 MoE block (on-device routing + batched experts)
- **P0**: Remove per-token `from_list()` allocations (device-side metadata)
- **P1**: Remove unconditional per-token `syncStream()` + minimize D2H
- **P1**: Narrow `RankWorker` mutex scope around output publication only
- **P2**: Add stable NVTX ranges for `forward`, `attention`, `moe`, `sampling`

---

## Notes / gotchas

- Keep vLLM experiments isolated in `.venv-vllm` as documented in `minicpm5_moe_inference_profiling.md`.
- When comparing runs, keep prompt/template and decoding parameters fixed.

