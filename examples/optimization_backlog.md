# Optimization backlog (compute-first)

This backlog is based on the balanced single-request run:

- InfiniLM `bench_balanced.py` (paged-attn): prompt=256, new=256, greedy
- Trace: `InfiniLM/examples/profiles/20260414_092627_paged/infinilm_balanced.nsys-rep`
- Kernel summaries: `stats_cuda_gpu_kern_sum.csv`, `stats_nvtx_kern_sum.csv`

## Key observations

- **Decode is dominated by tiny GEMV kernels**: `internal::gemvx::kernel<...bf16...>` accounts for **~77% of total GPU kernel time** and is launched **~797k times** in this run.
  - This strongly suggests **launch overhead + poor arithmetic intensity** during per-token decode (many small matvecs instead of larger matmuls).
- **Paged attention decode kernels are not the bottleneck** in this run:
  - `op::paged_attention::nvidia::flashAttentionDecodeHd128SplitKvCta` is **~2.4%** of GPU kernel time.
  - `...SplitKvCombine` is **~0.5%**.
- **Python-side CPU “prep” is small (~0.9 ms/step)** compared to GPU forward time (~56–60 ms/step), so the next big wins are in the **model compute / kernel strategy**, not Python list building.

## P0 (largest expected decode ITL wins)

1. **Eliminate GEMV-for-decode: switch decode linear path to batched GEMM**
   - **Why**: `gemvx` is the top kernel by a large margin and the launch count is extreme.
   - **Approach**:
     - Audit where GEMV is selected (likely “M=1” matmul/linear fast-paths).
     - Prefer a **single fused/packed GEMM** per linear (or per block) over many GEMVs:
       - batch over heads / channels where possible
       - batch multiple projections (Q/K/V, gate) when shapes align
     - If the underlying library has both GEMV and GEMM backends, add a policy:
       - “**never GEMV on GPU for decode** unless explicitly requested”
   - **Expected impact**:
     - **Decode ITL**: large reduction (target: from ~61 ms/token toward the 15–30 ms/token range depending on MoE cost).
     - **TTFT**: modest-to-large improvement if prefill also uses GEMV-heavy paths for small batch.

2. **Fix MoE execution strategy (current correctness-first implementation)**
   - **Why**: `MiniCPM5MoeSparseMoeBlock::forward` is explicitly CPU-heavy (router logits to CPU, per-token loops).
   - **Approach**:
     - Keep routing/top-k on device (use an on-device top-k router; InfiniCore already has `topkrouter` op).
     - Pack tokens by expert and use **grouped GEMM** for expert MLPs.
     - Avoid per-token host loops and repeated host↔device transfers.
   - **Expected impact**:
     - **TTFT**: large (prefill traverses many MoE layers).
     - **Decode ITL**: large if routing/expert execution is still on the critical path per token.

## P1 (medium wins / latency stability)

3. **Remove unconditional per-step stream sync in `RankWorker`**
   - **Where**: `InfiniLM/csrc/engine/rank_worker.cpp` calls `syncStream()` every step after sampling.
   - **Approach**:
     - Only synchronize when the host must consume results.
     - Consider returning device output ids and deferring D2H conversion.
   - **Expected impact**:
     - **Decode ITL**: medium (depends on overlap/async behavior), also reduces jitter.

4. **Reduce per-token tensor allocations / conversions**
   - **Where**: `InferEngine.generate()` creates several small metadata tensors each step (`from_list`).
   - **Approach**:
     - Pre-allocate device tensors for metadata and update in-place.
     - Cache `slot_mapping` patterns for decode steps where they follow a deterministic stride.
   - **Expected impact**:
     - **Decode ITL**: small-to-medium (current prep is ~0.9 ms/step; still worthwhile once GPU work is reduced).

## P2 (trace/observability + correctness guardrails)

5. **Make NVTX ranges show up consistently**
   - **Why**: NVTX range summaries are not yet reflecting the intended high-level ranges.
   - **Approach**:
     - Ensure NVTX is actually enabled in the build and that ranges are recorded on the thread executing the work.
     - Add ranges around attention backend calls and MoE routing/expert blocks.

6. **Stabilize flash-attn path for InfiniCore tensors**
   - **Why**: in this environment, flash-attn prefill/decode via torch extensions can fail with “Tensor doesn’t have storage”.
   - **Approach**:
     - Either ensure the tensors passed into flash-attn ops are torch-backed (ATen adaptor), or keep using InfiniCore native paged-attn kernels for now.

