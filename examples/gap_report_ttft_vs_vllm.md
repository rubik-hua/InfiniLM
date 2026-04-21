# TTFT gap report: vLLM vs InfiniLM (MiniCPM5 MoE, tp=1, A100)

## Apples-to-apples baseline runs (prompt=256, new=256, greedy)

### vLLM (Transformers fallback)

- **Metrics JSON**: `InfiniLM/examples/profiles/20260415_104318_vllm_256_log/vllm_metrics.json`
- Key numbers:
  - `prompt_tokens=256`, `n_generated=256`
  - **TTFT**: `ttft_ms=159.997`
  - **Prefill engine time** (scheduled → first token): `prefill_engine_ms=144.237`
  - Decode avg ITL: `avg_decode_itl_ms=51.519`

### InfiniLM (paged-attn)

- **Metrics JSON**: `InfiniLM/examples/profiles/20260415_103820_infinilm_breakdown/bench_infinilm.json`
- Key numbers:
  - `prompt_tokens_actual=256`, `max_new_tokens=256`
  - **TTFT**: `ttft_ms=9487.461`
  - **TTFT breakdown**:
    - `ttft_cpu_prep_ms=1.048`
    - `ttft_gpu_forward_ms=9485.334`
    - `ttft_gpu_sampling_ms=0.102`
    - `ttft_gpu_d2h_ms=0.019`
    - `ttft_unaccounted_ms=0.957`
  - Decode avg ITL: `avg_decode_itl_ms=47.673`

## Highest-impact gaps (TTFT drivers)

### 1) Prefill forward path dominates InfiniLM TTFT (not CPU prep, not D2H)

- Evidence: InfiniLM TTFT is ~9.49s and is almost entirely **GPU forward** (~9.49s) with negligible CPU prep (~1ms) and negligible first-token D2H (~0.02ms).
- Implication: focus should be on **prefill GPU kernels** (MoE routing/dispatch + expert MLP compute + projection/GEMV selection), not Python overhead or host sync.

### 2) MoE implementation differences (routing/experts dispatch)

- vLLM is running MiniCPM5 MoE via **`TransformersMoEForCausalLM` fallback** and uses fused MoE helpers internally.
- InfiniLM is executing its own MoE implementation (MiniCPM5 MoE C++ path) where prefill forward is currently very expensive.
- Evidence (vLLM engine logs from the repo’s earlier run; same vLLM version/config family):
  - `Resolved architecture: TransformersMoEForCausalLM`
  - `Using TRITON backend for Unquantized MoE`
  - vLLM also warns it is a fallback path and “performance may not be optimal”, but it is still orders of magnitude faster on prefill TTFT vs current InfiniLM.
  - Source: `InfiniLM/examples/profiles/20260414_095106_vllm/vllm_metrics.json`
- Implication: the biggest TTFT win for InfiniLM is likely **moving MoE routing/dispatch fully on-device** and executing experts as **grouped GEMM/fused experts**, minimizing per-expert loops/launches and avoiding GEMV-heavy paths.

### 3) Attention backend is likely not the TTFT bottleneck right now

- vLLM chooses FlashAttention 2 backend in the earlier log:
  - `Using FLASH_ATTN attention backend ...`
  - `Using FlashAttention version 2`
  - Source: `InfiniLM/examples/profiles/20260414_095106_vllm/vllm_metrics.json`
- InfiniLM run here uses `attn_backend=paged-attn`.
- Given InfiniLM TTFT is dominated by forward time and prior kernel summaries showed GEMV dominance, attention backend parity is probably **secondary** until MoE/projection kernels are fixed.

### 4) Prefill scheduling / chunked prefill / prefix cache

- vLLM enables:
  - **Chunked prefill**: `Chunked prefill is enabled ...` (log)
  - **Asynchronous scheduling** (log)
  - **Prefix caching**: `enable_prefix_caching=True` in engine config (log)
  - Source: `InfiniLM/examples/profiles/20260414_095106_vllm/vllm_metrics.json`
- For the single-request prompt=256 case, these features can reduce overhead but are unlikely to explain a ~\(9.3\) second TTFT gap by themselves; they matter more for multi-request throughput and long prompts.
- Still, InfiniLM currently does not appear to have comparable **chunked prefill** behavior exposed at the engine level.

### 5) Graph / compile / CUDAGraphs

- vLLM run is in eager mode (`--enforce-eager`), so it explicitly disables torch.compile/CUDAGraphs in the log.
- InfiniLM `bench_balanced.py` currently sets `enable_graph_compiling=False`.
- Since vLLM is *also* not using graphs here, graphs are not a plausible explanation for the large TTFT difference in this configuration.

## Concrete next actions (scoped)

1. **MoE prefill speed** (primary):
   - Implement/enable on-device token→expert routing and grouped expert execution.
   - Re-profile TTFT breakdown; success means `ttft_gpu_forward_ms` drops materially.

2. **Projection/GEMV selection** (secondary after MoE):
   - After MoE improvements, re-run `nsys` for prefill and confirm GEMV is no longer dominating.

3. **Engine features parity** (nice-to-have for TTFT; more for throughput):
   - Evaluate adding chunked prefill and/or prefix cache pathways for repeated prompts / multi-request workloads.

