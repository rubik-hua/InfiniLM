## vLLM kernel reuse evaluation (InfiniLM / MiniCPM5 MoE)

### Current state in this repo

- **InfiniCore <-> torch views exist** when built with `--aten=y`:
  - `infinicore.from_torch(torch_tensor)` creates an InfiniCore view on the same storage.
  - `infinicore.to_torch(infinicore_tensor)` creates a torch view on the same storage.
- **MoE block is correctness-first and CPU-heavy** today: `InfiniLM/csrc/models/minicpm5_moe/minicpm5_moe_sparse_moe_block.cpp` routes on CPU, loops tokens, and copies per-token outputs back to CPU.

### The impedance: stream semantics (and what we changed)

Reusing torch/vLLM ops safely requires correct CUDA stream ordering in both directions:

- **InfiniCore -> torch**: torch must not read an aliasing view before InfiniCore kernels finish.
- **torch -> InfiniCore**: InfiniCore must not read an aliasing view before torch kernels finish.

To avoid a global sync boundary, we implement **event-based stream bridging**:

- `to_torch()` now records an event on the InfiniCore context stream and makes the **current torch stream** wait on it.
- `from_torch()` (when the ATen bridge is present) records an event on the **current torch stream** and makes the InfiniCore context stream wait on it.

This clears the most important “framework boundary” hazard for a torch-bridge spike without introducing unconditional synchronizes.

### Option A — Vendor / port CUDA (vLLM fused MoE into InfiniCore)

- **Pros**: keeps hot path inside the existing C++ engine; avoids torch runtime overhead; best for **per-token decode**.
- **Cons**: more engineering + ongoing maintenance; must match layouts, quantization, and router behavior.
- **Verdict**: **Best long-term** if the goal is decode ITL.

### Option B — Torch subgraph bridge (call vLLM fused MoE via `to_torch()` / `from_torch()`)

- **Pros**: fastest path to prototype; can reuse vLLM implementations directly.
- **Cons**: still introduces boundary overhead; may be fine for **prefill** or batched layers but risky for **decode** unless batching is substantial.
- **Verdict**: **Viable for a spike now** because the stream impedance is addressed; must micro-benchmark.

### Option C — Shared building blocks (CUTLASS/Triton/FlashInfer patterns without vLLM)

- **Pros**: performance alignment without tight coupling to vLLM.
- **Cons**: still non-trivial engineering; not a drop-in.
- **Verdict**: sensible middle ground if full vendor is too heavy.

### Option D — Native vLLM model

- **Pros**: best performance in vLLM’s serving stack; gets paged-attn + fused MoE end-to-end.
- **Cons**: separate product path from InfiniLM engine; larger integration.
- **Verdict**: strategic for serving, not a direct InfiniLM optimization unless the target runtime changes.

### Spike artifact

- `InfiniLM/examples/vllm_moe_bridge_spike.py`: microbench to quantify bridge + torch compute overhead on decode-like and prefill-like shapes.

**InfiniCore bridge + tests:** `InfiniCore/python/infinicore/vllm_fused_moe_bridge.py` (`fused_experts_ic` via `to_torch` / `from_torch`); operator test `InfiniCore/test/infinicore/ops/vllm_fused_experts_bridge.py` (run `python run.py --ops vllm_fused_experts_bridge --nvidia` from `InfiniCore/test/infinicore`, with vLLM in the **same** interpreter as `_infinicore`). Optional pip extra: `InfiniCore/pyproject.toml` `[project.optional-dependencies] vllm`.

**InfiniLM stub:** `InfiniLM/csrc/models/minicpm5_moe_fused_stub/` registers `model_type` **`minicpm5_moe_fused_stub`**. The MoE path uses **`MiniCPM5MoeVllmFusedSparseMoeBlock`**: HF-aligned CPU routing, then **`infinicore.vllm_fused_moe_bridge.fused_experts_ic`** (via `InfiniLM/csrc/utils/vllm_fused_moe_dispatch.cpp`, GIL + PyBind) when vLLM/ATen are importable; otherwise it falls back to the per-expert C++ loop in `MiniCPM5MoeSparseMoeBlock`. Set **`INFINILM_DISABLE_VLLM_FUSED_MOE=1`** to force the reference path. Use full depth (`--mini-layers 0` in `minicpm5_moe_fused_stub_ckpt.py`) for sensible `jiuge.py` output.

### Python environments and PyTorch ABI (InfiniCore bridge vs vLLM venv)

This repo intentionally keeps **two Python stacks** (see `minicpm5_moe_inference_profiling.md` and `setup_vllm_venv.sh`):

| Interpreter | Role |
|-------------|------|
| Container **system** `python3` | InfiniLM (`jiuge.py`), HF / `logit_sanity_minicpm5_moe.py`, InfiniCore — pin **`transformers==4.57.1`** for checkpoint parity. |
| **`$REPO/.venv-vllm`** | **`vllm==0.19.0`**; use **`setup_vllm_venv.sh --moe`** if you need **`transformers>=5`** for `TransformersMoEForCausalLM` fallback. **Do not** install vLLM into the system interpreter used for InfiniLM. |

**Why a single process cannot casually mix both:**

- InfiniCore’s `_infinicore` / `libinfinicore_cpp_api.so` is built with **`--aten=y`** against the **system** PyTorch headers/libs (`torch::` / `libtorch`).
- The venv installs its **own** PyTorch build for vLLM. Loading **system** `infinicore` and **venv** `vllm` in one interpreter risks duplicate `libtorch`, pybind11 type confusion, and CUDA custom-op registration clashes.

**Practical options:**

1. **Split env (default)** — InfiniLM + `to_torch` / `from_torch` on **system** torch; vLLM + `fused_experts` spike on **venv** torch only (plain `torch.Tensor`, no InfiniCore in that process). Validate kernels and packing here; integrate across processes only if needed (IPC, second-stage server, etc.).
2. **Second InfiniCore build** — Re-run `InfiniCore/scripts/install.py` / `xmake` with **`venv`’s `python`** and the same `--aten=y` flags so `_infinicore` links **venv** `libtorch`. Then one process can call `to_torch` → `torch.ops.vllm.*` / `fused_experts`. You now **maintain two artifacts**: system-torch build for `jiuge.py` and venv-torch build for the bridge experiment.
3. **Pure-torch spike in venv** — No InfiniCore; exercise `fused_experts` / weight layout only. Fastest for Triton path and shapes; no ABI coupling.

**Linker note:** Prefer the **non-HPCX** `LD_LIBRARY_PATH` layout from the workspace perf rule when running InfiniLM + flash-attn; avoid prepending partial `/opt/hpcx` trees. Use **`ctypes.CDLL(..., RTLD_GLOBAL)`** on `flash_attn_2_cuda*.so` (as in `jiuge.py` / `logit_sanity_minicpm5_moe.py`) instead of `LD_PRELOAD` + HPCX ordering.

### CPU-only MoE baseline (git reference)

For a **trusted CPU MoE accumulation path** without experimental batched GPU dispatch (`INFINILM_MOE_USE_BATCHED_DISPATCH`, `index_add_rows_f32`, etc.), the known-good **InfiniLM** tree is commit **`3c4fe49`** for:

- `csrc/models/minicpm5_moe/minicpm5_moe_sparse_moe_block.cpp`
- `csrc/models/minicpm5_moe/minicpm5_moe_sparse_moe_block.hpp`
- `xmake.lua` (`_infinilm` target without `add_rules("cuda")` / `csrc/**.cu`)

**InfiniCore:** no separate revert was required beyond matching the committed branch (discard local tensor/op experiments if any). Rebuild `_infinicore` and `_infinilm`, then smoke with `jiuge.py --nvidia` and `logit_sanity_minicpm5_moe.py` (requires **`INFINILM_RETURN_LAST_LOGITS=1`** so `RankWorker` publishes last-token logits).

