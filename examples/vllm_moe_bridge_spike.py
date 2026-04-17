"""
Spike: evaluate feasibility of reusing vLLM (or torch) MoE kernels from InfiniLM via ATen bridge.

This script is intentionally lightweight:
- It uses InfiniCore <-> torch zero-copy views when InfiniCore is built with --aten=y.
- It measures the overhead of crossing the boundary and (optionally) running a vLLM fused MoE op.

Usage (inside the minicpm5-moe container):

  python3 InfiniLM/examples/vllm_moe_bridge_spike.py

Optional:
- If `vllm` is installed and exposes a fused MoE op, we try to call it.
- Otherwise, we run a small "expert MLP" stand-in in pure torch to measure bridge + torch compute.
"""

from __future__ import annotations

import time

import infinicore
from infinicore.lib import _infinicore


def _require_aten_bridge() -> None:
    if not hasattr(_infinicore, "_tensor_as_torch"):
        raise SystemExit("InfiniCore built without ATen bridge. Rebuild with --aten=y.")


def _maybe_import_vllm():
    try:
        import vllm  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    _require_aten_bridge()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")

    device_index = int((__import__("os").environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"))
    t_dev = torch.device("cuda", device_index)
    ic_dev = infinicore.device("cuda", device_index)

    # Shapes chosen to mimic "decode-ish" (M small) and "prefill-ish" (M larger).
    cases = [
        ("decode_like", 1, 4096),
        ("prefill_like", 256, 4096),
    ]

    have_vllm = _maybe_import_vllm()
    if have_vllm:
        print("vLLM detected: will attempt a fused MoE call if available.")
    else:
        print("vLLM not detected: running torch stand-in only.")

    torch.manual_seed(0)
    dtype = torch.bfloat16

    # Simple torch expert stand-in: (x @ w1) -> silu -> ( @ w2)
    def torch_expert_mlp(x, w1, w2):
        y = torch.nn.functional.linear(x, w1)
        y = torch.nn.functional.silu(y)
        y = torch.nn.functional.linear(y, w2)
        return y

    for name, m, h in cases:
        x_t = torch.randn(m, h, device=t_dev, dtype=dtype)
        w1_t = torch.randn(h * 2, h, device=t_dev, dtype=dtype)
        w2_t = torch.randn(h, h * 2, device=t_dev, dtype=dtype)

        # InfiniCore views (zero-copy) + stream-bridged.
        x_ic = infinicore.from_torch(x_t)
        w1_ic = infinicore.from_torch(w1_t)
        w2_ic = infinicore.from_torch(w2_t)

        # Warm-up bridge + torch path.
        for _ in range(5):
            x_view = infinicore.to_torch(x_ic)
            w1_view = infinicore.to_torch(w1_ic)
            w2_view = infinicore.to_torch(w2_ic)
            y = torch_expert_mlp(x_view, w1_view, w2_view)
            _ = infinicore.from_torch(y)
        torch.cuda.synchronize()

        iters = 50 if name == "decode_like" else 20
        t0 = time.perf_counter()
        for _ in range(iters):
            x_view = infinicore.to_torch(x_ic)
            w1_view = infinicore.to_torch(w1_ic)
            w2_view = infinicore.to_torch(w2_ic)
            y = torch_expert_mlp(x_view, w1_view, w2_view)
            y_ic = infinicore.from_torch(y)
            # Touch y_ic shape to ensure it stays alive.
            _ = y_ic.size(0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000.0 / iters
        print(f"[{name}] bridge + torch-MLP avg: {ms:.3f} ms/iter (m={m}, h={h}, dtype={dtype})")

    print("\nNotes:")
    print("- With the event-based bridge, `to_torch()` / `from_torch()` should no longer do a full stream sync.")
    print("- If vLLM fused MoE is to be reused, it should be called on torch tensors produced by `to_torch()`.")


if __name__ == "__main__":
    main()

