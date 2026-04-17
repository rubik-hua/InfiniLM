#!/usr/bin/env python3
"""
Spike: benchmark HF-style MoE scatter (index_add) using torch vs InfiniCore matmul path.

Requires InfiniCore with --aten=y for ``infinicore.to_torch``.

Example:
  CUDA_VISIBLE_DEVICES=0 python3 InfiniLM/examples/moe_aten_spike.py
"""

from __future__ import annotations

import os
import time

import torch


def main() -> None:
    import infinicore
    from infinicore import sync_stream
    from infinicore.lib import _infinicore as _ic

    if not hasattr(_ic, "_tensor_as_torch"):
        raise SystemExit(
            "Install InfiniCore with ATen (e.g. install.py --aten=y) for to_torch / this spike."
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dev = torch.device("cuda", 0)
    ic_dev = infinicore.device("cuda", 0)

    n = 64
    h = 256
    iters = 200

    hidden = torch.randn(n, h, device=dev, dtype=torch.bfloat16)
    expert_w = torch.randn(h, h, device=dev, dtype=torch.bfloat16)

    # Torch baseline: batched GEMM
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = hidden @ expert_w
    torch.cuda.synchronize()
    dt_torch = (time.perf_counter() - t0) / iters * 1000

    h_ic = infinicore.from_torch(hidden)
    w_ic = infinicore.from_torch(expert_w)
    t0 = time.perf_counter()
    for _ in range(iters):
        y_ic = infinicore.matmul(h_ic, w_ic)
    sync_stream()
    dt_ic = (time.perf_counter() - t0) / iters * 1000

    y_t = infinicore.to_torch(y_ic)
    torch.cuda.synchronize()
    ref = hidden @ expert_w
    ok = torch.allclose(y_t.float(), ref.float(), rtol=0.05, atol=0.05)

    print(f"matmul bf16 [{n}x{h}] @ [{h}x{h}]: torch {dt_torch:.3f} ms/it  infinicore {dt_ic:.3f} ms/it")
    print(f"to_torch allclose vs torch matmul: {ok}")


if __name__ == "__main__":
    main()
