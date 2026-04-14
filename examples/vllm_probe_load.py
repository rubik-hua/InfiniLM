#!/usr/bin/env python3
"""Minimal vLLM `LLM(...)` load probe. Must be a real file: vLLM workers use spawn and re-exec this module.

Run from an isolated `$REPO/.venv-vllm` so system Python keeps HF parity (`transformers==4.57.1`).
MiniCPM5 MoE uses vLLM's Transformers fallback (`TransformersMoEForCausalLM`); see profiling doc."""
import argparse
import sys
import traceback


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile / CUDA graphs (often needed for remote-code MoE).",
    )
    args = p.parse_args()

    try:
        from vllm import LLM

        LLM(
            model=args.model_path,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
        )
        print("LOAD_OK", flush=True)
        return 0
    except Exception:
        print("LOAD_FAIL", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
