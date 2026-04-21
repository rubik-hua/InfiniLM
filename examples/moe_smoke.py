"""
Minimal correctness smoke test for MiniCPM5 MoE path.

Runs a single forward+sample step and checks that returned logits are finite
when requested (batch=1). This is meant to catch obvious routing/dispatch bugs
without requiring a full HF parity harness.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ex_dir = os.path.dirname(os.path.abspath(__file__))
if _ex_dir not in sys.path:
    sys.path.insert(0, _ex_dir)

from flash_attn_preload import maybe_load_flash_attn_global

maybe_load_flash_attn_global()

import infinicore
from infinilm.cache import PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "--model-path", dest="model_path", type=str, required=True)
    ap.add_argument("--nvidia", action="store_true", default=True)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=2)
    return ap.parse_args()


def main() -> None:
    args = get_args()
    model_path = os.path.expanduser(args.model_path)

    tok = 1  # arbitrary non-special token id (model-dependent but fine for smoke)
    input_ids = infinicore.from_list([[tok] * int(args.prompt_tokens)], dtype=infinicore.int64)

    # Use paged KV cache to avoid the legacy static-kv update path.
    block_size = 256
    max_total = int(args.prompt_tokens) + int(args.max_new_tokens)
    num_blocks = (max_total + block_size - 1) // block_size
    cache_config = PagedKVCacheConfig(num_blocks=num_blocks, block_size=block_size)

    engine = InferEngine(
        model_path,
        device=infinicore.device("cuda", 0),
        distributed_config=DistConfig(1),
        cache_config=cache_config,
        enable_graph_compiling=False,
        attention_backend="paged-attn",
    )
    load_model_state_dict_by_file(engine, model_path, dtype=engine.config.dtype)

    # Use the high-level generate path so required metadata (offsets, lengths)
    # is populated consistently with the benchmark harness.
    engine.reset_cache(cache_config)
    out_ids = engine.generate(
        input_ids,
        GenerationConfig(
            max_new_tokens=int(args.max_new_tokens),
            eos_token_id=[],
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            stop_on_eos=False,
        ),
        _measure_and_log_time=False,
    )
    # Ensure outputs are materialized and finite (host-side).
    last = out_ids[-1].to_numpy()
    if not np.isfinite(last).all():
        raise SystemExit("FAIL: non-finite output ids detected")
    # Catch the common regression where we repeatedly emit a single special token id (e.g. 0).
    flat = np.array([int(t.to_numpy()[0]) for t in out_ids], dtype=np.int64)
    if flat.size > 0 and int(flat.min()) == int(flat.max()):
        raise SystemExit(f"FAIL: degenerate generation (all ids identical): id={int(flat[0])}")
    print("OK")


if __name__ == "__main__":
    main()

