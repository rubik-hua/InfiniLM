"""
Build a HuggingFace-style directory that InfiniLM loads as ``minicpm5_moe_fused_stub``.

Copies tokenizer symlinks, rewrites ``config.json``, optionally shrinks depth, then filters
weights to the parameters declared by the stub graph (same MoE layout as ``minicpm5_moe``).

Example::

    python minicpm5_moe_fused_stub_ckpt.py \\
        --src /path/to/minicpm5_moe \\
        --out /tmp/minicpm5_fused_stub_ckpt \\
        --mini-layers 1

Then point ``jiuge.py`` at ``--model-path /tmp/minicpm5_fused_stub_ckpt``. Use
``--attn default`` with the default static KV cache, or
``--enable-paged-attn --attn flash-attn`` (the flash backend expects a paged cache config).
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import shutil
import sys

import torch


def _maybe_load_flash_attn_global() -> None:
    if os.environ.get("INFINILM_DISABLE_FLASH_ATTN_RTLD_GLOBAL") == "1":
        return
    fa = "/usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
    if os.path.isfile(fa):
        ctypes.CDLL(fa, mode=ctypes.RTLD_GLOBAL)


_maybe_load_flash_attn_global()

import infinicore  # noqa: E402

from infinilm.distributed import DistConfig  # noqa: E402
from infinilm.infer_engine import InferEngine  # noqa: E402

_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "configuration_minicpm.py",
    "modeling_minicpm.py",
]


def _load_source_state_dict(src_model_path: str) -> dict[str, torch.Tensor]:
    st_files = sorted(glob.glob(os.path.join(src_model_path, "*.safetensors")))
    if st_files:
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise RuntimeError("safetensors checkpoints require the safetensors package") from e
        out: dict[str, torch.Tensor] = {}
        for p in st_files:
            out.update(load_file(p))
        return out
    bin_path = os.path.join(src_model_path, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, weights_only=True, map_location="cpu")
    raise FileNotFoundError(
        f"No pytorch_model.bin or *.safetensors under {src_model_path!r}"
    )


def _symlink_tokenizer_assets(src_model_path: str, out_dir: str) -> None:
    for fn in _TOKENIZER_FILES:
        s = os.path.join(src_model_path, fn)
        d = os.path.join(out_dir, fn)
        if os.path.lexists(d):
            continue
        if os.path.exists(s):
            os.symlink(os.path.abspath(s), d)


def refine_weights_to_engine_keys(ckpt_dir: str) -> None:
    """Rewrite ``pytorch_model.bin`` in *ckpt_dir* to exactly the keys ``InferEngine`` expects."""
    device = infinicore.device("cuda", 0)
    eng = InferEngine(
        ckpt_dir,
        device=device,
        distributed_config=DistConfig(1),
        attention_backend="default",
    )
    keys = set(eng.state_dict_keyname())
    del eng

    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    sd = torch.load(bin_path, weights_only=True, map_location="cpu")
    filtered = {k: v for k, v in sd.items() if k in keys}
    torch.save(filtered, bin_path)
    infinicore.sync_device()


def prepare_minicpm5_moe_fused_stub_directory(
    src_model_path: str,
    out_dir: str,
    *,
    mini_layers: int = 0,
) -> None:
    """
    Create *out_dir* with ``model_type`` ``minicpm5_moe_fused_stub`` and loadable weights.

    If *mini_layers* > 0 and smaller than the source depth, keep only weights for
    ``model.layers.0`` .. ``model.layers.{mini_layers-1}`` plus embeddings / final norm / lm_head.
    """
    src_model_path = os.path.expanduser(src_model_path)
    os.makedirs(out_dir, exist_ok=True)
    _symlink_tokenizer_assets(src_model_path, out_dir)

    cfg_path = os.path.join(src_model_path, "config.json")
    cfg = json.load(open(cfg_path))
    full_layers = int(cfg.get("num_hidden_layers", 0))
    if mini_layers and mini_layers < full_layers:
        cfg["num_hidden_layers"] = int(mini_layers)
    elif mini_layers and mini_layers == full_layers:
        mini_layers = 0

    cfg["model_type"] = "minicpm5_moe_fused_stub"
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)

    sd = _load_source_state_dict(src_model_path)
    if mini_layers:
        prefixes = [
            "model.embed_tokens.",
            "model.norm.",
            "lm_head.",
        ] + [f"model.layers.{i}." for i in range(int(cfg["num_hidden_layers"]))]
        sd = {k: v for k, v in sd.items() if any(k.startswith(p) for p in prefixes)}

    torch.save(sd, os.path.join(out_dir, "pytorch_model.bin"))
    refine_weights_to_engine_keys(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Source MiniCPM5 MoE HF directory")
    ap.add_argument("--out", required=True, help="Output directory (created)")
    ap.add_argument(
        "--mini-layers",
        type=int,
        default=0,
        help=(
            "If >0 and < source depth, keep only this many decoder layers (fast load/logit checks; "
            "text from jiuge.py will not match a full model). Use 0 for normal generation."
        ),
    )
    args = ap.parse_args()
    if os.path.exists(args.out) and os.listdir(args.out):
        # Avoid mixing stale config/weights with a partial tree
        shutil.rmtree(args.out)
    prepare_minicpm5_moe_fused_stub_directory(
        args.src,
        args.out,
        mini_layers=args.mini_layers,
    )
    print(f"Wrote fused-stub checkpoint to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
