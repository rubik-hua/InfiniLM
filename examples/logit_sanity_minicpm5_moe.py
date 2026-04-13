import argparse
import os
import json
import tempfile

import numpy as np


def _hpcx_flash_env():
    # Keep this script self-contained for the container.
    torch_lib = "/usr/local/lib/python3.12/dist-packages/torch/lib"
    fa = "/usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [
            "/opt/hpcx/ucc/lib",
            "/opt/hpcx/ucx/lib",
            "/opt/hpcx/ompi/lib",
            "/root/.infini/lib",
            torch_lib,
            "/usr/local/lib/python3.12/dist-packages",
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/lib/x86_64-linux-gnu",
        ]
    )
    os.environ["LD_PRELOAD"] = fa


def topk(x: np.ndarray, k: int):
    idx = np.argpartition(-x, k)[:k]
    idx = idx[np.argsort(-x[idx])]
    return idx, x[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompt", default="Hi")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--mini-layers", type=int, default=1)
    args = ap.parse_args()

    _hpcx_flash_env()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import infinicore
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.cache import StaticKVCacheConfig
    from infinilm.modeling_utils import load_model_state_dict_by_file

    model_path = os.path.expanduser(args.model_path)

    # Create a temporary \"mini\" checkpoint with fewer layers to isolate layer-0 correctness.
    # This keeps both HF and InfiniLM constructing the same depth.
    if args.mini_layers is not None and args.mini_layers > 0:
        tmpdir = tempfile.mkdtemp(prefix="minicpm5_moe_mini_")
        mini_path = os.path.join(tmpdir, "ckpt")
        os.makedirs(mini_path, exist_ok=True)

        # Copy/symlink tokenizer assets.
        for fn in [
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "configuration_minicpm.py",
            "modeling_minicpm.py",
        ]:
            src = os.path.join(model_path, fn)
            dst = os.path.join(mini_path, fn)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

        cfg = json.load(open(os.path.join(model_path, "config.json")))
        cfg["num_hidden_layers"] = int(args.mini_layers)
        json.dump(cfg, open(os.path.join(mini_path, "config.json"), "w"), indent=2)

        # Filter weights to layer0 + embeddings + norm + lm_head.
        full_sd = torch.load(os.path.join(model_path, "pytorch_model.bin"), weights_only=True, map_location="cpu")
        keep_prefixes = (
            "model.embed_tokens.",
            "model.layers.0.",
            "model.norm.",
            "lm_head.",
        )
        mini_sd = {k: v for k, v in full_sd.items() if k.startswith(keep_prefixes)}
        torch.save(mini_sd, os.path.join(mini_path, "pytorch_model.bin"))
        model_path = mini_path

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text = tok.apply_chat_template(
        conversation=[{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    enc = tok(text, return_tensors="pt")
    input_ids_pt = enc["input_ids"]

    hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda").eval()

    with torch.no_grad():
        out = hf(input_ids=input_ids_pt.to("cuda"))
        hf_logits = out.logits[0, -1].float().cpu().numpy()

    device = infinicore.device("cuda", 0)
    eng = InferEngine(
        model_path,
        device=device,
        distributed_config=DistConfig(1),
        attention_backend="default",
    )
    load_model_state_dict_by_file(eng, model_path, dtype=eng.config.dtype)

    input_ids_inf = infinicore.from_list(input_ids_pt.cpu().tolist(), dtype=infinicore.int64)
    # Use engine.forward (single step) to fetch logits + sampled id.
    # Provide offsets so RankWorker can pick last position.
    bsz, seqlen = input_ids_pt.shape
    # Ensure KV cache exists for attention backends.
    eng.reset_cache(StaticKVCacheConfig(max_batch_size=int(bsz), max_cache_len=int(seqlen)))
    pos = infinicore.from_list([list(range(seqlen))], dtype=infinicore.int64)
    past = infinicore.from_list([0], dtype=infinicore.int32)
    total = infinicore.from_list([seqlen], dtype=infinicore.int32)
    cu = infinicore.from_list([0, seqlen], dtype=infinicore.int32)
    offsets = infinicore.from_list([0, seqlen], dtype=infinicore.int32)

    # Call the bound C++ forward so we can read `.logits` (Python wrapper `InferEngine.forward`
    # returns only output_ids).
    from infinilm.lib import _infinilm

    out = _infinilm.InferEngine.forward(
        eng,
        _infinilm.InferEngine.Input(
            input_ids_inf._underlying,
            position_ids=pos._underlying,
            past_sequence_lengths=past._underlying,
            total_sequence_lengths=total._underlying,
            input_offsets=offsets._underlying,
            cu_seqlens=cu._underlying,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
        ),
    )
    inf_logits = infinicore.Tensor(out.logits).to_numpy().astype(np.float32)

    # Metrics
    hf_norm = np.linalg.norm(hf_logits) + 1e-12
    inf_norm = np.linalg.norm(inf_logits) + 1e-12
    cos = float(np.dot(hf_logits, inf_logits) / (hf_norm * inf_norm))
    max_abs = float(np.max(np.abs(hf_logits - inf_logits)))

    print("== Logit sanity (last position) ==")
    print(f"cosine:   {cos:.6f}")
    print(f"max_abs:  {max_abs:.6f}")

    k = args.topk
    hf_i, hf_v = topk(hf_logits, k)
    inf_i, inf_v = topk(inf_logits, k)
    overlap = len(set(hf_i.tolist()) & set(inf_i.tolist()))
    print(f"top{k} overlap: {overlap}/{k}")

    print("\n-- HF topk --")
    for i, v in zip(hf_i, hf_v):
        print(int(i), float(v), tok.decode([int(i)]))
    print("\n-- InfiniLM topk --")
    for i, v in zip(inf_i, inf_v):
        print(int(i), float(v), tok.decode([int(i)]))


if __name__ == "__main__":
    main()

