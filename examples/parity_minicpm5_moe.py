import argparse
import ctypes
import importlib.util
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _preload_flash_attn_cuda_global():
    """Expose FlashAttention CUDA symbols globally for InfiniCore .so (undefined at dlopen otherwise)."""
    spec = importlib.util.find_spec("flash_attn_2_cuda")
    if spec is None or not spec.origin:
        return
    try:
        ctypes.CDLL(spec.origin, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


def _preload_libfmt_global():
    """Ensure libfmt vtable symbols resolve for _infinilm when LD_LIBRARY_PATH is curated."""
    for name in ("libfmt.so.9", "libfmt.so"):
        for d in ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu"):
            path = os.path.join(d, name)
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    return
                except OSError:
                    continue


def _hf_device_map(device: str):
    """Pin HF to one device; avoid device_map=auto sharding across all visible GPUs."""
    if device == "cpu":
        return {"": "cpu"}
    if device.startswith("cuda:"):
        return {"": int(device.split(":")[-1])}
    if device == "cuda":
        return {"": 0}
    return "auto"


def build_input_ids(tokenizer, prompt: str):
    # Prefer HF chat template if available.
    try:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        return input_ids
    except Exception:
        return tokenizer(prompt, return_tensors="pt").input_ids


@torch.inference_mode()
def hf_next_token_id(model, input_ids: torch.Tensor) -> int:
    out = model(input_ids=input_ids)
    logits = out.logits[:, -1, :]
    return int(torch.argmax(logits, dim=-1).item())


def hf_prefill_last_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids)
    return out.logits[:, -1, :].detach().float().cpu()


def _hf_equivalent_param_name(inf_name: str) -> str:
    """Map InfiniLM state_dict names to HF checkpoint names where they differ."""
    # HF MiniCPM5 MoE keeps the router buffer on `mlp.gate`; InfiniLM flattens to `mlp`.
    suf = ".mlp.e_score_correction_bias"
    if inf_name.endswith(suf):
        return inf_name[: -len(suf)] + ".mlp.gate.e_score_correction_bias"
    return inf_name


def check_strict_weight_keys(engine, hf_keys: set[str]) -> None:
    """Ensure every InfiniLM parameter name exists in the HF checkpoint key set."""
    inf_keys = set(engine.state_dict_keyname())
    missing_in_checkpoint = sorted(
        k for k in inf_keys if _hf_equivalent_param_name(k) not in hf_keys
    )
    if missing_in_checkpoint:
        raise SystemExit(
            "Strict weight check failed: InfiniLM expects tensors missing from HF checkpoint: "
            + ", ".join(missing_in_checkpoint[:32])
            + (" ..." if len(missing_in_checkpoint) > 32 else "")
        )
    print(
        f"Strict weight check OK ({len(inf_keys)} InfiniLM keys found in checkpoint)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompt", default="Hello!")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--hf-on-cpu",
        action="store_true",
        help="Load the HF reference model on CPU (slow but avoids OOM when the GPU is already occupied).",
    )
    ap.add_argument(
        "--strict-weights",
        action="store_true",
        help="Fail if any InfiniLM parameter name is absent from the loaded HF state_dict.",
    )
    ap.add_argument(
        "--prefill-logits",
        action="store_true",
        help=(
            "Log HF prefill last-position logits sanity (finite values). "
            "InfiniLM logits are not exposed on InferEngine.Output in Python yet, "
            "so there is no cross-framework logits tensor comparison."
        ),
    )
    args = ap.parse_args()

    model_path = os.path.abspath(args.model_path)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = build_input_ids(tok, args.prompt)

    hf_device = "cpu" if args.hf_on_cpu else args.device
    input_ids_hf = input_ids.to(hf_device)

    dtype = torch.float32 if hf_device == "cpu" else torch.bfloat16
    hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=_hf_device_map(hf_device),
    )
    hf.eval()

    hf_id = hf_next_token_id(hf, input_ids_hf)

    if args.prefill_logits:
        logits = hf_prefill_last_logits(hf, input_ids_hf)
        if not torch.isfinite(logits).all():
            raise SystemExit("HF prefill logits contain NaN/Inf")
        print(
            f"HF prefill last logits: shape={tuple(logits.shape)} "
            f"min={logits.min().item():.4f} max={logits.max().item():.4f} "
            f"argmax_token={int(torch.argmax(logits, dim=-1).item())}"
        )
        print(
            "Note: InferEngine Python binding only returns output_ids; "
            "prefill logits parity vs InfiniLM would need Output.logits exposed in pybind."
        )

    hf_keys = set(hf.state_dict().keys()) if args.strict_weights else None

    del hf
    if args.device != "cpu":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # InfiniLM path (token-level parity; logits are not exposed yet).
    _preload_flash_attn_cuda_global()
    _preload_libfmt_global()
    from infinilm.infer_engine import GenerationConfig, InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file
    import infinicore

    engine = InferEngine(model_path=model_path, device=infinicore.device(args.device))
    if hf_keys is not None:
        check_strict_weight_keys(engine, hf_keys)

    load_model_state_dict_by_file(engine, model_path, dtype=engine.config.dtype)

    gen_cfg = GenerationConfig(max_new_tokens=1, temperature=1.0, top_k=1, top_p=1.0)
    out_ids = engine.generate(
        infinicore.from_torch(input_ids.to("cpu")), gen_cfg, _measure_and_log_time=False
    )
    inf_id = int(out_ids[0].to_numpy()[0])

    print("HF next_token_id:", hf_id)
    print("InfiniLM next_token_id:", inf_id)
    if hf_id != inf_id:
        raise SystemExit("Mismatch")


if __name__ == "__main__":
    main()

