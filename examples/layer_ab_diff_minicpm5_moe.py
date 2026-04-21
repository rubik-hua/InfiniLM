import argparse
import os
import re
import subprocess
import sys


def _stable_env():
    env = os.environ.copy()
    # Match the known-stable recipe (avoid /opt/hpcx).
    torch_lib = subprocess.check_output(
        [sys.executable, "-c", 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'],
        text=True,
    ).strip()
    fa = "/usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
    env.pop("LD_LIBRARY_PATH", None)
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            "/root/.infini/lib",
            torch_lib,
            "/usr/local/lib/python3.12/dist-packages",
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/lib/x86_64-linux-gnu",
        ]
    )
    env["LD_PRELOAD"] = fa
    return env


def _run_jiuge(env, model_path, prompt, max_new_tokens, extra_env):
    env2 = env.copy()
    env2.update(extra_env)
    cmd = [
        sys.executable,
        "-u",
        "jiuge.py",
        "--nvidia",
        "--model-path",
        model_path,
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--top-k",
        "1",
        "--attn",
        "flash-attn",
        "--enable-paged-attn",
        "--paged-kv-block-size",
        "256",
    ]
    p = subprocess.run(cmd, env=env2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def _parse_stats(txt: str):
    pat = re.compile(
        r"\[INFINILM_LAYER_STATS\] layer=(\d+) tag=(\w+) nonfinite=(\d+) checked=(\d+) max_abs=([0-9eE+\.\-]+) sum_abs=([0-9eE+\.\-]+)"
    )
    out = {}
    for line in txt.splitlines():
        m = pat.search(line)
        if not m:
            continue
        layer = int(m.group(1))
        tag = m.group(2)
        out[(layer, tag)] = dict(
            nonfinite=int(m.group(3)),
            checked=int(m.group(4)),
            max_abs=float(m.group(5)),
            sum_abs=float(m.group(6)),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompt", default="Hi")
    ap.add_argument("--max-new-tokens", type=int, default=2)
    ap.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()

    env = _stable_env()
    base = {
        "INFINILM_DEBUG_NAN_LAYERS": args.layers,
        "INFINILM_DEBUG_LAYER_STATS": "1",
    }

    rc_b, out_b = _run_jiuge(
        env, args.model_path, args.prompt, args.max_new_tokens, {**base, "INFINILM_MOE_USE_BATCHED_DISPATCH": "1"}
    )
    rc_r, out_r = _run_jiuge(env, args.model_path, args.prompt, args.max_new_tokens, {**base, "INFINILM_MOE_FORCE_FALLBACK": "1"})

    if rc_b != 0 or rc_r != 0:
        print("Run failed.")
        print("== batched ==")
        print(out_b)
        print("== reference ==")
        print(out_r)
        raise SystemExit(1)

    st_b = _parse_stats(out_b)
    st_r = _parse_stats(out_r)
    keys = sorted(set(st_b.keys()) & set(st_r.keys()))
    first = None
    for k in keys:
        b = st_b[k]
        r = st_r[k]
        db = abs(b["sum_abs"] - r["sum_abs"])
        dm = abs(b["max_abs"] - r["max_abs"])
        if b["nonfinite"] or r["nonfinite"] or db > 1e-3 or dm > 1e-4:
            first = (k, b, r, db, dm)
            break

    print("== A/B layer stats (jiuge) ==")
    if first is None:
        print("No divergence detected in collected stats.")
    else:
        (layer, tag), b, r, db, dm = first
        print(f"first_diff: layer={layer} tag={tag}")
        print(f"  batched: nonfinite={b['nonfinite']} checked={b['checked']} max_abs={b['max_abs']:.6g} sum_abs={b['sum_abs']:.6g}")
        print(f"  ref:    nonfinite={r['nonfinite']} checked={r['checked']} max_abs={r['max_abs']:.6g} sum_abs={r['sum_abs']:.6g}")
        print(f"  deltas: dmax={dm:.6g} dsum={db:.6g}")


if __name__ == "__main__":
    main()

