# MiniCPM5 MoE — inference profiling (InfiniLM vs HF)

Use this document as a **continuous profiling** log: fill the environment block each session, then append or update the run table with fresh numbers. Workload settings should stay fixed when comparing engines.

---

## Environment (per run)

| Field | Value |
|--------|--------|
| Date | |
| Host / GPU | |
| CUDA / driver | |
| PyTorch | |
| Transformers | |
| Git commit / branch | |
| Model path | |

---

## Workload (keep identical across engines)

| Field | Typical value |
|--------|----------------|
| Prompt | `Hi` (or any fixed string) |
| Chat template | `apply_chat_template` + `add_generation_prompt=True` (same as `jiuge.py`) |
| Prompt token count | (recorded per run) |
| `max_new_tokens` |16 |
| Batch size | 1 |
| `top_k` / `top_p` / `temperature` | 1 / 1.0 / 1.0 |
| Activations dtype | bfloat16 |

---

## Metrics

### Weight load

| Engine | Metric | Notes |
|--------|--------|--------|
| InfiniLM | Wall time from loader start until ready to generate (`jiuge.py` “load weights over”) | Dominated by custom load path / I/O |
| Hugging Face | Wall time for `from_pretrained` + `.to(cuda)` + `eval()` (`hf_bench_match_jiuge.py`) | Different format and pipeline; not apples-to-apples with InfiniLM load |

### Generation (after weights are resident)

| Metric | Definition |
|--------|------------|
| **Total generation** | Wall time around the full generate path (InfiniLM: `jiuge` timer; HF: `model.generate` with CUDA sync) |
| **Prefill** | One forward over the full prompt (`use_cache=True`); HF bench reports this explicitly |
| **TTFT** | InfiniLM engine-reported time to first token (includes prefill-related work as implemented in engine) |
| **Decode total** | Sum of per-step decode forwards (HF manual loop) |
| **Decode avg / step** | `decode_total / max_new_tokens` |

### Example snapshot (replace on every profile pass)

Numbers below are **examples only** from one session; do not treat them as baselines.

| Metric | InfiniLM (`jiuge.py`) | HF manual (prefill + greedy steps) | HF `model.generate()` |
|--------|----------------------|-------------------------------------|------------------------|
| Weight load | ~120 s | ~24 s | — |
| Total generation | ~1486 ms | — | ~5008 ms |
| Prefill | (see TTFT) | ~925 ms | (inside `generate`) |
| TTFT | ~669 ms | — | — |
| Decode total (16 steps) | — | ~4810 ms | — |
| Decode avg / step | ~54 ms | ~301 ms | — |
| Prefill + decode (manual) | — | ~5734 ms | — |

---

## Reproduce

### InfiniLM

```bash
# Set REPO, PYTHONPATH, and linker env per your container / workspace rules.
python3 -u InfiniLM/examples/jiuge.py --nvidia \
  --model-path /path/to/minicpm5 \
  --prompt "Hi" --max-new-tokens 16 --batch-size 1 \
  --top-k 1 --top-p 1.0 --temperature 1.0 --attn default
```

### Hugging Face (matched tokenization and sampling knobs)

```bash
python3 -u InfiniLM/examples/hf_bench_match_jiuge.py \
  --model-path /path/to/minicpm5 \
  --prompt "Hi" --max-new-tokens 16 --batch-size 1 \
  --top-k 1 --top-p 1.0 --temperature 1.0
```

---

## Continuous run log

| Run | Date | Engine | Load (s) | Prefill (ms) | Dec avg (ms) | Gen total (ms) | Notes |
|-----|------|--------|----------|--------------|--------------|----------------|-------|
| | | | | | | | |

---

## Related files

- `InfiniLM/examples/jiuge.py` — InfiniLM generation driver
- `InfiniLM/examples/hf_bench_match_jiuge.py` — HF timing with jiuge-aligned tokenization
- `InfiniLM/examples/logit_sanity_minicpm5_moe.py` — correctness / logit sanity vs HF
