# scripts/ — Deprecated

This directory contains legacy inference scripts that use the `libinfinicore_infer`
ctypes-based API (built from `src/`).

**Status:** Deprecated as of 2026-03-25. Scheduled for removal in v0.2.0 (Q2 2026).

**Use instead:**
- `examples/jiuge.py` — general LLaMA-family inference via `python/infinilm`
- `python/infinilm/server/inference_server.py` — production inference server

**Known issues (not fixed because this code is being removed):**
- `launch_server.py` non-streaming path returns `chunk_json(...)` instead of
  `completion_json(...)`, resulting in incorrect `object` field in the response.
