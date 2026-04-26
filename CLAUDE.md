# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InfiniLM is an inference engine built on [InfiniCore](https://github.com/InfiniTensor/InfiniCore). It provides LLM inference across multiple hardware backends (CPU, NVIDIA, Cambricon, Ascend, MetaX, Moore, Iluvatar, Kunlun, Hygon, Ali/QY). The project has two layers: a C++ core library (`infinicore_infer`) and a Python package (`infinilm`) with pybind11 bindings.

## Prerequisites

- **InfiniCore** must be compiled and installed first. The `INFINI_ROOT` env var (default `$HOME/.infini`) must point to the installation.
- Submodules required: `git submodule update --init --recursive` (pulls `third_party/spdlog` and `third_party/json`).
- Python >= 3.10.

## Build Systems

**Two parallel build systems coexist** — pick the one that matches the target:

- **xmake (`xmake.lua`)** — builds the legacy `infinicore_infer` shared lib (from `src/`) for the `scripts/`-based ctypes path. Also has an `_infinilm` target, but the canonical way to build the pybind module is via CMake/`pip install -e .`.
- **CMake (`CMakeLists.txt`)** — drives `pip install -e .` via `setup.py`. Builds three vendored subprojects (`runtime/` → `libinfinirt.so`, `ccl/` → `libinfiniccl.so`, `ops/` → `libinfinicore.so` + `_infinicore` Python module) plus the `_infinilm` pybind11 module from `csrc/`. Co-locates all `.so`s under `python/infinilm/lib/` so RPATH=`$ORIGIN` works without `LD_LIBRARY_PATH`.

## Build Commands

```bash
# Legacy path: build/install infinicore_infer to $INFINI_ROOT
xmake && xmake install

# xmake build options
xmake f --use-kv-caching=true -cv      # KV-caching op (nvidia/ali/iluvatar/metax/hygon/qy)
xmake f --use-classic-llama=true -cv   # classic LlamaForCausalLM path

# Modern path: builds runtime/ccl/ops subprojects + _infinilm pybind module via CMake
pip install -e .

# Build-time env knobs (consumed by setup.py)
INFINILM_BUILD_FLASH_ATTN=1 pip install -e .          # enable FlashAttention backend (auto-clones third_party/flash-attention)
INFINILM_FLASH_ATTN_ARCHS=80 pip install -e .         # restrict CUDA archs (default 80;86;89;90)
INFINILM_BUILD_TYPE=Debug pip install -e .            # Debug | Release | RelWithDebInfo
INFINILM_BUILD_JOBS=8 pip install -e .                # parallel compile jobs (default nproc)
```

The CMake build also accepts `-DINFINILM_ENABLE_KV_CACHING=ON` and `-DINFINILM_USE_CLASSIC_LLAMA=ON` as the equivalents of the xmake flags above.

## Running Inference

The new `examples/` + `python/infinilm/` path takes `--device <name>`; the legacy `scripts/` path takes `--<name>` flags. Devices: `cpu | nvidia | qy | metax | moore | iluvatar | ali | cambricon | hygon | ascend | kunlun`.

```bash
# Single inference (new path)
python examples/jiuge.py --device=nvidia --model=/path/to/model

# Distributed inference (tensor parallelism)
python examples/jiuge.py --device=nvidia --model=/path/to/model --backend=cpp --tp=4 --batch-size=16

# Experimental: warmup, paged attention, CUDA graph, attention backend
python examples/bench.py --device=nvidia --model=/path/to/model --warmup
python examples/bench.py --device=nvidia --model=/path/to/model --enable-paged-attn --enable-graph
python examples/bench.py --device=nvidia --model=/path/to/model --enable-paged-attn --attn=flash-attn

# Inference server (OpenAI-compatible API)
python python/infinilm/server/inference_server.py --device=nvidia --model=/path/to/model --tp=1

# Benchmark (C-Eval / MMLU). --cache-dir avoids HF network calls.
python test/bench/test_benchmark.py --device nvidia /path/to/model --bench ceval --subject all --backend cpp --tp 1 --output-csv results.csv

# Legacy path (scripts/, ctypes, --device-as-flag style)
python scripts/jiuge.py --nvidia /path/to/model 4
python scripts/launch_server.py --model /path/to/model --dev nvidia --ndev 4
python scripts/test_ppl.py --model /path/to/model
```

## Architecture

### C++ Core

Two C++ trees exist side-by-side, corresponding to the two build systems:

1. **Legacy `src/` → `infinicore_infer.so`** (xmake target). Public headers in `include/infinicore_infer/`. Contains model implementations (Jiuge/LLaMA variants, DeepSeek V3, Qwen3VL), tensor utilities, allocators, data loaders, cache manager. Consumed by `scripts/` via ctypes (`scripts/libinfinicore_infer/`).

2. **Modern `csrc/` → `_infinilm` pybind11 module** (CMake target, also has an xmake target). Layout:
   - `csrc/models/` — LLaMA-based model implementations using the `InfinilmModel` base class; new models register via `InfinilmModelFactory`.
   - `csrc/engine/` — `InferEngine` orchestrates distributed inference using `RankWorker` threads with barrier synchronization.
   - `csrc/cache/` — KV cache management, including paged-attention support.
   - `csrc/config/` — `ModelConfig` and `QuantConfig`.
   - `csrc/backends/` — Attention backend abstraction (default vs FlashAttention).
   - `csrc/layers/` — Shared layers (e.g., fused linear).
   - `csrc/global_state/`, `csrc/pybind11/` — global state and pybind glue.

3. **Vendored InfiniCore subprojects** (CMake-only; each has its own `CMakeLists.txt`):
   - `runtime/` → `libinfinirt.so` (runtime abstraction)
   - `ccl/` → `libinfiniccl.so` (collective communication)
   - `ops/` → `libinfinicore.so` + `_infinicore` Python module (operator library)
   These get co-located under `python/infinilm/lib/` so the wheel is self-contained.

**Deprecation note:** The legacy `InfinilmModel::Config`-based API is deprecated and scheduled for removal in v0.2.0 (Q2 2026). New code must use the `ModelConfig`-based polymorphic overloads.

### Python Package (`python/infinilm/`)

- `models/` — Python model wrappers
- `llm/` — LLM-specific logic
- `generation/` — Text generation pipeline
- `server/` — Inference server (OpenAI-compatible API)
- `distributed/` — Distributed/tensor-parallel support
- `cache/` and `cache_utils.py` — Cache management
- `infer_engine.py` — Python-side engine interface
- `lib/` — populated at build time with co-located `.so` files (do not commit)

`setup.py` also packages a sibling `infinicore` Python package (with `.nn`, `.ops` submodules) sourced from `python/`, exposed by the `_infinicore` module built from `ops/`.

### Tests (`test/`)

- `test/bench/test_benchmark.py` — C-Eval / MMLU accuracy benchmarks (see Running Inference above).
- `test/models/` — model-level test cases.

There is no top-level test runner wired up; invoke individual test scripts directly with `python`.

### Legacy Scripts (`scripts/`)

ctypes-based inference scripts that target the legacy `src/`-built `infinicore_infer.so` (loaded via `scripts/libinfinicore_infer/`):
- `jiuge.py`, `jiuge_awq.py`, `jiuge_gptq.py`, `jiuge_ppl.py` — Jiuge/LLaMA variants and quantized paths
- `deepseek.py`, `qwen3vl.py` — DeepSeek V3 and Qwen3-VL
- `launch_server.py`, `test_perf.py`, `test_ppl.py`, `test_ceval.py` — server launch and evaluation utilities
- `kvcache_pool.py`, `infer_task.py` — engine-side helpers

New work should target the `examples/` + `python/infinilm/` path; touch `scripts/` only when maintaining the legacy entry points.

## C++ Conventions

- C++17 standard, GCC toolchain
- Warnings treated as errors (`-Wall -Werror`)
- Namespace: `infinilm` (sub-namespaces: `engine`, `cache`, `config`, `backends`)
- Dependencies: InfiniCore (`infiniop`, `infinirt`, `infiniccl`), spdlog, nlohmann/json
