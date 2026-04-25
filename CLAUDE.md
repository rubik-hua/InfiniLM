# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InfiniLM is an inference engine built on [InfiniCore](https://github.com/InfiniTensor/InfiniCore). It provides LLM inference across multiple hardware backends (CPU, NVIDIA, Cambricon, Ascend, MetaX, Moore, Iluvatar, Kunlun, Hygon, Ali/QY). The project has two layers: a C++ core library (`infinicore_infer`) and a Python package (`infinilm`) with pybind11 bindings.

## Prerequisites

- **InfiniCore** must be compiled and installed first. The `INFINI_ROOT` env var (default `$HOME/.infini`) must point to the installation.
- Submodules required: `git submodule update --init --recursive` (pulls `third_party/spdlog` and `third_party/json`).
- Python >= 3.10.

## Build Commands

```bash
# Build the C++ shared library (infinicore_infer)
xmake

# Install to INFINI_ROOT
xmake install

# Build with KV caching support (for supported platforms)
xmake f --use-kv-caching=true -cv
xmake && xmake install

# Install Python package (also builds the _infinilm pybind11 module)
pip install -e .
```

## Running Inference

```bash
# Single inference test
python examples/jiuge.py --nvidia --model_path=/path/to/model

# Distributed inference (tensor parallelism)
python examples/jiuge.py --nvidia --model_path=/path/to/model --backend=cpp --tp=4

# Launch inference server
python python/infinilm/server/inference_server.py --nvidia --model_path=/path/to/model --tp=1

# Benchmark (C-Eval / MMLU)
python test/bench/test_benchmark.py --nvidia /path/to/model --bench ceval --subject all --backend cpp --ndev 1
```

## Architecture

### C++ Core (`csrc/` and `src/`)

Two C++ build targets exist in `xmake.lua`:

1. **`infinicore_infer`** — the main shared library built from `src/`. Contains model implementations (Jiuge/LLaMA variants, DeepSeek V3, Qwen3VL), tensor utilities, memory allocators, data loaders, and cache management. Public headers live in `include/`.

2. **`_infinilm`** — pybind11 module built from `csrc/`. This is the "new" C++ backend with a different architecture:
   - `csrc/models/` — Model implementations (LLaMA-based) using `InfinilmModel` base class. New models are registered via `InfinilmModelFactory`.
   - `csrc/engine/` — `InferEngine` orchestrates distributed inference using `RankWorker` threads with barrier synchronization.
   - `csrc/cache/` — KV cache management including paged attention support.
   - `csrc/config/` — `ModelConfig` and `QuantConfig` for model/quantization configuration.
   - `csrc/backends/` — Attention backend abstraction (default vs flash-attention).
   - `csrc/layers/` — Shared layers (e.g., fused linear).

**Deprecation note:** The legacy `InfinilmModel::Config`-based API is deprecated and scheduled for removal in v0.2.0 (Q2 2026). New code must use the `ModelConfig`-based polymorphic overloads.

### Python Package (`python/infinilm/`)

- `models/` — Python model wrappers
- `llm/` — LLM-specific logic
- `generation/` — Text generation pipeline
- `server/` — Inference server (OpenAI-compatible API)
- `distributed/` — Distributed/tensor-parallel support
- `cache/` and `cache_utils.py` — Cache management
- `infer_engine.py` — Python-side engine interface

### Legacy Scripts (`scripts/`)

Older inference scripts (`jiuge.py`, `deepseek.py`, etc.) that use `src/`-based models via ctypes (`scripts/libinfinicore_infer/`). The newer path uses `examples/` + `python/infinilm/`.

## C++ Conventions

- C++17 standard, GCC toolchain
- Warnings treated as errors (`-Wall -Werror`)
- Namespace: `infinilm` (sub-namespaces: `engine`, `cache`, `config`, `backends`)
- Dependencies: InfiniCore (`infiniop`, `infinirt`, `infiniccl`), spdlog, nlohmann/json
