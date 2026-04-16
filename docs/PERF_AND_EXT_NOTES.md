# Performance & Extensibility Notes

This document tracks issues flagged by review agents (`/simplify` runs) that
are **not** immediately fixed in the current change. They require larger
refactors, or are worth measuring before investing in a fix.

Format: each item has a severity estimate, the primary evidence, and a
sketched fix. The list is ordered from highest concern to informational.

---

## ⚡ PERF (significant)

### 1. Qwen3VL static barrier is shared across all model instances

- **Location:** `src/models/qwen3vl/qwen3vl_impl.hpp:67-71`
  (`inline static std::mutex mtx_sync; inline static int sync_cnt = 0; inline static std::condition_variable cv_sync;`)
- **Why it matters:** the vision-text sync barrier uses **class-level statics**,
  so two `Qwen3vlModel` instances in the same process share the same counter.
  Any multi-tenant / ensemble / A-B-test deployment that holds more than one
  Qwen3VL at once will deadlock or corrupt the other's vision sync.
- **Fix sketch:** move the three fields to **instance members** of
  `Qwen3vlInferState`. Add a `virtual void ModelBase::pre_dispatch(Request&)`
  hook that subclasses override (`Qwen3vlModel::pre_dispatch` would reset
  `state_.sync_cnt = ndev()`). This also cleans up the leaky protocol where C
  API free functions today must write `Qwen3vlInferState::sync_cnt` from
  outside before calling `dispatch()`.
- **Severity:** latent correctness bug; becomes **critical** the moment anyone
  instantiates two Qwen3VL models concurrently.

### 2. `req_ = req` copied per-dispatch under `states_[0].mtx`

- **Location:** `src/models/model_base.hpp:111` (inside the dispatch loop).
- **Why it matters:** for `Qwen3vlInferRequest` (~19 scalar fields + pointer
  arrays) and `BaseInferRequest` (~12 pointer fields), this is a per-batch
  struct copy. The byte cost is negligible, but the **happens-before** is
  subtle — `req_` is written under `states_[0].mtx` and read by device thread
  `idev` under `states_[idev].mtx`, relying on the `proceed=true` / cv_start
  chain for ordering. A future refactor could easily break this.
- **Fix sketch:** pass the request by `const Request&` captured once outside
  the loop, or use `std::atomic<const Request*>` to make the handoff explicit.
- **Severity:** medium (performance), low (correctness today).

### 3. Per-step device-to-host copy in attention forward

- **Location:** `csrc/models/llama/llama_attention.cpp:212-216` (and analogous
  in other attention variants). `total_sequence_lengths.value()->to(infinicore::Device::cpu())->data()[0]`
  is pulled **once per forward** — already hoisted out of the per-layer loop
  in commit `39f1804`.
- **Status:** fixed as-of HEAD. Left in this doc as a reminder: **any future
  change** that reintroduces the `.to(cpu)` call inside a per-layer or
  per-token context will reintroduce the regression.
- **Fix sketch:** keep the hoisted variable; consider adding a
  `total_seq_len` parameter to the inner forward signature across all
  attention variants so the cpu read can never drift back into the hot path.

### 4. JSON chunk allocation per streamed token

- **Location:** `python/infinilm/server/inference_server.py:334-354`.
  `chunk_json(...)` builds a 5-level nested dict + calls `int(time.time())`
  + `json.dumps` per emitted token.
- **Why it matters:** at 500 tok/s aggregate (batched), this is measurable
  Python overhead. At 30 tok/s per stream, it's background noise.
- **Fix sketch:** template the static fields (`"object"`, `"model"`,
  `"system_fingerprint"`) once per request; fill only `choices[0].delta`
  per chunk.
- **Severity:** medium at large batch sizes, low per-stream.

### 5. `_step_loop` busy-sleeps 10 ms when idle

- **Location:** `python/infinilm/llm/llm.py:617-628`.
- **Why it matters:** when the scheduler has no pending requests, the loop
  does `time.sleep(0.01)`, so a newly arrived request waits up to 10 ms
  before the next iteration picks it up — pure TTFT overhead.
- **Fix sketch:** replace the sleep with a `threading.Event` that
  `add_request` sets; `_step_loop` does `event.wait(timeout=0.01)` and
  clears on wakeup.
- **Severity:** medium for latency-sensitive traffic.

## 🔒 SEC (low-to-medium)

### 6. Unbounded `janus.Queue()` in scheduler/request paths

- **Locations:** `python/infinilm/llm/scheduler.py:139-140`,
  `python/infinilm/llm/static_scheduler.py:104`,
  `python/infinilm/llm/request.py:161`.
- **Why it matters:** a stuck/slow streaming client holds the queue open while
  tokens accumulate. N slow clients × long generations → unbounded heap
  growth.
- **Fix sketch:** `janus.Queue(maxsize=...)` with a back-pressure path in
  `_batch_put`.

### 7. C API pointer lifetime between `KVCache*` and `Model*`

- **Locations:** `src/models/jiuge/jiuge.cpp:349-358` (and analogues in awq /
  gptq / deepseek_v3 / qwen3vl — the extern-C `create*Model` / `destroy*Model`
  functions).
- **Why it matters:** if Python holds a `KVCache*` returned from model N and
  then destroys model N before the cache, any subsequent `inferBatch*` call
  dereferences a freed memory-pool-backed buffer. There is no reference link
  between the cache and its owning model.
- **Fix sketch:** document the ordering requirement in header comments;
  ideally, have the Python side own a shared handle so the cache is freed
  before the model.

## 📐 EXT (extensibility)

### 8. C API wrappers duplicated across 3 jiuge variants

- **Locations:** `src/models/jiuge/jiuge.cpp:309-347`,
  `src/models/jiuge_awq/jiuge_awq.cpp:267-305`,
  `src/models/jiuge_gptq/jiuge_gptq.cpp:267-305`. Each set of 4 extern-C
  functions (`infer` / `forward` / `create` / `destroy`) is byte-for-byte
  identical modulo type suffix. DeepSeekV3 and Qwen3VL follow the same shape
  with different Request types.
- **Fix sketch:** a `DEFINE_MODEL_C_API(ModelName, ReqType)` macro that emits
  the four functions, calling a templated `dispatchInfer<Model, Req>` /
  `dispatchForward<Model, Req>` helper. Estimated ~120 LOC removed across 5
  files.
- **Severity:** pure maintenance — no functional impact — but the extern-C
  boundary is exactly where copy-paste bugs stay silent the longest.

### 9. `llama_compatible` hardcoded set in `model_factory.cpp`

- **Location:** `csrc/models/model_factory.cpp` second `createModel` overload.
  The set `{llama, qwen2, minicpm, fm9g, fm9g7b}` hardcodes which model_types
  fall through to `LlamaForCausalLM` in `USE_CLASSIC_LLAMA` builds.
- **Status:** already **sync'd with `rank_worker.cpp`** and `qwen3` removed
  (qwen3 has its own `Qwen3ForCausalLM` class). The hardcoded set remains
  because it is only exercised when the registry-driven path is disabled.
- **Fix sketch (future):** drive the set from a registry flag
  (`INFINILM_REGISTER_LLAMA_COMPATIBLE(name)` or an
  `is_llama_compatible(model_type)` helper) so a new llama-family model
  needs only one registration instead of two edits.

### 10. `python/infinilm/utils/` is a grab-bag package

- **Location:** `python/infinilm/utils/{tokenizer.py, openai_format.py}`.
  The two modules are unrelated — one fixes HuggingFace tokenizer decoders
  (pre-inference concern), the other builds OpenAI-format JSON (HTTP concern).
- **Fix sketch:** move `openai_format.py` → `python/infinilm/server/openai_format.py`
  (its only consumer is the server). Move `tokenizer.py` →
  `python/infinilm/tokenization.py` at the top level. Delete the empty
  `utils/` package. Severity: cosmetic, but worth doing **before** more
  modules land in `utils/` and the name sticks.

---

## Fixed in the same /simplify pass

- Removed redundant `classic_models` gate in `csrc/engine/rank_worker.cpp`
  and the divergent `qwen3` entry in `llama_compatible` — single source of
  truth now lives in `model_factory.cpp`.
- Added `[[deprecated]]` attribute to the legacy `createModel(LlamaConfig...)`
  overload in `csrc/models/model_factory.hpp` — the `InfinilmModel::Config`
  signature is proven unreachable (rank_worker throws), and new call sites
  now get a compile-time warning.
- Added a `seq_len` bounds check in `duplicateKVCache` — prevents silent OOB
  read/write when caller passes an overly large length.
- Narrowed `except Exception` in `python/infinilm/__init__.py` to
  `(ImportError, TypeError)` with a `logging.warning` diagnostic and
  `INFINILM_DEBUG=1` opt-in for full tracebacks.
- Removed narration / policy-emoji comments in `model_factory.{hpp,cpp}` and
  a commented-out 6-line call block in `rank_worker.cpp`.
