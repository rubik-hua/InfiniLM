#include "vllm_fused_moe_dispatch.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <cstdlib>
#include <cstdio>
#include <atomic>

namespace py = pybind11;

namespace infinilm::vllm_fused_moe_dispatch {

// -1: unavailable (do not attempt again), 0: unknown, 1: available
static std::atomic<int> g_fused_available{0};

bool fused_experts_ic_available() {
    int s = g_fused_available.load(std::memory_order_relaxed);
    if (s != 0) {
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                static bool printed = false;
                if (!printed) {
                    printed = true;
                    std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic_available cached=%d\n", s);
                    std::fflush(stderr);
                }
            }
        }
        return s > 0;
    }
    // Probe imports once under the GIL.
    py::gil_scoped_acquire gil;
    try {
        // Import vLLM itself; if it is missing, we want to disable this path entirely.
        (void)py::module_::import("vllm.model_executor.layers.fused_moe");
        (void)py::module_::import("infinicore.vllm_fused_moe_bridge");
        g_fused_available.store(1, std::memory_order_relaxed);
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic_available probed=1\n");
                std::fflush(stderr);
            }
        }
        return true;
    } catch (const py::error_already_set &e) {
        g_fused_available.store(-1, std::memory_order_relaxed);
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic_available probed=-1: %s\n", e.what());
                std::fflush(stderr);
            }
        }
        PyErr_Clear();
        return false;
    } catch (...) {
        g_fused_available.store(-1, std::memory_order_relaxed);
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic_available probed=-1: unknown\n");
                std::fflush(stderr);
            }
        }
        return false;
    }
}

std::optional<infinicore::Tensor> try_fused_experts_ic(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &w1_stacked,
    const infinicore::Tensor &w2_stacked,
    const infinicore::Tensor &topk_weights,
    const infinicore::Tensor &topk_ids) {
    if (!fused_experts_ic_available()) {
        return std::nullopt;
    }
    // Hold GIL for the whole function so catch handlers that touch Python APIs
    // (e.g. PyErr_Clear) never run after gil_scoped_acquire has unwound.
    py::gil_scoped_acquire gil;
    try {
        py::object tensor_mod = py::module_::import("infinicore.tensor");
        py::object TensorCls = tensor_mod.attr("Tensor");
        py::object bridge = py::module_::import("infinicore.vllm_fused_moe_bridge");
        py::object fn = bridge.attr("fused_experts_ic");

        py::object h_py = TensorCls(py::cast(hidden_states));
        py::object w1_py = TensorCls(py::cast(w1_stacked));
        py::object w2_py = TensorCls(py::cast(w2_stacked));
        py::object tw_py = TensorCls(py::cast(topk_weights));
        py::object id_py = TensorCls(py::cast(topk_ids));

        py::object out_py = fn(h_py, w1_py, w2_py, tw_py, id_py);
        py::object und = out_py.attr("_underlying");
        return und.cast<infinicore::Tensor>();
    } catch (const py::error_already_set &e) {
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic failed: %s\n", e.what());
            }
        }
        // If vLLM is missing (common in the HF/InfiniLM interpreter), disable further attempts
        // to avoid paying per-token Python exception overhead.
        if (std::string(e.what()).find("requires vLLM to be installed") != std::string::npos) {
            g_fused_available.store(-1, std::memory_order_relaxed);
        }
        PyErr_Clear();
        return std::nullopt;
    } catch (...) {
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic failed: unknown exception\n");
            }
        }
        return std::nullopt;
    }
}

} // namespace infinilm::vllm_fused_moe_dispatch
