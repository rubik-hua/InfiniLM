#pragma once

#include "infinicore/tensor.hpp"

#include <optional>

namespace infinilm::vllm_fused_moe_dispatch {

/// Returns whether the vLLM fused experts bridge is usable in this interpreter.
/// This is cached: once it is detected as unavailable (e.g. vLLM not installed),
/// it stays disabled for the rest of the process to avoid per-token Python errors.
bool fused_experts_ic_available();

/// Acquires the GIL, wraps tensors as Python ``infinicore.tensor.Tensor``, calls
/// ``infinicore.vllm_fused_moe_bridge.fused_experts_ic``. Returns nullopt on any failure
/// (missing vLLM, ATen bridge, import error, etc.).
std::optional<infinicore::Tensor> try_fused_experts_ic(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &w1_stacked,
    const infinicore::Tensor &w2_stacked,
    const infinicore::Tensor &topk_weights,
    const infinicore::Tensor &topk_ids);

} // namespace infinilm::vllm_fused_moe_dispatch
