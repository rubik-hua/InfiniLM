#pragma once

#include "../minicpm5_moe/minicpm5_moe_sparse_moe_block.hpp"

#include <optional>

namespace infinilm::models::minicpm5_moe_fused_stub {

/**
 * MiniCPM5 MoE block that keeps HF-aligned CPU routing but runs routed experts via
 * vLLM ``fused_experts`` (``infinicore.vllm_fused_moe_bridge``) when the Python stack is available.
 * Falls back to ``MiniCPM5MoeSparseMoeBlock::forward`` if dispatch fails or
 * ``INFINILM_DISABLE_VLLM_FUSED_MOE=1``.
 */
class MiniCPM5MoeVllmFusedSparseMoeBlock : public infinilm::models::minicpm5_moe::MiniCPM5MoeSparseMoeBlock {
public:
    using MiniCPM5MoeSparseMoeBlock::MiniCPM5MoeSparseMoeBlock;

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const override;

private:
    void rebuild_stacked_expert_weights(const infinicore::Device &dev, const infinicore::DataType &dt) const;

    mutable std::optional<infinicore::Tensor> w1_stacked_;
    mutable std::optional<infinicore::Tensor> w2_stacked_;
    mutable std::optional<infinicore::Device> stacked_dev_;
    mutable std::optional<infinicore::DataType> stacked_dt_;
};

} // namespace infinilm::models::minicpm5_moe_fused_stub
