#pragma once

#include "../../layers/common_modules.hpp"

#include <optional>

namespace infinilm::models::minicpm5_moe {

using MiniCPM5MoeMLP = infinilm::layers::MoeMLP;

/**
 * MiniCPM5-MoE sparse block: HF-aligned grouped top-k routing + routed sum in float32
 * (matching `dtype=topk_weights.dtype`) then cast to activation dtype before shared experts.
 * TODO(opt): fuse routing, device-side fp32 cast, and batched expert dispatch.
 */
class MiniCPM5MoeSparseMoeBlock : public infinicore::nn::Module {
public:
    MiniCPM5MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                              const infinicore::Device &device);

    virtual infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, gate);
    INFINICORE_NN_PARAMETER(e_score_correction_bias);
    INFINICORE_NN_MODULE_VEC(MiniCPM5MoeMLP, experts);
    INFINICORE_NN_MODULE(MiniCPM5MoeMLP, shared_experts);

    /// Lazily built F32 copy of `gate` weights on device for `linear` router logits (HF: matmul in float32).
    mutable std::optional<infinicore::Tensor> gate_weight_f32_device_;
};

} // namespace infinilm::models::minicpm5_moe

