#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::minicpm5_moe {

using MiniCPM5MoeMLP = infinilm::layers::MoeMLP;

/**
 * NOTE: This is a minimal Sparse-MoE block implementation intended to unblock
 * model registration and end-to-end engine wiring. It currently does not
 * implement expert routing; it runs a single expert MLP.
 */
class MiniCPM5MoeSparseMoeBlock : public infinicore::nn::Module {
public:
    MiniCPM5MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, gate);
    INFINICORE_NN_MODULE_VEC(MiniCPM5MoeMLP, experts);
    INFINICORE_NN_MODULE(MiniCPM5MoeMLP, shared_expert);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, shared_expert_gate);
};

} // namespace infinilm::models::minicpm5_moe

