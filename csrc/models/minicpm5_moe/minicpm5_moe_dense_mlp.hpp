#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/module.hpp"

namespace infinilm::models::minicpm5_moe {

/**
 * Dense MLP with parameter names matching HF (`gate_proj`, `up_proj`, `down_proj`),
 * but using `intermediate_size` from config.
 */
class MiniCPM5DenseMLP : public infinicore::nn::Module {
public:
    MiniCPM5DenseMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, gate_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, up_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, down_proj);
};

} // namespace infinilm::models::minicpm5_moe

