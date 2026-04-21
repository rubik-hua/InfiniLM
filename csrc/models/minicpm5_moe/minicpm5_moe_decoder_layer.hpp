#pragma once

#include "../../layers/common_modules.hpp"
#include "minicpm5_moe_attention.hpp"
#include "minicpm5_moe_dense_mlp.hpp"
#include "minicpm5_moe_sparse_moe_block.hpp"
#include "minicpm5_moe_vllm_fused_sparse_moe_block.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include <memory>

namespace infinilm::models::minicpm5_moe {

class MiniCPM5MoeDecoderLayer : public infinicore::nn::Module {
public:
    MiniCPM5MoeDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiniCPM5MoeAttention, self_attn);

    std::shared_ptr<MiniCPM5DenseMLP> dense_mlp_;
    std::shared_ptr<MiniCPM5MoeSparseMoeBlock> moe_mlp_;

    size_t layer_idx_;
};

} // namespace infinilm::models::minicpm5_moe

