#include "minicpm5_moe_decoder_layer.hpp"

#include "infinicore/ops.hpp"

namespace infinilm::models::minicpm5_moe {

MiniCPM5MoeDecoderLayer::MiniCPM5MoeDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 size_t layer_idx,
                                                 const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    input_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("input_layernorm", hidden_size, rms_norm_eps, dtype, device);
    post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("post_attention_layernorm", hidden_size, rms_norm_eps, dtype, device);

    self_attn_ = this->register_module<MiniCPM5MoeAttention>("self_attn", model_config, layer_idx, device);

    // HF: `first_k_dense_replace` shallow layers use dense `MiniCPM5MoEMLP`; deeper layers use MoE.
    const size_t first_k_dense_replace = model_config->get_or<size_t>("first_k_dense_replace", 0);
    if (layer_idx < first_k_dense_replace) {
        dense_mlp_ = this->register_module<MiniCPM5DenseMLP>("mlp", model_config, device);
    } else {
        moe_mlp_ = this->register_module<MiniCPM5MoeSparseMoeBlock>("mlp", model_config, device);
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> MiniCPM5MoeDecoderLayer::forward(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual) {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = dense_mlp_ ? dense_mlp_->forward(hidden_states) : moe_mlp_->forward(hidden_states);
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor MiniCPM5MoeDecoderLayer::forward(const infinicore::Tensor &positions,
                                                    infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = dense_mlp_ ? dense_mlp_->forward(hidden_states) : moe_mlp_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

} // namespace infinilm::models::minicpm5_moe

