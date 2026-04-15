#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../layers/common_modules.hpp"
#include "../../layers/linear/fused_linear.hpp"

namespace infinilm::models::minicpm5_moe {

/**
 * MiniCPM5-MoE attention module (HF parity for gated attention).
 *
 * HF `MiniCPM5MoEAttention` supports `use_gated_attention` where `q_proj` outputs
 * 2x the query dimension (query + gate). The attention output is gated by
 * `sigmoid(gate_score)` before the final `o_proj`.
 */
class MiniCPM5MoeAttention : public infinicore::nn::Module {
public:
    MiniCPM5MoeAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    size_t layer_idx() const { return layer_idx_; }

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::QKVParallelLinear, qkv_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);

    size_t layer_idx_;
    size_t hidden_size_{0};
    size_t head_dim_{0};
    size_t num_attention_heads_{0};
    size_t num_key_value_heads_{0};
    bool use_gated_attention_{false};

    ::infinilm::backends::AttentionBackend attention_backend_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;

    // For off-line kv cache quantization
    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);
};

} // namespace infinilm::models::minicpm5_moe

