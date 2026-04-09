#include "minicpm_sala_decoder_layer.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/context/context.hpp"
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace infinilm::models::minicpm_sala {


MiniCPMSALADecoderLayer::MiniCPMSALADecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 const infinicore::Device &device,
                                                 size_t layer_idx,
                                                 const std::string &mixer_type,
                                                 engine::distributed::RankInfo rank_info,
                                                 backends::AttentionBackend attention_backend) {
    layer_idx_ = layer_idx;
    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config->get_dtype();
    const double eps = model_config->get<double>("rms_norm_eps");

    // MuP residual scaling at forward (o_proj/down_proj not scaled in loader for minicpm_sala).
    const double scale_depth = model_config->get_or<double>("scale_depth", 1.0);
    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    residual_scale_ = scale_depth / std::sqrt(static_cast<double>(num_layers));

    INFINICORE_NN_MODULE_INIT(input_layernorm, model_config->get<size_t>("hidden_size"), eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attn, model_config, device, layer_idx, mixer_type, rank_info, attention_backend);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, model_config->get<size_t>("hidden_size"), eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, model_config, device);
}

void MiniCPMSALADecoderLayer::set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
    self_attn_->set_rotary_emb(rotary_emb);
}

infinicore::Tensor MiniCPMSALADecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                                    const infinicore::Tensor &position_ids,
                                                    std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                                    std::optional<infinicore::Tensor> past_sequence_lengths,
                                                    std::optional<infinicore::Tensor> total_sequence_lengths,
                                                    std::optional<infinicore::Tensor> input_offsets,
                                                    std::optional<infinicore::Tensor> cu_seqlens,
                                                    std::optional<infinicore::Tensor> block_tables,
                                                    std::optional<infinicore::Tensor> slot_mapping) const {
    // Match `layers/attention/Attention`: stash attention metadata in global forward context.
    infinilm::global_state::get_forward_context().attn_metadata =
        infinilm::global_state::AttentionMetadata(past_sequence_lengths,
                                                  total_sequence_lengths,
                                                  input_offsets,
                                                  cu_seqlens,
                                                  block_tables,
                                                  slot_mapping);

    // Pre-norm attention
    auto hs1 = input_layernorm_->forward(hidden_states);
    (void)kv_cache;
    auto attn_out = self_attn_->forward(position_ids, hs1);

    // residual + scale_down * attn_out (MuP)
    auto ones_attn = infinicore::Tensor::empty(attn_out->shape(), attn_out->dtype(), attn_out->device());
    infinicore::op::ones_(ones_attn);
    auto out1 = infinicore::op::addcmul(hidden_states, attn_out, ones_attn, static_cast<float>(residual_scale_));

    // Pre-norm MLP
    auto hs2 = post_attention_layernorm_->forward(out1);
    auto mlp_out = mlp_->forward(hs2);
    // residual + scale_down * mlp_out (MuP)
    auto ones_mlp = infinicore::Tensor::empty(mlp_out->shape(), mlp_out->dtype(), mlp_out->device());
    infinicore::op::ones_(ones_mlp);
    auto out2 = infinicore::op::addcmul(out1, mlp_out, ones_mlp, static_cast<float>(residual_scale_));

    return out2;
}

} // namespace infinilm::models::minicpm_sala
