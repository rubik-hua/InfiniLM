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
                                                 const std::string &mixer_type) {
    layer_idx_ = layer_idx;
    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config->get_dtype();
    const double eps = model_config->get<double>("rms_norm_eps");

    // MuP residual scaling at forward (o_proj/down_proj not scaled in loader for minicpm_sala).
    const double scale_depth = model_config->get_or<double>("scale_depth", 1.0);
    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    residual_scale_ = scale_depth / std::sqrt(static_cast<double>(num_layers));

    INFINICORE_NN_MODULE_INIT(input_layernorm, model_config->get<size_t>("hidden_size"), eps, dtype, device);
    if (mixer_type == "minicpm4") {
        self_attn_ = this->register_module<MiniCPMSALAMinicpm4Attention>(
            "self_attn", model_config, device, layer_idx);
    } else {
        self_attn_ = this->register_module<MiniCPMSALALightningAttention>(
            "self_attn", model_config, device, layer_idx);
    }
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, model_config->get<size_t>("hidden_size"), eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, model_config, device);
}

infinicore::Tensor MiniCPMSALADecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                                    const infinicore::Tensor &position_ids) const {
    // Pre-norm attention
    auto hs1 = input_layernorm_->forward(hidden_states);
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
