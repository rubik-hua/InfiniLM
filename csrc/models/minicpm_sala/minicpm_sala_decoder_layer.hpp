#pragma once

#include "minicpm_sala_attention.hpp"
#include "minicpm_sala_mlp.hpp"

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"

#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <string>

namespace infinilm::models::minicpm_sala {

class MiniCPMSALAModel;

class MiniCPMSALADecoderLayer : public infinicore::nn::Module {
public:
    MiniCPMSALADecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            const infinicore::Device &device,
                            size_t layer_idx,
                            const std::string &mixer_type,
                            engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
                            backends::AttentionBackend attention_backend = backends::AttentionBackend::Default);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &position_ids) const;

private:
    friend class MiniCPMSALAModel;

    double residual_scale_ = 1.0;
    size_t layer_idx_ = 0;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(MiniCPMSALAAttention, self_attn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiniCPMSALAMLP, mlp);
};

} // namespace infinilm::models::minicpm_sala
