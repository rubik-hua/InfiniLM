#pragma once

#include "minicpm_sala_decoder_layer.hpp"

#include "../../backends/attention_backends.hpp"
#include "../../cache/cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"

#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace infinilm::models::minicpm_sala {

class MiniCPMSALAModel : public infinicore::nn::Module {
public:
    MiniCPMSALAModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                               const infinicore::Tensor &position_ids) const;

    void reset_state();

    size_t hidden_size() const { return hidden_size_; }
    double dim_model_base() const { return dim_model_base_; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(MiniCPMSALADecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

private:
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    infinicore::Device compute_device_;

    size_t hidden_size_;
    double scale_emb_;
    double dim_model_base_;
};

} // namespace infinilm::models::minicpm_sala

