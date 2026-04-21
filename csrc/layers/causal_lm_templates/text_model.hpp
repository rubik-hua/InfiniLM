#pragma once

#include "../../config/model_config.hpp"
#include "../../models/infinilm_model.hpp"
#include "../../ops_shim/ops_shim.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <vector>

namespace infinilm::layers::causal_lm_templates {

/**
 * @brief Text model architecture (without language modeling head).
 *
 * Generic transformer model consisting of:
 * - Token embeddings
 * - Multiple decoder layers
 * - Final layer normalization
 *
 * @tparam DecoderLayer The decoder layer type (e.g., Qwen3DecoderLayer)
 */
template <typename DecoderLayer>
class TextModel : public infinicore::nn::Module {
public:
    TextModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              const infinicore::Device &device) {
        const auto &dtype{model_config->get_dtype()};
        size_t vocab_size = model_config->get<size_t>("vocab_size");
        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        double rope_theta = model_config->get<double>("rope_theta");
        double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        embed_tokens_ = this->register_module<infinicore::nn::Embedding>("embed_tokens", vocab_size, hidden_size, std::nullopt, dtype, device);

        layers_.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            layers_.push_back(this->register_module<DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
        }

        norm_ = this->register_module<infinicore::nn::RMSNorm>("norm", hidden_size, rms_norm_eps, dtype, device);
    }

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const {
        auto input_ids = input.input_ids.value();
        auto positions = input.position_ids.value();
        // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
        auto hidden_states = infinilm::ops_shim::embedding(input_ids, embed_tokens_->weight());

        // 2. Process through all decoder layers
        size_t num_layers = layers_.size();
        infinicore::Tensor residual;
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.at(i)->forward(
                positions,
                hidden_states,
                residual);
        }

        infinilm::ops_shim::rms_norm_forward_inplace(
            hidden_states, residual,
            static_cast<const infinicore::Tensor &>(norm_->weight()),
            static_cast<float>(norm_->eps()));
        return hidden_states;
    }

    infinicore::Tensor forward_naive(const infinilm::InfinilmModel::Input &input) const {
        auto input_ids = input.input_ids.value();
        auto positions = input.position_ids.value();
        auto hidden_states = infinilm::ops_shim::embedding(input_ids, embed_tokens_->weight());
        size_t num_layers = layers_.size();
        for (size_t i = 0; i < num_layers; ++i) {
            hidden_states = layers_.at(i)->forward(positions, hidden_states);
        }
        hidden_states = norm_->forward(hidden_states);
        return hidden_states;
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
};

} // namespace infinilm::layers::causal_lm_templates
