#include "minicpm_sala_model.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace infinilm::models::minicpm_sala {

MiniCPMSALAModel::MiniCPMSALAModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   const infinicore::Device &device,
                                   engine::distributed::RankInfo rank_info,
                                   backends::AttentionBackend attention_backend)
    : model_config_(std::move(model_config)),
      rank_info_(rank_info),
      attention_backend_(attention_backend) {

    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config_->get_dtype();
    compute_device_ = device;

    hidden_size_ = model_config_->get<size_t>("hidden_size");
    dim_model_base_ = model_config_->get_or<double>("dim_model_base", static_cast<double>(hidden_size_));
    scale_emb_ = model_config_->get_or<double>("scale_emb", 1.0);

    const size_t vocab_size = model_config_->get<size_t>("vocab_size");
    const size_t num_layers = model_config_->get<size_t>("num_hidden_layers");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size_, std::nullopt, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, model_config_->get<double>("rms_norm_eps"), dtype, device);

    // Shared rotary embedding (used by lightning layers only)
    INFINICORE_NN_MODULE_INIT(rotary_emb,
                              model_config_->get_head_dim(),
                              model_config_->get<size_t>("max_position_embeddings"),
                              model_config_->get<double>("rope_theta"),
                              infinicore::nn::RoPE::Algo::GPT_NEOX,
                              dtype,
                              device,
                              model_config_->get_rope_scaling());

    // Mixer types per-layer decide attention flavor (minicpm4 vs lightning-attn).
    std::vector<std::string> mixer_types;
    try {
        mixer_types = model_config_->get<std::vector<std::string>>("mixer_types");
    } catch (...) {
        mixer_types.assign(num_layers, "minicpm4");
    }
    if (mixer_types.size() != num_layers) {
        mixer_types.resize(num_layers, mixer_types.empty() ? "minicpm4" : mixer_types.back());
    }
    mixer_types_ = mixer_types;

    layers_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.push_back(this->register_module<MiniCPMSALADecoderLayer>(
            "layers." + std::to_string(i), model_config_, device, i, mixer_types[i], rank_info_, attention_backend_));
        layers_.back()->set_rotary_emb(rotary_emb_);
    }
}

void MiniCPMSALAModel::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        kv_cache_minicpm4_ = nullptr;
        kv_cache_lightning_ = nullptr;
        for (auto &layer : layers_) {
            layer->reset_cache();
        }
        return;
    }

    if (auto static_cfg = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        // Allocate separate caches by KV shape to avoid per-layer padding copies.
        const size_t num_hidden_layers = model_config_->get<size_t>("num_hidden_layers");
        // mixer_types_ is filled in ctor from model_config_->get("mixer_types").
        const size_t minicpm4_layer_count =
            !mixer_types_.empty() ? std::count(mixer_types_.begin(), mixer_types_.end(), "minicpm4") : num_hidden_layers;
        const size_t lightning_layer_count = num_hidden_layers - minicpm4_layer_count;

        const size_t base_kv_heads = model_config_->get<size_t>("num_key_value_heads");
        const size_t base_head_dim = model_config_->get<size_t>("head_dim");
        const size_t lightning_kv_heads = model_config_->get_or<size_t>("lightning_nkv", base_kv_heads);
        const size_t lightning_head_dim = model_config_->get_or<size_t>("lightning_head_dim", base_head_dim);

        kv_cache_minicpm4_ = (minicpm4_layer_count > 0)
                                 ? std::make_shared<cache::StaticKVCache>(
                                       /*k_dim=*/base_head_dim,
                                       /*v_dim=*/base_head_dim,
                                       /*num_k_heads=*/base_kv_heads,
                                       /*num_v_heads=*/base_kv_heads,
                                       /*num_layers=*/minicpm4_layer_count,
                                       /*max_positional_embedding=*/model_config_->get<size_t>("max_position_embeddings"),
                                       /*dtype=*/model_config_->get_dtype(),
                                       *static_cfg,
                                       rank_info_)
                                 : nullptr;

        kv_cache_lightning_ = (lightning_layer_count > 0)
                                   ? std::make_shared<cache::StaticKVCache>(
                                         /*k_dim=*/lightning_head_dim,
                                         /*v_dim=*/lightning_head_dim,
                                         /*num_k_heads=*/lightning_kv_heads,
                                         /*num_v_heads=*/lightning_kv_heads,
                                         /*num_layers=*/lightning_layer_count,
                                         /*max_positional_embedding=*/model_config_->get<size_t>("max_position_embeddings"),
                                         /*dtype=*/model_config_->get_dtype(),
                                         *static_cfg,
                                         rank_info_)
                                 : nullptr;
    } else {
        // This refactor implements HF-like dense caching only.
        throw std::runtime_error("MiniCPMSALAModel::reset_cache: Unsupported cache type (expected StaticKVCacheConfig)");
    }

    for (auto &layer : layers_) {
        layer->reset_cache();
    }
}

infinicore::Tensor MiniCPMSALAModel::forward(const infinicore::Tensor &input_ids,
                                             const infinicore::Tensor &position_ids,
                                             std::optional<infinicore::Tensor> past_sequence_lengths,
                                             std::optional<infinicore::Tensor> total_sequence_lengths,
                                             std::optional<infinicore::Tensor> input_offsets,
                                             std::optional<infinicore::Tensor> cu_seqlens,
                                             std::optional<infinicore::Tensor> block_tables,
                                             std::optional<infinicore::Tensor> slot_mapping) const {
    // MuP scaling baked into weights at load time for minicpm_sala; no forward scaling here.
    auto hs = embed_tokens_->forward(input_ids);

    for (size_t i = 0; i < layers_.size(); ++i) {
        std::shared_ptr<cache::Cache> layer_cache;
        if (!mixer_types_.empty() && mixer_types_[i] == "minicpm4") {
            layer_cache = kv_cache_minicpm4_;
        } else {
            layer_cache = kv_cache_lightning_;
        }
        hs = layers_[i]->forward(hs,
                                 position_ids,
                                 layer_cache,
                                 past_sequence_lengths,
                                 total_sequence_lengths,
                                 input_offsets,
                                 cu_seqlens,
                                 block_tables,
                                 slot_mapping);
        if (const char *env = std::getenv("MINICPM_SALA_LAYER_TRACE")) {
            if (env[0] != '\0' && env[0] != '0') {
                fprintf(stderr, "[minicpm_sala][layer_trace] layer=%zu mixer=%s\n",
                        i,
                        mixer_types_.empty() ? "unknown" : mixer_types_[i].c_str());
                fflush(stderr);
            }
        }
    }

    hs = norm_->forward(hs);
    return hs;
}

} // namespace infinilm::models::minicpm_sala
