#include "model_factory.hpp"
#include <unordered_set>
#include "llama/llama_for_causal_lm.hpp"
#include "models_registry.hpp"

namespace infinilm {
// Deprecated legacy-config overload; removed in v0.2.0. New code must use
// the ModelConfig-based overloads below.
std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    const InfinilmModel::Config &config,
    engine::distributed::RankInfo rank_info,
    const cache::CacheConfig *cache,
    backends::AttentionBackend attention_backend) {
    std::shared_ptr<InfinilmModel> model;
    if (const auto llama_config_ptr = dynamic_cast<const models::llama::LlamaConfig *>(&config)) {
        const auto &llama_config = *llama_config_ptr;
        model = std::make_shared<models::llama::LlamaForCausalLM>(
            llama_config, rank_info.device, rank_info, attention_backend);
    } else {
        throw std::invalid_argument("InfinilmModelFactory::createModel: Unsupported model config type");
    }

    if (cache) {
        model->reset_cache(cache);
    }

    return model;
}

std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    engine::distributed::RankInfo rank_info,
    const cache::CacheConfig *cache,
    backends::AttentionBackend attention_backend) {

    const auto model_type = model_config->get_or<std::string>("model_type", "");
    if (model_type.empty()) {
        throw std::invalid_argument(
            "InfinilmModelFactory::createModel: 'model_type' field is missing or empty in config");
    }
    // model_types whose weight layout is compatible with LlamaForCausalLM.
    // qwen3 is intentionally excluded: it has its own Qwen3ForCausalLM class
    // (registered via INFINILM_REGISTER_CAUSAL_LM_MODEL) and is handled by the
    // registry-driven overload below. Only keep entries here that are registered
    // to fall through to LlamaForCausalLM when USE_CLASSIC_LLAMA is set.
    static const std::unordered_set<std::string> llama_compatible = {
        "llama", "qwen2", "minicpm", "fm9g", "fm9g7b"
    };

    std::shared_ptr<InfinilmModel> model;
    if (llama_compatible.count(model_type)) {
        model = std::make_shared<models::llama::LlamaForCausalLM>(
            model_config, rank_info.device, rank_info, attention_backend);
    } else {
        throw std::invalid_argument(
            "InfinilmModelFactory::createModel: Unsupported model_type '" + model_type + "'");
    }

    if (cache) {
        model->reset_cache(cache);
    }

    return model;
}

std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device,
    const cache::CacheConfig *cache) {
    const std::string model_type = model_config->get<std::string>("model_type");
    std::shared_ptr<InfinilmModel> model;
    const auto &model_map = models::get_causal_lm_model_map();
    auto it = model_map.find(model_type);
    if (it != model_map.end()) {
        auto &model_creator = it->second;
        model = model_creator(model_config, device);
    } else {
        throw std::invalid_argument(
            "InfinilmModelFactory::createModel: Unsupported model_type '" + model_type + "'");
    }

    if (cache) {
        model->reset_cache(cache);
    }
    return model;
}
} // namespace infinilm
