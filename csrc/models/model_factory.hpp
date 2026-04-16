#pragma once

#include "../backends/attention_backends.hpp"
#include "../engine/distributed/distributed.hpp"
#include "infinilm_model.hpp"

namespace infinilm {
class InfinilmModelFactory {
public:
    // Legacy LlamaConfig-based overload; removed in v0.2.0. The [[deprecated]]
    // attribute fires a compile-time warning at every call site.
    [[deprecated("Use the ModelConfig-based createModel; removed in v0.2.0")]]
    static std::shared_ptr<InfinilmModel> createModel(
        const InfinilmModel::Config &config,
        engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
        const cache::CacheConfig *cache = nullptr,
        backends::AttentionBackend attention_backend = backends::AttentionBackend::Default);

    // RankInfo-based overload for the USE_CLASSIC_LLAMA build (llama / qwen2 /
    // minicpm / fm9g / fm9g7b fall through to LlamaForCausalLM here). The
    // registry-driven overload below is preferred for all other model_types.
    static std::shared_ptr<InfinilmModel> createModel(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
        const cache::CacheConfig *cache = nullptr,
        backends::AttentionBackend attention_backend = backends::AttentionBackend::Default);

    static std::shared_ptr<InfinilmModel> createModel(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device,
        const cache::CacheConfig *cache = nullptr);
};
} // namespace infinilm
