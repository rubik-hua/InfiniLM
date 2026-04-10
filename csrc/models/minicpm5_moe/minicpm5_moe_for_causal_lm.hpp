#pragma once

#include "../infinilm_model.hpp"

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"

#include "infinicore/device.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace infinilm::models::minicpm5_moe {

class MiniCPM5MoEAttention;
class MiniCPM5MoEDecoderLayer;
class MiniCPM5MoEModel;

class MiniCPM5MoEForCausalLM : public infinilm::InfinilmModel {
public:
    MiniCPM5MoEForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device,
                           engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
                           backends::AttentionBackend attention_backend = backends::AttentionBackend::Default);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;
    const cache::CacheConfig *get_cache_config() const override;

protected:
    INFINICORE_NN_MODULE(MiniCPM5MoEModel, model);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, lm_head);

    std::unique_ptr<cache::CacheConfig> cache_config_;
};

std::shared_ptr<infinilm::config::ModelConfig>
create_minicpm5_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpm5_moe

