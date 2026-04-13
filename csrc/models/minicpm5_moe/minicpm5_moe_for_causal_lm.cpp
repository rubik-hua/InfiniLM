#include "minicpm5_moe_for_causal_lm.hpp"
#include "../models_registry.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm5_moe {

std::shared_ptr<infinilm::config::ModelConfig>
create_minicpm5_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("minicpm5_moe" != model_type) {
        throw std::runtime_error("create_minicpm5_moe_model_config: model_type is not minicpm5_moe");
    }

    // Provide defaults expected by the current attention implementation.
    auto &j = model_config->get_config_json();
    if (!j.contains("rope_theta")) {
        j["rope_theta"] = 10000.0;
    }

    // Map HF MoE naming to InfiniLM fields used by the MoE block.
    if (!j.contains("num_experts") && j.contains("n_routed_experts")) {
        j["num_experts"] = j["n_routed_experts"];
    }
    return model_config;
}

} // namespace infinilm::models::minicpm5_moe

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minicpm5_moe,
    infinilm::models::minicpm5_moe::MiniCPM5MoeForCausalLM,
    infinilm::models::minicpm5_moe::create_minicpm5_moe_model_config);
} // namespace

