#include "minicpm5_moe_sparse_moe_block.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm5_moe {

MiniCPM5MoeSparseMoeBlock::MiniCPM5MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                     const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t num_experts = model_config->get_or<size_t>("num_experts", 0);

    // If the checkpoint doesn't declare experts, fall back to a single MLP expert.
    if (num_experts == 0) {
        num_experts = 1;
    }

    INFINICORE_NN_MODULE_INIT(gate, hidden_size, num_experts, false, dtype, device);

    experts_.reserve(num_experts);
    for (size_t i = 0; i < num_experts; ++i) {
        experts_.push_back(this->register_module<MiniCPM5MoeMLP>("experts." + std::to_string(i), model_config, device));
    }

    // Optional shared expert (kept for compatibility with Qwen3-MoE-style configs).
    size_t shared_expert_intermediate_size = model_config->get_or<size_t>("shared_expert_intermediate_size", 0);
    if (shared_expert_intermediate_size > 0) {
        INFINICORE_NN_MODULE_INIT(shared_expert, model_config, device);
        INFINICORE_NN_MODULE_INIT(shared_expert_gate, hidden_size, 1, false, dtype, device);
    }
}

infinicore::Tensor MiniCPM5MoeSparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    if (experts_.empty()) {
        throw std::runtime_error("MiniCPM5MoeSparseMoeBlock: no experts initialized");
    }
    // Minimal non-routing behavior: execute a single expert.
    return experts_[0]->forward(hidden_states);
}

} // namespace infinilm::models::minicpm5_moe

