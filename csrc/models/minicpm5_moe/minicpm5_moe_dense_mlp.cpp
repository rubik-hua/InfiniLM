#include "minicpm5_moe_dense_mlp.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::minicpm5_moe {

MiniCPM5DenseMLP::MiniCPM5DenseMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("intermediate_size");
    const bool use_bias = model_config->get_or<bool>("mlp_bias", false);

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;

    auto quant_scheme = model_config->get_quant_scheme();
    auto quantization_method = model_config->get_quantization_method();
    switch (quant_scheme) {
    case infinicore::quantization::QuantScheme::NONE: {
        INFINICORE_NN_MODULE_INIT(gate_proj, hidden_size, intermediate_size, quantization_method,
                                  use_bias, dtype, device, tp_rank, tp_size);
        INFINICORE_NN_MODULE_INIT(up_proj, hidden_size, intermediate_size, quantization_method,
                                  use_bias, dtype, device, tp_rank, tp_size);
        INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size, hidden_size, quantization_method,
                                  use_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    default:
        throw std::runtime_error("MiniCPM5DenseMLP: unsupported quantization scheme");
    }
}

infinicore::Tensor MiniCPM5DenseMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto gate = gate_proj_->forward(hidden_states_mutable);
    auto up = up_proj_->forward(hidden_states_mutable);
    auto intermediate = infinicore::op::swiglu(up, gate);
    return down_proj_->forward(intermediate);
}

} // namespace infinilm::models::minicpm5_moe

