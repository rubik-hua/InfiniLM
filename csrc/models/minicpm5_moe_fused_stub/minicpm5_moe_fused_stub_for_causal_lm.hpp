#pragma once

#include "../../layers/common_modules.hpp"
#include "minicpm5_moe_fused_stub_decoder_layer.hpp"
#include <memory>

namespace infinilm::models::minicpm5_moe_fused_stub {

using MiniCPM5MoeFusedStubModel =
    infinilm::layers::causal_lm_templates::TextModel<MiniCPM5MoeFusedStubDecoderLayer>;

using MiniCPM5MoeFusedStubForCausalLM =
    infinilm::layers::causal_lm_templates::TextCausalLM<MiniCPM5MoeFusedStubModel>;

std::shared_ptr<infinilm::config::ModelConfig>
create_minicpm5_moe_fused_stub_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpm5_moe_fused_stub
