#pragma once

#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::internlm3 {

std::shared_ptr<infinilm::config::ModelConfig> create_internlm3_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

}
