# 拆分 fm9g 注册的模型类型 + attention_bias 规范化

## Context

fm9g_for_causal_lm.cpp 当前注册了 5 种 model_type（llama、qwen2、fm9g、fm9g7b、minicpm），共用同一个 config creator，没有处理 attention_bias。但不同模型在 HuggingFace transformers 中的 attention_bias 设置不同：
- **LLaMA**: QKV bias=False（LlamaConfig 默认值）
- **Qwen2**: QKV bias=True（硬编码在 Qwen2Attention 中）
- **MiniCPM**: QKV bias=True
- **fm9g/fm9g7b**: QKV bias=True（内部模型）

目标：拆成独立目录，每个模型用自己的 config creator 设置正确的 attention_bias。

## PR 拆分方案

### PR1: 拆出 qwen2

**新建文件：**
- `csrc/models/qwen2/qwen2_for_causal_lm.hpp`
- `csrc/models/qwen2/qwen2_for_causal_lm.cpp`

hpp 内容：跟 mistral 相同模式，类型别名组合（复用标准 Attention + MLP）

cpp 关键逻辑：
```cpp
create_qwen2_model_config:
  - 校验 model_type == "qwen2"
  - 补全 head_dim（如果没有）
  - if (!contains("attention_bias")) → attention_bias = true   // 对齐 transformers Qwen2Attention
  - if (!contains("attention_output_bias")) → attention_output_bias = false
```

**修改文件：**
- `csrc/models/fm9g/fm9g_for_causal_lm.cpp` — 删除 qwen2 的注册宏

### PR2: 拆出 minicpm

**新建文件：**
- `csrc/models/minicpm/minicpm_for_causal_lm.hpp`
- `csrc/models/minicpm/minicpm_for_causal_lm.cpp`

cpp 关键逻辑：
```cpp
create_minicpm_model_config:
  - 校验 model_type == "minicpm"
  - 补全 head_dim
  - if (!contains("attention_bias")) → attention_bias = true
  - if (!contains("attention_output_bias")) → attention_output_bias = false
```

**修改文件：**
- `csrc/models/fm9g/fm9g_for_causal_lm.cpp` — 删除 minicpm 的注册宏

### PR3: 替换旧 llama + 清理 fm9g

**重命名：**
- `csrc/models/llama/` → `csrc/models/llama_legacy/`（旧实现保留但不再作为主路径）

**新建文件：**
- `csrc/models/llama/llama_for_causal_lm.hpp`
- `csrc/models/llama/llama_for_causal_lm.cpp`

cpp 关键逻辑：
```cpp
create_llama_model_config:
  - 校验 model_type == "llama"
  - 补全 head_dim
  - if (!contains("attention_bias")) → attention_bias = false   // 对齐 transformers LlamaConfig
  - if (!contains("attention_output_bias")) → attention_output_bias = false
```

**修改文件：**
- `csrc/models/fm9g/fm9g_for_causal_lm.cpp` — 删除 llama 注册宏，只保留 fm9g 和 fm9g7b
- `csrc/models/llama_legacy/` 中的文件 — 可能需要调整避免与新 llama/ 编译冲突（视 USE_CLASSIC_LLAMA 宏而定）

## attention_bias 对照表

| model_type | config creator 中默认值 | 依据（transformers） |
|------------|----------------------|-------------------|
| llama | false | LlamaConfig 默认 attention_bias=False |
| qwen2 | true | Qwen2Attention 硬编码 bias=True |
| mistral | false | 已实现 |
| minicpm | true | MiniCPM 模型有权重中有 bias |
| fm9g | true（保持 get_or 默认值） | 内部模型，不改动 |
| fm9g7b | true（保持 get_or 默认值） | 内部模型，不改动 |

## 各模型文件模板（以 qwen2 为例）

**qwen2_for_causal_lm.hpp：**
```cpp
#pragma once
#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::qwen2 {

using Qwen2MLP = infinilm::layers::MLP;
using Qwen2Attention = infinilm::layers::attention::Attention;
using Qwen2DecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<Qwen2Attention, Qwen2MLP>;
using Qwen2Model = infinilm::layers::causal_lm_templates::TextModel<Qwen2DecoderLayer>;
using Qwen2ForCausalLM = infinilm::layers::causal_lm_templates::TextCausalLM<Qwen2Model>;

std::shared_ptr<infinilm::config::ModelConfig> create_qwen2_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen2
```

**qwen2_for_causal_lm.cpp：**
```cpp
#include "qwen2_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::qwen2 {

std::shared_ptr<infinilm::config::ModelConfig> create_qwen2_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("qwen2" != model_type) {
        throw std::runtime_error("...");
    }
    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = model_config->get<size_t>("hidden_size")
            / model_config->get<size_t>("num_attention_heads");
    }
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = true;  // Qwen2 硬编码 bias=True
    }
    if (!config_json.contains("attention_output_bias")) {
        config_json["attention_output_bias"] = false;
    }
    return model_config;
}

} // namespace infinilm::models::qwen2

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen2,
    infinilm::models::qwen2::Qwen2ForCausalLM,
    infinilm::models::qwen2::create_qwen2_model_config);
} // namespace
```

## 验证方式

每个 PR 合入后：

```bash
# 编译
cd InfiniLM && xmake build

# PR1 后：测试 qwen2 模型
CUDA_VISIBLE_DEVICES=1 python examples/jiuge.py --nvidia \
  --model=/data-aisoft/mechdancer/models/Qwen2.5-0.5B-Instruct-AWQ

# PR2 后：测试 minicpm 模型（如有的话）

# PR3 后：测试 llama 模型
CUDA_VISIBLE_DEVICES=1 python examples/jiuge.py --nvidia \
  --model=/data-aisoft/mechdancer/models/Llama-3.2-1B-Instruct

# PR3 后：测试 Yi-6B（config 无 attention_bias，默认 false）
CUDA_VISIBLE_DEVICES=1 python examples/jiuge.py --nvidia \
  --model=/data-models/llms/hf_llm_models/01-ai_Yi-6B

# 全部完成后：测试 mistral（确保不受影响）
CUDA_VISIBLE_DEVICES=1 python examples/jiuge.py --nvidia \
  --model=/data-models/llms/hf_llm_models/mistralai_Mistral-7B-Instruct-v0.2
```
