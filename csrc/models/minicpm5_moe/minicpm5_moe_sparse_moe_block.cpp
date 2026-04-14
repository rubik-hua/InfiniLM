#include "minicpm5_moe_sparse_moe_block.hpp"

#include "infinicore/ops/add.hpp"

#include <stdexcept>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>

namespace infinilm::models::minicpm5_moe {

namespace {
inline uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(uint32_t));
    // truncate (no rounding) for scalar fill
    return static_cast<uint16_t>(u >> 16);
}

inline uint16_t f32_to_f16(float x) {
    // fp32 -> fp16 (minimal; for scalar fill only)
    uint32_t u;
    std::memcpy(&u, &x, sizeof(uint32_t));
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x007FFFFFu;
    if (exp <= 0) {
        // underflow to zero
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 0x1F) {
        // overflow to inf
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    uint16_t out = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    return out;
}

inline float bf16_to_f32(uint16_t x) {
    uint32_t u = uint32_t(x) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

inline float f16_to_f32(uint16_t x) {
    // IEEE fp16 -> fp32 conversion (minimal, for scalar reads only)
    uint32_t sign = (x & 0x8000u) << 16;
    uint32_t exp = (x & 0x7C00u) >> 10;
    uint32_t mant = (x & 0x03FFu);
    uint32_t u = 0;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            // subnormal
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            u = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

inline float scalar_to_f32(const infinicore::Tensor &t, size_t idx) {
    auto dtype = t->dtype();
    const std::byte *p = t->data();
    if (dtype == infinicore::DataType::F32) {
        float v;
        std::memcpy(&v, p + idx * sizeof(float), sizeof(float));
        return v;
    }
    if (dtype == infinicore::DataType::BF16) {
        uint16_t v;
        std::memcpy(&v, p + idx * sizeof(uint16_t), sizeof(uint16_t));
        return bf16_to_f32(v);
    }
    if (dtype == infinicore::DataType::F16) {
        uint16_t v;
        std::memcpy(&v, p + idx * sizeof(uint16_t), sizeof(uint16_t));
        return f16_to_f32(v);
    }
    throw std::runtime_error("MiniCPM5MoeSparseMoeBlock: unsupported scalar dtype");
}

inline int32_t scalar_to_i32(const infinicore::Tensor &t, size_t idx) {
    const std::byte *p = t->data();
    int32_t v;
    std::memcpy(&v, p + idx * sizeof(int32_t), sizeof(int32_t));
    return v;
}

/// Write one float into `t` at flat index `idx` using `t`'s dtype (HF casts f32 routed sums back to activation dtype).
inline void write_f32_as_element(infinicore::Tensor t, size_t idx, float v) {
    auto dtype = t->dtype();
    std::byte *p = t->data();
    if (dtype == infinicore::DataType::F32) {
        std::memcpy(p + idx * sizeof(float), &v, sizeof(float));
    } else if (dtype == infinicore::DataType::BF16) {
        uint16_t x = f32_to_bf16(v);
        std::memcpy(p + idx * sizeof(uint16_t), &x, sizeof(uint16_t));
    } else if (dtype == infinicore::DataType::F16) {
        uint16_t x = f32_to_f16(v);
        std::memcpy(p + idx * sizeof(uint16_t), &x, sizeof(uint16_t));
    } else {
        throw std::runtime_error("MiniCPM5MoeSparseMoeBlock: write_f32_as_element unsupported dtype");
    }
}

inline float sigmoid_f32(float x) {
    // Stable enough for our typical router logits.
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

inline void topk_indices_desc(const std::vector<float> &vals, size_t k, std::vector<int32_t> &out_idx) {
    // Select k indices with largest vals. Not sorted, matches HF `sorted=False`.
    const size_t n = vals.size();
    out_idx.resize(k);
    std::vector<int32_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = static_cast<int32_t>(i);
    if (k >= n) {
        out_idx.assign(idx.begin(), idx.end());
        return;
    }
    auto nth = idx.begin() + static_cast<std::ptrdiff_t>(k);
    std::nth_element(idx.begin(), nth, idx.end(), [&](int32_t a, int32_t b) { return vals[a] > vals[b]; });
    out_idx.assign(idx.begin(), nth);
}
} // namespace

MiniCPM5MoeSparseMoeBlock::MiniCPM5MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                     const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t num_experts = model_config->get_or<size_t>("n_routed_experts", 0);
    if (num_experts == 0) {
        num_experts = 1;
    }

    INFINICORE_NN_MODULE_INIT(gate, hidden_size, num_experts, false, dtype, device);
    // Keep bias in the model compute dtype to avoid mixed-dtype add producing NaNs.
    // (We can revisit a clean cast-to-f32 path later if needed.)
    INFINICORE_NN_PARAMETER_INIT(e_score_correction_bias, ({num_experts}, dtype, device, 0, 0, 1));

    experts_.reserve(num_experts);
    for (size_t i = 0; i < num_experts; ++i) {
        experts_.push_back(this->register_module<MiniCPM5MoeMLP>("experts." + std::to_string(i), model_config, device));
    }

    // HF: `shared_experts = MiniCPM5MoEMLP(..., intermediate_size=moe_intermediate_size * n_shared_experts)`.
    const size_t n_shared_experts = model_config->get_or<size_t>("n_shared_experts", 1);
    const size_t moe_intermediate = model_config->get<size_t>("moe_intermediate_size");
    const size_t shared_intermediate = moe_intermediate * n_shared_experts;
    INFINICORE_NN_MODULE_INIT(shared_experts, model_config, device, shared_intermediate);
}

infinicore::Tensor MiniCPM5MoeSparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    // Correctness-first (slow) CPU-style dispatch.
    //
    // TODO(opt): batch tokens per expert and fuse router+dispatch on-device.
    // This per-token loop is only for bringing up correctness and logit sanity.

    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hidden_size = shape[2];
    const size_t n_tokens = batch_size * seq_len;

    const size_t top_k = infinilm::global_state::get_infinilm_config().model_config->get<size_t>("num_experts_per_tok");
    const bool norm_topk_prob = infinilm::global_state::get_infinilm_config().model_config->get_or<bool>("norm_topk_prob", true);
    const float routed_scaling_factor = static_cast<float>(
        infinilm::global_state::get_infinilm_config().model_config->get<double>("routed_scaling_factor"));
    const size_t n_group = infinilm::global_state::get_infinilm_config().model_config->get_or<size_t>("n_group", 1);
    const size_t topk_group = infinilm::global_state::get_infinilm_config().model_config->get_or<size_t>("topk_group", 1);

    const size_t n_routed_experts = experts_.size();
    if (n_group == 0 || topk_group == 0 || n_routed_experts == 0 || (n_routed_experts % n_group) != 0) {
        throw std::runtime_error("MiniCPM5MoeSparseMoeBlock: invalid n_group/topk_group/n_routed_experts");
    }
    const size_t experts_per_group = n_routed_experts / n_group;

    auto hs2d = hidden_states->view({n_tokens, hidden_size});

    // HF reference routing:
    // router_logits = linear(hidden_states.float32, weight.float32)
    // scores = sigmoid(router_logits)
    // scores_for_choice = scores + e_score_correction_bias
    // group_scores = sum(top2(scores_for_choice per-group))
    // select topk_group groups, mask others to 0 in scores_for_choice
    // topk_indices = topk(scores_for_choice, k=top_k, sorted=False)
    // topk_weights = scores.gather(topk_indices)
    // optional renorm + scaling
    //
    // TODO(opt): move this router back to device / use `topkrouter` op.

    auto hs_cpu = hs2d->to(infinicore::Device::cpu());
    hs_cpu = hs_cpu->contiguous();
    auto w_cpu = gate_->weight()->to(infinicore::Device::cpu());
    w_cpu = w_cpu->contiguous();
    auto bias_cpu = e_score_correction_bias_->to(infinicore::Device::cpu());
    bias_cpu = bias_cpu->contiguous();

    std::vector<std::vector<int32_t>> topk_indices_cpu(n_tokens);
    std::vector<std::vector<float>> topk_weights_cpu(n_tokens);
    std::vector<float> scores_row(n_routed_experts);
    std::vector<float> scores_for_choice(n_routed_experts);
    std::vector<float> group_scores(n_group);
    std::vector<int32_t> chosen_groups;
    std::vector<int32_t> chosen_experts;

    for (size_t t = 0; t < n_tokens; ++t) {
        // Compute router logits in f32 on CPU from (possibly bf16/f16) hs + weight.
        for (size_t e = 0; e < n_routed_experts; ++e) {
            float acc = 0.0f;
            for (size_t i = 0; i < hidden_size; ++i) {
                float x = scalar_to_f32(hs_cpu, t * hidden_size + i);
                float w = scalar_to_f32(w_cpu, e * hidden_size + i);
                acc += x * w;
            }
            scores_row[e] = sigmoid_f32(acc);
        }

        // scores_for_choice = scores + e_score_correction_bias
        for (size_t e = 0; e < n_routed_experts; ++e) {
            float b = scalar_to_f32(bias_cpu, e);
            scores_for_choice[e] = scores_row[e] + b;
        }

        // group_scores: sum of top2 per group (using scores_for_choice)
        for (size_t g = 0; g < n_group; ++g) {
            float m1 = -std::numeric_limits<float>::infinity();
            float m2 = -std::numeric_limits<float>::infinity();
            size_t base = g * experts_per_group;
            for (size_t j = 0; j < experts_per_group; ++j) {
                float v = scores_for_choice[base + j];
                if (v > m1) {
                    m2 = m1;
                    m1 = v;
                } else if (v > m2) {
                    m2 = v;
                }
            }
            group_scores[g] = m1 + m2;
        }

        // choose topk_group groups
        topk_indices_desc(group_scores, topk_group, chosen_groups);

        // mask scores_for_choice: groups not selected -> 0
        std::vector<uint8_t> group_keep(n_group, 0);
        for (auto g : chosen_groups) group_keep[static_cast<size_t>(g)] = 1;
        for (size_t g = 0; g < n_group; ++g) {
            if (group_keep[g]) continue;
            size_t base = g * experts_per_group;
            for (size_t j = 0; j < experts_per_group; ++j) {
                scores_for_choice[base + j] = 0.0f;
            }
        }

        // topk experts on masked scores_for_choice
        topk_indices_desc(scores_for_choice, top_k, chosen_experts);

        // topk_weights from unmasked sigmoid scores (HF uses `scores.gather`)
        topk_indices_cpu[t] = chosen_experts;
        topk_weights_cpu[t].resize(top_k);
        float denom = 0.0f;
        for (size_t j = 0; j < top_k; ++j) {
            float w = scores_row[static_cast<size_t>(chosen_experts[j])];
            topk_weights_cpu[t][j] = w;
            denom += w;
        }
        if (norm_topk_prob) {
            denom += 1e-20f;
            float inv = 1.0f / denom;
            for (size_t j = 0; j < top_k; ++j) topk_weights_cpu[t][j] *= inv;
        }
        for (size_t j = 0; j < top_k; ++j) topk_weights_cpu[t][j] *= routed_scaling_factor;
    }

    // HF `moe()`: `final_hidden_states = zeros_like(..., dtype=topk_weights.dtype)` — float32 accumulator.
    // Keep the full routed buffer on CPU (float32) and upload once to avoid fragile D2D `copy_from` into narrowed views.
    auto out2d_cpu = infinicore::Tensor::zeros({n_tokens, hidden_size}, infinicore::DataType::F32, infinicore::Device::cpu());
    float *out_flat = reinterpret_cast<float *>(out2d_cpu->data());

    for (size_t t = 0; t < n_tokens; ++t) {
        float *row_acc = out_flat + t * hidden_size;
        std::memset(row_acc, 0, hidden_size * sizeof(float));
        for (size_t j = 0; j < top_k; ++j) {
            int32_t expert_id = topk_indices_cpu[t][j];
            if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) {
                continue;
            }
            float w = topk_weights_cpu[t][j];
            if (w == 0.0f) {
                continue;
            }

            auto token_in = hs2d->narrow({{0, t, 1}}); // [1, H]
            auto token_out = experts_.at(static_cast<size_t>(expert_id))->forward(token_in); // [1, H]

            auto tok_on_cpu = token_out->to(infinicore::Device::cpu());
            tok_on_cpu = tok_on_cpu->contiguous();
            for (size_t i = 0; i < hidden_size; ++i) {
                row_acc[i] += scalar_to_f32(tok_on_cpu, i) * w;
            }
        }
    }

    // HF `moe()` return: `.type(hidden_states.dtype)` before adding shared experts.
    // Cast on CPU then one H2D upload — avoids fragile device-side rearrange on fp32 buffers.
    infinicore::Tensor routed;
    if (hidden_states->dtype() == infinicore::DataType::F32) {
        routed = out2d_cpu->to(hidden_states->device())->view({batch_size, seq_len, hidden_size});
    } else {
        auto routed_cpu =
            infinicore::Tensor::empty({n_tokens, hidden_size}, hidden_states->dtype(), infinicore::Device::cpu());
        const size_t numel = n_tokens * hidden_size;
        for (size_t i = 0; i < numel; ++i) {
            write_f32_as_element(routed_cpu, i, scalar_to_f32(out2d_cpu, i));
        }
        routed = routed_cpu->to(hidden_states->device())->view({batch_size, seq_len, hidden_size});
    }

    auto shared = shared_experts_->forward(hidden_states);
    return infinicore::op::add(routed, shared);
}

} // namespace infinilm::models::minicpm5_moe

