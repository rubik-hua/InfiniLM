#include "minicpm5_moe_sparse_moe_block.hpp"

#include "infinicore/ops/add.hpp"
#include "infinicore/ops/convert_to_f32.hpp"
#include "infinicore/ops/index_add.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/sigmoid.hpp"
#include "infinicore/ops/take.hpp"
#include "infinicore/ops/topk.hpp"
#include "infinicore/ops/topkrouter.hpp"
#include "../../utils/nvtx.hpp"

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
    infinilm::utils::NvtxRange nvtx_moe("MiniCPM5MoeSparseMoeBlock::forward");
    // Routing + MoE dispatch.
    //
    // - Fast path: `topkrouter` (CUDA) + per-expert batching (only for the fixed 256-expert kernel).
    // - Correctness path: CPU routing + per-token expert dispatch (used for arbitrary expert counts),
    //   matching the older bring-up implementation and HF semantics.

    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t hidden_size = shape[2];
    const size_t n_tokens = batch_size * seq_len;

    const size_t top_k = infinilm::global_state::get_infinilm_config().model_config->get<size_t>("num_experts_per_tok");
    const bool norm_topk_prob = infinilm::global_state::get_infinilm_config().model_config->get_or<bool>("norm_topk_prob", true);
    const float routed_scaling_factor = static_cast<float>(
        infinilm::global_state::get_infinilm_config().model_config->get<double>("routed_scaling_factor"));
    // The current `topkrouter` backend implements the MiniCPM5/DeepSeek-style grouped routing
    // with fixed (n_experts=256, n_group=8, topk_group=4). Ensure config matches.
    const size_t n_group = infinilm::global_state::get_infinilm_config().model_config->get_or<size_t>("n_group", 8);
    const size_t topk_group = infinilm::global_state::get_infinilm_config().model_config->get_or<size_t>("topk_group", 4);

    const size_t n_routed_experts = experts_.size();
    if (n_group == 0 || topk_group == 0 || n_routed_experts == 0 || (n_routed_experts % n_group) != 0) {
        throw std::runtime_error("MiniCPM5MoeSparseMoeBlock: invalid n_group/topk_group/n_routed_experts");
    }

    auto hs2d = hidden_states->view({n_tokens, hidden_size});

    // Router logits: do matmul in F32 for stability; `topkrouter` applies sigmoid + correction bias
    // and returns routed weights (already scaled) + expert indices on device.

    if (!gate_weight_f32_device_.has_value()) {
        gate_weight_f32_device_ =
            infinicore::op::convert_to_f32(gate_->weight()->contiguous());
    }
    if (!e_score_correction_bias_f32_device_.has_value()) {
        e_score_correction_bias_f32_device_ =
            infinicore::op::convert_to_f32(e_score_correction_bias_->contiguous());
    }

    auto hs_f32 = infinicore::op::convert_to_f32(hs2d->contiguous());
    auto router_logits =
        infinicore::op::linear(hs_f32, gate_weight_f32_device_.value(), std::nullopt);

    const bool force_fallback = (std::getenv("INFINILM_MOE_FORCE_FALLBACK") != nullptr);
    const bool can_use_topkrouter = (n_routed_experts == 256 && n_group == 8 && topk_group == 4);

    // -------------------------
    // Correctness-first fallback
    // -------------------------
    if (force_fallback || !can_use_topkrouter) {
        // CPU routing (HF-aligned, sorted=False) + per-token expert dispatch.
        auto logits_cpu = router_logits->to(infinicore::Device::cpu())->contiguous();
        auto bias_cpu = e_score_correction_bias_f32_device_.value()->to(infinicore::Device::cpu())->contiguous();

        std::vector<std::vector<int32_t>> topk_indices_cpu(n_tokens);
        std::vector<std::vector<float>> topk_weights_cpu_vec(n_tokens);
        std::vector<float> scores_row(n_routed_experts);
        std::vector<float> scores_for_choice(n_routed_experts);
        std::vector<int32_t> chosen_experts;

        for (size_t t = 0; t < n_tokens; ++t) {
            for (size_t e = 0; e < n_routed_experts; ++e) {
                float logit = scalar_to_f32(logits_cpu, t * n_routed_experts + e);
                // sigmoid
                float s;
                if (logit >= 0.0f) {
                    float z = std::exp(-logit);
                    s = 1.0f / (1.0f + z);
                } else {
                    float z = std::exp(logit);
                    s = z / (1.0f + z);
                }
                scores_row[e] = s;
            }

            for (size_t e = 0; e < n_routed_experts; ++e) {
                scores_for_choice[e] = scores_row[e] + scalar_to_f32(bias_cpu, e);
            }

            // Select k indices with largest vals. Not sorted (HF sorted=False).
            chosen_experts.resize(top_k);
            std::vector<int32_t> idx(n_routed_experts);
            for (size_t i = 0; i < n_routed_experts; ++i) idx[i] = static_cast<int32_t>(i);
            if (top_k >= n_routed_experts) {
                chosen_experts.assign(idx.begin(), idx.end());
            } else {
                auto nth = idx.begin() + static_cast<std::ptrdiff_t>(top_k);
                std::nth_element(
                    idx.begin(), nth, idx.end(),
                    [&](int32_t a, int32_t b) {
                        return scores_for_choice[static_cast<size_t>(a)] > scores_for_choice[static_cast<size_t>(b)];
                    });
                chosen_experts.assign(idx.begin(), nth);
            }

            topk_indices_cpu[t] = chosen_experts;
            topk_weights_cpu_vec[t].resize(top_k);
            float denom = 0.0f;
            for (size_t j = 0; j < top_k; ++j) {
                float w = scores_row[static_cast<size_t>(chosen_experts[j])];
                topk_weights_cpu_vec[t][j] = w;
                denom += w;
            }
            if (norm_topk_prob) {
                denom += 1e-20f;
                float inv = 1.0f / denom;
                for (size_t j = 0; j < top_k; ++j) topk_weights_cpu_vec[t][j] *= inv;
            }
            for (size_t j = 0; j < top_k; ++j) topk_weights_cpu_vec[t][j] *= routed_scaling_factor;
        }

        // Dispatch:
        // - default: per-token expert forwards + CPU FP32 row accumulation (reference numerics).
        // - INFINILM_MOE_USE_BATCHED_DISPATCH=1: experimental batched gather path (may diverge; debug only).
        //
        // GPU-side row `add_` into a dense buffer was still incorrect in end-to-end tests even after
        // including `data()` in InfiniCore tensor hashes; keep accumulation on CPU until that path is
        // root-caused (likely elsewhere in the InfiniOP / graph stack).
        const bool use_batched_dispatch = (std::getenv("INFINILM_MOE_USE_BATCHED_DISPATCH") != nullptr);

        infinicore::Tensor out2d_cpu;

        if (!use_batched_dispatch) {
            out2d_cpu =
                infinicore::Tensor::zeros({n_tokens, hidden_size}, infinicore::DataType::F32, infinicore::Device::cpu());
            float *out_flat = reinterpret_cast<float *>(out2d_cpu->data());
            for (size_t t = 0; t < n_tokens; ++t) {
                float *row_acc = out_flat + t * hidden_size;
                std::memset(row_acc, 0, hidden_size * sizeof(float));
                for (size_t j = 0; j < top_k; ++j) {
                    int32_t expert_id = topk_indices_cpu[t][j];
                    if (expert_id < 0 || static_cast<size_t>(expert_id) >= experts_.size()) continue;
                    float w = topk_weights_cpu_vec[t][j];
                    if (w == 0.0f) continue;

                    auto token_in = hs2d->narrow({{0, t, 1}}); // [1, H]
                    auto token_out = experts_.at(static_cast<size_t>(expert_id))->forward(token_in); // [1, H]
                    auto tok_cpu = token_out->to(infinicore::Device::cpu())->contiguous();
                    for (size_t i = 0; i < hidden_size; ++i) {
                        row_acc[i] += scalar_to_f32(tok_cpu, i) * w;
                    }
                }
            }
        } else {
            // Build per-expert token lists from CPU routing results.
            std::vector<std::vector<int32_t>> expert_token_ids(n_routed_experts);
            std::vector<std::vector<float>> expert_token_weights(n_routed_experts);
            for (size_t t = 0; t < n_tokens; ++t) {
                for (size_t j = 0; j < top_k; ++j) {
                    int32_t e = topk_indices_cpu[t][j];
                    if (e < 0 || static_cast<size_t>(e) >= n_routed_experts) continue;
                    float w = topk_weights_cpu_vec[t][j];
                    if (w == 0.0f) continue;
                    expert_token_ids[static_cast<size_t>(e)].push_back(static_cast<int32_t>(t));
                    expert_token_weights[static_cast<size_t>(e)].push_back(w);
                }
            }

            auto device = hidden_states->device();
            auto out2d_f32 = infinicore::Tensor::zeros({n_tokens, hidden_size}, infinicore::DataType::F32, device);
            auto hs_flat = hs2d->contiguous()->view({n_tokens * hidden_size});

            for (size_t e = 0; e < n_routed_experts; ++e) {
                const auto &tok_ids = expert_token_ids[e];
                const auto &tok_w = expert_token_weights[e];
                const size_t m = tok_ids.size();
                if (m == 0) continue;

                // 1D gather indices into flattened [N*H]
                auto gather_idx_cpu = infinicore::Tensor::empty({m * hidden_size}, infinicore::DataType::I64, infinicore::Device::cpu());
                int64_t *gptr = reinterpret_cast<int64_t *>(gather_idx_cpu->data());
                for (size_t i = 0; i < m; ++i) {
                    int64_t base = static_cast<int64_t>(tok_ids[i]) * static_cast<int64_t>(hidden_size);
                    for (size_t h = 0; h < hidden_size; ++h) {
                        gptr[i * hidden_size + h] = base + static_cast<int64_t>(h);
                    }
                }
                auto gather_idx_dev = gather_idx_cpu->to(device);

                auto expert_in = infinicore::op::take(hs_flat, gather_idx_dev)->view({m, hidden_size});
                auto expert_out = experts_.at(e)->forward(expert_in);
                auto expert_out_f32 = infinicore::op::convert_to_f32(expert_out->contiguous());

                auto w_full_cpu = infinicore::Tensor::empty({m * hidden_size}, infinicore::DataType::F32, infinicore::Device::cpu());
                float *wptr = reinterpret_cast<float *>(w_full_cpu->data());
                for (size_t i = 0; i < m; ++i) {
                    float w = tok_w[i];
                    for (size_t h = 0; h < hidden_size; ++h) {
                        wptr[i * hidden_size + h] = w;
                    }
                }
                auto w_full_dev = w_full_cpu->to(device)->view({m, hidden_size});
                auto weighted_f32 = infinicore::op::mul(expert_out_f32, w_full_dev);

                auto tok_idx_cpu = infinicore::Tensor::empty({m}, infinicore::DataType::I32, infinicore::Device::cpu());
                std::memcpy(tok_idx_cpu->data(), tok_ids.data(), m * sizeof(int32_t));
                auto tok_idx_dev = tok_idx_cpu->to(device);
                infinicore::op::index_add_(out2d_f32, out2d_f32, 0, tok_idx_dev, weighted_f32, 1.0f);
            }

            // Copy device FP32 accumulation back to the CPU buffer expected below.
            out2d_cpu = out2d_f32->to(infinicore::Device::cpu())->contiguous();
        }

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
    // Two routing paths:
    // - Fast path: `topkrouter` (grouped) for the fixed 256-expert kernel.
    // - Generic path: sigmoid + bias + topk (no grouping) for arbitrary expert counts.
    infinicore::Tensor topk_indices_cpu;
    infinicore::Tensor topk_weights_cpu;
    if (can_use_topkrouter) {
        auto [topk_weights_dev, topk_indices_dev] = infinicore::op::topkrouter(
            router_logits,
            e_score_correction_bias_f32_device_.value(),
            routed_scaling_factor,
            top_k);
        topk_indices_cpu = topk_indices_dev->to(infinicore::Device::cpu())->contiguous();
        topk_weights_cpu = topk_weights_dev->to(infinicore::Device::cpu())->contiguous();
    } else {
        // Generic HF-like routing (no grouped masking):
        // scores = sigmoid(logits)
        // scores_for_choice = scores + bias
        // indices = topk(scores_for_choice, k=top_k, sorted=False)
        // weights = scores.gather(indices), optional renorm, then scaling
        auto scores = infinicore::op::sigmoid(router_logits); // float32
        auto bias2d = e_score_correction_bias_f32_device_.value()->as_strided(
            {n_tokens, n_routed_experts}, {0, 1});
        auto scores_for_choice = infinicore::op::add(scores, bias2d);
        auto _topk = infinicore::op::topk(scores_for_choice, top_k, 1, /*largest=*/true, /*sorted=*/false);
        // topk values from scores_for_choice are not the gating weights; gather from `scores` on CPU.
        auto topk_indices_dev = _topk.second;

        auto scores_cpu = scores->to(infinicore::Device::cpu())->contiguous();
        topk_indices_cpu = topk_indices_dev->to(infinicore::Device::cpu())->contiguous();

        // Build topk_weights_cpu as float32 [N, top_k] on CPU.
        topk_weights_cpu =
            infinicore::Tensor::empty({n_tokens, top_k}, infinicore::DataType::F32, infinicore::Device::cpu());
        for (size_t t = 0; t < n_tokens; ++t) {
            float denom = 0.0f;
            for (size_t j = 0; j < top_k; ++j) {
                const int32_t e = scalar_to_i32(topk_indices_cpu, t * top_k + j);
                float w = 0.0f;
                if (e >= 0 && static_cast<size_t>(e) < n_routed_experts) {
                    w = scalar_to_f32(scores_cpu, t * n_routed_experts + static_cast<size_t>(e));
                }
                write_f32_as_element(topk_weights_cpu, t * top_k + j, w);
                denom += w;
            }
            if (norm_topk_prob) {
                denom += 1e-20f;
                float inv = 1.0f / denom;
                for (size_t j = 0; j < top_k; ++j) {
                    float w = scalar_to_f32(topk_weights_cpu, t * top_k + j) * inv;
                    write_f32_as_element(topk_weights_cpu, t * top_k + j, w);
                }
            }
            for (size_t j = 0; j < top_k; ++j) {
                float w = scalar_to_f32(topk_weights_cpu, t * top_k + j) * routed_scaling_factor;
                write_f32_as_element(topk_weights_cpu, t * top_k + j, w);
            }
        }
    }

    std::vector<std::vector<int32_t>> expert_token_ids(n_routed_experts);
    std::vector<std::vector<float>> expert_token_weights(n_routed_experts);
    for (size_t t = 0; t < n_tokens; ++t) {
        for (size_t j = 0; j < top_k; ++j) {
            const int32_t e = scalar_to_i32(topk_indices_cpu, t * top_k + j);
            if (e < 0 || static_cast<size_t>(e) >= n_routed_experts) {
                continue;
            }
            const float w = scalar_to_f32(topk_weights_cpu, t * top_k + j);
            if (w == 0.0f) {
                continue;
            }
            expert_token_ids[static_cast<size_t>(e)].push_back(static_cast<int32_t>(t));
            expert_token_weights[static_cast<size_t>(e)].push_back(w);
        }
    }

    auto device = hidden_states->device();
    // HF accumulates routed expert contributions in float32 (dtype=topk_weights.dtype),
    // then casts back to activation dtype before adding shared experts.
    auto out2d_f32 = infinicore::Tensor::zeros({n_tokens, hidden_size}, infinicore::DataType::F32, device);
    auto hs2d_contig = hs2d->contiguous();
    auto hs_flat = hs2d_contig->view({n_tokens * hidden_size});

    for (size_t e = 0; e < n_routed_experts; ++e) {
        const auto &tok_ids = expert_token_ids[e];
        const auto &tok_w = expert_token_weights[e];
        const size_t m = tok_ids.size();
        if (m == 0) continue;

        // token index tensor: [m]
        auto tok_idx_cpu = infinicore::Tensor::empty({m}, infinicore::DataType::I32, infinicore::Device::cpu());
        std::memcpy(tok_idx_cpu->data(), tok_ids.data(), m * sizeof(int32_t));
        auto tok_idx_dev = tok_idx_cpu->to(device);

        // gather indices into flattened [N*H] buffer: shape [m, H] of int64 offsets
        auto gather_idx_cpu = infinicore::Tensor::empty({m, hidden_size}, infinicore::DataType::I64, infinicore::Device::cpu());
        int64_t *gather_ptr = reinterpret_cast<int64_t *>(gather_idx_cpu->data());
        for (size_t i = 0; i < m; ++i) {
            const int64_t base = static_cast<int64_t>(tok_ids[i]) * static_cast<int64_t>(hidden_size);
            int64_t *row = gather_ptr + i * static_cast<int64_t>(hidden_size);
            for (size_t h = 0; h < hidden_size; ++h) {
                row[h] = base + static_cast<int64_t>(h);
            }
        }
        auto gather_idx_dev = gather_idx_cpu->to(device);

        // expert input: [m, H]
        auto expert_in = infinicore::op::take(hs_flat, gather_idx_dev)->view({m, hidden_size});
        auto expert_out = experts_.at(e)->forward(expert_in); // [m, H] (activation dtype)
        auto expert_out_f32 = infinicore::op::convert_to_f32(expert_out->contiguous());

        // weights expanded to [m, H] in float32 (mul has no broadcast).
        auto w_full_cpu = infinicore::Tensor::empty({m, hidden_size}, infinicore::DataType::F32, infinicore::Device::cpu());
        for (size_t i = 0; i < m; ++i) {
            float w = tok_w[i];
            size_t base = i * hidden_size;
            for (size_t h = 0; h < hidden_size; ++h) {
                write_f32_as_element(w_full_cpu, base + h, w);
            }
        }
        auto w_full_dev = w_full_cpu->to(device);
        auto weighted_f32 = infinicore::op::mul(expert_out_f32, w_full_dev);

        // out2d_f32[tok_ids] += weighted_f32
        infinicore::op::index_add_(out2d_f32, out2d_f32, 0, tok_idx_dev, weighted_f32, 1.0f);
    }

    // Add shared experts in float32, then cast back to activation dtype if needed.
    auto shared = shared_experts_->forward(hidden_states);
    auto shared_f32 = infinicore::op::convert_to_f32(shared->contiguous());
    auto summed_f32 = infinicore::op::add(
        out2d_f32->view({batch_size, seq_len, hidden_size}),
        shared_f32);

    if (hidden_states->dtype() == infinicore::DataType::F32) {
        return summed_f32;
    }

    // Cast f32 -> activation dtype on CPU (one-time per layer output).
    auto summed_cpu = summed_f32->to(infinicore::Device::cpu())->contiguous();
    auto cast_cpu = infinicore::Tensor::empty({n_tokens, hidden_size}, hidden_states->dtype(), infinicore::Device::cpu());
    const size_t numel = n_tokens * hidden_size;
    for (size_t i = 0; i < numel; ++i) {
        write_f32_as_element(cast_cpu, i, scalar_to_f32(summed_cpu, i));
    }
    return cast_cpu->to(device)->view({batch_size, seq_len, hidden_size});
}

} // namespace infinilm::models::minicpm5_moe

