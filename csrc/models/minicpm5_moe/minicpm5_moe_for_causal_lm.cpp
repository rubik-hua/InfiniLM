#include "minicpm5_moe_for_causal_lm.hpp"

#include "../models_registry.hpp"
#include "../../utils.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mul.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm5_moe {

// ----------------------------- Dense MLP -----------------------------
class DenseMLP : public infinicore::nn::Module {
public:
    DenseMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
             const infinicore::Device &device,
             size_t intermediate_size)
        : model_config_(std::move(model_config)) {
        const auto &dtype = model_config_->get_dtype();
        const size_t hidden_size = model_config_->get<size_t>("hidden_size");
        const bool mlp_bias = model_config_->get_or<bool>("mlp_bias", false);

        INFINICORE_NN_MODULE_INIT(gate_proj, hidden_size, intermediate_size, mlp_bias, dtype, device);
        INFINICORE_NN_MODULE_INIT(up_proj, hidden_size, intermediate_size, mlp_bias, dtype, device);
        INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size, hidden_size, mlp_bias, dtype, device);
    }

    infinicore::Tensor forward(const infinicore::Tensor &x) const {
        auto x_nc = x; // Linear::forward expects non-const Tensor&
        auto gate = gate_proj_->forward(x_nc);
        auto up = up_proj_->forward(x_nc);
        gate = infinicore::op::silu(gate);
        auto prod = infinicore::op::mul(gate, up);
        return down_proj_->forward(prod);
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, gate_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, up_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, down_proj);
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
};

// ----------------------------- Sparse MoE -----------------------------
class SparseMoE : public infinicore::nn::Module {
public:
    SparseMoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              const infinicore::Device &device)
        : model_config_(std::move(model_config)) {
        const auto &dtype = model_config_->get_dtype();
        const size_t hidden_size = model_config_->get<size_t>("hidden_size");

        n_routed_experts_ = model_config_->get<size_t>("n_routed_experts");
        n_shared_experts_ = model_config_->get<size_t>("n_shared_experts");
        num_experts_per_tok_ = model_config_->get<size_t>("num_experts_per_tok");
        moe_intermediate_size_ = model_config_->get<size_t>("moe_intermediate_size");
        routed_scaling_factor_ = model_config_->get_or<double>("routed_scaling_factor", 1.0);
        norm_topk_prob_ = model_config_->get_or<bool>("norm_topk_prob", true);

        // Router: F.linear(hidden, weight) where weight is [n_routed_experts, hidden]
        INFINICORE_NN_MODULE_INIT(gate, hidden_size, n_routed_experts_, false, dtype, device);

        experts_.reserve(n_routed_experts_);
        for (size_t i = 0; i < n_routed_experts_; ++i) {
            experts_.push_back(this->register_module<DenseMLP>(
                "experts." + std::to_string(i),
                model_config_,
                device,
                moe_intermediate_size_));
        }

        // Shared experts: one MLP with intermediate_size = moe_intermediate_size * n_shared_experts
        const size_t shared_intermediate = moe_intermediate_size_ * n_shared_experts_;
        shared_experts_ = this->register_module<DenseMLP>("shared_experts", model_config_, device, shared_intermediate);

        // HF has an e_score_correction_bias buffer; we model it as a parameter tensor for naming parity.
        INFINICORE_NN_PARAMETER_INIT(e_score_correction_bias, ({n_routed_experts_}, infinicore::DataType::F32, device, 0, 0, 1));
    }

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const {
        // hidden_states: [B, S, H]
        auto shape = hidden_states->shape();
        const size_t batch = shape[0];
        const size_t seq = shape[1];
        const size_t hidden = shape[2];
        const size_t ntok = batch * seq;

        // Flatten tokens: [ntok, H]
        auto hs = hidden_states->view({ntok, hidden});

        // Router logits: [ntok, n_routed_experts]
        auto router_logits = gate_->forward(hs);

        // Correctness-first routing on CPU to avoid depending on optional TopK operator builds.
        auto router_logits_cpu = router_logits->to(infinicore::Device::cpu());
        auto bias_cpu = e_score_correction_bias_->to(infinicore::Device::cpu());

        const auto *logits_ptr = reinterpret_cast<const float *>(router_logits_cpu->data());
        const auto *bias_ptr = reinterpret_cast<const float *>(bias_cpu->data());

        struct Choice {
            int idx;
            float score;
        };
        std::vector<Choice> choices;
        choices.resize(n_routed_experts_);

        // Accumulate routed expert outputs in fp32 for stability, then cast back.
        auto out = infinicore::Tensor::zeros({ntok, hidden}, hidden_states->dtype(), hidden_states->device());

        for (size_t t = 0; t < ntok; ++t) {
            // token slice: [1, H]
            auto x_t = hs->narrow({{0, t, 1}});
            auto acc_t = out->narrow({{0, t, 1}});

            // Compute sigmoid scores on CPU and select top-k experts.
            const float *row = logits_ptr + t * n_routed_experts_;
            for (size_t e = 0; e < n_routed_experts_; ++e) {
                const float z = row[e] + bias_ptr[e];
                const float s = 1.0f / (1.0f + std::exp(-z));
                choices[e] = Choice{int(e), s};
            }
            std::nth_element(
                choices.begin(),
                choices.begin() + num_experts_per_tok_,
                choices.end(),
                [](const Choice &a, const Choice &b) { return a.score > b.score; });
            choices.resize(num_experts_per_tok_);

            float sum = 0.f;
            for (const auto &c : choices) {
                sum += c.score;
            }
            const float denom = (norm_topk_prob_ ? std::max(sum, 1e-20f) : 1.f);

            for (const auto &c : choices) {
                auto y = experts_.at(size_t(c.idx))->forward(x_t);
                infinicore::op::add_(acc_t, acc_t, y);
            }

            choices.resize(n_routed_experts_);
        }

        auto shared = shared_experts_->forward(hs);
        auto sum = infinicore::op::add(out, shared);
        return sum->view({batch, seq, hidden});
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Linear, gate);
    INFINICORE_NN_MODULE(DenseMLP, shared_experts);
    INFINICORE_NN_MODULE_VEC(DenseMLP, experts);

    INFINICORE_NN_PARAMETER(e_score_correction_bias);

    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    size_t n_routed_experts_{0};
    size_t n_shared_experts_{0};
    size_t num_experts_per_tok_{0};
    size_t moe_intermediate_size_{0};
    double routed_scaling_factor_{1.0};
    bool norm_topk_prob_{true};
};

// ----------------------------- Gated Attention -----------------------------
class MiniCPM5MoEAttention : public infinicore::nn::Module {
public:
    MiniCPM5MoEAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device,
                         size_t layer_idx,
                         engine::distributed::RankInfo rank_info,
                         backends::AttentionBackend attention_backend)
        : model_config_(std::move(model_config)),
          layer_idx_(layer_idx),
          rank_info_(rank_info),
          attention_backend_(attention_backend) {
        const auto &dtype = model_config_->get_dtype();
        hidden_size_ = model_config_->get<size_t>("hidden_size");
        num_attention_heads_ = model_config_->get<size_t>("num_attention_heads");
        num_key_value_heads_ = model_config_->get<size_t>("num_key_value_heads");
        head_dim_ = model_config_->get_head_dim();
        use_gated_attention_ = model_config_->get_or<bool>("use_gated_attention", false);
        const bool use_bias = model_config_->get_or<bool>("attention_bias", false);
        const bool use_output_bias = model_config_->get_or<bool>("attention_output_bias", false);

        // NOTE: correctness-first: only tp_size==1 supported here.
        if (rank_info_.tp_size != 1) {
            throw std::runtime_error("MiniCPM5MoEAttention: tp_size!=1 not supported yet");
        }

        const size_t q_out = num_attention_heads_ * head_dim_ * (use_gated_attention_ ? 2 : 1);
        const size_t kv_out = num_key_value_heads_ * head_dim_;

        INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, q_out, use_bias, dtype, device);
        INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, kv_out, use_bias, dtype, device);
        INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, kv_out, use_bias, dtype, device);
        INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads_ * head_dim_, hidden_size_, use_output_bias, dtype, device);

        scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    }

    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) { rotary_emb_ = rotary_emb; }

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &position_ids,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache,
                               std::optional<infinicore::Tensor> past_sequence_lengths,
                               std::optional<infinicore::Tensor> total_sequence_lengths,
                               std::optional<infinicore::Tensor> input_offsets,
                               std::optional<infinicore::Tensor> cu_seqlens,
                               std::optional<infinicore::Tensor> block_tables,
                               std::optional<infinicore::Tensor> slot_mapping) const {
        if (!rotary_emb_) {
            throw std::runtime_error("MiniCPM5MoEAttention: rotary_emb not configured");
        }
        if (auto paged_kv_cache = std::dynamic_pointer_cast<cache::PagedKVCache>(kv_cache)) {
            return forward_paged_(hidden_states, position_ids, paged_kv_cache, total_sequence_lengths, input_offsets, cu_seqlens, block_tables, slot_mapping);
        }
        return forward_static_(hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths);
    }

private:
    infinicore::Tensor forward_static_(const infinicore::Tensor &hidden_states,
                                      const infinicore::Tensor &position_ids,
                                      std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                      std::optional<infinicore::Tensor> past_sequence_lengths,
                                      std::optional<infinicore::Tensor> total_sequence_lengths) const {
        auto shape = hidden_states->shape();
        const size_t batch = shape[0];
        const size_t seq = shape[1];

        auto hs_nc = hidden_states; // Linear::forward expects non-const Tensor&
        auto q_all = q_proj_->forward(hs_nc);
        infinicore::Tensor q;
        infinicore::Tensor gate_score;
        if (use_gated_attention_) {
            // q_all: [B, S, 2*Hq]
            auto q_reshaped = q_all->view({batch, seq, num_attention_heads_, head_dim_ * 2});
            // split last dim
            q = q_reshaped->narrow({{3, 0, head_dim_}})->contiguous()->view({batch, seq, num_attention_heads_ * head_dim_});
            gate_score = q_reshaped->narrow({{3, head_dim_, head_dim_}})->contiguous()->view({batch, seq, num_attention_heads_ * head_dim_});
        } else {
            q = q_all;
        }

        auto k = k_proj_->forward(hs_nc);
        auto v = v_proj_->forward(hs_nc);

        auto q_reshaped = q->view({batch, seq, num_attention_heads_, head_dim_});
        auto k_reshaped = k->view({batch, seq, num_key_value_heads_, head_dim_});
        auto v_reshaped = v->view({batch, seq, num_key_value_heads_, head_dim_});

        // Prepare position ids for RoPE (same handling as llama)
        auto pos_shape = position_ids->shape();
        infinicore::Tensor pos_ids_for_rope = position_ids;
        if (pos_shape.size() == 2) {
            auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
            pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
        } else if (pos_shape.size() == 1) {
            pos_ids_for_rope = position_ids->contiguous();
        } else {
            throw std::runtime_error("MiniCPM5MoEAttention: Unexpected position_ids shape");
        }

        // Apply RoPE
        auto q_rope = infinicore::Tensor::empty({batch, num_attention_heads_, seq, head_dim_}, q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});
        rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

        auto q_heads = q_rope->permute({0, 2, 1, 3});
        auto k_heads = k_reshaped->permute({0, 2, 1, 3});
        auto v_heads = v_reshaped->permute({0, 2, 1, 3});

        infinicore::Tensor k_total;
        infinicore::Tensor v_total;
        if (kv_cache == nullptr) {
            k_total = k_heads;
            v_total = v_heads;
        } else if (auto static_kv_cache = std::dynamic_pointer_cast<cache::StaticKVCache>(kv_cache)) {
            auto [k_total_tmp, v_total_tmp] = static_kv_cache->update(layer_idx_, k_heads, v_heads, past_sequence_lengths.value());
            k_total = k_total_tmp;
            v_total = v_total_tmp;
        } else {
            throw std::runtime_error("MiniCPM5MoEAttention: Unsupported kvcache type");
        }

        // Metadata total can exceed the materialized KV length when no cache is used (e.g. decode
        // steps with seq==1 but cumulative lengths in total_sequence_lengths). Clamp to tensor dim.
        const size_t meta_total = static_cast<size_t>(
            reinterpret_cast<int32_t *>(
                total_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0]);
        const size_t k_len = k_total->shape()[2];
        const size_t total_seq_len = std::min(meta_total, k_len);
        k_total = k_total->narrow({{2, 0, total_seq_len}});
        v_total = v_total->narrow({{2, 0, total_seq_len}});

        // GQA attention (same as llama)
        const size_t ngroup = num_attention_heads_ / num_key_value_heads_;
        auto Q = q_heads->view({batch * num_key_value_heads_, ngroup * seq, head_dim_});
        auto K = k_total->view({batch * num_key_value_heads_, total_seq_len, head_dim_});
        auto V = v_total->view({batch * num_key_value_heads_, total_seq_len, head_dim_});

        auto Kt = K->permute({0, 2, 1});
        auto attn_weight = infinicore::op::matmul(Q, Kt, scaling_);
        auto attn_weight_softmax = attn_weight->view({batch * num_attention_heads_, seq, total_seq_len});
        infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);
        auto out = infinicore::op::matmul(attn_weight, V);
        auto attn_output = out->view({batch, num_attention_heads_, seq, head_dim_})
                               ->permute({0, 2, 1, 3})
                               ->contiguous()
                               ->view({batch, seq, num_attention_heads_ * head_dim_});

        (void)gate_score;

        return o_proj_->forward(attn_output);
    }

    infinicore::Tensor forward_paged_(const infinicore::Tensor &hidden_states,
                                     const infinicore::Tensor &position_ids,
                                     std::shared_ptr<infinilm::cache::PagedKVCache> paged_kv_cache,
                                     std::optional<infinicore::Tensor> total_sequence_lengths,
                                     std::optional<infinicore::Tensor> input_offsets,
                                     std::optional<infinicore::Tensor> cu_seqlens,
                                     std::optional<infinicore::Tensor> block_tables,
                                     std::optional<infinicore::Tensor> slot_mapping) const {
        // Keep parity with llama: batch==1 and flattened tokens.
        auto shape = hidden_states->shape();
        const size_t batch = shape[0];
        const size_t seq = shape[1];
        ASSERT_EQ(batch, 1);
        ASSERT(block_tables.has_value());
        ASSERT(slot_mapping.has_value());

        bool is_prefill = (seq != total_sequence_lengths.value()->shape()[0]);

        auto hs_nc = hidden_states; // Linear::forward expects non-const Tensor&
        auto q_all = q_proj_->forward(hs_nc);
        infinicore::Tensor q;
        infinicore::Tensor gate_score;
        if (use_gated_attention_) {
            auto q_reshaped2 = q_all->view({seq, num_attention_heads_, head_dim_ * 2});
            q = q_reshaped2->narrow({{2, 0, head_dim_}})->contiguous()->view({seq, num_attention_heads_, head_dim_});
            gate_score = q_reshaped2->narrow({{2, head_dim_, head_dim_}})->contiguous()->view({1, seq, num_attention_heads_ * head_dim_});
        } else {
            q = q_all->view({seq, num_attention_heads_, head_dim_});
        }
        auto k = k_proj_->forward(hs_nc)->view({seq, num_key_value_heads_, head_dim_});
        auto v = v_proj_->forward(hs_nc)->view({seq, num_key_value_heads_, head_dim_});

        // RoPE
        auto pos_shape = position_ids->shape();
        infinicore::Tensor pos_ids_for_rope = position_ids;
        if (pos_shape.size() == 2) {
            auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
            pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
        } else if (pos_shape.size() == 1) {
            pos_ids_for_rope = position_ids;
        } else {
            throw std::runtime_error("MiniCPM5MoEAttention: Unexpected position_ids shape");
        }
        rotary_emb_->forward(q, pos_ids_for_rope, true);
        rotary_emb_->forward(k, pos_ids_for_rope, true);

        auto [k_total, v_total] = paged_kv_cache->update(layer_idx_, k, v, slot_mapping.value());

        infinicore::Tensor attn_out = infinicore::Tensor::empty({seq, num_attention_heads_, head_dim_}, q->dtype(), q->device());
        if (is_prefill) {
            infinicore::op::paged_attention_prefill_(
                attn_out,
                q,
                k_total,
                v_total,
                block_tables.value(),
                total_sequence_lengths.value(),
                input_offsets.value(),
                std::nullopt,
                scaling_);
        } else {
            infinicore::op::paged_attention_(
                attn_out,
                q,
                k_total,
                v_total,
                block_tables.value(),
                total_sequence_lengths.value(),
                std::nullopt,
                scaling_);
        }
        auto attn_output = attn_out->view({1, seq, num_attention_heads_ * head_dim_});
        (void)gate_score;
        return o_proj_->forward(attn_output);
    }

private:
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t num_attention_heads_{0};
    size_t num_key_value_heads_{0};
    size_t head_dim_{0};
    bool use_gated_attention_{false};
    float scaling_{1.f};

    INFINICORE_NN_MODULE(infinicore::nn::Linear, q_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, k_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, v_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, o_proj);

    engine::distributed::RankInfo rank_info_;
    backends::AttentionBackend attention_backend_;
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
};

// ----------------------------- Decoder Layer -----------------------------
class MiniCPM5MoEDecoderLayer : public infinicore::nn::Module {
public:
    MiniCPM5MoEDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            const infinicore::Device &device,
                            size_t layer_idx,
                            engine::distributed::RankInfo rank_info,
                            backends::AttentionBackend attention_backend)
        : model_config_(std::move(model_config)), layer_idx_(layer_idx) {
        const auto &dtype = model_config_->get_dtype();
        const size_t hidden_size = model_config_->get<size_t>("hidden_size");
        const double eps = model_config_->get<double>("rms_norm_eps");

        INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, eps, dtype, device);
        self_attn_ = this->register_module<MiniCPM5MoEAttention>("self_attn", model_config_, device, layer_idx, rank_info, attention_backend);

        const size_t first_k_dense_replace = model_config_->get_or<size_t>("first_k_dense_replace", 0);
        if (layer_idx_ >= first_k_dense_replace) {
            moe_ = this->register_module<SparseMoE>("mlp", model_config_, device);
        } else {
            dense_mlp_ = this->register_module<DenseMLP>("mlp", model_config_, device, model_config_->get<size_t>("intermediate_size"));
        }
    }

    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
        if (self_attn_) {
            self_attn_->set_rotary_emb(rotary_emb);
        }
    }

    void forward(infinicore::Tensor &hidden_states,
                 infinicore::Tensor &residual,
                 const infinicore::Tensor &position_ids,
                 std::shared_ptr<infinilm::cache::Cache> kv_cache,
                 std::optional<infinicore::Tensor> past_sequence_lengths,
                 std::optional<infinicore::Tensor> total_sequence_lengths,
                 std::optional<infinicore::Tensor> input_offsets,
                 std::optional<infinicore::Tensor> cu_seqlens,
                 std::optional<infinicore::Tensor> block_tables,
                 std::optional<infinicore::Tensor> slot_mapping) const {
        if (!residual) {
            residual = hidden_states;
        } else {
            residual = infinicore::op::add(residual, hidden_states);
        }

        input_layernorm_->forward_inplace(hidden_states, residual);
        auto attn_out = self_attn_->forward(hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths, input_offsets, cu_seqlens, block_tables, slot_mapping);
        hidden_states = attn_out;

        post_attention_layernorm_->forward_inplace(hidden_states, residual);
        if (moe_) {
            hidden_states = moe_->forward(hidden_states);
        } else {
            hidden_states = dense_mlp_->forward(hidden_states);
        }
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiniCPM5MoEAttention, self_attn);
    INFINICORE_NN_MODULE(DenseMLP, dense_mlp);
    INFINICORE_NN_MODULE(SparseMoE, moe);

    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    size_t layer_idx_{0};
};

// ----------------------------- Model -----------------------------
class MiniCPM5MoEModel : public infinicore::nn::Module {
public:
    MiniCPM5MoEModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device,
                     engine::distributed::RankInfo rank_info,
                     backends::AttentionBackend attention_backend)
        : model_config_(std::move(model_config)), rank_info_(rank_info) {
        const auto &dtype = model_config_->get_dtype();

        const size_t vocab_size = model_config_->get<size_t>("vocab_size");
        const size_t hidden_size = model_config_->get<size_t>("hidden_size");
        const size_t num_layers = model_config_->get<size_t>("num_hidden_layers");
        const double eps = model_config_->get<double>("rms_norm_eps");

        INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);

        layers_.reserve(num_layers);
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.push_back(this->register_module<MiniCPM5MoEDecoderLayer>(
                "layers." + std::to_string(i), model_config_, device, i, rank_info, attention_backend));
        }

        INFINICORE_NN_MODULE_INIT(norm, hidden_size, eps, dtype, device);

        // RoPE
        const size_t head_dim = model_config_->get_head_dim();
        const size_t max_pos = model_config_->get<size_t>("max_position_embeddings");
        const double rope_theta = model_config_->get_or<double>("rope_theta", 10000.0);
        INFINICORE_NN_MODULE_INIT(rotary_emb, head_dim, max_pos, rope_theta, infinicore::nn::RoPE::Algo::GPT_NEOX, dtype, device, model_config_->get_rope_scaling());
        for (auto &layer : layers_) {
            layer->set_rotary_emb(rotary_emb_);
        }
    }

    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                               const infinicore::Tensor &position_ids,
                               std::optional<infinicore::Tensor> past_sequence_lengths,
                               std::optional<infinicore::Tensor> total_sequence_lengths,
                               std::optional<infinicore::Tensor> input_offsets,
                               std::optional<infinicore::Tensor> cu_seqlens,
                               std::optional<infinicore::Tensor> block_tables,
                               std::optional<infinicore::Tensor> slot_mapping) const {
        auto hidden_states = embed_tokens_->forward(input_ids);
        infinicore::Tensor residual;
        for (auto &layer : layers_) {
            layer->forward(hidden_states, residual, position_ids, kv_cache_, past_sequence_lengths, total_sequence_lengths, input_offsets, cu_seqlens, block_tables, slot_mapping);
        }
        norm_->forward_inplace(hidden_states, residual);
        return hidden_states;
    }

    void reset_cache(const cache::CacheConfig *cache_config) {
        if (cache_config == nullptr) {
            kv_cache_ = nullptr;
            return;
        }
        if (auto kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
            kv_cache_ = std::make_shared<cache::StaticKVCache>(
                model_config_->get_head_dim(),
                model_config_->get_head_dim(),
                model_config_->get<size_t>("num_key_value_heads"),
                model_config_->get<size_t>("num_key_value_heads"),
                model_config_->get<size_t>("num_hidden_layers"),
                model_config_->get<size_t>("max_position_embeddings"),
                model_config_->get_kv_cache_dtype(),
                *kv_cache_config,
                rank_info_);
        } else if (auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)) {
            kv_cache_ = std::make_shared<cache::PagedKVCache>(
                model_config_->get_head_dim(),
                model_config_->get_head_dim(),
                model_config_->get<size_t>("num_key_value_heads"),
                model_config_->get<size_t>("num_key_value_heads"),
                model_config_->get<size_t>("num_hidden_layers"),
                model_config_->get_kv_cache_dtype(),
                *paged_kv_cache_config,
                rank_info_);
        } else {
            throw std::runtime_error("MiniCPM5MoEModel: Unsupported cache type");
        }
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(MiniCPM5MoEDecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(infinicore::nn::RoPE, rotary_emb);

    engine::distributed::RankInfo rank_info_;
    std::shared_ptr<infinilm::cache::Cache> kv_cache_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
};

MiniCPM5MoEForCausalLM::MiniCPM5MoEForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device,
                                               engine::distributed::RankInfo rank_info,
                                               backends::AttentionBackend attention_backend) {
    device_ = device;

    const auto &dtype = model_config->get_dtype();
    INFINICORE_NN_MODULE_INIT(model, model_config, device, rank_info, attention_backend);
    INFINICORE_NN_MODULE_INIT(lm_head, model_config->get<size_t>("hidden_size"), model_config->get<size_t>("vocab_size"), false, dtype, device);
}

MiniCPM5MoEForCausalLM::Output MiniCPM5MoEForCausalLM::forward(const Input &input) const {
    auto input_ids = input.input_ids.value();
    auto position_ids = input.position_ids.value();

    auto hidden_states = model_->forward(
        input_ids,
        position_ids,
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping);

    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void MiniCPM5MoEForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        cache_config_.reset();
        model_->reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();
    model_->reset_cache(cache_config_.get());
}

const cache::CacheConfig *MiniCPM5MoEForCausalLM::get_cache_config() const {
    return cache_config_.get();
}

std::shared_ptr<infinilm::config::ModelConfig>
create_minicpm5_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if (model_type != "minicpm5_moe") {
        throw std::runtime_error("create_minicpm5_moe_model_config: model_type is not minicpm5_moe");
    }

    // Fill required defaults that HF configs sometimes omit but our C++ expects.
    auto &j = model_config->get_config_json();
    if (!j.contains("rope_theta")) {
        j["rope_theta"] = 10000.0;
    }
    if (!j.contains("attention_bias")) {
        j["attention_bias"] = false;
    }
    if (!j.contains("attention_output_bias")) {
        j["attention_output_bias"] = false;
    }
    if (!j.contains("mlp_bias")) {
        j["mlp_bias"] = false;
    }
    if (!j.contains("norm_topk_prob")) {
        j["norm_topk_prob"] = true;
    }
    return model_config;
}

} // namespace infinilm::models::minicpm5_moe

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minicpm5_moe,
    infinilm::models::minicpm5_moe::MiniCPM5MoEForCausalLM,
    infinilm::models::minicpm5_moe::create_minicpm5_moe_model_config);
} // namespace

