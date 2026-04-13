#include "minicpm5_moe_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/sigmoid.hpp"

#include <stdexcept>

namespace infinilm::models::minicpm5_moe {

MiniCPM5MoeAttention::MiniCPM5MoeAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t layer_idx,
                                           const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    use_gated_attention_ = model_config->get_or<bool>("use_gated_attention", false);

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;

    const auto quantization_method = model_config->get_quantization_method();
    // HF config default: attention_bias = False.
    const bool use_bias = model_config->get_or<bool>("attention_bias", false);
    const bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);

    if ((total_num_kv_heads < static_cast<size_t>(tp_size)) || (0 != (total_num_kv_heads % static_cast<size_t>(tp_size)))) {
        throw std::runtime_error("MiniCPM5MoeAttention: num_key_value_heads must be divisible by tp_size");
    }

    num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
    num_key_value_heads_ = total_num_kv_heads / static_cast<size_t>(tp_size);

    // HF parity:
    // - gated attention doubles q_proj output dim (query + gate).
    const size_t q_out_dim = (use_gated_attention_ ? 2 : 1) * total_num_heads * head_dim_;

    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, q_out_dim, quantization_method,
                              use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, total_num_kv_heads * head_dim_, quantization_method,
                              use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, total_num_kv_heads * head_dim_, quantization_method,
                              use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * head_dim_, hidden_size_, quantization_method,
                              use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);
    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    auto kv_quant_scheme = infinilm::global_state::get_infinilm_config().model_config->get_kv_quant_scheme();
    switch (kv_quant_scheme) {
    case (infinicore::quantization::KVQuantAlgo::NONE): {
        break;
    }
    case (infinicore::quantization::KVQuantAlgo::INT8): {
        INFINICORE_NN_PARAMETER_INIT(kv_cache_k_scale, ({1}, infinicore::DataType::F32, device, 0, 0, 1));
        INFINICORE_NN_PARAMETER_INIT(kv_cache_v_scale, ({1}, infinicore::DataType::F32, device, 0, 0, 1));
        break;
    }
    default: {
        throw std::runtime_error("MiniCPM5MoeAttention: unsupported kv_quant_scheme");
    }
    }
}

infinicore::Tensor MiniCPM5MoeAttention::forward(const infinicore::Tensor &position_ids,
                                                 const infinicore::Tensor &hidden_states) const {
    // Mirror `qwen3::Qwen3Attention` tensor shaping and RoPE application, with MiniCPM5 gated attention.
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        auto hidden_states_mutable = hidden_states;

        // Project Q (and gate), K, V.
        auto qg = q_proj_->forward(hidden_states_mutable);
        auto k = k_proj_->forward(hidden_states_mutable);
        auto v = v_proj_->forward(hidden_states_mutable);

        // Normalize projection outputs to 3D [B,S,dim] (linear kernels may flatten to 2D).
        const size_t q_local_dim = (use_gated_attention_ ? 2 : 1) * num_attention_heads_ * head_dim_;
        const size_t kv_local_dim = num_key_value_heads_ * head_dim_;
        if (qg->shape().size() == 2) {
            qg = qg->view({batch_size, seq_len, q_local_dim});
        }
        if (k->shape().size() == 2) {
            k = k->view({batch_size, seq_len, kv_local_dim});
        }
        if (v->shape().size() == 2) {
            v = v->view({batch_size, seq_len, kv_local_dim});
        }

        infinicore::Tensor q;
        infinicore::Tensor gate_score;
        if (use_gated_attention_) {
            // qg: [B,S, 2*H*D] -> [B,S,H,2*D] then split on last dim.
            auto qg_view = qg->view({batch_size, seq_len, num_attention_heads_, 2 * head_dim_});
            q = qg_view->narrow({{3, 0, head_dim_}})->contiguous();
            auto gate = qg_view->narrow({{3, head_dim_, head_dim_}})->contiguous();
            gate_score = gate->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
        } else {
            q = qg->view({batch_size, seq_len, num_attention_heads_, head_dim_});
        }

        auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
        auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

        // Prepare position ids for RoPE (same as qwen3).
        auto pos_shape = position_ids->shape();
        infinicore::Tensor pos_ids_for_rope = position_ids;
        if (pos_shape.size() == 2) {
            auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
            pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
        } else if (pos_shape.size() == 1) {
            pos_ids_for_rope = position_ids->contiguous();
        } else {
            throw std::runtime_error("MiniCPM5MoeAttention: Unexpected position_ids shape");
        }

        // Apply RoPE to Q and K.
        auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_},
                                                q->dtype(), q->device())
                          ->permute({0, 2, 1, 3}); // -> [B,S,H,D]
        rotary_emb_->forward(q_rope, q, pos_ids_for_rope);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

        // Attention.
        auto attn_output = attn_->forward(q_rope, k_reshaped, v_reshaped);

        // Gate (HF parity): attn_out *= sigmoid(gate_score)
        if (use_gated_attention_) {
            auto gate = infinicore::op::sigmoid(gate_score);
            attn_output = infinicore::op::mul(attn_output, gate);
        }

        return o_proj_->forward(attn_output);
    }

    // Paged/flash path (batch_size must be 1, flattened along seq_len).
    if (batch_size != 1) {
        throw std::runtime_error("MiniCPM5MoeAttention: paged attention only supports batch_size=1");
    }

    auto hidden_states_mutable = hidden_states;
    auto qg = q_proj_->forward(hidden_states_mutable);
    auto k = k_proj_->forward(hidden_states_mutable);
    auto v = v_proj_->forward(hidden_states_mutable);

    const size_t q_local_dim = (use_gated_attention_ ? 2 : 1) * num_attention_heads_ * head_dim_;
    const size_t kv_local_dim = num_key_value_heads_ * head_dim_;
    if (qg->shape().size() == 2) {
        qg = qg->view({seq_len, q_local_dim});
    }
    if (k->shape().size() == 2) {
        k = k->view({seq_len, kv_local_dim});
    }
    if (v->shape().size() == 2) {
        v = v->view({seq_len, kv_local_dim});
    }

    infinicore::Tensor q;
    infinicore::Tensor gate_score;
    if (use_gated_attention_) {
        // ColumnParallelLinear outputs [S, 2*H*D] for batch=1 flattened.
        auto qg_view = qg->view({seq_len, num_attention_heads_, 2 * head_dim_});
        q = qg_view->narrow({{2, 0, head_dim_}})->contiguous();
        auto gate = qg_view->narrow({{2, head_dim_, head_dim_}})->contiguous();
        gate_score = gate->view({seq_len, num_attention_heads_ * head_dim_});
    } else {
        q = qg->view({seq_len, num_attention_heads_, head_dim_});
    }
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    // Prepare position ids for RoPE.
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("MiniCPM5MoeAttention: Unexpected position_ids shape");
    }

    rotary_emb_->forward(q, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q, k_reshaped, v_reshaped);
    if (use_gated_attention_) {
        auto gate = infinicore::op::sigmoid(gate_score);
        attn_output = infinicore::op::mul(attn_output, gate);
    }
    return o_proj_->forward(attn_output);
}

} // namespace infinilm::models::minicpm5_moe

