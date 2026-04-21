#include "minicpm5_moe_sparse_moe_block.hpp"

#include "minicpm5_moe_router_cpu_detail.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/convert_to_f32.hpp"
#include "infinicore/ops/linear.hpp"
#include "../../utils/nvtx.hpp"

#include <stdexcept>
#include <string>
#include <cstring>

namespace infinilm::models::minicpm5_moe {

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

    auto hs2d = hidden_states->view({n_tokens, hidden_size});

    if (!gate_weight_f32_device_.has_value()) {
        gate_weight_f32_device_ =
            infinicore::op::convert_to_f32(gate_->weight()->contiguous());
    }

    auto hs_f32 = infinicore::op::convert_to_f32(hs2d->contiguous());
    auto router_logits =
        infinicore::op::linear(hs_f32, gate_weight_f32_device_.value(), std::nullopt);
    auto logits_cpu = router_logits->to(infinicore::Device::cpu());
    logits_cpu = logits_cpu->contiguous();

    auto bias_cpu = e_score_correction_bias_->to(infinicore::Device::cpu());
    bias_cpu = bias_cpu->contiguous();

    router_cpu_detail::RouterTopkCpuResult router_out;
    router_cpu_detail::run_router_topk_cpu(
        logits_cpu,
        bias_cpu,
        n_tokens,
        n_routed_experts,
        top_k,
        norm_topk_prob,
        routed_scaling_factor,
        n_group,
        topk_group,
        router_out);
    const auto &topk_indices_cpu = router_out.topk_indices;
    const auto &topk_weights_cpu = router_out.topk_weights;

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
                row_acc[i] += router_cpu_detail::scalar_to_f32(tok_on_cpu, i) * w;
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
            router_cpu_detail::write_f32_as_element(routed_cpu, i, router_cpu_detail::scalar_to_f32(out2d_cpu, i));
        }
        routed = routed_cpu->to(hidden_states->device())->view({batch_size, seq_len, hidden_size});
    }

    auto shared = shared_experts_->forward(hidden_states);
    return infinicore::op::add(routed, shared);
}

} // namespace infinilm::models::minicpm5_moe
