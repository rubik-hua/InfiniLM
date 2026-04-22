#include "ops_shim.hpp"

#include "ops_shim_cuda.hpp"

#include "../global_state/forward_context.hpp"

#include <infinicore/context/context.hpp>

#include <config.h>
#include <data_type.h>
#include <device.h>
#include <handle.h>
#include <operator.h>
#include <base/add.h>
#include <base/embedding.h>
#include <base/gemm.h>
#include <base/mha_kvcache.h>
#include <base/mha_varlen.h>
#include <base/paged_caching.h>
#include <base/random_sample.h>
#include <base/rms_norm.h>
#include <base/rope.h>
#include <base/swiglu.h>
#include <tensor.h>

// SFINAE-based discovery of `Operator<Op, Dev, N>` specializations happens at
// template instantiation time in this translation unit, so each backend
// implementation header must be visible here. Without these includes,
// `ActiveImplementations<Op, Dev>::type` resolves to `List<>` and
// `Operator<Op>::Call` fails with
// "no allowed values registered for value N in the context:
// Operator::Make(implementation_index)".
#include <torch/add/add.h>
#include <torch/embedding/embedding.h>
#include <torch/gemm/gemm.h>
#include <torch/mha_kvcache/mha_kvcache.h>
#include <torch/mha_varlen/mha_varlen.h>
#include <torch/paged_caching/paged_caching.h>
#include <torch/random_sample/random_sample.h>
#include <torch/rms_norm/rms_norm.h>
#include <torch/rope/rope.h>
#include <torch/swiglu/swiglu.h>

// Each enabled-platform marker specializes `DeviceEnabled<Device::Type::X>`
// so that `ActiveDevices<Key>` picks up `X` at template instantiation time.
// These specializations are header-only, so they must be visible in this
// translation unit — linking against `libinfiniops.so` alone is not enough.
// Mirror the platforms enabled by `InfiniOps`' own build.
#include <cpu/device_.h>
#if defined(WITH_NVIDIA)
#include <cuda/nvidia/device_.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace infinilm::ops_shim {

// Lightweight aggregate timing controlled by the `INFINILM_OPS_PROFILE` env
// var. When set, every shim entry point accumulates `(count, ns)` into a
// small per-op atomic counter and dumps the totals at program exit. Disabled
// by default (single static check per call is negligible).
namespace profile {

enum OpId : std::size_t {
    kAdd = 0,
    kSwiglu,
    kPagedCaching,
    kMhaKvcache,
    kMhaVarlen,
    kRandomSample,
    kEmbedding,
    kLinear,
    kRmsNorm,
    kRmsNormForwardInplace,
    kRopeForward,
    kCount,
};

inline constexpr std::array<const char *, kCount> kOpNames{
    "add",        "swiglu",      "paged_caching_",     "mha_kvcache",
    "mha_varlen", "random_sample_", "embedding",       "linear",
    "rms_norm",   "rms_norm_forward_inplace",          "rope_forward",
};

struct Counter {
    std::atomic<std::uint64_t> count{0};
    std::atomic<std::uint64_t> total_ns{0};
};

inline std::array<Counter, kCount> &counters() {
    static std::array<Counter, kCount> instance;
    return instance;
}

inline bool enabled() {
    static const bool value = [] {
        const char *env = std::getenv("INFINILM_OPS_PROFILE");
        const bool on = env != nullptr && env[0] != '\0' && env[0] != '0';
        if (on) {
            std::atexit([] {
                std::fprintf(stderr,
                             "\n=== `infinilm::ops_shim` profile ===\n");
                std::fprintf(stderr, "%-28s %12s %12s %12s\n", "op", "count",
                             "total_ms", "avg_us");
                std::uint64_t total_ns = 0;
                for (std::size_t i = 0; i < kCount; ++i) {
                    auto count = counters()[i].count.load();
                    auto ns = counters()[i].total_ns.load();
                    total_ns += ns;
                    if (count == 0) {
                        continue;
                    }
                    std::fprintf(stderr, "%-28s %12llu %12.3f %12.3f\n",
                                 kOpNames[i],
                                 static_cast<unsigned long long>(count),
                                 ns / 1e6, ns / static_cast<double>(count) / 1e3);
                }
                std::fprintf(stderr, "%-28s %12s %12.3f\n", "TOTAL", "",
                             total_ns / 1e6);
            });
        }
        return on;
    }();
    return value;
}

class Scope {
 public:
    explicit Scope(OpId op) : op_{op}, start_{} {
        if (enabled()) {
            start_ = std::chrono::steady_clock::now();
        }
    }

    ~Scope() {
        if (enabled()) {
            auto end = std::chrono::steady_clock::now();
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          end - start_)
                          .count();
            counters()[op_].count.fetch_add(1, std::memory_order_relaxed);
            counters()[op_].total_ns.fetch_add(ns, std::memory_order_relaxed);
        }
    }

 private:
    OpId op_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace profile

#define INFINILM_OPS_SHIM_PROFILE(op_id) profile::Scope _profile_scope_{profile::OpId::op_id}


namespace {

infini::ops::DataType to_ops_dtype(infinicore::DataType dtype) {
    switch (dtype) {
    case infinicore::DataType::I8:
        return infini::ops::DataType::kInt8;
    case infinicore::DataType::I16:
        return infini::ops::DataType::kInt16;
    case infinicore::DataType::I32:
        return infini::ops::DataType::kInt32;
    case infinicore::DataType::I64:
        return infini::ops::DataType::kInt64;
    case infinicore::DataType::U8:
        return infini::ops::DataType::kUInt8;
    case infinicore::DataType::U16:
        return infini::ops::DataType::kUInt16;
    case infinicore::DataType::U32:
        return infini::ops::DataType::kUInt32;
    case infinicore::DataType::U64:
        return infini::ops::DataType::kUInt64;
    case infinicore::DataType::F16:
        return infini::ops::DataType::kFloat16;
    case infinicore::DataType::BF16:
        return infini::ops::DataType::kBFloat16;
    case infinicore::DataType::F32:
        return infini::ops::DataType::kFloat32;
    case infinicore::DataType::F64:
        return infini::ops::DataType::kFloat64;
    default:
        throw std::invalid_argument(
            "infinilm::ops_shim: unsupported `infinicore::DataType` for `infini::ops` conversion");
    }
}

infini::ops::Device to_ops_device(const infinicore::Device &device) {
    infini::ops::Device::Type type{};
    switch (device.getType()) {
    case infinicore::Device::Type::CPU:
        type = infini::ops::Device::Type::kCpu;
        break;
    case infinicore::Device::Type::NVIDIA:
        type = infini::ops::Device::Type::kNvidia;
        break;
    case infinicore::Device::Type::CAMBRICON:
        type = infini::ops::Device::Type::kCambricon;
        break;
    case infinicore::Device::Type::ASCEND:
        type = infini::ops::Device::Type::kAscend;
        break;
    case infinicore::Device::Type::METAX:
        type = infini::ops::Device::Type::kMetax;
        break;
    case infinicore::Device::Type::MOORE:
        type = infini::ops::Device::Type::kMoore;
        break;
    case infinicore::Device::Type::ILUVATAR:
        type = infini::ops::Device::Type::kIluvatar;
        break;
    case infinicore::Device::Type::KUNLUN:
        type = infini::ops::Device::Type::kKunlun;
        break;
    case infinicore::Device::Type::HYGON:
        type = infini::ops::Device::Type::kHygon;
        break;
    case infinicore::Device::Type::QY:
        type = infini::ops::Device::Type::kQy;
        break;
    default:
        throw std::invalid_argument(
            "infinilm::ops_shim: unsupported `infinicore::Device` for `infini::ops` conversion");
    }
    return infini::ops::Device{type, static_cast<int>(device.getIndex())};
}

infini::ops::Tensor to_ops_tensor(const infinicore::Tensor &tensor) {
    return infini::ops::Tensor{
        const_cast<std::byte *>(tensor->data()),
        tensor->shape(),
        to_ops_dtype(tensor->dtype()),
        to_ops_device(tensor->device()),
        tensor->strides()};
}

// Per-op implementation index: when `InfiniOps` has a native CUDA kernel
// for an op, we prefer it over the PyTorch fallback to skip ATen dispatch
// and `from_blob` wrapping on the hot path. The PyTorch fallback
// (currently index `1` for every op that has one) is used only when no
// native kernel exists.
//
// Measured overhead for small elementwise ops is ~12 us/call via the
// PyTorch fallback; native kernels eliminate most of that.
constexpr std::size_t kFallbackTorchIndex = 1;
constexpr std::size_t kGemmTorchImplementationIndex = 2;

infini::ops::Handle make_handle() {
    infini::ops::Handle handle;
    handle.set_stream(infinicore::context::getStream());
    return handle;
}

infini::ops::Config make_config(std::size_t implementation_index = kFallbackTorchIndex) {
    infini::ops::Config config;
    config.set_implementation_index(implementation_index);
    return config;
}

} // namespace

infinicore::Tensor add(const infinicore::Tensor &a, const infinicore::Tensor &b) {
    INFINILM_OPS_SHIM_PROFILE(kAdd);
    auto c = infinicore::Tensor::empty(a->shape(), a->dtype(), a->device());
    cuda_dispatch::add(to_ops_tensor(a), to_ops_tensor(b), to_ops_tensor(c),
                       infinicore::context::getStream());
    return c;
}

infinicore::Tensor swiglu(const infinicore::Tensor &input, const infinicore::Tensor &gate) {
    INFINILM_OPS_SHIM_PROFILE(kSwiglu);
    auto out = infinicore::Tensor::empty(input->shape(), input->dtype(), input->device());
    cuda_dispatch::swiglu(to_ops_tensor(input), to_ops_tensor(gate),
                          to_ops_tensor(out), infinicore::context::getStream());
    return out;
}

void paged_caching_(infinicore::Tensor k_cache, infinicore::Tensor v_cache,
                    const infinicore::Tensor &k, const infinicore::Tensor &v,
                    const infinicore::Tensor &slot_mapping) {
    INFINILM_OPS_SHIM_PROFILE(kPagedCaching);
    cuda_dispatch::paged_caching(
        to_ops_tensor(k_cache), to_ops_tensor(v_cache), to_ops_tensor(k),
        to_ops_tensor(v), to_ops_tensor(slot_mapping),
        infinicore::context::getStream());
}

infinicore::Tensor mha_kvcache(const infinicore::Tensor &q,
                               const infinicore::Tensor &k_cache,
                               const infinicore::Tensor &v_cache,
                               const infinicore::Tensor &seqlens_k,
                               const infinicore::Tensor &block_table,
                               float scale) {
    INFINILM_OPS_SHIM_PROFILE(kMhaKvcache);
    auto out = infinicore::Tensor::empty(q->shape(), q->dtype(), q->device());
    auto ops_q = to_ops_tensor(q);
    auto ops_k_cache = to_ops_tensor(k_cache);
    auto ops_v_cache = to_ops_tensor(v_cache);
    auto ops_seqlens_k = to_ops_tensor(seqlens_k);
    auto ops_block_table = to_ops_tensor(block_table);
    auto ops_out = to_ops_tensor(out);

    const auto &host_seqlens =
        infinilm::global_state::get_forward_context()
            .attn_metadata.total_sequence_lengths_host;
    infini::ops::MhaKvcacheHostSeqlensHint hint{
        host_seqlens.empty() ? nullptr : host_seqlens.data()};

    infini::ops::Operator<infini::ops::MhaKvcache>::Call(
        make_handle(), make_config(), ops_q, ops_k_cache, ops_v_cache,
        ops_seqlens_k, ops_block_table, scale, ops_out);
    return out;
}

void mha_varlen_(infinicore::Tensor out,
                 const infinicore::Tensor &q,
                 const infinicore::Tensor &k_cache,
                 const infinicore::Tensor &v_cache,
                 const infinicore::Tensor &cum_seqlens_q,
                 const infinicore::Tensor &cum_seqlens_k,
                 const infinicore::Tensor &block_table,
                 float scale) {
    INFINILM_OPS_SHIM_PROFILE(kMhaVarlen);
    auto ops_q = to_ops_tensor(q);
    auto ops_k_cache = to_ops_tensor(k_cache);
    auto ops_v_cache = to_ops_tensor(v_cache);
    auto ops_cum_seqlens_q = to_ops_tensor(cum_seqlens_q);
    auto ops_cum_seqlens_k = to_ops_tensor(cum_seqlens_k);
    auto ops_block_table = to_ops_tensor(block_table);
    auto ops_out = to_ops_tensor(out);
    infini::ops::Operator<infini::ops::MhaVarlen>::Call(
        make_handle(), make_config(), ops_q, ops_k_cache, ops_v_cache,
        ops_cum_seqlens_q, ops_cum_seqlens_k, ops_block_table, scale, ops_out);
}

void random_sample_(infinicore::Tensor out, const infinicore::Tensor &logits,
                    float random_val, float topp, int topk,
                    float temperature) {
    INFINILM_OPS_SHIM_PROFILE(kRandomSample);
    auto ops_logits = to_ops_tensor(logits);
    auto ops_out = to_ops_tensor(out);
    infini::ops::Operator<infini::ops::RandomSample>::Call(
        make_handle(), make_config(), ops_logits, random_val, topp, topk,
        temperature, ops_out);
}

infinicore::Tensor embedding(const infinicore::Tensor &indices,
                             const infinicore::Tensor &weight) {
    INFINILM_OPS_SHIM_PROFILE(kEmbedding);
    // Output shape: `indices.shape() + [embedding_dim]`.
    auto out_shape = indices->shape();
    out_shape.push_back(weight->shape().back());
    auto out = infinicore::Tensor::empty(out_shape, weight->dtype(), weight->device());

    auto ops_indices = to_ops_tensor(indices);
    auto ops_weight = to_ops_tensor(weight);
    auto ops_out = to_ops_tensor(out);
    infini::ops::Operator<infini::ops::Embedding>::Call(
        make_handle(), make_config(), ops_indices, ops_weight, ops_out);
    return out;
}

infinicore::Tensor linear(const infinicore::Tensor &input,
                          const infinicore::Tensor &weight,
                          const std::optional<infinicore::Tensor> &bias) {
    INFINILM_OPS_SHIM_PROFILE(kLinear);
    const auto &input_shape = input->shape();
    const auto &weight_shape = weight->shape();
    const auto out_features = weight_shape[0];
    const auto in_features = weight_shape[1];

    auto out_shape = input_shape;
    out_shape.back() = out_features;
    auto out = infinicore::Tensor::empty(out_shape, input->dtype(), input->device());

    // Flatten the leading dims for a 2D `addmm`. `input`/`weight` must be
    // contiguous so the 2D view is sound.
    infinicore::Size n = 1;
    for (size_t i = 0; i + 1 < input_shape.size(); ++i) {
        n *= input_shape[i];
    }
    auto input_contig = input->is_contiguous() ? input : input->contiguous();
    auto weight_contig = weight->is_contiguous() ? weight : weight->contiguous();
    auto input_2d = input_contig->view({n, in_features});
    auto out_2d = out->view({n, out_features});

    float beta = 0.0f;
    if (bias.has_value()) {
        // Broadcast `bias` into `out_2d` (shape `[n, out_features]`) by
        // copying through a strided view, then blend it into the `Gemm`
        // via `beta = 1`.
        auto bias_broadcast =
            bias.value()->as_strided({n, out_features}, {0, 1});
        out_2d->copy_from(bias_broadcast);
        beta = 1.0f;
    }

    // `Gemm(a, b, alpha, beta, trans_a, trans_b, c)`:
    //   `c = alpha * op(a) @ op(b) + beta * c`.
    // With `trans_b = 1` we compute `input @ weight.T` in one step.
    cuda_dispatch::gemm(to_ops_tensor(input_2d), to_ops_tensor(weight_contig),
                        /*alpha=*/1.0f, /*beta=*/beta, /*trans_a=*/0,
                        /*trans_b=*/1, to_ops_tensor(out_2d),
                        infinicore::context::getStream());

    return out;
}

infinicore::Tensor rms_norm(const infinicore::Tensor &input,
                            const infinicore::Tensor &weight, float eps) {
    INFINILM_OPS_SHIM_PROFILE(kRmsNorm);
    auto out = infinicore::Tensor::empty(input->shape(), input->dtype(), input->device());
    cuda_dispatch::rms_norm(to_ops_tensor(input), to_ops_tensor(weight), eps,
                            to_ops_tensor(out),
                            infinicore::context::getStream());
    return out;
}

void rms_norm_forward_inplace(infinicore::Tensor &hidden_states,
                              infinicore::Tensor &residual,
                              const infinicore::Tensor &weight, float eps) {
    INFINILM_OPS_SHIM_PROFILE(kRmsNormForwardInplace);
    if (!residual) {
        // First layer: `residual` captures the pre-norm activations; we only
        // normalize `hidden_states`.
        residual = hidden_states;
        hidden_states = rms_norm(hidden_states, weight, eps);
    } else {
        // Subsequent layers: fuse `residual += hidden_states` with the norm
        // via `AddRmsNorm`. One CUDA kernel instead of two, and the residual
        // is written in-place so the single-pass kernel can share the load.
        auto out = infinicore::Tensor::empty(hidden_states->shape(),
                                             hidden_states->dtype(),
                                             hidden_states->device());
        auto new_residual = infinicore::Tensor::empty(hidden_states->shape(),
                                                      hidden_states->dtype(),
                                                      hidden_states->device());
        cuda_dispatch::add_rms_norm(
            to_ops_tensor(hidden_states), to_ops_tensor(residual),
            to_ops_tensor(weight), eps, to_ops_tensor(out),
            to_ops_tensor(new_residual), infinicore::context::getStream());
        hidden_states = out;
        residual = new_residual;
    }
}

infinicore::Tensor rope_forward(const infinicore::nn::RoPE &module,
                                const infinicore::Tensor &x,
                                const infinicore::Tensor &positions,
                                std::optional<infinicore::Tensor> out) {
    INFINILM_OPS_SHIM_PROFILE(kRopeForward);
    // Keep RoPE on `nn::RoPE::forward` rather than routing it through
    // InfiniOps. The op is cheap (~µs / layer on decode), and its
    // pre-computed sin/cos tables live as protected `Module` buffers that
    // aren't reachable from outside — forwarding through the module avoids
    // adding public accessors just to feed a different kernel.
    auto destination = out.value_or(x);
    module.forward(destination, x, positions);
    return destination;
}

} // namespace infinilm::ops_shim
