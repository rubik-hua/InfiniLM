#include "ops_shim.hpp"

#include <infinicore/context/context.hpp>

#include <config.h>
#include <data_type.h>
#include <device.h>
#include <handle.h>
#include <operator.h>
#include <base/add.h>
#include <base/mha_kvcache.h>
#include <base/mha_varlen.h>
#include <base/paged_caching.h>
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
#include <torch/mha_kvcache/mha_kvcache.h>
#include <torch/mha_varlen/mha_varlen.h>
#include <torch/paged_caching/paged_caching.h>
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

#include <stdexcept>

namespace infinilm::ops_shim {

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

// The `PyTorch` backend is registered as implementation index `1`. We route
// all calls through that backend for consistency, which also guarantees that
// ops like `Embedding` (which currently only exist there) stay reachable.
constexpr std::size_t kTorchImplementationIndex = 1;

infini::ops::Handle make_handle() {
    infini::ops::Handle handle;
    handle.set_stream(infinicore::context::getStream());
    return handle;
}

infini::ops::Config make_config() {
    infini::ops::Config config;
    config.set_implementation_index(kTorchImplementationIndex);
    return config;
}

} // namespace

infinicore::Tensor add(const infinicore::Tensor &a, const infinicore::Tensor &b) {
    auto c = infinicore::Tensor::empty(a->shape(), a->dtype(), a->device());
    auto ops_a = to_ops_tensor(a);
    auto ops_b = to_ops_tensor(b);
    auto ops_c = to_ops_tensor(c);
    infini::ops::Operator<infini::ops::Add>::Call(make_handle(), make_config(), ops_a, ops_b, ops_c);
    return c;
}

infinicore::Tensor swiglu(const infinicore::Tensor &input, const infinicore::Tensor &gate) {
    auto out = infinicore::Tensor::empty(input->shape(), input->dtype(), input->device());
    auto ops_input = to_ops_tensor(input);
    auto ops_gate = to_ops_tensor(gate);
    auto ops_out = to_ops_tensor(out);
    infini::ops::Operator<infini::ops::Swiglu>::Call(make_handle(), make_config(), ops_input, ops_gate, ops_out);
    return out;
}

void paged_caching_(infinicore::Tensor k_cache, infinicore::Tensor v_cache,
                    const infinicore::Tensor &k, const infinicore::Tensor &v,
                    const infinicore::Tensor &slot_mapping) {
    auto ops_k_cache = to_ops_tensor(k_cache);
    auto ops_v_cache = to_ops_tensor(v_cache);
    auto ops_k = to_ops_tensor(k);
    auto ops_v = to_ops_tensor(v);
    auto ops_slot_mapping = to_ops_tensor(slot_mapping);
    infini::ops::Operator<infini::ops::PagedCaching>::Call(
        make_handle(), make_config(), ops_k_cache, ops_v_cache, ops_k, ops_v,
        ops_slot_mapping);
}

infinicore::Tensor mha_kvcache(const infinicore::Tensor &q,
                               const infinicore::Tensor &k_cache,
                               const infinicore::Tensor &v_cache,
                               const infinicore::Tensor &seqlens_k,
                               const infinicore::Tensor &block_table,
                               float scale) {
    auto out = infinicore::Tensor::empty(q->shape(), q->dtype(), q->device());
    auto ops_q = to_ops_tensor(q);
    auto ops_k_cache = to_ops_tensor(k_cache);
    auto ops_v_cache = to_ops_tensor(v_cache);
    auto ops_seqlens_k = to_ops_tensor(seqlens_k);
    auto ops_block_table = to_ops_tensor(block_table);
    auto ops_out = to_ops_tensor(out);
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

} // namespace infinilm::ops_shim
