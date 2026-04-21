#include "ops_shim.hpp"

#include <infinicore/context/context.hpp>

#include <config.h>
#include <data_type.h>
#include <device.h>
#include <handle.h>
#include <operator.h>
#include <base/add.h>
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

} // namespace infinilm::ops_shim
