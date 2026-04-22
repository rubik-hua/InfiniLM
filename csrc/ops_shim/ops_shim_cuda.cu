// CUDA dispatch helpers used by `ops_shim.cpp`.
//
// This translation unit is compiled with `nvcc` so it can include
// `InfiniOps`' `cuda/nvidia/*/kernel.h` headers (which transitively pull
// in `__device__` / `__global__` declarations). It deliberately avoids
// including any `infinicore` headers — those pull in `spdlog`, whose
// recent C++23 `_BitInt` usage breaks `nvcc`'s host frontend. The host
// `ops_shim.cpp` does the `infinicore::Tensor` → `infini::ops::Tensor`
// conversion and then calls into the helpers declared here.

#include "ops_shim_cuda.hpp"

#include <config.h>
#include <data_type.h>
#include <device.h>
#include <handle.h>
#include <operator.h>
#include <tensor.h>

#include <base/add.h>
#include <base/add_rms_norm.h>
#include <base/gemm.h>
#include <base/paged_caching.h>
#include <base/rms_norm.h>
#include <base/swiglu.h>

#include <cuda/nvidia/add/kernel.h>
#include <cuda/nvidia/add_rms_norm/kernel.h>
#include <cuda/nvidia/gemm/cublas.h>
#include <cuda/nvidia/gemm/cublaslt.h>
#include <cuda/nvidia/paged_caching/kernel.h>
#include <cuda/nvidia/rms_norm/kernel.h>
#include <cuda/nvidia/swiglu/kernel.h>

#include <cpu/device_.h>
#include <cuda/nvidia/device_.h>

namespace infinilm::ops_shim::cuda_dispatch {

namespace {

// Route through `Operator::Call` (not direct construction). Its cache
// reuses the `Operator` instance across same-shape calls, which matters
// a lot here: the native CUDA `CudaAdd` / `CudaRmsNorm` etc. constructors
// `cudaMalloc` + `cudaMemcpy` GPU-side metadata, and rebuilding that
// per call wipes out any native-kernel win.
infini::ops::Handle make_handle(void *stream) {
    infini::ops::Handle handle;
    handle.set_stream(stream);
    return handle;
}

infini::ops::Config native_config() {
    infini::ops::Config config;
    config.set_implementation_index(0);
    return config;
}

} // namespace

void add(const infini::ops::Tensor &a, const infini::ops::Tensor &b,
         infini::ops::Tensor c, void *stream) {
    infini::ops::Operator<infini::ops::Add>::Call(
        make_handle(stream), native_config(), a, b, c);
}

void swiglu(const infini::ops::Tensor &input,
            const infini::ops::Tensor &gate, infini::ops::Tensor out,
            void *stream) {
    infini::ops::Operator<infini::ops::Swiglu>::Call(
        make_handle(stream), native_config(), input, gate, out);
}

void rms_norm(const infini::ops::Tensor &input,
              const infini::ops::Tensor &weight, float eps,
              infini::ops::Tensor out, void *stream) {
    infini::ops::Operator<infini::ops::RmsNorm>::Call(
        make_handle(stream), native_config(), input, weight, eps, out);
}

void add_rms_norm(const infini::ops::Tensor &input,
                  const infini::ops::Tensor &other,
                  const infini::ops::Tensor &weight, float eps,
                  infini::ops::Tensor out,
                  infini::ops::Tensor residual_out, void *stream) {
    infini::ops::Operator<infini::ops::AddRmsNorm>::Call(
        make_handle(stream), native_config(), input, other, weight, eps, out,
        residual_out);
}

void paged_caching(infini::ops::Tensor k_cache, infini::ops::Tensor v_cache,
                   const infini::ops::Tensor &k,
                   const infini::ops::Tensor &v,
                   const infini::ops::Tensor &slot_mapping, void *stream) {
    infini::ops::Operator<infini::ops::PagedCaching>::Call(
        make_handle(stream), native_config(), k_cache, v_cache, k, v,
        slot_mapping);
}

void gemm(const infini::ops::Tensor &a, const infini::ops::Tensor &b,
          float alpha, float beta, int trans_a, int trans_b,
          infini::ops::Tensor c, void *stream) {
    // `cublasLt` (index `1`) caches matmul descriptor + heuristic per
    // `Operator` instance, which on the hot decode path beats the plain
    // `cublas` backend (index `0`) that repeats algo selection every call.
    infini::ops::Config config;
    config.set_implementation_index(1);
    infini::ops::Operator<infini::ops::Gemm>::Call(
        make_handle(stream), config, a, b,
        std::optional<float>{alpha}, std::optional<float>{beta},
        std::optional<int>{trans_a}, std::optional<int>{trans_b}, c);
}

} // namespace infinilm::ops_shim::cuda_dispatch
