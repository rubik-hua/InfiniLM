#pragma once

#include <tensor.h>

namespace infinilm::ops_shim::cuda_dispatch {

// Direct-instantiation entry points for the `InfiniOps` NVIDIA CUDA
// backends (implementation index `0`). They skip `Operator::Call`'s
// cache / SFINAE dispatch, which costs ~5-10 us per call on the hot
// path of small element-wise ops.
//
// These live in a separate `.cu` TU because the `cuda/nvidia/*/kernel.h`
// headers they need transitively include `__device__` / `__global__`
// declarations that the host compiler cannot parse. The host-side
// `ops_shim.cpp` builds `infini::ops::Tensor` objects from
// `infinicore::Tensor`s and dispatches here.

void add(const infini::ops::Tensor &a, const infini::ops::Tensor &b,
         infini::ops::Tensor c, void *stream);

void swiglu(const infini::ops::Tensor &input,
            const infini::ops::Tensor &gate, infini::ops::Tensor out,
            void *stream);

void rms_norm(const infini::ops::Tensor &input,
              const infini::ops::Tensor &weight, float eps,
              infini::ops::Tensor out, void *stream);

void add_rms_norm(const infini::ops::Tensor &input,
                  const infini::ops::Tensor &other,
                  const infini::ops::Tensor &weight, float eps,
                  infini::ops::Tensor out,
                  infini::ops::Tensor residual_out, void *stream);

void gemm(const infini::ops::Tensor &a, const infini::ops::Tensor &b,
          float alpha, float beta, int trans_a, int trans_b,
          infini::ops::Tensor c, void *stream);

} // namespace infinilm::ops_shim::cuda_dispatch
