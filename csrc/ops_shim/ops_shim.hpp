#pragma once

#include <infinicore/tensor.hpp>

namespace infinilm::ops_shim {

// Compute `a + b` via the `InfiniOps` `Add` operator.
infinicore::Tensor add(const infinicore::Tensor &a, const infinicore::Tensor &b);

// Compute `input * SiLU(gate)` via the `InfiniOps` `Swiglu` operator.
infinicore::Tensor swiglu(const infinicore::Tensor &input, const infinicore::Tensor &gate);

// Scatter `(k, v)` into `(k_cache, v_cache)` according to `slot_mapping` via
// the `InfiniOps` `PagedCaching` operator.
void paged_caching_(infinicore::Tensor k_cache, infinicore::Tensor v_cache,
                    const infinicore::Tensor &k, const infinicore::Tensor &v,
                    const infinicore::Tensor &slot_mapping);

} // namespace infinilm::ops_shim
