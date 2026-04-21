#pragma once

#include <infinicore/tensor.hpp>

namespace infinilm::ops_shim {

// Compute `a + b` via the `InfiniOps` `Add` operator.
infinicore::Tensor add(const infinicore::Tensor &a, const infinicore::Tensor &b);

// Compute `input * SiLU(gate)` via the `InfiniOps` `Swiglu` operator.
infinicore::Tensor swiglu(const infinicore::Tensor &input, const infinicore::Tensor &gate);

} // namespace infinilm::ops_shim
