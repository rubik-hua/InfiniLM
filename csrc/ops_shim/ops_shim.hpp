#pragma once

#include <infinicore/tensor.hpp>

namespace infinilm::ops_shim {

// Compute `a + b` via the `InfiniOps` `Add` operator.
infinicore::Tensor add(const infinicore::Tensor &a, const infinicore::Tensor &b);

} // namespace infinilm::ops_shim
