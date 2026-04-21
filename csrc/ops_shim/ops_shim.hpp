#pragma once

#include <infinicore/nn/rope.hpp>
#include <infinicore/tensor.hpp>

#include <optional>

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

// Decode-time multi-head attention over a paged KV cache.
infinicore::Tensor mha_kvcache(const infinicore::Tensor &q,
                               const infinicore::Tensor &k_cache,
                               const infinicore::Tensor &v_cache,
                               const infinicore::Tensor &seqlens_k,
                               const infinicore::Tensor &block_table,
                               float scale);

// Prefill-time variable-length multi-head attention over a paged KV cache.
void mha_varlen_(infinicore::Tensor out,
                 const infinicore::Tensor &q,
                 const infinicore::Tensor &k_cache,
                 const infinicore::Tensor &v_cache,
                 const infinicore::Tensor &cum_seqlens_q,
                 const infinicore::Tensor &cum_seqlens_k,
                 const infinicore::Tensor &block_table,
                 float scale);

// Sample one index from `logits` into a 0D `out` tensor.
void random_sample_(infinicore::Tensor out, const infinicore::Tensor &logits,
                    float random_val, float topp, int topk,
                    float temperature);

// Embedding lookup `weight[indices]`. `indices` can have any shape; the
// output shape is `indices.shape() + [embedding_dim]`.
infinicore::Tensor embedding(const infinicore::Tensor &indices,
                             const infinicore::Tensor &weight);

// Dense linear: `out = input @ weight.T [+ bias]`.
//   `input`:  `[..., in_features]`.
//   `weight`: `[out_features, in_features]`.
//   `bias`:   `[out_features]` (optional).
// Returns a tensor of shape `input.shape()[:-1] + [out_features]`.
infinicore::Tensor linear(const infinicore::Tensor &input,
                          const infinicore::Tensor &weight,
                          const std::optional<infinicore::Tensor> &bias = std::nullopt);

// Returns `rms_norm(input, weight, eps)`.
infinicore::Tensor rms_norm(const infinicore::Tensor &input,
                            const infinicore::Tensor &weight, float eps);

// Fused add + RMSNorm with in-place residual semantics, matching
// `infinicore::nn::RMSNorm::forward_inplace`:
//   - On first call (`residual` is null), `residual` takes `hidden_states`
//     and `hidden_states` becomes `rms_norm(hidden_states, weight, eps)`.
//   - Otherwise `residual = hidden_states + residual`, then
//     `hidden_states = rms_norm(residual, weight, eps)`.
void rms_norm_forward_inplace(infinicore::Tensor &hidden_states,
                              infinicore::Tensor &residual,
                              const infinicore::Tensor &weight, float eps);

// Apply the `InfiniOps` `Rope` op using `module`'s pre-computed `sin` /
// `cos` cache. When `out` is omitted, runs in place on `x` and returns
// it; when provided, writes into `out` and returns it. This covers both
// `infinicore::nn::RoPE::forward(x, pos, in_place)` and
// `infinicore::nn::RoPE::forward(y, x, pos)`.
infinicore::Tensor rope_forward(const infinicore::nn::RoPE &module,
                                const infinicore::Tensor &x,
                                const infinicore::Tensor &positions,
                                std::optional<infinicore::Tensor> out = std::nullopt);

} // namespace infinilm::ops_shim
