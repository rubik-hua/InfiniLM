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

} // namespace infinilm::ops_shim
