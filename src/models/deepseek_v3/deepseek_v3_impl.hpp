// src/models/deepseek_v3/deepseek_v3_impl.hpp
#ifndef DEEPSEEK_V3_IMPL_H
#define DEEPSEEK_V3_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"
#include "../model_base.hpp"

#include <memory>
#include <vector>

// ─── Weight structs (unchanged) ────────────────────────────────────────────
struct QuantLinearWeight {
    std::shared_ptr<Tensor> w, s, z;
};
struct MLAWeight {
    std::shared_ptr<Tensor> kv_a_norm, q_a_norm;
    std::shared_ptr<QuantLinearWeight> kv_a_proj, kv_b_proj, o_proj, q_a_proj, q_b_proj;
};
struct GateWeight {
    std::shared_ptr<Tensor> w, b;
};
struct MLPWeight {
    std::shared_ptr<QuantLinearWeight> gate, up, down;
};
struct LayerWeight {
    std::shared_ptr<Tensor> mla_norm, mlp_norm;
    std::shared_ptr<MLAWeight> mla;
    std::shared_ptr<MLPWeight> dense_mlp, share_expert;
    std::shared_ptr<GateWeight> route;
    std::vector<std::shared_ptr<MLPWeight>> experts;
};
struct DeepSeekV3DeviceWeights {
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table, cos_table;
    std::vector<LayerWeight> w_layers;
    infiniDevice_t device;
    int dev_id;
    infinirtStream_t load_stream;
};
struct DeepSeekV3Weights {
    std::vector<std::shared_ptr<DeepSeekV3DeviceWeights>> device_weights;
    DeepSeekV3Weights(const DeepSeekV3Meta *meta, infiniDevice_t device,
                      int ndev, const int *dev_ids);
};

// ─── Cache (unchanged) ─────────────────────────────────────────────────────
struct DeepSeekV3Cache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> kv_pass, k_rot;
};

// ─── Custom request type for DeepSeekV3 ───────────────────────────────────
// Uses DeepSeekV3Cache** instead of KVCache** — cannot reuse BaseInferRequest.
struct DSInferRequest {
    const uint32_t          *tokens;
    uint32_t                 ntok;
    const uint32_t          *req_lens;
    uint32_t                 nreq;
    const uint32_t          *req_pos;
    struct DeepSeekV3Cache **kv_caches;
    const float             *temperature;
    const uint32_t          *topk;
    const float             *topp;
    uint32_t                *output;
    void                    *logits;
};

// ─── Device resource ───────────────────────────────────────────────────────
struct DeepSeekV3DeviceResource : public DeviceResourceBase {
    std::shared_ptr<DeepSeekV3DeviceWeights> weights;
};

// ─── Model ─────────────────────────────────────────────────────────────────
struct DeepSeekV3Model
    : public ModelBase<DeepSeekV3Meta, DeepSeekV3DeviceResource, DSInferRequest>
{
    DeepSeekV3Model(const DeepSeekV3Meta *meta, const DeepSeekV3Weights *weights);

protected:
    void createDeviceResource(DeepSeekV3DeviceResource *rsrc,
                              int idev, int ndev,
                              int dev_id, infinicclComm_t comm) override;
    void releaseDeviceResource(DeepSeekV3DeviceResource &rsrc) override;
    void inferDeviceBatch(DeepSeekV3DeviceResource &rsrc,
                          int idev, int ndev,
                          const DSInferRequest &req) override;

private:
    const DeepSeekV3Weights *weights_;
};

#endif
