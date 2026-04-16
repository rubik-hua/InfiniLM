#pragma once
#include "infinicore_infer/models/jiuge_gptq.h"

#include "../../cache.hpp"
#include "../../dataloader/weights_loader.hpp"
#include "../model_base.hpp"

#include <memory>

struct QuantInt4Weight {
    std::shared_ptr<Tensor> w, s, z, g_idx;  // g_idx distinguishes GPTQ from AWQ
};

struct JiugeGPTQDeviceWeight {
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table, cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, b_attn_q, b_attn_k, b_attn_v, w_ffn_norm;
    std::vector<std::shared_ptr<QuantInt4Weight>> w_attn_q, w_attn_k, w_attn_v,
        w_attn_out, w_ffn_gate, w_ffn_up, w_ffn_down;
};

class JiugeGPTQWeights : public infinicore::weights::Loader {
private:
    std::vector<std::shared_ptr<JiugeGPTQDeviceWeight>> _device_weights;
public:
    JiugeGPTQWeights(const JiugeGPTQMeta *meta, infiniDevice_t device,
                     const std::vector<int> &dev_ids);
    std::vector<std::shared_ptr<JiugeGPTQDeviceWeight>> &device_weights() {
        return _device_weights;
    }
};

struct GPTQDeviceResource : public DeviceResourceBase {
    std::shared_ptr<JiugeGPTQDeviceWeight> weights;
};

struct JiugeGPTQModel : public ModelBase<JiugeGPTQMeta, GPTQDeviceResource> {
    JiugeGPTQModel(const JiugeGPTQMeta *meta, const ModelWeights *weights);

    // See JiugeModel::~JiugeModel: shutdown() must run before the base
    // destructor to keep the derived vtable alive for releaseDeviceResource.
    ~JiugeGPTQModel() override { shutdown(); }

protected:
    void createDeviceResource(GPTQDeviceResource *rsrc,
                              int idev, int ndev,
                              int dev_id, infinicclComm_t comm) override;
    void releaseDeviceResource(GPTQDeviceResource &rsrc) override;
    void inferDeviceBatch(GPTQDeviceResource &rsrc,
                          int idev, int ndev,
                          const BaseInferRequest &req) override;

private:
    const ModelWeights *weights_;
};
