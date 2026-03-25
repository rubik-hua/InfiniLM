#pragma once
#include "infinicore_infer/models/jiuge_awq.h"

#include "../../cache.hpp"
#include "../../dataloader/weights_loader.hpp"
#include "../model_base.hpp"

#include <memory>

struct QuantInt4Weight {
    std::shared_ptr<Tensor> w, s, z;
};

struct JiugeAWQDeviceWeight {
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table, cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, b_attn_q, b_attn_k, b_attn_v, w_ffn_norm;
    std::vector<std::shared_ptr<QuantInt4Weight>> w_attn_q, w_attn_k, w_attn_v,
        w_attn_out, w_ffn_gate, w_ffn_up, w_ffn_down;
};

class JiugeAWQWeights : public infinicore::weights::Loader {
private:
    std::vector<std::shared_ptr<JiugeAWQDeviceWeight>> _device_weights;
public:
    JiugeAWQWeights(const JiugeAWQMeta *meta, infiniDevice_t device,
                    const std::vector<int> &dev_ids);
    std::vector<std::shared_ptr<JiugeAWQDeviceWeight>> &device_weights() {
        return _device_weights;
    }
};

struct AWQDeviceResource : public DeviceResourceBase {
    std::shared_ptr<JiugeAWQDeviceWeight> weights;
};

struct JiugeAWQModel : public ModelBase<JiugeAWQMeta, AWQDeviceResource> {
    JiugeAWQModel(const JiugeAWQMeta *meta, const ModelWeights *weights);

protected:
    void createDeviceResource(AWQDeviceResource *rsrc,
                              int idev, int ndev,
                              int dev_id, infinicclComm_t comm) override;
    void releaseDeviceResource(AWQDeviceResource &rsrc) override;
    void inferDeviceBatch(AWQDeviceResource &rsrc,
                          int idev, int ndev,
                          const BaseInferRequest &req) override;

private:
    const ModelWeights *weights_;
};
