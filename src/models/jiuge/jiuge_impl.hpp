#ifndef JIUGE_IMPL_H
#define JIUGE_IMPL_H

#include "infinicore_infer.h"
#include "infinicore_infer/models/jiuge.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"
#include "../model_base.hpp"

#include <memory>
#include <vector>

struct JiugeDeviceResource : public DeviceResourceBase {
    // Weights (base holds: device, device_id, handle, stream, comm, memory_pool)
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table, cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv,
        w_attn_q_norm, w_attn_k_norm, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
};

#include "../../cache.hpp"

struct JiugeModel : public ModelBase<JiugeMeta, JiugeDeviceResource> {
    JiugeModel(const JiugeMeta *meta, const JiugeWeights *weights,
               infiniDevice_t device, std::vector<int> dev_ids);

    // Stop worker threads while the derived vtable is still live so
    // releaseDeviceResource() dispatches correctly. ~ModelBase intentionally
    // does not call shutdown() to avoid a pure-virtual call.
    ~JiugeModel() override { shutdown(); }

protected:
    void createDeviceResource(JiugeDeviceResource *rsrc,
                              int idev, int ndev,
                              int dev_id, infinicclComm_t comm) override;
    void releaseDeviceResource(JiugeDeviceResource &rsrc) override;
    void inferDeviceBatch(JiugeDeviceResource &rsrc,
                          int idev, int ndev,
                          const BaseInferRequest &req) override;

private:
    const JiugeWeights *weights_;  // non-owning; lifetime managed by caller
};

#endif
