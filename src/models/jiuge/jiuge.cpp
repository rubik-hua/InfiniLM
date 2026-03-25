#include "jiuge_impl.hpp"
#include "jiuge_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h"

#include <random>
#include <thread>
#include <vector>

JiugeModel::JiugeModel(const JiugeMeta *meta, const JiugeWeights *weights,
                        infiniDevice_t device, std::vector<int> dev_ids)
    : ModelBase(*meta, device, std::move(dev_ids)), weights_(weights)
{
    launch();  // starts threads; must be last — threads access weights_ via this
}

void JiugeModel::createDeviceResource(JiugeDeviceResource *rsrc,
                                       int idev, int ndev,
                                       int dev_id, infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device_, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv,
        w_attn_q_norm, w_attn_k_norm, w_attn_out, w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    for (size_t layer = 0; layer < meta_.nlayer; layer++) {
        w_attn_norm.push_back(getAttnNorm(&meta_, weights_, layer));
        w_attn_qkv.push_back(getAttnQKV(&meta_, weights_, layer, idev, ndev));
        if (weights_->attn_qkv_b != nullptr)
            b_attn_qkv.push_back(getAttnQKVBias(&meta_, weights_, layer, idev, ndev));
        if (weights_->attn_q_norm != nullptr) {
            w_attn_q_norm.push_back(getAttnQNorm(&meta_, weights_, layer));
            w_attn_k_norm.push_back(getAttnKNorm(&meta_, weights_, layer));
        }
        w_attn_out.push_back(getAttnO(&meta_, weights_, layer, idev, ndev));
        w_ffn_norm.push_back(getFFNNorm(&meta_, weights_, layer));
        w_ffn_gate_up.push_back(getFFNGateUp(&meta_, weights_, layer, idev, ndev));
        w_ffn_down.push_back(getFFNDown(&meta_, weights_, layer, idev, ndev));
    }

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    // Set DeviceResourceBase fields
    rsrc->device       = device_;
    rsrc->device_id    = dev_id;
    rsrc->handle       = handle;
    rsrc->stream       = stream;
    rsrc->comm         = comm;
    rsrc->memory_pool  = memory_pool;
    // Set JiugeDeviceResource-specific fields
    rsrc->w_in_embd     = getInEmbd(&meta_, weights_);
    rsrc->w_out_norm    = getOutNorm(&meta_, weights_);
    rsrc->w_out_embd    = getOutEmbd(&meta_, weights_);
    rsrc->sin_table     = getSinTable(&meta_);
    rsrc->cos_table     = getCosTable(&meta_);
    rsrc->w_attn_norm   = std::move(w_attn_norm);
    rsrc->w_attn_qkv    = std::move(w_attn_qkv);
    rsrc->b_attn_qkv    = std::move(b_attn_qkv);
    rsrc->w_attn_q_norm = std::move(w_attn_q_norm);
    rsrc->w_attn_k_norm = std::move(w_attn_k_norm);
    rsrc->w_attn_out    = std::move(w_attn_out);
    rsrc->w_ffn_norm    = std::move(w_ffn_norm);
    rsrc->w_ffn_gate_up = std::move(w_ffn_gate_up);
    rsrc->w_ffn_down    = std::move(w_ffn_down);

    RUN_INFINI(infinirtDeviceSynchronize());
}

void JiugeModel::releaseDeviceResource(JiugeDeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors
    res.w_in_embd.reset();
    res.w_out_norm.reset();
    res.w_out_embd.reset();
    res.sin_table.reset();
    res.cos_table.reset();
    for (auto &t : res.w_attn_norm) {
        t.reset();
    }
    res.w_attn_norm.clear();
    for (auto &t : res.w_attn_qkv) {
        t.reset();
    }
    res.w_attn_qkv.clear();
    for (auto &t : res.b_attn_qkv) {
        t.reset();
    }
    res.b_attn_qkv.clear();
    for (auto &t : res.w_attn_out) {
        t.reset();
    }
    res.w_attn_out.clear();
    for (auto &t : res.w_ffn_norm) {
        t.reset();
    }
    res.w_ffn_norm.clear();
    for (auto &t : res.w_ffn_gate_up) {
        t.reset();
    }
    res.w_ffn_gate_up.clear();
    for (auto &t : res.w_ffn_down) {
        t.reset();
    }
    res.w_ffn_down.clear();
    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

void JiugeModel::inferDeviceBatch(JiugeDeviceResource &rsrc,
                                   int idev, int ndev,
                                   const BaseInferRequest &req) {
    auto nlayer = meta_.nlayer;
    auto nkvh = meta_.nkvh / ndev;
    auto nh = meta_.nh / ndev;
    auto ngroup = nh / nkvh;
    // auto dctx = meta_.dctx;
    auto dh = meta_.dh;
    auto d = meta_.d;
    auto dt_logits = meta_.dt_logits;
    auto di = meta_.di / ndev;
    auto dvoc = meta_.dvoc;
    auto stream = rsrc.stream;
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;
    bool has_qk_norm = rsrc.w_attn_q_norm.size() > 0 && rsrc.w_attn_k_norm.size() > 0;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {req.ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {req.ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {req.ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dt_logits, {req.ntok, 2 * di}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {req.ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {req.nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {req.nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(req.nreq);

    auto qkv_rope = qkv_buf->view({req.ntok, nh + nkvh * 2, dh});
    auto q_buf = qkv_rope->slice(1, 0, nh);
    auto k_buf = qkv_rope->slice(1, nh, nkvh);

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(req.ntok);
    size_t req_start = 0;
    for (uint32_t r = 0; r < req.nreq; r++) {
        for (uint32_t i = 0; i < req.req_lens[r]; i++) {
            batch_pos_ids[req_start + i] = req.req_pos[r] + i;
        }
        req_start += req.req_lens[r];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {req.ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {req.ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * req.ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    for (uint32_t i = 0; i < req.ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(req.tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // Attention
    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t r = 0; r < req.nreq; r++) {
        auto past_len = req.req_pos[r];
        auto seq_len = req.req_lens[r];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }

    auto qk_buf = Tensor::buffer(dt_logits, {nh * max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    // MLP buffers
    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);

    // Compute
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, rsrc.w_attn_norm[layer], meta_.epsilon);
        // qkv_proj
        linear(qkv_buf, logits_out, rsrc.w_attn_qkv[layer], 1.0, 0.0, nullptr, has_qkv_bias ? rsrc.b_attn_qkv[layer] : nullptr);

        if (has_qk_norm) {
            rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer], meta_.epsilon);
            rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer], meta_.epsilon);
        }

        // rope
        rope(q_buf, q_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
        rope(k_buf, k_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);

        size_t token_offset = 0;
        for (uint32_t r = 0; r < req.nreq; r++) {
            auto past_len = req.req_pos[r];
            auto seq_len = req.req_lens[r];
            auto total_len = past_len + seq_len;
            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = qkv_rope->slice({{0, token_offset, seq_len}, {1, 0, nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
            auto v = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});

            // self attention
            // concat
            rearrange(req.kv_caches[r]->k[idev][layer]->slice(0, past_len, seq_len), k);
            rearrange(req.kv_caches[r]->v[idev][layer]->slice(0, past_len, seq_len), v);
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = req.kv_caches[r]->k[idev][layer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);
            auto v_gemm = req.kv_caches[r]->v[idev][layer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

            token_offset += seq_len;
        }

        // o_proj
        linear(logits_in, o_buf, rsrc.w_attn_out[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), req.ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
        // 2. FFN
        rmsnorm(logits_out, logits_in, rsrc.w_ffn_norm[layer], meta_.epsilon);
        linear(gate_up_buf, logits_out, rsrc.w_ffn_gate_up[layer], 1.0, 0.0, nullptr, nullptr);
        swiglu(gate_buf, up_buf, gate_buf);
        linear(logits_in, gate_buf, rsrc.w_ffn_down[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), req.ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }
    // Sample and Output
    if (idev == 0) {
        if (req.logits != nullptr) {
            rmsnorm(logits_out, logits_in, rsrc.w_out_norm, meta_.epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {req.ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, rsrc.w_out_embd, 1.0, 0.0, nullptr, nullptr);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(req.logits, last_logits_buf->data(), dsize(dt_logits) * req.ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (req.output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t r = 0; r < req.nreq; r++) {
                auto seq_len = req.req_lens[r];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, r, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        rsrc.w_out_norm,
                        meta_.epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, req.nreq), rsrc.w_out_embd, 1.0, 0.0, nullptr, nullptr);
            std::random_device _rd;
            std::mt19937 gen(_rd());
            token_offset = 0;
            for (uint32_t r = 0; r < req.nreq; r++) {
                auto seq_len = req.req_lens[r];
                float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
                randomSample(result_buf->slice(0, r, 1)->view_as({}, {}),
                             prob_buf->slice(0, r, 1)->view_as({dvoc}, {1}),
                             random_val, req.topp[r], req.topk[r], req.temperature[r]);
                token_offset += seq_len;
            }
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                      sizeof(int64_t) * req.nreq, INFINIRT_MEMCPY_D2H));
            for (uint32_t r = 0; r < req.nreq; r++) {
                req.output[r] = uint32_t(result_cpu[r]);
            }
        }
    }
}

__INFINI_C void
inferBatchJiuge(struct JiugeModel *model,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                struct KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                uint32_t *output) {
    BaseInferRequest req{};
    req.tokens      = tokens;
    req.ntok        = ntok;
    req.req_lens    = req_lens;
    req.nreq        = nreq;
    req.req_pos     = req_pos;
    req.kv_caches   = kv_caches;
    req.temperature = temperature;
    req.topk        = topk;
    req.topp        = topp;
    req.output      = output;
    req.logits      = nullptr;
    model->dispatch(req);
}

__INFINI_C void
forwardBatchJiuge(struct JiugeModel *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  void *logits) {
    BaseInferRequest req{};
    req.tokens    = tokens;
    req.ntok      = ntok;
    req.req_lens  = req_lens;
    req.nreq      = nreq;
    req.req_pos   = req_pos;
    req.kv_caches = kv_caches;
    req.logits    = logits;
    req.output    = nullptr;
    model->dispatch(req);
}

__INFINI_C struct JiugeModel *
createJiugeModel(const JiugeMeta *meta, const JiugeWeights *weights,
                 infiniDevice_t device, int ndev, const int *dev_ids) {
    std::vector<int> device_ids(dev_ids, dev_ids + ndev);
    return new JiugeModel(meta, weights, device, std::move(device_ids));
}

__INFINI_C void destroyJiugeModel(struct JiugeModel *model) {
    delete model;  // ~ModelBase() calls shutdown(), which joins all threads
}
