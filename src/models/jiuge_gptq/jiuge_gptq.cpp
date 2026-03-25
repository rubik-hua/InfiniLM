#include "jiuge_gptq.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"

#include <random>
#include <thread>
#include <vector>

JiugeGPTQModel::JiugeGPTQModel(const JiugeGPTQMeta *meta, const ModelWeights *weights)
    : ModelBase(*meta, ((JiugeGPTQWeights *)(weights))->device(),
                ((JiugeGPTQWeights *)(weights))->devIds()),
      weights_(weights)
{
    launch();
}

void JiugeGPTQModel::createDeviceResource(GPTQDeviceResource *rsrc,
                                           int idev, int ndev,
                                           int dev_id, infinicclComm_t comm) {
    auto weights = (JiugeGPTQWeights *)(weights_);

    RUN_INFINI(infinirtSetDevice(device_, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    rsrc->device      = device_;
    rsrc->device_id   = dev_id;
    rsrc->handle      = handle;
    rsrc->stream      = stream;
    rsrc->comm        = comm;
    rsrc->memory_pool = memory_pool;
    rsrc->weights     = weights->device_weights()[idev];

    RUN_INFINI(infinirtDeviceSynchronize());
}

void JiugeGPTQModel::releaseDeviceResource(GPTQDeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors
    res.weights.reset();

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

void JiugeGPTQModel::inferDeviceBatch(GPTQDeviceResource &rsrc,
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
    auto weight = rsrc.weights;
    bool has_qkv_bias = meta_.has_qkv_bias;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {req.ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {req.ntok, d}, rsrc.memory_pool);
    auto q_buf = Tensor::buffer(dt_logits, {req.ntok, nh * dh}, rsrc.memory_pool);
    auto k_buf = Tensor::buffer(dt_logits, {req.ntok, nkvh * dh}, rsrc.memory_pool);
    auto v_buf = Tensor::buffer(dt_logits, {req.ntok, nkvh * dh}, rsrc.memory_pool);

    auto gate_buf = Tensor::buffer(dt_logits, {req.ntok, di}, rsrc.memory_pool);
    auto up_buf = Tensor::buffer(dt_logits, {req.ntok, di}, rsrc.memory_pool);

    auto o_buf = Tensor::buffer(dt_logits, {req.ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {req.nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {req.nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(req.nreq);

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
                                       weight->w_in_embd->data(req.tokens[i] * d),
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

    // Compute
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, weight->w_attn_norm[layer], meta_.epsilon);
        // qkv_proj
        dequant_linear(q_buf, logits_out,
                       weight->w_attn_q[layer]->w, weight->w_attn_q[layer]->s, weight->w_attn_q[layer]->z,
                       1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_q[layer] : nullptr,
                       QuantType::GPTQ, weight->w_attn_q[layer]->g_idx);
        dequant_linear(k_buf, logits_out,
                       weight->w_attn_k[layer]->w, weight->w_attn_k[layer]->s, weight->w_attn_k[layer]->z,
                       1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_k[layer] : nullptr,
                       QuantType::GPTQ, weight->w_attn_k[layer]->g_idx);
        dequant_linear(v_buf, logits_out,
                       weight->w_attn_v[layer]->w, weight->w_attn_v[layer]->s, weight->w_attn_v[layer]->z,
                       1.0, 0.0, nullptr, has_qkv_bias ? weight->b_attn_v[layer] : nullptr,
                       QuantType::GPTQ, weight->w_attn_v[layer]->g_idx);
        // rope
        rope_v2(q_buf->view({req.ntok, nh, dh}), q_buf->view({req.ntok, nh, dh}), pos_ids_buf, weight->sin_table, weight->cos_table);
        rope_v2(k_buf->view({req.ntok, nkvh, dh}), k_buf->view({req.ntok, nkvh, dh}), pos_ids_buf, weight->sin_table, weight->cos_table);
        size_t token_offset = 0;
        for (uint32_t r = 0; r < req.nreq; r++) {
            auto past_len = req.req_pos[r];
            auto seq_len = req.req_lens[r];
            auto total_len = past_len + seq_len;
            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = q_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = k_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});
            auto v = v_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});

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
        dequant_linear(logits_in, o_buf,
                       weight->w_attn_out[layer]->w, weight->w_attn_out[layer]->s, weight->w_attn_out[layer]->z,
                       1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr,
                       QuantType::GPTQ, weight->w_attn_out[layer]->g_idx);
        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), req.ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
        // 2. FFN
        rmsnorm(logits_out, logits_in, weight->w_ffn_norm[layer], meta_.epsilon);
        dequant_linear(gate_buf, logits_out,
                       weight->w_ffn_gate[layer]->w, weight->w_ffn_gate[layer]->s, weight->w_ffn_gate[layer]->z,
                       1.0, 0.0, nullptr, nullptr,
                       QuantType::GPTQ, weight->w_ffn_gate[layer]->g_idx);
        dequant_linear(up_buf, logits_out,
                       weight->w_ffn_up[layer]->w, weight->w_ffn_up[layer]->s, weight->w_ffn_up[layer]->z,
                       1.0, 0.0, nullptr, nullptr,
                       QuantType::GPTQ, weight->w_ffn_up[layer]->g_idx);
        swiglu(gate_buf, up_buf, gate_buf);
        dequant_linear(logits_in, gate_buf,
                       weight->w_ffn_down[layer]->w, weight->w_ffn_down[layer]->s, weight->w_ffn_down[layer]->z,
                       1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr,
                       QuantType::GPTQ, weight->w_ffn_down[layer]->g_idx);
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
            rmsnorm(logits_out, logits_in, weight->w_out_norm, meta_.epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {req.ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, weight->w_out_embd, 1.0, 0.0, nullptr, nullptr);
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
                        weight->w_out_norm,
                        meta_.epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, req.nreq), weight->w_out_embd, 1.0, 0.0, nullptr, nullptr);
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

__INFINI_C struct JiugeGPTQModel *
createJiugeGPTQModel(const JiugeGPTQMeta *meta,
                     const ModelWeights *weights) {
    return new JiugeGPTQModel(meta, weights);
}

__INFINI_C void destroyJiugeGPTQModel(struct JiugeGPTQModel *model) {
    delete model;  // ~ModelBase() calls shutdown(), which joins all threads
}

__INFINI_C void
inferBatchJiugeGPTQ(struct JiugeGPTQModel *model,
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
forwardBatchJiugeGPTQ(struct JiugeGPTQModel *model,
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
