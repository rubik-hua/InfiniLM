// src/models/model_base.hpp
#pragma once

#include "../cache.hpp"
#include "inference_context.hpp"
#include "infinicore_infer.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

// ─── Common base structs ─────────────────────────────────────────────────────

// All model DeviceResource structs must inherit this.
// Provides common fields accessed by ModelBase::threadLoop().
struct DeviceResourceBase {
    infiniDevice_t       device;
    int                  device_id;
    infiniopHandle_t     handle;
    infinirtStream_t     stream;
    infinicclComm_t      comm;
    std::shared_ptr<MemoryPool> memory_pool;
};

// Per-device state (mutex + condition variables + flags).
// Used by ModelBase for thread startup and infer-loop coordination.
// Qwen3VL extends this with static barrier members in Qwen3vlInferState.
struct ModelInferState {
    std::mutex               mtx;
    std::condition_variable  cv_load, cv_start, cv_done;
    bool loaded    = false;
    bool proceed   = false;
    bool exit_flag = false;
};

// Base inference request used by Jiuge, JiugeAWQ, JiugeGPTQ.
// DeepSeekV3 and Qwen3VL define their own request types (different cache ptr types
// and, for Qwen3VL, additional vision fields).
struct BaseInferRequest {
    const uint32_t *tokens;
    uint32_t        ntok;
    const uint32_t *req_lens;
    uint32_t        nreq;
    const uint32_t *req_pos;
    struct KVCache **kv_caches;
    const float    *temperature;
    const uint32_t *topk;
    const float    *topp;
    uint32_t       *output;
    void           *logits;
};

// ─── ModelBase template ───────────────────────────────────────────────────────

// Template base class for all src/ inference models.
//
// Manages per-device threads, startup synchronization, the infer-dispatch loop,
// and clean shutdown. Subclasses provide model-specific behavior by overriding
// three pure virtual methods.
//
// Template parameters:
//   Meta           — model metadata/config struct (JiugeMeta, DeepSeekV3Meta, …)
//   DeviceResource — must publicly inherit DeviceResourceBase
//   Request        — inference request struct; defaults to BaseInferRequest
//   State          — per-device state; defaults to ModelInferState
//
// Usage pattern in subclass constructor:
//   SubclassModel(...) : ModelBase(meta, device, dev_ids), weights_(weights) {
//       launch();  // ← MUST be the last statement in the constructor
//   }
//
// launch() starts threads, which immediately call createDeviceResource().
// Because createDeviceResource() accesses 'this' (for meta_, weights_, etc.),
// all subclass members must be initialised before launch() is called.
template<
    typename Meta,
    typename DeviceResource,
    typename Request = BaseInferRequest,
    typename State   = ModelInferState
>
class ModelBase {
    static_assert(std::is_base_of<DeviceResourceBase, DeviceResource>::value,
                  "DeviceResource must inherit DeviceResourceBase");
    static_assert(std::is_base_of<ModelInferState, State>::value,
                  "State must inherit ModelInferState");

public:
    ModelBase(const Meta &meta, infiniDevice_t device, std::vector<int> dev_ids)
        : meta_(meta), device_(device), dev_ids_(std::move(dev_ids))
    {
        int ndev = static_cast<int>(dev_ids_.size());
        dev_resources_.resize(ndev);
        states_.resize(ndev);
        threads_.resize(ndev);
    }

    // Joining threads in shutdown() requires the model to still be alive.
    // Subclasses must NOT call shutdown() themselves; the destructor does it.
    virtual ~ModelBase() { shutdown(); }

    ModelBase(const ModelBase &)            = delete;
    ModelBase &operator=(const ModelBase &) = delete;

protected:
    // ── Subclass interface (pure virtual) ────────────────────────────────────

    // Initialise *rsrc for device `dev_id`. Called from the per-device thread.
    // Must set all DeviceResourceBase fields (handle, stream, comm, memory_pool)
    // in addition to model-specific weight pointers.
    // `idev` is the rank index (0…ndev-1); `dev_id` is the physical device ID.
    virtual void createDeviceResource(DeviceResource *rsrc,
                                      int idev, int ndev,
                                      int dev_id, infinicclComm_t comm) = 0;

    // Release all resources in rsrc. Called from the per-device thread after
    // the infer loop exits. Must destroy handle, stream, comm, and tensors.
    virtual void releaseDeviceResource(DeviceResource &rsrc) = 0;

    // Execute one inference batch on device `idev`. Called from the per-device
    // thread while holding the infer lock (proceed == true).
    virtual void inferDeviceBatch(DeviceResource &rsrc,
                                  int idev, int ndev,
                                  const Request &req) = 0;

    // ── Thread management ────────────────────────────────────────────────────

    // Start per-device threads and wait until every thread signals loaded.
    // MUST be the last call in the subclass constructor.
    void launch() {
        int ndev = static_cast<int>(dev_ids_.size());
        RUN_INFINI(infinirtInit());
        std::vector<infinicclComm_t> comms(ndev, nullptr);
        if (ndev > 1) {
            RUN_INFINI(infinicclCommInitAll(device_, comms.data(), ndev, dev_ids_.data()));
        }
        for (int i = 0; i < ndev; i++) {
            threads_[i] = std::thread(&ModelBase::threadLoop, this, i, comms[i]);
        }
        for (int i = 0; i < ndev; i++) {
            std::unique_lock<std::mutex> lock(states_[i].mtx);
            states_[i].cv_load.wait(lock, [&] { return states_[i].loaded; });
        }
    }

    // Copy req into req_, signal all device threads, and wait until all finish.
    void dispatch(const Request &req) {
        req_ = req;
        int ndev = static_cast<int>(dev_ids_.size());
        for (int idev = 0; idev < ndev; idev++) {
            std::unique_lock<std::mutex> lock(states_[idev].mtx);
            states_[idev].proceed = true;
            lock.unlock();
            states_[idev].cv_start.notify_one();
        }
        // Wait in reverse order (matches original inferBatchJiuge pattern)
        for (int i = ndev; i > 0; i--) {
            int idev = i - 1;
            std::unique_lock<std::mutex> lock(states_[idev].mtx);
            states_[idev].cv_done.wait(lock, [&] { return !states_[idev].proceed; });
        }
    }

    // Signal all threads to stop and join them.
    void shutdown() {
        int ndev = static_cast<int>(dev_ids_.size());
        for (int idev = 0; idev < ndev; idev++) {
            if (!threads_[idev].joinable()) continue;
            {
                std::unique_lock<std::mutex> lock(states_[idev].mtx);
                states_[idev].exit_flag = true;
            }
            states_[idev].cv_start.notify_one();
        }
        for (int idev = 0; idev < ndev; idev++) {
            if (threads_[idev].joinable()) threads_[idev].join();
        }
    }

    Meta                        meta_;
    infiniDevice_t              device_;
    std::vector<int>            dev_ids_;
    std::vector<DeviceResource> dev_resources_;
    std::vector<State>          states_;
    std::vector<std::thread>    threads_;
    Request                     req_;

private:
    void threadLoop(int idev, infinicclComm_t comm) {
        int ndev = static_cast<int>(dev_ids_.size());
        createDeviceResource(&dev_resources_[idev], idev, ndev, dev_ids_[idev], comm);

        CacheManager     cache_manager(100);
        InferenceContext ctx(dev_resources_[idev].handle,
                             dev_resources_[idev].memory_pool,
                             &cache_manager,
                             dev_resources_[idev].stream);
        setInferenceContext(&ctx);

        {
            std::unique_lock<std::mutex> lock(states_[idev].mtx);
            states_[idev].loaded = true;
            lock.unlock();
            states_[idev].cv_load.notify_one();
        }

        while (true) {
            std::unique_lock<std::mutex> lock(states_[idev].mtx);
            states_[idev].cv_start.wait(lock,
                [&] { return states_[idev].proceed || states_[idev].exit_flag; });
            if (states_[idev].exit_flag) break;

            inferDeviceBatch(dev_resources_[idev], idev, ndev, req_);

            states_[idev].proceed = false;
            lock.unlock();
            states_[idev].cv_done.notify_one();
        }

        releaseDeviceResource(dev_resources_[idev]);
        setInferenceContext(nullptr);
    }
};
