#pragma once

// Lightweight NVTX wrapper for Nsight Systems trace readability.
// If NVTX headers are unavailable, these become no-ops.

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define INFINILM_NVTX_AVAILABLE 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define INFINILM_NVTX_AVAILABLE 1
#else
#define INFINILM_NVTX_AVAILABLE 0
#endif
#else
#define INFINILM_NVTX_AVAILABLE 0
#endif

namespace infinilm::utils {

struct NvtxRange {
    explicit NvtxRange(const char *name) {
#if INFINILM_NVTX_AVAILABLE
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~NvtxRange() {
#if INFINILM_NVTX_AVAILABLE
        nvtxRangePop();
#endif
    }
    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};

} // namespace infinilm::utils

