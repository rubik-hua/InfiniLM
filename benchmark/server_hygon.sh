#!/usr/bin/env bash
# Hygon DCU variant of server.sh.
#
# Differences from the NVIDIA version:
#   - Sources DTK env (HIP runtime + CUDA-compat shim).
#   - Builds with INFINILM_ENABLE_HYGON=1 (no flash-attn checkout needed —
#     flash-attn comes from the system flash_attn_2_cuda*.so via dlsym).
#   - Uses HIP_VISIBLE_DEVICES (DTK reads this; CUDA_VISIBLE_DEVICES is also
#     respected by DTK's nvcc-compat shim, so we set both to be safe).
#   - --device hygon (CLI maps via BaseConfig.get_device_str → "cuda" backend).
#   - --block-size 64 (DTK flash-attn requires block_size divisible by 64;
#     this is the canonical value, tested green at the smallest setting).

set -e

# Default to single-card; override via GPU=0,1,2,3 for tp>1.
GPU="${GPU:-0}"
PORT="${PORT:-8102}"
ROOT="${ROOT:-/root/InfiniLM}"
MODEL="${MODEL:-/root/models/9g_8b_thinking_llama/}"
TP="${TP:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"
NUM_BLOCKS="${NUM_BLOCKS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"

# DTK environment — both are needed: env.sh sets up HIP/hipcc, the cuda/env.sh
# overlay pulls in the CUDA-compat nvcc shim that our build uses.
source /opt/dtk/env.sh
source /opt/dtk/cuda/cuda-12/env.sh

# DTK's caching allocator + flash-attn needs libtorch resolvable from the
# process at runtime.
export LD_LIBRARY_PATH="$(python3 -c 'import torch,os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):${LD_LIBRARY_PATH}"

# Both CUDA_ and HIP_ for DTK.
export HIP_VISIBLE_DEVICES="${GPU}"
export CUDA_VISIBLE_DEVICES="${GPU}"

# Hygon DCU TP>1 deadlock workaround (verified: TP=2 8B 55 tok/s, TP=4 8B 54
# tok/s, TP=4 70B 14.6 tok/s; without these all hang at first decode).
#  - HSA_ENABLE_SDMA=0: SDMA hangs in Segment::Freeze BlitKernel signal-wait
#    when concurrent rank threads do first-iter HIP module load.
#  - HSA_FORCE_FINE_GRAIN_PCIE=1: RCCL warns this is required; without it ring
#    builds but kernels deadlock.
#  - NCCL_P2P_DISABLE=1: required at TP>=4 — RCCL P2P over PCIe deadlocks the
#    ring at first allreduce. Routes through shared-memory staging instead.
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
if [ "${TP}" -gt 1 ]; then
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
fi

# Build for Hygon (idempotent — only re-runs cmake / make if sources changed).
INFINILM_ENABLE_HYGON=1 \
    pip install -e "${ROOT}" --no-build-isolation > /dev/null

cd "${ROOT}"

exec python python/infinilm/server/inference_server.py \
    --device hygon \
    --model "${MODEL}" \
    --temperature 1.0 \
    --top-p 0.8 \
    --top-k 1 \
    --port "${PORT}" \
    --tp "${TP}" \
    --block-size "${BLOCK_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-blocks "${NUM_BLOCKS}" \
    --max-batch-size "${MAX_BATCH_SIZE}" \
    --enable-graph \
    --cache-type paged \
    --attn flash-attn
