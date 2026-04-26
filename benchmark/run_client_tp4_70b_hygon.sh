#!/usr/bin/env bash
# Hygon DCU tp=4 client sweep for the 70B Qwen2.
# Pair with: GPU=0,1,2,3 TP=4 PORT=8103 MAX_BATCH_SIZE=16 NUM_BLOCKS=128 \
#            MAX_NEW_TOKENS=1024 \
#            MODEL=/root/models/FM9G_70B_SFT_MHA_qwen2/ \
#            ./server_hygon.sh
#
# NOTE: as of this commit /root/models/FM9G_70B_SFT_MHA_qwen2 is incomplete on
# disk (single 58 GB safetensors for what should be ~140 GB; tokenizer_config.json
# is empty). Re-download / restore the checkpoint before this script will work.

set -e

unset http_proxy https_proxy all_proxy ALL_PROXY

OUT_DIR="${OUT_DIR:-$(dirname "$0")/results}"
mkdir -p "${OUT_DIR}"

PORT="${PORT:-8201}"
MODEL_ID="${MODEL_ID:-fm9g_70b_qwen2}"
TOKENIZER="${TOKENIZER:-/root/models/FM9G_70B_SFT_MHA_qwen2}"

batch_size_list=(1 4 16)
random_input_len_list=(256)
random_output_len_list=(128)

seed=42
prompts_per_concurrency=10

for batch_size in "${batch_size_list[@]}"; do
    for input_len in "${random_input_len_list[@]}"; do
        for output_len in "${random_output_len_list[@]}"; do
            num_prompts=$(( batch_size * prompts_per_concurrency ))
            if [ "${num_prompts}" -lt 100 ]; then
                num_prompts=100
            fi

            tag="bs=${batch_size}_in=${input_len}_out=${output_len}_n=${num_prompts}_seed=${seed}"
            json_file="${OUT_DIR}/infinilm_HygonDCUx4_model=FM9G_70B_qwen2_${tag}.json"

            echo "==========================================="
            echo "Hygon tp=4  bs=${batch_size}  in=${input_len}  out=${output_len}  n=${num_prompts}"
            echo "==========================================="

            python "$(dirname "$0")/bench_client.py" \
                --tokenizer "${TOKENIZER}" \
                --model "${MODEL_ID}" \
                --port "${PORT}" \
                --seed "${seed}" \
                --num-prompts "${num_prompts}" \
                --max-concurrency "${batch_size}" \
                --random-input-len "${input_len}" \
                --random-output-len "${output_len}" \
                --ignore-eos \
                > "${json_file}" 2>&1 || true

            tail -n 8 "${json_file}"
            echo
        done
    done
done

echo "All Hygon TP=4 70B benchmarks done; JSON results under ${OUT_DIR}"
