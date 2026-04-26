#!/usr/bin/env bash
# Hygon DCU client sweep for the 8B llama on a single DCU.
# Pair with: GPU=0 ./server_hygon.sh

set -e

unset http_proxy https_proxy all_proxy ALL_PROXY

OUT_DIR="${OUT_DIR:-$(dirname "$0")/results}"
mkdir -p "${OUT_DIR}"

PORT="${PORT:-8200}"
MODEL_ID="${MODEL_ID:-9g_8b_thinking}"
TOKENIZER="${TOKENIZER:-/root/models/9g_8b_thinking_llama}"

batch_size_list=(1 4 16 64 128)
random_input_len_list=(256)
random_output_len_list=(256)

seed=42
prompts_per_concurrency=20

for batch_size in "${batch_size_list[@]}"; do
    for input_len in "${random_input_len_list[@]}"; do
        for output_len in "${random_output_len_list[@]}"; do
            num_prompts=$(( batch_size * prompts_per_concurrency ))
            if [ "${num_prompts}" -lt 200 ]; then
                num_prompts=200
            fi

            tag="bs=${batch_size}_in=${input_len}_out=${output_len}_n=${num_prompts}_seed=${seed}"
            json_file="${OUT_DIR}/infinilm_HygonDCU_model=9g_8b_thinking_llama_${tag}.json"

            echo "==========================================="
            echo "Hygon  bs=${batch_size}  in=${input_len}  out=${output_len}  n=${num_prompts}"
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

echo "All Hygon 8B benchmarks done; JSON results under ${OUT_DIR}"
