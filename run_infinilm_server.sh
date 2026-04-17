
export CUDA_VISIBLE_DEVICES=0

clear

xmake build _infinilm && xmake install _infinilm

 python python/infinilm/server/inference_server.py \
--nvidia \
--model_path=/home/ubuntu/models/Qwen/Qwen3-0.6B/ \
--temperature 1.0 \
--top_p 0.8 \
--top_k 1 \
--port 8103 \
--tp 1  \
--block_size 256 \
--max_tokens 128 \
--num_blocks 8 \
--max_batch_size 1 \
--attn paged-attn \
--cache_type paged \



## --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
## --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'