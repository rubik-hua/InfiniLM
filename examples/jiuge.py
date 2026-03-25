import infinicore
import transformers
from transformers import AutoTokenizer
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.utils.tokenizer import fix_llama_tokenizer_decoder
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
import argparse
import sys
import time
import os
import numpy as np
from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig
from packaging import version
from infinilm.base_config import BaseConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

_PAGED_KV_BLOCK_SIZE = 256


def test(
    prompts: str | list[str],
    model_path,
    max_new_tokens=100,
    infini_device=infinicore.device("cpu", 0),
    tp=1,
    enable_paged_attn=False,
    enable_graph=False,
    top_k=1,
    top_p=1.0,
    temperature=1.0,
    attn_backend="default",
):
    model_path = os.path.expanduser(model_path)
    # ---------------------------------------------------------------------------- #
    #                        Create Model
    # ---------------------------------------------------------------------------- #
    if enable_paged_attn and attn_backend == "default":
        attn_backend = "paged-attn"

    model = InferEngine(
        model_path,
        device=infini_device,
        distributed_config=DistConfig(tp),
        enable_graph_compiling=enable_graph,
        attention_backend=attn_backend,
        kv_cache_dtype=cfg.kv_cache_dtype,
    )
    # ---------------------------------------------------------------------------- #
    #                        Load Weights
    # ---------------------------------------------------------------------------- #
    load_model_state_dict_by_file(model, model_path, dtype=model.dtype)

    # ---------------------------------------------------------------------------- #
    #                        create tokenizer
    # ---------------------------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    fix_llama_tokenizer_decoder(tokenizer, model.model_type)

    # ---------------------------------------------------------------------------- #
    #                        tokenize
    # ---------------------------------------------------------------------------- #
    # prompt = "山东最高的山是？"
    if isinstance(prompts, str):
        prompts = [prompts]
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        input_contents = [
            tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]
    else:
        input_contents = prompts

    # input_ids_list = tokenizer.batch_encode_plus(input_contents)[
    #     "input_ids"
    # ]  # List: [[1, 1128, 526, 366, 29892]]
    if version.parse(transformers.__version__) < version.parse("5.0.0"):
        # Ideally this is solved by upgrading transformers. However, doing so causes version mismatch between transformers and mlu pytorch on devices with Phytium CPU. So a branch is temporarily used.
        input_ids_list = [
            tokenizer.encode_plus(
                text, truncation=True, max_length=2048, add_special_tokens=True
            )["input_ids"]
            for text in input_contents
        ]
    else:
        input_ids_list = [
            tokenizer._encode_plus(
                text, truncation=True, max_length=2048, add_special_tokens=True
            )["input_ids"]
            for text in input_contents
        ]

    # ---------------------------------------------------------------------------- #
    #                       Create KVCache
    # ---------------------------------------------------------------------------- #
    if enable_paged_attn:
        batch_size = 1 if prompts is str else len(prompts)
        max_total_tokens = max_new_tokens + len(input_ids_list[0])
        cache_config = PagedKVCacheConfig(
            num_blocks=(
                (max_total_tokens + (_PAGED_KV_BLOCK_SIZE - 1)) // _PAGED_KV_BLOCK_SIZE
            )
            * batch_size,
            block_size=_PAGED_KV_BLOCK_SIZE,
        )
    else:
        batch_size = 1 if prompts is str else len(prompts)
        initial_capacity = max_new_tokens + len(input_ids_list[0])
        cache_config = StaticKVCacheConfig(
            max_batch_size=batch_size, max_cache_len=initial_capacity
        )

    model.reset_cache(cache_config)

    # ---------------------------------------------------------------------------- #
    #                        Generate
    # ---------------------------------------------------------------------------- #
    print(input_contents[0], end="", flush=True)
    input_ids_infini = infinicore.from_list(input_ids_list)

    t1 = time.time()
    print("=================== start generate ====================")
    output_ids = model.generate(
        input_ids_infini,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ),
        _measure_and_log_time=True,
    )
    t2 = time.time()

    numpy_output_ids = np.array([output_id.to_numpy()[0] for output_id in output_ids])
    print(tokenizer.decode(numpy_output_ids, skip_special_tokens=True))

    print(
        f"total_time: {round((t2 - t1) * 1000, 2)} ms",
    )


if __name__ == "__main__":
    cfg = BaseConfig()

    device_str = cfg.get_device_str(cfg.device)

    prompts = [cfg.prompt for _ in range(cfg.batch_size)]

    _PAGED_KV_BLOCK_SIZE = cfg.paged_kv_block_size

    model_path = cfg.model

    max_new_tokens = cfg.max_new_tokens

    backend = cfg.backend

    tp = cfg.tp

    enable_paged_attn = cfg.enable_paged_attn

    enable_graph = cfg.enable_graph

    if backend != "cpp":
        raise ValueError(f"Unsupported backend: {backend}.")

    infini_device = infinicore.device(device_str, 0)

    test(
        prompts,
        model_path,
        max_new_tokens,
        infini_device=infini_device,
        tp=tp,
        enable_paged_attn=enable_paged_attn,
        enable_graph=enable_graph,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        attn_backend=cfg.attn,
    )
