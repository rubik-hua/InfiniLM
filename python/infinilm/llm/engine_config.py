"""
Engine configuration — shared by LLMEngine, Worker, ModelRunner.
"""

from dataclasses import dataclass, field
from typing import Any
import os
import json
import uuid
from typing import Any, Literal, get_args

KVProducer = Literal["kv_producer", "kv_both"]
KVConsumer = Literal["kv_consumer", "kv_both"]
KVRole = Literal[KVProducer, KVConsumer]


@dataclass
class KVTransferConfig:
    """Configuration for distributed KV cache transfer."""

    #  The KV connector to transmit KV caches.
    kv_connector: str | None = None  # kv_connector = "LMCacheConnectorV1"

    # The engine id for KV transfers.
    engine_id: str | None = None  # engine_id: str | None = "d0b90a4d"

    #  Choices  are 'kv_producer', 'kv_consumer', and 'kv_both'.
    kv_role = None  # kv_role = "kv_producer"

    # any extra config that the connector may need.
    kv_connector_extra_config = {"mooncake_protocol": "tcp"}

    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        if self.kv_role is not None and self.kv_role not in get_args(KVRole):
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role}. "
                f"Supported roles are {get_args(KVRole)}"
            )

        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError(
                "Please specify kv_role when kv_connector "
                f"is set, supported roles are {get_args(KVRole)}"
            )


@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""

    # world_size is TPxPP, it affects the number of workers we create.
    tensor_parallel_size: int = 1

    tensor_parallel_rank: int = 0

    world_size: int = 1

    #  Global rank in distributed setup.
    rank: int = 0


@dataclass
class EngineConfig:
    """Configuration for LLM Engine.

    Attributes:
        model_path: Path to the model directory.
        device: Device type string ('cpu', 'cuda', 'mlu', etc.).
        dtype: Data type string ('float16', 'bfloat16', 'float32').
        tensor_parallel_size: Number of devices for tensor parallelism.
        cache_type: Cache type ('paged' or 'static').
        max_batch_size: Maximum batch size for inference (only for paged cache).
        max_tokens: Default maximum tokens to generate.
        num_blocks: Number of KV cache blocks (only for paged cache).
        block_size: Size of each KV cache block (only for paged cache).
        max_cache_len: Maximum sequence length (only for static cache).
        temperature: Default sampling temperature.
        top_p: Default top-p sampling parameter.
        top_k: Default top-k sampling parameter.
        enable_graph: Whether to enable graph compiling.
        attn_backend: Attention backend to use ('default', 'flash-attn').
        kv_connector_type: KV connector type for PD separation ('null', etc.).
        kv_connector_role: KV connector role ('none', 'sender', 'receiver', 'both').
        kv_connector_kwargs: Extra keyword arguments for the KV connector.
    """

    model_path: str
    device: str = "cuda"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    cache_type: str = "paged"  # "paged" or "static"
    max_batch_size: int = 16
    max_tokens: int = 4096
    num_blocks: int = 512
    block_size: int = 256
    max_cache_len: int = 4096
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 1
    enable_graph: bool = False
    attn_backend: str = "default"
    # ---- PD separation ----
    kv_transfer_config: KVTransferConfig = field(default_factory=KVTransferConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)

    # hf_config
    hf_config = None

    def __post_init__(self):
        path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"config.json not found in {self.model_path}")

        with open(path, "r") as f:
            self.hf_config = json.load(f)

    def get_total_num_kv_heads(self) -> int:
        return self.hf_config["num_key_value_heads"]
