"""
Engine configuration — shared by LLMEngine, Worker, ModelRunner.
"""

from dataclasses import dataclass, field
from typing import Any
import os
import json
import uuid

# Valid kv_role strings for PD / Mooncake wiring.
KV_ROLE_CHOICES = frozenset({"kv_producer", "kv_consumer", "kv_both"})


@dataclass(init=False)
class KVTransferConfig:
    """Configuration for distributed KV cache transfer.

    Constructor keyword arguments default to ``None`` / built-in extras; callers may omit
    fields. JSON parsing for CLI should reject unknown keys (see ``parse_kv_transfer_config``).
    """

    kv_connector: str | None
    engine_id: str | None
    kv_role: str | None
    kv_connector_extra_config: dict[str, Any]

    def __init__(
        self,
        *,
        kv_connector: str | None = None,
        engine_id: str | None = None,
        kv_role: str | None = None,
        kv_connector_extra_config: dict[str, Any] | None = None,
    ) -> None:
        self.kv_connector = kv_connector
        self.engine_id = engine_id
        self.kv_role = kv_role
        if kv_connector_extra_config is not None:
            self.kv_connector_extra_config = dict(kv_connector_extra_config)
        else:
            self.kv_connector_extra_config = {"mooncake_protocol": "tcp"}
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.kv_role is not None and self.kv_role not in KV_ROLE_CHOICES:
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role!r}. "
                f"Supported roles are {sorted(KV_ROLE_CHOICES)}"
            )

        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError(
                "Please specify kv_role when kv_connector is set; "
                f"supported roles are {sorted(KV_ROLE_CHOICES)}"
            )

        if self.engine_id is None:
            role_key = self.kv_role if self.kv_role is not None else "unset"
            self.engine_id = f"{role_key}_" + str(uuid.uuid4())


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
