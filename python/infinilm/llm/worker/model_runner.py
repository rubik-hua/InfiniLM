"""
Model Runner - Wraps model execution with KV connector support.

The ModelRunner handles:
1. Loading the model (InferEngine + weights)
2. Creating and managing the KV connector
3. Initializing KV cache
4. Building model inputs from scheduler output
5. Running the model forward pass

Architecture:

    Worker
    └── ModelRunner(KVConnectorModelRunnerMixin)
            ├── load_model(device)
            │       ├── InferEngine(...)           — create model
            │       ├── load_model_state_dict(...) — load weights
            │       └── _init_kv_connector()       — create KV connector
            ├── initialize_cache(cache_config)
            │       └── model_engine.reset_cache(...)
            ├── execute_model(scheduler_output)
            │       ├── build model inputs
            │       ├── [KV connector hooks]
            │       ├── _model_forward(...)
            │       └── return sampled_tokens
            └── _model_forward(**model_input)
                    └── model_engine.forward(...)
"""

import logging
from contextlib import contextmanager
from typing import Any, List, Optional

import infinicore

from infinilm.llm.engine_config import EngineConfig
from infinilm.llm.kv_connector.base import (
    KVConnectorBase,
    KVConnectorMetadata,
    KVConnectorRole,
    NullKVConnector,
)
from infinilm.llm.kv_connector import create_kv_connector

from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV connector mixin (mirrors vLLM's KVConnectorModelRunnerMixin)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from infinilm.llm.kv_connector.mooncake.mooncake_connector import (
    MooncakeConnectorOutput,
)


@dataclass
class ModelRunnerOutput:
    # [num_reqs]
    req_ids: list[str] = field(default_factory=list)

    # req_id -> index
    req_id_to_index: dict[str, int] = field(default_factory=dict)

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.

    #  sampled_token_ids 就是 infinilm中的 sampled_tokens_list
    sampled_token_ids: list[int] = field(default_factory=list)

    kv_connector_output: MooncakeConnectorOutput | None = None


class KVConnectorModelRunnerMixin:
    """Mixin that adds KV connector support to ModelRunner.

    Provides a context manager ``maybe_get_kv_connector_output`` that wraps
    the model forward pass with KV connector hooks.  When the connector is
    a ``NullKVConnector`` (or ``None``), the context manager is a no-op so
    there is zero overhead on the normal (non-PD) code path.

    Expects the host class to have a ``kv_connector`` attribute.
    """

    kv_connector: Optional[KVConnectorBase]

    @contextmanager
    def maybe_get_kv_connector_output(
        self,
        scheduler_output: Any,
        defer_finalize: bool = False,
    ):
        """Context manager for KV connector operations around model forward.

        Usage::

            with self.maybe_get_kv_connector_output(sched_out) as kv_meta:
                output = self._model_forward(...)

        Args:
            scheduler_output: The scheduler output for the current step.
            defer_finalize: If True, skip ``finalize()`` inside this context
                (the caller is responsible for calling it later).

        Yields:
            ``KVConnectorMetadata`` or ``None`` when no active connector.
        """
        # Fast path — no connector or NullKVConnector
        if self.kv_connector is None or isinstance(self.kv_connector, NullKVConnector):
            yield None
            return

        output = MooncakeConnectorOutput()

        # 1. TODO: Mooncake: Metadata
        assert scheduler_output.kv_connector_metadata is not None

        self.kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata
        )

        # 2. Pre-forward: start loading KV caches (receiver / decode side)
        # TODO: Mooncake: 从mooncake拉取数据
        self.kv_connector.start_load_kv(forward_context="forward_context")

        try:
            # forward 操作
            yield output
        finally:
            # 3. Post-forward: wait for all saves (sender / prefill side)
            # TODO: Mooncake: 等待存取数据
            self.kv_connector.wait_for_save()

            # TODO: Mooncake: 调用 get_finished
            output.finished_sending, output.finished_recving = (
                self.kv_connector.get_finished("finished_req_ids")
            )
            output.test_flag = "ok"

            if False:
                # TODO: Mooncake: 调用 get_kv_connector_stats
                # 好像不用掉用
                # kv_connector.get_kv_connector_stats()

                # TODO: Mooncake: 调用 get_kv_connector_kv_cache_events
                # 好像不用掉用
                # kv_connector.get_kv_connector_kv_cache_events()

                # TODO: Mooncake: 调用 build_connector_worker_meta
                # 好像不用掉用
                # kv_connector.build_connector_worker_meta()

                # TODO: Mooncake: 调用 clear_connector_metadata
                # self.kv_connector.clear_connector_metadata()
                pass


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------


class ModelRunner(KVConnectorModelRunnerMixin):
    """Model runner that executes model forward with KV connector support.

    Aligned with vLLM v1, the ModelRunner is responsible for:
    - Loading the model (``load_model``)
    - Creating the KV connector (``_init_kv_connector``)
    - Initializing the KV cache (``initialize_cache``)
    - Executing the forward pass (``execute_model``)

    The ModelRunner is created by ``Worker`` and should not be created
    directly by ``LLMEngine``.

    Args:
        config: The ``EngineConfig`` instance with all configuration.
    """

    def __init__(self, config: EngineConfig):
        self.config = config

        # Populated by load_model()
        self.model_engine: Optional[InferEngine] = None
        self.kv_connector: Optional[KVConnectorBase] = None

        # Sampling defaults
        self.default_temperature = config.temperature
        self.default_top_p = config.top_p
        self.default_top_k = config.top_k

    # ------------------------------------------------------------------
    # Staged initialisation (aligned with vLLM v1)
    # ------------------------------------------------------------------

    def load_model(self, device: Any) -> None:
        """Load the model and initialise KV connector.

        This corresponds to ``GPUModelRunner.load_model()`` in vLLM v1.

        Args:
            device: The infinicore device object to load the model onto.
        """
        # 1. Create InferEngine
        if self.config.cache_type == "static":
            cache_config = StaticKVCacheConfig(
                max_batch_size=1, max_cache_len=self.config.max_cache_len
            )
        elif self.config.cache_type == "paged":
            cache_config = PagedKVCacheConfig(
                num_blocks=self.config.num_blocks, block_size=self.config.block_size
            )
        else:
            raise ValueError(f"Unsupported cache_type: {self.config.cache_type}")

        logger.info(
            f"ModelRunner: KV cache initialized ({type(cache_config).__name__})"
        )

        self.model_engine = InferEngine(
            model_path=self.config.model_path,
            device=device,
            distributed_config=DistConfig(self.config.tensor_parallel_size),
            cache_config=cache_config,
            enable_graph_compiling=self.config.enable_graph,
            attention_backend=self.config.attn_backend,
        )

        # 2. Load model weights
        load_model_state_dict_by_file(
            self.model_engine,
            self.config.model_path,
            dtype=self.model_engine.config.dtype,
        )

        # 3. Initialise KV connector
        self._init_kv_connector()

        logger.info(f"ModelRunner: model loaded from {self.config.model_path}")

    def _init_kv_connector(self) -> None:
        """Create KV connector based on configuration.

        This corresponds to ``GPUModelRunner._maybe_init_kv_connector()``
        in vLLM v1.
        """
        self.kv_connector = None
        if self.config.kv_transfer_config is not None:
            self.kv_connector = create_kv_connector(
                self.config, role=KVConnectorRole.WORKER
            )

        if self.kv_connector is not None:
            # TODO: Mooncake: 注册kvcache
            kv_cache_list = self.model_engine.get_kv_cache()
            assert len(kv_cache_list) == self.config.tensor_parallel_size

            # TODO: 构造输入  # KV cache layer model.layers.0.self_attn.attn has shape torch.Size([2, 3572, 16, 8, 128])
            kv_caches = {}
            for rank_idx, kv_cache_vec in enumerate(kv_cache_list):
                # per layer kv cache
                for layer_idx, layer_kv_cache in enumerate(kv_cache_vec):
                    # print(layer_kv.shape)  # shape：[2, 8, 8, 256, 128]
                    key_name = (
                        f"rank.{rank_idx}.model.layers.{layer_idx}.self_attn.attn"
                    )
                    kv_caches[key_name] = layer_kv_cache

            self.kv_connector.register_kv_caches(kv_caches)

    # ------------------------------------------------------------------
    # Model config access
    # ------------------------------------------------------------------

    @property
    def model_config(self) -> Any:
        """Return the model configuration (from InferEngine)."""
        if self.model_engine is None:
            raise RuntimeError(
                "ModelRunner.load_model() must be called before accessing model_config"
            )
        return self.model_engine.config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_model(
        self,
        scheduler_output: Any,
    ) -> List[int]:
        """Execute model forward pass for the given scheduler output.

        Steps:
        1. Build model inputs from scheduler output.
        2. Wrap the forward pass with KV connector hooks.
        3. Return sampled token IDs.

        Args:
            scheduler_output: Output from the scheduler containing
                scheduled requests and their metadata.

        Returns:
            List of sampled token IDs, one per request in the batch.
        """

        # Execute forward with KV connector hooks
        with self.maybe_get_kv_connector_output(
            scheduler_output,
        ) as kv_connector_output:
            if scheduler_output.num_requests > 0:
                # Build model inputs from scheduler output
                model_input_dict = scheduler_output.build_model_inputs(
                    self.default_temperature,
                    self.default_top_p,
                    self.default_top_k,
                )
                model_input = self._prepare_model_input(model_input_dict)
                sampled_tokens = self._model_forward(**model_input)
            else:
                empty_output = ModelRunnerOutput()
                return empty_output

        if False:
            # Build model inputs from scheduler output
            model_input_dict = scheduler_output.build_model_inputs(
                self.default_temperature,
                self.default_top_p,
                self.default_top_k,
            )
            model_input = self._prepare_model_input(model_input_dict)
            sampled_tokens = self._model_forward(**model_input)

        # Convert to Python list
        sampled_tokens_list = sampled_tokens.to_numpy().tolist()

        assert len(sampled_tokens_list) == scheduler_output.num_requests

        model_runner_output = ModelRunnerOutput()
        model_runner_output.kv_connector_output = kv_connector_output
        for i in range(scheduler_output.num_requests):
            model_runner_output.req_ids.append(
                scheduler_output.scheduled_requests[i].request_id
            )
            model_runner_output.sampled_token_ids.append(sampled_tokens_list[i])

        return model_runner_output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_forward(self, **model_input) -> Any:
        """Run the actual model forward pass.

        Separated from ``execute_model`` so that subclasses can override
        the forward logic (e.g. for pipeline parallelism or custom
        layer-level KV connector hooks).

        Args:
            **model_input: Model input tensors.

        Returns:
            Sampled token tensor from the model engine.
        """
        return self.model_engine.forward(**model_input)

    @staticmethod
    def _prepare_model_input(model_input_dict: dict) -> dict:
        """Convert a raw model input dict to infinicore tensors.

        Args:
            model_input_dict: Raw model input dictionary produced by
                ``SchedulerOutput.build_model_inputs()``.

        Returns:
            Dictionary with values converted to infinicore tensors where
            appropriate.
        """
        model_input = {}
        for key, value in model_input_dict.items():
            if value is None:
                model_input[key] = None
            elif key in ["input_ids", "position_ids", "slot_mapping"]:
                model_input[key] = infinicore.from_list(value, dtype=infinicore.int64)
            elif key in [
                "past_kv_lengths",
                "total_kv_lengths",
                "input_offsets",
                "cu_seqlens",
                "block_tables",
            ]:
                model_input[key] = infinicore.from_list(value, dtype=infinicore.int32)
            else:
                model_input[key] = value
        return model_input
