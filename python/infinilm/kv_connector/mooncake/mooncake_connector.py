import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Optional

from infinilm.kv_connector import (
    KVConnectorBase,
    KVConnectorMetadata,
    KVConnectorRole,
)
from infinilm.config.kv_transfer import KVTransferConfig
from infinilm.llm import InferenceRequest, RequestStatus, SchedulerOutput

logger = logging.getLogger(__name__)

ReqId = str
TransferId = str
EngineId = str


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_ids: TransferId
    local_block_ids: list[int]
    remote_engine_id: str
    remote_bootstrap_addr: str
    expire_time: float = float("inf")
    pull_tasks_count: int = 0


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[int]]] = {}
        self.reqs_not_processed: set[TransferId] = set()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, str],
        load_remote_cache: bool = True,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_ids=transfer_id,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


class MooncakeConnector(KVConnectorBase):
    def __init__(
        self,
        role: KVConnectorRole,
        kv_transfer_config: KVTransferConfig | None = None,
    ):
        cfg = kv_transfer_config or KVTransferConfig()
        super().__init__(
            role=role,
            kv_transfer_config=cfg,
        )

        self.engine_id: EngineId | None = cfg.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(cfg, engine_id=self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        else:
            self.connector_scheduler = None
            self.connector_worker = None

    def get_num_new_matched_tokens(
        self, request: InferenceRequest, num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self,
        request: InferenceRequest,
        block_ids: list[int],
        num_external_tokens: int,
        block_size: Optional[int] = None,
    ) -> None:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, block_ids, num_external_tokens, block_size
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata | None:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: InferenceRequest,
        block_ids: list[int],
        block_size: Optional[int] = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids, block_size)


class MooncakeConnectorScheduler:
    def __init__(self, kv_transfer_config: KVTransferConfig, engine_id: Optional[str]):
        self.kv_transfer_config = kv_transfer_config
        self.engine_id = engine_id

        self.is_kv_producer = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer = kv_transfer_config.kv_role == "kv_consumer"

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[InferenceRequest, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[InferenceRequest, list[int]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()

    def get_num_new_matched_tokens(
        self, request: InferenceRequest, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        Args:
            request(InferenceRequest): the request object.
            num_computed_tokens(int): the number of locally computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded asynchronously (between scheduler steps).
        """
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - 1 - num_computed_tokens
            if count > 0:
                return count, True

        return 0, False

    def update_state_after_alloc(
        self,
        request: InferenceRequest,
        block_ids: list[int],
        num_external_tokens: int,
        block_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            request: the request object.
            block_ids: the list of block IDs allocated for the request,
                specifically for the tokens that will be loaded from the external KV cache.
            num_external_tokens: the number of tokens that will be loaded
                from the external KV cache.
        """
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
            ):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                # remote_block_ids = block_ids if num_external_tokens > 0 else []
                if num_external_tokens > 0:
                    assert (
                        block_size is not None
                    ), "block_size must be provided when num_external_tokens > 0"
                    prompt_len = request.get_prompt_length()
                    local_computed_tokens = prompt_len - 1 - num_external_tokens
                    assert (
                        local_computed_tokens % block_size == 0
                    ), "local_computed_tokens must be divisible by block_size"
                    start_idx = local_computed_tokens // block_size
                    end_idx = (prompt_len - 2) // block_size + 1
                    remote_block_ids = block_ids[start_idx:end_idx]
                else:
                    remote_block_ids = []
                self._reqs_need_recv[request.request_id] = (request, remote_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                self._reqs_need_send[request.request_id] = (request, [])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata | None:
        meta = MooncakeConnectorMetadata()

        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()
        meta.reqs_not_processed = self._reqs_not_processed
        self._reqs_not_processed = set()

        return meta

    def request_finished(
        self,
        request: InferenceRequest,
        block_ids: list[int],
        block_size: Optional[int] = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.

        Returns:
            A tuple of (delay_free_blocks, extra_info)
            - delay_free_blocks: whether to delay freeing blocks until async transfer is done.
            - extra_info: additional info for the caller, currently unused.
        """
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished: req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )

        if not params or not params.get("transfer_id"):
            return False, None

        # Consumer-side error handling.
        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None

        # Producer-side error and normal handling.
        assert not self.is_kv_consumer
        if request.status != RequestStatus.FINISHED:
            self._reqs_not_processed.add(params["transfer_id"])
            return False, None

        delay_free_blocks = len(block_ids) > 0
        if delay_free_blocks:
            assert (
                block_size is not None
            ), "block_size must be provided when delay_free_blocks is True"
            prompt_len = request.get_prompt_length()
            last_token_idx = prompt_len - 2
            num_blocks_to_send = last_token_idx // block_size + 1
            self._reqs_need_send[request.request_id] = (
                request,
                block_ids[:num_blocks_to_send],
            )
        return delay_free_blocks, None


class MooncakeConnectorWorker:
    pass
