from infinilm.llm.kv_connector.base import (
    KVConnectorBase,
    KVConnectorRole,
    KVConnectorMetadata,
)

from typing import TYPE_CHECKING, Any, Optional

import infinicore
import httpx
import msgspec
import numpy as np
from enum import IntEnum
from typing import TYPE_CHECKING, Any
import zmq
import zmq.asyncio
import asyncio
import threading
from dataclasses import dataclass

ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)
EngineId = str

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0


class MooncakeAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    remote_hostname: str
    remote_port: int
    request_ids: list[ReqId]
    kv_caches_base_addr: list[int]
    block_ids: list[list[int]]


@dataclass
class RecvReqMeta:
    local_block_ids: list[int]
    remote_host: str
    remote_port: int


@dataclass
class SendBlockMeta:
    local_block_ids: list[int]
    ready: asyncio.Event
    expire_time: float = float("inf")


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, RecvReqMeta] = {}
        self.reqs_to_send: dict[ReqId, list[int]] = {}

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        if load_remote_cache:
            self.reqs_to_recv[request_id] = RecvReqMeta(
                local_block_ids=local_block_ids,
                remote_host=kv_transfer_params["remote_host"],
                remote_port=kv_transfer_params["remote_port"],
            )
        else:
            self.reqs_to_send[request_id] = local_block_ids


class MooncakeConnector(KVConnectorBase):
    def __init__(self, config: "EngineConfig", role: KVConnectorRole):
        super().__init__(role)
        # TODO: 一些初始化代码
        # 为了能够初始化，应该需要修 infinilm 中的代码调整一下config参数
        engine_id = config.kv_transfer_config.engine_id  # TODO 再确认下engine_id的含义

        logger.info(
            f" MooncakeConnector::__init__   config.kv_transfer_config: {config.kv_transfer_config} MooncakeConnector: role={role}, engine_id={engine_id}"
        )

        if role == KVConnectorRole.SCHEDULER:
            # TODO: MooncakeConnectorScheduler 类未实现
            from .mooncake_connector_scheduler import MooncakeConnectorScheduler

            self.connector_scheduler = MooncakeConnectorScheduler(
                config, engine_id
            )  # TODO: 后续修改为 class MooncakeConnectorScheduler的对象
            self.connector_worker = None

        elif role == KVConnectorRole.WORKER:
            # TODO: MooncakeConnectorWorker 类未实现
            from .mooncake_connector_worker import MooncakeConnectorWorker

            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
                config, engine_id
            )  # TODO: 后续修改为 class MooncakeConnectorScheduler的对象

    ############################################################
    # Scheduler Side Methods
    ############################################################
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        # TODO：
        # request的数据类型为 class Request. # /vllm/v1/request.py
        # 相当于 InfiniLm中是InferRequest类。与服务侧再确认。
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        # TODO：
        # blocks的数据类型为 class KVCacheBlocks. # vllm/v1/core/kv_cache_manager.py 中
        # 与服务侧再确认这个类
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> MooncakeConnectorMetadata:
        # TODO：
        # scheduler_output的数据类型为 class SchedulerOutput. # infinilm/llm/scheduler.py 中
        # scheduler_output虽然需要传递给了 build_connector_meta(), 但是build_connector_meta中没有使用
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # TODO
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, infinicore.Tensor]):
        # TODO:
        # kv_caches的创建在C++里面。通过函数拿到了python端，数据类型是 infinicore.Tensor
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Get the finished recving and sending requests."""

        # TODO: finished_req_ids的参数没有被用到
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        # TODO:
        # forward_context虽然出现在 start_load_kv函数的参数中，但实现中未被使用。
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        # TODO: 应该是无操作
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: infinicore.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        # TODO: 应该是无操作
        pass

    def wait_for_save(self):
        # TODO: 应该是无操作

        pass


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups
