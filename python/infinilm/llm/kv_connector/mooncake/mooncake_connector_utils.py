import zmq
import zmq.asyncio
from typing import TYPE_CHECKING, Any
import ipaddress
import psutil
import os
from dataclasses import dataclass, field
import infinicore
from urllib3.util import parse_url

EngineId = str
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_ip() -> str:
    host_ip = os.getenv("INFINILM_HOST_IP")
    if not host_ip:
        host_ip = "127.0.0.1"
        logger.warning("INFINILM_HOST_IP is not set, using default IP address")

    logger.warning(f"INFINILM_HOST_IP is set to {host_ip}")
    return host_ip


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def make_zmq_path(scheme: str, host: str, port: int | None = None) -> str:
    """Make a ZMQ path from its parts.

    Args:
        scheme: The ZMQ transport scheme (e.g. tcp, ipc, inproc).
        host: The host - can be an IPv4 address, IPv6 address, or hostname.
        port: Optional port number, only used for TCP sockets.

    Returns:
        A properly formatted ZMQ path string.
    """
    if port is None:
        return f"{scheme}://{host}"
    if is_valid_ipv6_address(host):
        return f"{scheme}://[{host}]:{port}"
    return f"{scheme}://{host}:{port}"


def split_zmq_path(path: str) -> tuple[str, str, str]:
    """Split a zmq path into its parts."""

    parsed = parse_url(path)

    if not parsed.scheme:
        raise ValueError(f"Invalid zmq path: {path}")

    scheme = parsed.scheme
    host = parsed.hostname or ""
    port = str(parsed.port or "")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]  # Remove brackets for IPv6 address

    if scheme == "tcp" and not all((host, port)):
        # The host and port fields are required for tcp
        raise ValueError(f"Invalid zmq path: {path}")

    if scheme != "tcp" and port:
        # port only makes sense with tcp
        raise ValueError(f"Invalid zmq path: {path}")

    return scheme, host, port


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783 # noqa: E501
def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
    router_handover: bool = False,
) -> zmq.Socket | zmq.asyncio.Socket:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics."""

    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if socket_type == zmq.ROUTER and router_handover:
        # Let a new connection take over an identity left behind by a dead one.
        socket.setsockopt(zmq.ROUTER_HANDOVER, 1)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

    # Determine if the path is a TCP socket with an IPv6 address.
    # Enable IPv6 on the zmq socket if so.
    scheme, host, _ = split_zmq_path(path)
    if scheme == "tcp" and is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


@dataclass
class TpKVTopology:
    """
    Helper class for tensor parallel and KV topology information for
    mapping between local and remote TP workers.
    """

    tp_rank: int
    remote_tp_size: dict[EngineId, int]
    is_mla: bool
    total_num_kv_heads: int
    engine_id: EngineId
    remote_block_size: dict[EngineId, int]
    tensor_shape = None
    is_mamba: bool = False

    def __post_init__(self):
        logger.debug(" -------------> TpKVTopology()  %s", self.tp_rank)
        logger.debug("Test self.remote_tp_size: %s", self.remote_tp_size)
        logger.debug("Test self.is_mla: %s", self.is_mla)
        logger.debug("Test self.total_num_kv_heads: %s", self.total_num_kv_heads)
        logger.debug("Test self.engine_id: %s", self.engine_id)
        logger.debug("Test self.remote_block_size: %s", self.remote_block_size)
        logger.debug("Test self.tensor_shape: %s", self.tensor_shape)
        logger.debug("Test self.is_mamba: %s", self.is_mamba)

        """
        -------------> TpKVTopology()  0
         Test self.remote_tp_size: {'0e41b091-40f3-49b9-8cd8-17ee5a28cc9b': 1}
         Test self.is_mla: False
         Test self.total_num_kv_heads: 8
         Test self.attn_backends: [<class 'vllm.v1.attention.backends.flash_attn.FlashAttentionBackend'>]
         Test self.engine_id: 0e41b091-40f3-49b9-8cd8-17ee5a28cc9b
         Test self.remote_block_size: {'0e41b091-40f3-49b9-8cd8-17ee5a28cc9b': 16}
         Test self.tensor_shape: None
         Test self.is_mamba: False
         Test kv_cache_shape: (2, 1, 16, 1, 1)
        
        """

        # Figure out whether the first dimension of the cache is K/V
        # or num_blocks. This is used to register the memory regions correctly.
        if not self.is_mamba:
            kv_cache_shape = (2, 1, 256, 1, 1)  # TODO: 再确认
            logger.debug("Test kv_cache_shape: %s", kv_cache_shape)

        # Non-MLA backends caches have 5 dims [2, num_blocks, H,N,D],
        # we just mock num_blocks to 1 for the dimension check below.
        # Hybrid SSM models assume a single blocks_first layout
        self._is_kv_layout_blocks_first = False
        self._cross_layers_blocks = False

    @property
    def is_kv_layout_blocks_first(self) -> bool:
        return self._is_kv_layout_blocks_first

    @property
    def split_k_and_v(self) -> bool:
        # Whether to register regions for K and V separately (when present).
        # return not (
        #     self._cross_layers_blocks or self.is_mla or self.is_kv_layout_blocks_first
        # )
        # TODO: 再确认是否需要修改
        return True

    @property
    def tp_size(self) -> int:
        return self.remote_tp_size[self.engine_id]

    @property
    def block_size(self) -> int:
        return self.remote_block_size[self.engine_id]

    @property
    def cross_layers_blocks(self) -> bool:
        return self._cross_layers_blocks

    def tp_ratio(
        self,
        remote_tp_size: int,
    ) -> int:
        """
        Calculate the tensor parallel ratio between local and remote TP.
        We can think of it as the number of local TP workers-per-remote TP
        workers. Local workers will read from the same remote TP worker in
        groups of size `tp_ratio`.If remote tp_size > local tp_size, the
        ratio is flipped (remote_size/local_size) and the returned value is
        negative.
        """
        if self.tp_size >= remote_tp_size:
            assert self.tp_size % remote_tp_size == 0, (
                f"Local tensor parallel size {self.tp_size} is not divisible "
                f"by remote tensor parallel size {remote_tp_size}."
            )
            return self.tp_size // remote_tp_size

        assert remote_tp_size % self.tp_size == 0, (
            f"Remote tensor parallel size {remote_tp_size} is not divisible "
            f"by local tensor parallel size {self.tp_size}."
        )
        # P TP > D TP case, return the ratio as negative
        return -remote_tp_size // self.tp_size

    def block_size_ratio(
        self,
        remote_block_size: int,
    ) -> int:
        """
        Calculate the block size ratio between local and remote TP.
        """
        assert self.block_size % remote_block_size == 0, (
            f"Local block size {self.block_size} is not divisible "
            f"by remote block size {remote_block_size} or vice versa."
        )
        return self.block_size // remote_block_size

    def tp_ratio_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> int:
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        return self.tp_ratio(remote_tp_size)

    def block_size_ratio_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> int:
        remote_block_size = self.remote_block_size[remote_engine_id]
        return self.block_size_ratio(remote_block_size)

    def is_kv_replicated(self, engine_id: EngineId) -> bool:
        """
        Whether the KV cache is replicated across TP workers due to the
        number of TP workers being greater than the number of KV heads.
        """
        tp_size = self.remote_tp_size[engine_id]
        return tp_size // self.total_num_kv_heads >= 1

    def replicates_kv_cache(self, remote_engine_id: EngineId) -> bool:
        # MLA is always replicated as the hidden dim can't be split.
        return self.is_mla or self.is_kv_replicated(remote_engine_id)

    def get_target_remote_ranks(
        self,
        remote_tp_size: int,
    ) -> list[int]:
        """
        Get the remote TP rank (on P) that the current local TP rank
        (on D) will read from. When remote tp_size > local tp_size, we
        read from multiple remote ranks.
        """
        tp_ratio = self.tp_ratio(remote_tp_size)
        if tp_ratio > 0:
            return [self.tp_rank // tp_ratio]

        # P TP > D TP case, D reads from |tp_ratio| remote workers.
        tp_ratio = -tp_ratio
        return [self.tp_rank * tp_ratio + i for i in range(tp_ratio)]

    def get_target_remote_ranks_from_engine_id(
        self,
        remote_engine_id: EngineId,
    ) -> list[int]:
        remote_tp_size = self.remote_tp_size[remote_engine_id]
        return self.get_target_remote_ranks(remote_tp_size)

    def get_transfer_cache_regions(
        self, cache: infinicore.Tensor, layer_spec: "KVCacheSpec"
    ) -> list[infinicore.Tensor] | infinicore.Tensor:
        """Return the cache tensor(s) to register as NIXL memory regions,
        also accounting for hybrid SSM models specificities.
        """
        # Regular case: backends like FA register K/V in separate regions
        return cache if self.split_k_and_v else [cache]
