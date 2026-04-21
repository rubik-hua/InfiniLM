from mooncake.engine import TransferEngine
import logging

import infinicore

# 子 logger 设为 DEBUG 不够：记录会 propagate 到 root，root 默认 WARNING，会滤掉 DEBUG。
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import time
import msgspec

from python.infinilm.llm.kv_connector.mooncake.mooncake_connector import (
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
)
import zmq

from python.infinilm.llm.kv_connector.mooncake.mooncake_connector_utils import (
    get_ip,
    make_zmq_path,
    EngineId,
    make_zmq_socket,
)


class Test:
    def __init__(self):
        self.engine = TransferEngine()
        self.hostname = "127.0.0.1"
        protocol = "tcp"
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        print(f"Hostname: {self.hostname}")
        print(f"Protocol: {protocol}")
        print(f"RPC port: {self.rpc_port}")

        self.block_size = 256

        # 同步测试脚本必须用 zmq.Context；asyncio Context 的 socket.recv() 返回 Future，不能当 bytes 解码。
        self.zmq_ctx = zmq.Context()

        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    def register_kv_caches(self, kv_caches: dict[str, infinicore.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches.")

        kv_data_ptrs = []
        kv_data_lens = []
        seen_base_addresses = []

        split_k_and_v = True
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            logger.debug(
                "registering layer %s with shape %s", layer_name, cache_or_caches.shape
            )

            assert split_k_and_v, "split_k_and_v must be True"
            cache_list = [
                cache_or_caches.narrow(0, 0, 1).squeeze(0),  # k
                cache_or_caches.narrow(0, 1, 1).squeeze(0),  # v
            ]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)

                if True:
                    if cache.dtype == infinicore.bfloat16:
                        dtype_size = 2
                    else:
                        raise ValueError(f"Unsupported dtype: {cache.dtype}")

                    numel = cache.numel()
                    # curr_tensor_size_bytes = cache.nbytes
                    curr_tensor_size_bytes = numel * dtype_size

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert tensor_size_bytes == curr_tensor_size_bytes, (
                    "All kv cache tensors must have the same size"
                )

                # TODO: 再确认, block_size的shape再倒数第二维度.
                kernel_block_size = cache.shape[-2]
                assert self.block_size == kernel_block_size
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(tensor_size_bytes)

        self.kv_caches_base_addr = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_len=%d", self.num_blocks, self.block_len
        )

    def transfer_test(self):
        print("==========================> transfer_test")

        time.sleep(1)

        worker_addr = "tcp://127.0.0.1:36031"  # cmpl- tcp://127.0.0.1: cmpl-
        metadata = MooncakeXferMetadata(
            remote_hostname="127.0.0.1",
            remote_port=self.rpc_port,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={
                "cmpl-d7f8e5c6659b4919938077bc1832e1f0": (
                    "xfer-cmpl-d7f8e5c6659b4919938077bc1832e1f0",
                    [0],
                )
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
        )

        encoded_data = self._encoder.encode(metadata)

        with make_zmq_socket(
            self.zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
        ) as sock:
            # If something goes wrong, let P wait timeout first (in asyncio.wait()).

            print("==========================> send ....... ")
            sock.setsockopt(zmq.RCVTIMEO, 3 * 1000)
            sock.send(encoded_data)
            print("==========================> send  over ")

            while True:
                time.sleep(1)

                print("==========================> recv ....... ")

                ret_msg = sock.recv()
                print("==========================> recv  over ")

                response = self._xfer_resp_decoder.decode(ret_msg)
                print(response)

                if response.status == MooncakeXferResponseStatus.ERROR:
                    logger.error(
                        "Error happens during transferring kvcache for : %s",
                        response.err_msg,
                    )
                    return

                if response.status == MooncakeXferResponseStatus.FINISH:
                    break


if __name__ == "__main__":
    kv_caches = {}
    for layer_idx in range(28):
        layer_kv_cache = infinicore.empty(
            (2, 8, 8, 256, 128),
            dtype=infinicore.bfloat16,
            device=infinicore.device("cuda", 0),
        )

        key_name = f"model.layers.{layer_idx}.self_attn.attn"
        kv_caches[key_name] = layer_kv_cache
    
    # for layer_idx in range(28, 28*2):
    #     layer_kv_cache = infinicore.empty(
    #         (2, 8, 8, 256, 128),
    #         dtype=infinicore.bfloat16,
    #         device=infinicore.device("cuda", 3),
    #     )

    #     key_name = f"model.layers.{layer_idx}.self_attn.attn"
    #     kv_caches[key_name] = layer_kv_cache
        


    print(kv_caches["model.layers.0.self_attn.attn"])

    test = Test()
    test.register_kv_caches(kv_caches)
    test.transfer_test()
    print(kv_caches["model.layers.0.self_attn.attn"])
    # print(kv_caches["model.layers.28.self_attn.attn"])