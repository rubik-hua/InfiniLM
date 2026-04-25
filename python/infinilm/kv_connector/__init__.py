"""
KV connector package.

This module:
- Exposes core KV connector abstractions (base, role, metadata)
- Provides the KVConnectorFactory
- Registers built-in connectors (e.g. MooncakeConnector)

Note:
Importing this module will trigger connector registration.
"""

from infinilm.kv_connector.base import (
    KVConnectorBase,
    KVConnectorRole,
    KVConnectorMetadata,
    KVConnectorHandshakeMetadata,
    KVConnectorWorkerMetadata,
)
from infinilm.kv_connector.factory import KVConnectorFactory

KVConnectorFactory.register_connector(
    "MooncakeConnector",
    "infinilm.kv_connector.mooncake.mooncake_connector",
    "MooncakeConnector",
)


def create_kv_transfer(
    role: KVConnectorRole,
    kv_transfer_config,
) -> KVConnectorBase:
    assert kv_transfer_config is not None

    connector_name = kv_transfer_config.kv_connector
    assert connector_name in ["MooncakeConnector"]

    return KVConnectorFactory.create_connector(connector_name, role, kv_transfer_config)


__all__ = [
    "KVConnectorBase",
    "KVConnectorRole",
    "KVConnectorMetadata",
    "KVConnectorHandshakeMetadata",
    "KVConnectorWorkerMetadata",
    "create_kv_transfer",
]
