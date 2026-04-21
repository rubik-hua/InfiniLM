"""
KV Connector package for Prefill-Decode disaggregated inference.
"""

from infinilm.llm.engine_config import EngineConfig
from infinilm.llm.kv_connector.base import (
    KVConnectorBase,
    KVConnectorMetadata,
    KVConnectorRole,
    NullKVConnector,
)

__all__ = [
    "KVConnectorBase",
    "KVConnectorMetadata",
    "KVConnectorRole",
    "NullKVConnector",
    "create_kv_connector",
]


def create_kv_connector(config: EngineConfig, role: KVConnectorRole) -> KVConnectorBase:
    """Factory function to create KV connectors.

    Args:
        config: EngineConfig containing `kv_transfer_config`.

    Returns:
        A KVConnectorBase instance.

    Raises:
        ValueError: If connector_type is not recognized.
    """
    assert config.kv_transfer_config is not None

    kv_transfer_config = config.kv_transfer_config

    assert kv_transfer_config.kv_connector is not None
    assert kv_transfer_config.kv_role is not None

    kv_connector = kv_transfer_config.kv_connector
    
    if kv_connector in (None, "", "null"):
        return NullKVConnector()

    if kv_connector == "MooncakeConnector":
        from infinilm.llm.kv_connector.mooncake.mooncake_connector import (
            MooncakeConnector,
        )

        connector = MooncakeConnector(config, role=role)
    else:
        raise ValueError(
            f"Unknown KV connector type: {kv_connector!r}. Supported types: ['null', 'MooncakeConnector']"
        )
    return connector
