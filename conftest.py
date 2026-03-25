"""Root conftest.py – stubs out optional C-extension modules so that
pure-Python unit tests can be collected without a full InfiniCore build."""
import sys
from unittest.mock import MagicMock

# Stub heavy C-extension / hardware-specific modules that are not available
# in a pure-Python test environment.
_STUB_MODULES = [
    "infinicore",
    "_infinilm",
]

for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
