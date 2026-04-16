import logging as _logging
import os as _os

try:
    from .models import AutoLlamaModel
    from . import distributed
    from . import cache
    from . import llm
    from . import base_config

    from .llm import (
        LLM,
        AsyncLLMEngine,
        SamplingParams,
        RequestOutput,
        TokenOutput,
    )

    __all__ = [
        "AutoLlamaModel",
        "distributed",
        "cache",
        "llm",
        "base_config",
        # LLM classes
        "LLM",
        "AsyncLLMEngine",
        "SamplingParams",
        "RequestOutput",
        "TokenOutput",
    ]
except (ImportError, TypeError) as _e:
    # ImportError: C-extensions (infinicore / _infinilm) absent, e.g. in
    # pure-Python test environments that haven't built the pybind11 module.
    # TypeError: when _infinilm is replaced by a MagicMock stub, classes that
    # inherit from both PretrainedConfig and _infinilm.LlamaConfig raise a
    # metaclass conflict. Always log the failure so production setups get a
    # diagnostic rather than a silent empty module.
    _logging.getLogger(__name__).warning(
        "infinilm top-level imports failed (%s: %s); only subpackages like "
        "infinilm.utils are available. Set INFINILM_DEBUG=1 for a traceback.",
        type(_e).__name__,
        _e,
    )
    if _os.environ.get("INFINILM_DEBUG"):
        raise
