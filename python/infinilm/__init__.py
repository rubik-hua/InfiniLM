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
except Exception:  # noqa: BLE001
    # C-extension modules (infinicore / _infinilm) may be absent in
    # pure-Python test environments.  A bare ImportError is insufficient:
    # mocking _infinilm via MagicMock causes a TypeError (metaclass conflict)
    # in configuration_llama.py's class LlamaConfig(PretrainedConfig, _infinilm.LlamaConfig).
    # Subpackages such as infinilm.utils remain usable in that case.
    pass
