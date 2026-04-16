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
except (ImportError, AttributeError, TypeError) as _e:
    # Recognised failure modes:
    #   ImportError     — C-extensions (infinicore / _infinilm) absent, e.g.
    #                     pure-Python test envs that haven't built pybind11.
    #   AttributeError  — _infinilm loaded but missing expected exports (ABI
    #                     skew / stale .so); lets us degrade to subpackages
    #                     instead of failing the whole import.
    #   TypeError       — only the metaclass conflict triggered when tests
    #                     replace _infinilm with a MagicMock stub. Any other
    #                     TypeError is a real bug and must not be swallowed.
    if isinstance(_e, TypeError) and "metaclass conflict" not in str(_e):
        raise
    import logging, os
    logging.getLogger(__name__).warning(
        "infinilm top-level exports unavailable (%s: %s); subpackages such as "
        "infinilm.utils remain importable. Set INFINILM_DEBUG=1 for a traceback.",
        type(_e).__name__,
        _e,
    )
    if os.environ.get("INFINILM_DEBUG"):
        raise
