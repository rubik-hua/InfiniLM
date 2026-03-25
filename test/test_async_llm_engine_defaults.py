"""Ensure AsyncLLMEngine.__init__ default max_tokens matches LLM default."""
import inspect
import sys
from unittest.mock import MagicMock

# Mock unavailable C++ extension and heavy dependencies before importing
sys.modules.setdefault("_infinilm", MagicMock())
sys.modules.setdefault("infinicore", MagicMock())
sys.modules.setdefault("infinilm.cache", MagicMock())
sys.modules.setdefault("infinilm.cache.cache", MagicMock())
sys.modules.setdefault("infinilm.models", MagicMock())
sys.modules.setdefault("infinilm.models.llama", MagicMock())
sys.modules.setdefault("infinilm.models.llama.configuration_llama", MagicMock())
sys.modules.setdefault("infinilm.auto_config", MagicMock())
sys.modules.setdefault("infinilm.infer_engine", MagicMock())
sys.modules.setdefault("infinilm.modeling_utils", MagicMock())

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "infinilm_llm_llm",
    str(
        importlib.util.find_spec("infinilm").submodule_search_locations[0]
        + "/llm/llm.py"
    ),
)


def _load_llm_module():
    import pathlib, importlib.util as ilu

    path = pathlib.Path(
        ilu.find_spec("infinilm").submodule_search_locations[0]
    ) / "llm" / "llm.py"
    spec = ilu.spec_from_file_location("_infinilm_llm_llm_isolated", path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_llm_module()
AsyncLLMEngine = _mod.AsyncLLMEngine
LLM = _mod.LLM


def test_async_llm_engine_default_max_tokens_matches_llm():
    llm_sig = inspect.signature(LLM.__init__)
    llm_default = llm_sig.parameters["max_tokens"].default

    async_sig = inspect.signature(AsyncLLMEngine.__init__)
    async_default = async_sig.parameters["max_tokens"].default

    assert async_default == llm_default, (
        f"AsyncLLMEngine.__init__ max_tokens default is {async_default}, "
        f"but LLM.__init__ uses {llm_default}; keep them consistent."
    )
    assert async_default == 4096, (
        f"AsyncLLMEngine.__init__ max_tokens default is {async_default}, expected 4096. "
        "LLM.__init__ uses 4096; keep them consistent."
    )
