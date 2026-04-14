"""
Auto-patch HF dynamic modules for vLLM workers.

This file is imported automatically by Python if its directory is on PYTHONPATH.
We use it to monkeypatch `transformers.dynamic_module_utils` so that when
Transformers copies remote-code modules into `HF_MODULES_CACHE`, we can apply a
small compatibility patch for MiniCPM5 MoE running under vLLM's
TransformersMoEForCausalLM backend.

Usage (must be in the environment inherited by vLLM EngineCore workers):

  export PYTHONPATH="$REPO/InfiniLM/examples/vllm_patches:${PYTHONPATH:-}"
"""

from __future__ import annotations

import importlib
import importlib.abc
import os
import sys
from pathlib import Path


def _patch_minicpm5_moe_modeling(path: str | os.PathLike) -> None:
    p = Path(path)
    if p.name != "modeling_minicpm.py":
        return
    if "minicpm5_dot_16a3_dot_v0314" not in str(p):
        return

    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return

    sentinel = "# vllm_autopatch_minicpm5_moe"
    if sentinel in text:
        return

    needle = "final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)"
    if needle not in text:
        return

    insert = (
        f"{sentinel}\n"
        "        # vLLM may replace `self.experts` with a fused MoE module (e.g. TransformersFusedMoE).\n"
        "        # In that case, call the fused implementation directly.\n"
        "        if not isinstance(self.experts, nn.ModuleList):\n"
        "            return self.experts(hidden_states, topk_indices, topk_weights).type(hidden_states.dtype)\n"
        "\n"
        "        "
    )

    patched = text.replace(needle, insert + needle, 1)
    if patched == text:
        return

    # Best-effort backup (once).
    bak = p.with_suffix(p.suffix + ".bak_autopatch")
    try:
        if not bak.exists():
            bak.write_text(text, encoding="utf-8")
    except Exception:
        pass

    try:
        p.write_text(patched, encoding="utf-8")
    except Exception:
        return


def _install_transformers_copy_hook() -> None:
    try:
        import transformers.dynamic_module_utils as dmu
    except Exception:
        return

    # Only patch once.
    if getattr(dmu, "_vllm_autopatch_installed", False):
        return

    orig_copy = dmu.shutil.copy

    def copy(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = orig_copy(src, dst, *args, **kwargs)
        try:
            _patch_minicpm5_moe_modeling(dst)
        finally:
            return out

    dmu.shutil.copy = copy  # type: ignore[assignment]
    dmu._vllm_autopatch_installed = True  # type: ignore[attr-defined]


_install_transformers_copy_hook()


# --- Import-time patch (more reliable than file rewrite) ---

_TARGET_SUBSTR = "transformers_modules.minicpm5_dot_16a3_dot_v0314.modeling_minicpm"


def _patch_minicpm5_moe_in_memory(module) -> None:  # type: ignore[no-untyped-def]
    try:
        import torch.nn as nn  # noqa: F401
    except Exception:
        return

    cls = getattr(module, "MiniCPM5MoEMoE", None)
    if cls is None:
        return
    if getattr(cls, "_vllm_autopatch_applied", False):
        return

    orig = getattr(cls, "moe", None)
    if orig is None:
        return

    import torch.nn as nn

    def moe(self, hidden_states, topk_indices, topk_weights):  # type: ignore[no-untyped-def]
        # If vLLM swapped experts with a fused module, call it directly.
        if not isinstance(self.experts, nn.ModuleList):
            return self.experts(hidden_states, topk_indices, topk_weights).type(hidden_states.dtype)
        return orig(self, hidden_states, topk_indices, topk_weights)

    setattr(cls, "moe", moe)
    setattr(cls, "_vllm_autopatch_applied", True)


class _PatchedLoader(importlib.abc.Loader):
    def __init__(self, base_loader):
        self._base = base_loader

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        if hasattr(self._base, "create_module"):
            return self._base.create_module(spec)
        return None

    def exec_module(self, module):  # type: ignore[no-untyped-def]
        self._base.exec_module(module)
        try:
            _patch_minicpm5_moe_in_memory(module)
        except Exception:
            pass


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
        if _TARGET_SUBSTR not in fullname:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _PatchedLoader(spec.loader)
        return spec


if not any(isinstance(f, _PatchFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _PatchFinder())


def _install_get_class_in_module_hook() -> None:
    """Patch at import time (spec_from_file_location path), which bypasses meta_path."""
    try:
        import transformers.dynamic_module_utils as dmu
    except Exception:
        return

    if getattr(dmu, "_vllm_autopatch_get_class_hook", False):
        return

    orig = dmu.get_class_in_module

    def get_class_in_module(class_name, module_path, *, force_reload=False):  # type: ignore[no-untyped-def]
        cls = orig(class_name, module_path, force_reload=force_reload)
        try:
            name = os.path.normpath(module_path).removesuffix(".py").replace(os.path.sep, ".")
            mod = sys.modules.get(name)
            if mod is not None and _TARGET_SUBSTR in name:
                _patch_minicpm5_moe_in_memory(mod)
        except Exception:
            pass
        return cls

    dmu.get_class_in_module = get_class_in_module  # type: ignore[assignment]
    dmu._vllm_autopatch_get_class_hook = True  # type: ignore[attr-defined]


_install_get_class_in_module_hook()


