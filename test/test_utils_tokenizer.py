"""Tests for infinilm.utils.tokenizer helper."""
import pytest
from unittest.mock import MagicMock
from tokenizers import decoders as _dec
from infinilm.utils.tokenizer import fix_llama_tokenizer_decoder


def _make_tokenizer(has_prepend: bool, has_strip: bool):
    norm = MagicMock()
    norm.__repr__ = lambda self: "Prepend(...)" if has_prepend else "NFC(...)"
    dec = MagicMock()
    dec.__repr__ = lambda self: "Strip(...)" if has_strip else "BPE(...)"
    inner = MagicMock()
    inner.normalizer = norm
    inner.decoder = dec
    backend = MagicMock()
    backend._tokenizer = inner
    tok = MagicMock()
    tok.backend_tokenizer = backend
    return tok, inner


def test_no_fix_for_non_llama():
    tok, inner = _make_tokenizer(has_prepend=True, has_strip=True)
    original_dec = inner.decoder
    fix_llama_tokenizer_decoder(tok, model_type="qwen")
    assert inner.decoder is original_dec


def test_no_fix_when_no_prepend():
    tok, inner = _make_tokenizer(has_prepend=False, has_strip=True)
    original_dec = inner.decoder
    fix_llama_tokenizer_decoder(tok, model_type="llama")
    assert inner.decoder is original_dec


def test_no_fix_when_no_strip():
    tok, inner = _make_tokenizer(has_prepend=True, has_strip=False)
    original_dec = inner.decoder
    fix_llama_tokenizer_decoder(tok, model_type="llama")
    assert inner.decoder is original_dec


def test_fix_applied_for_llama_with_prepend_and_strip():
    tok, inner = _make_tokenizer(has_prepend=True, has_strip=True)
    fix_llama_tokenizer_decoder(tok, model_type="llama")
    assert isinstance(inner.decoder, _dec.Sequence)
