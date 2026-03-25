"""Tokenizer utilities shared across examples and LLM engine."""

from tokenizers import decoders as _dec


def fix_llama_tokenizer_decoder(tokenizer, model_type: str) -> None:
    """Fix the BPE decoder for llama-family models that have both
    Prepend normalizer and Strip decoder, which causes spurious leading
    spaces in decoded output.

    Safe to call unconditionally: checks model_type and tokenizer state
    before making any changes.

    Args:
        tokenizer: HuggingFace AutoTokenizer instance.
        model_type: Model type string from model config (e.g. "llama").
    """
    if "llama" not in model_type.lower():
        return
    backend = getattr(tokenizer, "backend_tokenizer", None)
    target = getattr(backend, "_tokenizer", backend)
    norm = getattr(target, "normalizer", None)
    dec = getattr(target, "decoder", None)
    sn = repr(norm)[:800] if norm is not None else ""
    sd = repr(dec)[:800] if dec is not None else ""
    has_prepend = "Prepend" in sn
    has_strip = "Strip" in sd
    if has_prepend and has_strip:
        target.decoder = _dec.Sequence(
            [
                _dec.Replace("▁", " "),
                _dec.ByteFallback(),
                _dec.Fuse(),
            ]
        )
