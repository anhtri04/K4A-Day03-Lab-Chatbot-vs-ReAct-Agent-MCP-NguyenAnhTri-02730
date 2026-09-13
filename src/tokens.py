"""Exact token counting via tiktoken, with chars//4 fallback."""

from __future__ import annotations


def count_tokens(text: str, model: str = "") -> int:
    """Return token count for `text` under `model`'s encoding."""
    text = text or ""
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore
    except Exception:
        return len(text) // 4
    try:
        if model:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4
