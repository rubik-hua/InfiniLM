"""OpenAI-compatible JSON response builders.

Used by the inference server and any other code that needs to emit
OpenAI chat completion format.
"""

import time


def chunk_json(
    id_,
    content=None,
    role=None,
    finish_reason=None,
    model: str = "unknown",
):
    """Generate JSON chunk for streaming response."""
    delta = {}
    if content:
        delta["content"] = content
    if role:
        delta["role"] = role
    return {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "text": content,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def completion_json(
    id_,
    content,
    role="assistant",
    finish_reason="stop",
    model: str = "unknown",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = None,
):
    """Generate JSON response for non-streaming completion."""
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "id": id_,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": role,
                    "content": content,
                },
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
