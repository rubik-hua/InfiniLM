"""Normalize OpenAI-style chat messages before HuggingFace chat_template.

Kept separate from ``inference_server`` so this logic can be smoke-tested without
loading InfiniCore / CUDA (see ``__main__`` block).
"""


def normalize_openai_messages_for_hf_template(messages: list) -> list:
    """Strip lm-eval ``type: text`` wrappers; flatten multimodal text parts.

    lm-eval ``local-chat-completions`` with ``tokenized_requests=False`` JSON-encodes
    each turn with an extra top-level ``"type": "text"`` (see ``TemplateAPI.apply_chat_template``
    in lm-eval). HuggingFace ``--model hf`` passes plain ``{role, content}`` dicts into
    ``apply_chat_template``. Stripping unknown keys keeps server templating aligned with
    the HF harness for text-only tasks.
    """
    normalized: list = []
    for msg in messages:
        if not isinstance(msg, dict):
            normalized.append(msg)
            continue

        role = msg.get("role")
        if role is None:
            normalized.append(msg)
            continue

        content = msg.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                elif isinstance(part, str):
                    text_parts.append(part)
            merged = "".join(text_parts) if text_parts else ""
            core = {"role": role, "content": merged}
            if msg.get("name") is not None:
                core["name"] = msg["name"]
            normalized.append(core)
        elif isinstance(content, str):
            core = {"role": role, "content": content}
            if msg.get("name") is not None:
                core["name"] = msg["name"]
            normalized.append(core)
        else:
            normalized.append(msg)

    return normalized


if __name__ == "__main__":
    # Smoke test (no InfiniCore): run as
    #   python3 -m infinilm.server.chat_message_normalize
    lm_eval_style = [
        {"role": "system", "content": "sys", "type": "text"},
        {"role": "user", "content": "hi", "type": "text"},
    ]
    out = normalize_openai_messages_for_hf_template(lm_eval_style)
    assert out == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}], out
    mm = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
            ],
        }
    ]
    assert normalize_openai_messages_for_hf_template(mm) == [
        {"role": "user", "content": "ab"}
    ]
    print("chat_message_normalize: ok")
