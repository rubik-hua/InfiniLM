"""Tests for infinilm.utils.openai_format helpers."""
from infinilm.utils.openai_format import chunk_json, completion_json


def test_chunk_json_with_content():
    result = chunk_json("id1", content="hello", model="m")
    assert result["id"] == "id1"
    assert result["object"] == "chat.completion.chunk"
    assert result["choices"][0]["delta"]["content"] == "hello"
    assert result["choices"][0]["text"] == "hello"
    assert result["model"] == "m"


def test_chunk_json_with_role():
    result = chunk_json("id2", role="assistant")
    assert result["choices"][0]["delta"]["role"] == "assistant"
    assert "content" not in result["choices"][0]["delta"]


def test_chunk_json_with_finish_reason():
    result = chunk_json("id3", finish_reason="stop")
    assert result["choices"][0]["finish_reason"] == "stop"


def test_completion_json_structure():
    result = completion_json("id4", content="hi", prompt_tokens=5, completion_tokens=2)
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "hi"
    assert result["usage"]["prompt_tokens"] == 5
    assert result["usage"]["completion_tokens"] == 2
    assert result["usage"]["total_tokens"] == 7
