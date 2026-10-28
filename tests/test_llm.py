from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from agentlab.config import Settings
from agentlab.llm import LLMClient, LLMError

SETTINGS = Settings(api_key="test-key", base_url="http://llm.test/v1", model="fake-model")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "回显",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
    }
]


def make_client(handler: Any, **kwargs: Any) -> tuple[LLMClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    transport = httpx.MockTransport(wrapped)
    client = LLMClient(SETTINGS, client=httpx.Client(transport=transport), **kwargs)
    return client, requests


def test_chat_sends_expected_payload_and_parses_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "我来查一下",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"text": "hi"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 7,
                    "total_tokens": 37,
                    "prompt_tokens_details": {"cached_tokens": 24},
                },
            },
        )

    client, requests = make_client(handler)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    response = client.chat(messages, tools=TOOLS)

    payload = json.loads(requests[0].content)
    assert payload["model"] == "fake-model"
    assert payload["messages"] == messages
    assert payload["tools"] == TOOLS
    assert payload["tool_choice"] == "auto"
    assert "stream" not in payload
    assert requests[0].headers["Authorization"] == "Bearer test-key"

    assert response.content == "我来查一下"
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].parse_args() == {"text": "hi"}
    assert response.usage.cached_tokens == 24
    assert response.usage.cache_hit_rate == pytest.approx(0.8)


def test_extra_headers_and_reasoning_effort_are_sent() -> None:
    settings = Settings(
        api_key="test-key",
        base_url="http://llm.test/v1",
        model="fake-model",
        extra_headers={"x-opencode-session": "abc123"},
        reasoning_effort="none",
    )
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    client = LLMClient(settings, client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.chat([{"role": "user", "content": "hi"}])

    assert captured[0].headers["x-opencode-session"] == "abc123"
    assert json.loads(captured[0].content)["reasoning_effort"] == "none"


def test_stream_accumulates_fragmented_tool_arguments_and_usage() -> None:
    sse = "\n".join(
        [
            'data: {"choices":[{"delta":{"role":"assistant","content":"你"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"好"}}]}',
            "",
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a",'
            '"type":"function","function":{"name":"get_weather","arguments":"{\\"city\\":"}}]}}]}',
            "",
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"\\"北京\\"}"}}]},"finish_reason":"tool_calls"}]}',
            "",
            'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":9,'
            '"total_tokens":29,"prompt_tokens_details":{"cached_tokens":16}}}',
            "",
            "data: [DONE]",
            "",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )

    client, requests = make_client(handler)
    events = list(client.stream_chat([{"role": "user", "content": "hi"}], tools=TOOLS))

    texts = [event.text for event in events if event.type == "text"]
    assert "".join(texts) == "你好"

    usage_event = next(event for event in events if event.type == "usage")
    assert usage_event.usage is not None
    assert usage_event.usage.cached_tokens == 16

    final = events[-1]
    assert final.type == "final"
    assert final.response is not None
    assert final.response.content == "你好"
    assert final.response.finish_reason == "tool_calls"
    assert final.response.tool_calls[0].id == "call_a"
    assert final.response.tool_calls[0].name == "get_weather"
    assert final.response.tool_calls[0].parse_args() == {"city": "北京"}

    payload = json.loads(requests[0].content)
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("agentlab.llm.time.sleep", lambda seconds: sleeps.append(seconds))

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    client, requests = make_client(handler)
    response = client.chat([{"role": "user", "content": "hi"}])

    assert response.content == "ok"
    assert len(requests) == 2
    assert sleeps == [0.0]


def test_400_raises_without_retry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad tool schema"}})

    client, requests = make_client(handler)
    with pytest.raises(LLMError, match="400"):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(requests) == 1


def test_stream_drops_stream_options_when_server_rejects_it() -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if "stream_options" in payload:
            return httpx.Response(400, json={"error": "unknown field stream_options"})
        return httpx.Response(200, content=sse.encode())

    client, requests = make_client(handler)
    events = list(client.stream_chat([{"role": "user", "content": "hi"}]))

    assert len(requests) == 2
    assert "stream_options" in json.loads(requests[0].content)
    assert "stream_options" not in json.loads(requests[1].content)
    assert events[-1].response is not None
    assert events[-1].response.content == "hi"
