"""测试替身:用脚本假装模型,永远不联网。"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

from agentlab.llm import LLMResponse, StreamEvent, ToolCall, Usage


def usage() -> Usage:
    return Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


def text_script(text: str, *, cached_tokens: int = 0) -> Iterator[StreamEvent]:
    used = Usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cached_tokens=cached_tokens,
    )
    yield StreamEvent("text", text=text)
    yield StreamEvent("usage", usage=used)
    yield StreamEvent("final", response=LLMResponse(content=text, usage=used, finish_reason="stop"))


def tool_script(*calls: ToolCall, content: str = "") -> Iterator[StreamEvent]:
    response = LLMResponse(
        content=content,
        tool_calls=list(calls),
        usage=usage(),
        finish_reason="tool_calls",
    )
    yield StreamEvent("final", response=response)


class FakeLLM:
    """按顺序播放预设脚本,并记录每次收到的 messages / tools。"""

    def __init__(self, scripts: list[Iterator[StreamEvent]]) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, **kwargs})
        if not self.scripts:
            raise AssertionError("FakeLLM 没有更多脚本了")
        yield from self.scripts.pop(0)
