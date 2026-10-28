"""Agent = 带工具和状态的 while 循环。

看 run():把模型返回的 tool_calls 执行掉、把结果塞回 messages、
再调一次模型,直到模型不再要工具为止。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from .llm import LLMClient, LLMResponse, ToolArgumentError, Usage
from .tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """你是一个命令行助手,可以调用工具完成任务。

规则:
1. 需要天气、文件内容、数学计算时,必须调用对应工具,不要凭记忆编造。
2. 一次可以调用多个工具;拿到结果后先用中文给出结论,再说细节。
3. 工具报错时,读懂错误并修正参数重试,不要原样重复。
4. 回答保持简洁,不要输出 JSON,除非用户明确要求。"""


@dataclass(slots=True)
class AgentEvent:
    """Agent 运行时产生的事件,CLI 和 HTTP 层都消费它。"""

    type: str  # text | reasoning | tool_call | tool_result | usage | done | error
    text: str = ""
    name: str = ""
    call_id: str = ""
    arguments: dict[str, Any] | None = None
    result: str = ""
    error: str = ""
    usage: Usage | None = None
    steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = {key: value for key, value in asdict(self).items() if value not in ("", None)}
        if self.usage is not None:
            data["usage"] = asdict(self.usage)
        data["type"] = self.type
        return data


@dataclass(slots=True)
class Agent:
    llm: LLMClient
    tools: ToolRegistry
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_steps: int = 8
    messages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]

    # -- 状态 -------------------------------------------------------------
    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    @property
    def turns(self) -> int:
        return sum(1 for message in self.messages if message["role"] == "user")

    # -- 主循环 -----------------------------------------------------------
    def stream(self, user_input: str) -> Iterator[AgentEvent]:
        """跑一轮对话。yield 事件,顺便把消息历史维护好。"""
        self.messages.append({"role": "user", "content": user_input})

        for step in range(1, self.max_steps + 1):
            response: LLMResponse | None = None

            try:
                for event in self.llm.stream_chat(
                    self.messages, tools=self.tools.schemas() or None
                ):
                    if event.type == "text":
                        yield AgentEvent("text", text=event.text)
                    elif event.type == "reasoning":
                        yield AgentEvent("reasoning", text=event.text)
                    elif event.type == "usage" and event.usage is not None:
                        yield AgentEvent("usage", usage=event.usage, steps=step)
                    elif event.type == "final":
                        response = event.response
            except Exception as exc:  # noqa: BLE001 - 网络/接口错误要展示给用户,而不是崩掉
                yield AgentEvent("error", error=str(exc))
                return

            if response is None:
                yield AgentEvent("error", error="模型没有返回完整响应")
                return

            # 1) 把助手这一轮的消息原样存进历史(包括 tool_calls)
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    call.to_message() for call in response.tool_calls
                ]
            self.messages.append(assistant_message)

            # 2) 没有工具调用 = 任务完成
            if not response.tool_calls:
                yield AgentEvent("done", text=response.content, steps=step)
                return

            # 3) 有工具调用 = 执行 -> 回填 -> 继续循环
            for call in response.tool_calls:
                try:
                    arguments = call.parse_args()
                    preview: dict[str, Any] | None = arguments
                except ToolArgumentError:
                    arguments = {}
                    preview = {"__raw__": call.arguments}

                yield AgentEvent(
                    "tool_call",
                    name=call.name,
                    call_id=call.id,
                    arguments=preview,
                    steps=step,
                )

                result = self.tools.execute(call.name, call.arguments)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                yield AgentEvent(
                    "tool_result",
                    name=call.name,
                    call_id=call.id,
                    result=result,
                    steps=step,
                )

        yield AgentEvent("error", error=f"超过最大步数 {self.max_steps},已停止。")

    def run(self, user_input: str) -> str:
        """非流式封装:只关心最终文本(测试和 HTTP 非流式接口用)。"""
        final_text = ""
        for event in self.stream(user_input):
            if event.type == "done":
                final_text = event.text
        return final_text

    # -- 调试 -------------------------------------------------------------
    def dump_messages(self, indent: int = 2) -> str:
        return json.dumps(self.messages, ensure_ascii=False, indent=indent)
