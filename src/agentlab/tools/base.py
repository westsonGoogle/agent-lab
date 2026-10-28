"""工具协议:一个工具 = 名字 + 描述 + JSON Schema + 一个普通函数。

模型看不到你的代码,它只能看到 schema。所以 description 和参数描述
就是"给模型看的文档",写得越清楚,调用越准。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

MAX_RESULT_CHARS = 8000  # 工具结果太长会挤爆上下文,截断是必须的


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]
    # 给 CLI / 文档展示用的参数例子,不发给模型
    example: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        """转成 /chat/completions 的 tools 数组元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}")
        self._tools[tool.name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"没有这个工具: {name}(可用: {', '.join(self._tools)})")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, name: str, raw_arguments: str) -> str:
        """执行工具。永远返回字符串,永远不抛异常——错误也要回填给模型。

        这是 Agent 能自我纠错的关键:工具报错不是程序崩溃,而是又一条
        role=tool 的消息,模型看到后会换个参数重试。
        """
        try:
            tool = self.get(name)
        except KeyError:
            return f"错误: 没有这个工具: {name}(可用: {', '.join(self._tools)})"

        try:
            arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError:
            return f"错误: 参数不是合法 JSON: {raw_arguments[:200]!r}"

        if not isinstance(arguments, dict):
            return f"错误: 参数必须是 JSON 对象,收到 {type(arguments).__name__}"

        properties = tool.parameters.get("properties") or {}
        missing = [key for key in tool.parameters.get("required", []) if key not in arguments]
        if missing:
            return f"错误: 缺少必填参数 {missing}"
        unknown = [key for key in arguments if key not in properties]
        if unknown:
            return f"错误: 不认识的参数 {unknown},可用参数: {list(properties)}"

        try:
            result = tool.func(**arguments)
        except Exception as exc:  # noqa: BLE001 - 工具错误必须变成给模型的文本
            return f"工具执行失败: {type(exc).__name__}: {exc}"

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + f"\n...[已截断,原长 {len(result)} 字符]"
        return result
