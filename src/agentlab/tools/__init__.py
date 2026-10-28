"""工具注册表 + 默认工具集。"""

from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolRegistry
from .calc import calculate
from .files import build_file_tools
from .weather import get_weather

__all__ = ["Tool", "ToolRegistry", "default_registry"]


def default_registry(allowed_root: Path) -> ToolRegistry:
    tools = [
        Tool(
            name="calculator",
            description="精确计算数学表达式。所有算术都必须用它,不要心算。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式,例如 (123*456)/7、sqrt(2)、2**10",
                    }
                },
                "required": ["expression"],
            },
            func=calculate,
            example={"expression": "(123 * 456) / 7"},
        ),
        Tool(
            name="get_weather",
            description="查询城市当前天气和未来几天预报。中文城市名即可。",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名,例如 北京、Shanghai"},
                    "forecast_days": {
                        "type": "integer",
                        "description": "预报天数 1-7,默认 3",
                        "minimum": 1,
                        "maximum": 7,
                    },
                },
                "required": ["city"],
            },
            func=get_weather,
            example={"city": "北京", "forecast_days": 3},
        ),
        *build_file_tools(allowed_root),
    ]
    return ToolRegistry(tools)
