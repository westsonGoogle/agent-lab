from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agentlab.tools import Tool, ToolRegistry
from agentlab.tools.calc import calculate
from agentlab.tools.files import build_file_tools
from agentlab.tools.weather import get_weather

# ---------------------------------------------------------------------------
# 计算器
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1+2*3", 7),
        ("(123*456)/7", 8012.571428571428),
        ("2**10", 1024),
        ("sqrt(16)", 4.0),
        ("round(3.14159, 2)", 3.14),
        ("max(1, 9, 3)", 9),
        ("-5 + 2", -3),
    ],
)
def test_calculate_ok(expression: str, expected: float) -> None:
    assert calculate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "().__class__",
        "open('/etc/passwd')",
        "x + 1",
        "2**100000",
        "1e400",
        "",
    ],
)
def test_calculate_rejects_dangerous_or_invalid(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate(expression)


def test_calculator_tool_through_registry() -> None:
    registry = ToolRegistry(
        [
            Tool(
                name="calculator",
                description="",
                parameters={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                func=calculate,
            )
        ]
    )
    assert registry.execute("calculator", '{"expression": "6*7"}') == "42"


# ---------------------------------------------------------------------------
# 注册表的容错:错误必须变成字符串回填给模型
# ---------------------------------------------------------------------------


def make_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="echo",
                description="",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                func=lambda text: f"echo:{text}",
            )
        ]
    )


def test_registry_unknown_tool() -> None:
    assert make_registry().execute("nope", "{}").startswith("错误: 没有这个工具")


def test_registry_invalid_json() -> None:
    assert "不是合法 JSON" in make_registry().execute("echo", "{oops")


def test_registry_missing_and_unknown_params() -> None:
    registry = make_registry()
    assert "缺少必填参数" in registry.execute("echo", "{}")
    assert "不认识的参数" in registry.execute("echo", '{"text": "hi", "x": 1}')


def test_registry_tool_exception_is_returned_as_text() -> None:
    registry = ToolRegistry(
        [
            Tool(
                name="boom",
                description="",
                parameters={"type": "object", "properties": {}},
                func=lambda: 1 / 0,
            )
        ]
    )
    result = registry.execute("boom", "{}")
    assert result.startswith("工具执行失败: ZeroDivisionError")


def test_registry_truncates_long_result() -> None:
    registry = ToolRegistry(
        [
            Tool(
                name="long",
                description="",
                parameters={"type": "object", "properties": {}},
                func=lambda: "x" * 9000,
            )
        ]
    )
    result = registry.execute("long", "{}")
    assert "已截断" in result
    assert len(result) < 9000


# ---------------------------------------------------------------------------
# 文件沙箱
# ---------------------------------------------------------------------------


@pytest.fixture()
def sandbox(tmp_path: Path) -> tuple[Path, ToolRegistry]:
    root = tmp_path / "root"
    root.mkdir()
    (root / "hello.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "nested.txt").write_text("nested", encoding="utf-8")
    (root / ".env").write_text("SECRET=1", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    return root, ToolRegistry(build_file_tools(root))


def test_read_file(sandbox: tuple[Path, ToolRegistry]) -> None:
    _, registry = sandbox
    data = json.loads(registry.execute("read_file", '{"path": "hello.txt"}'))
    assert data["content"] == "line1\nline2\nline3"
    assert data["total_lines"] == 3
    assert data["truncated"] is False


def test_read_file_slice(sandbox: tuple[Path, ToolRegistry]) -> None:
    _, registry = sandbox
    data = json.loads(
        registry.execute("read_file", '{"path": "hello.txt", "start_line": 2, "max_lines": 1}')
    )
    assert data["content"] == "line2"
    assert data["end_line"] == 2
    assert data["truncated"] is True


def test_read_file_blocks_escape(sandbox: tuple[Path, ToolRegistry]) -> None:
    _, registry = sandbox
    assert "PermissionError" in registry.execute("read_file", '{"path": "../outside.txt"}')
    assert "PermissionError" in registry.execute("read_file", '{"path": "/etc/passwd"}')
    assert "PermissionError" in registry.execute("read_file", '{"path": ".env"}')


def test_read_file_missing(sandbox: tuple[Path, ToolRegistry]) -> None:
    _, registry = sandbox
    assert "FileNotFoundError" in registry.execute("read_file", '{"path": "nope.txt"}')


def test_list_dir(sandbox: tuple[Path, ToolRegistry]) -> None:
    _, registry = sandbox
    data = json.loads(registry.execute("list_dir", '{"path": "."}'))
    names = {entry["name"] for entry in data["entries"]}
    assert {"hello.txt", "sub"} <= names
    assert ".env" not in names  # 敏感文件不列出来


# ---------------------------------------------------------------------------
# 天气(httpx.MockTransport,零网络)
# ---------------------------------------------------------------------------


def test_get_weather_with_mock() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "geocoding" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "name": "北京",
                            "latitude": 39.9,
                            "longitude": 116.4,
                            "admin1": "北京市",
                            "country": "中国",
                        }
                    ]
                },
            )
        assert request.url.params["latitude"] == "39.9"
        return httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-10-28T12:00",
                    "temperature_2m": 18.4,
                    "apparent_temperature": 17.9,
                    "relative_humidity_2m": 45,
                    "wind_speed_10m": 8.2,
                    "weather_code": 2,
                },
                "daily": {
                    "time": ["2026-10-28"],
                    "temperature_2m_max": [21.0],
                    "temperature_2m_min": [11.0],
                    "precipitation_probability_max": [10],
                    "weather_code": [2],
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    data = get_weather("北京", client=client)
    assert data["city"] == "北京"
    assert data["current"]["weather"] == "多云"
    assert data["current"]["temperature"] == 18.4
    assert data["daily"][0]["temp_max"] == 21.0
    assert data["source"] == "open-meteo.com"


def test_get_weather_unknown_city() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    with pytest.raises(ValueError, match="找不到城市"):
        get_weather("不存在的城市", client=httpx.Client(transport=httpx.MockTransport(handler)))
