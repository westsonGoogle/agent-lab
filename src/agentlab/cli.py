"""命令行界面:流式打印 + 工具调用可视化。

运行:
    uv run agent
    uv run agent --once "北京现在多少度?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import Settings
from .llm import LLMClient, Usage
from .loop import Agent, AgentEvent
from .tools import default_registry

USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def _dim(text: str) -> str:
    return _c(text, "2")


def _cyan(text: str) -> str:
    return _c(text, "36")


def _yellow(text: str) -> str:
    return _c(text, "33")


def _green(text: str) -> str:
    return _c(text, "32")


def _red(text: str) -> str:
    return _c(text, "31")


def build_agent(settings: Settings) -> Agent:
    return Agent(
        llm=LLMClient(settings),
        tools=default_registry(settings.allowed_root),
        max_steps=settings.max_steps,
    )


def render(event: AgentEvent) -> None:
    if event.type == "text":
        print(event.text, end="", flush=True)
    elif event.type == "reasoning":
        print(_dim(event.text), end="", flush=True)
    elif event.type == "tool_call":
        args = json.dumps(event.arguments, ensure_ascii=False) if event.arguments else "{}"
        print(f"\n{_cyan('  ↳ 调用工具')} {_yellow(event.name)} {_dim(args)}", flush=True)
    elif event.type == "tool_result":
        preview = event.result.replace("\n", " ")[:300]
        suffix = "…" if len(event.result) > 300 else ""
        print(_dim(f"  ↳ 结果 {preview}{suffix}"), flush=True)
    elif event.type == "error":
        print(_red(f"\n[错误] {event.error}"), flush=True)


def print_usage(usage: Usage | None) -> None:
    if usage is None:
        return
    cached = f",缓存命中 {usage.cached_tokens}" if usage.cached_tokens else ""
    print(
        _dim(
            f"\n[tokens] 输入 {usage.prompt_tokens}{cached} / "
            f"输出 {usage.completion_tokens} / 合计 {usage.total_tokens}"
        )
    )


BANNER = f"""
{_cyan("agent-lab")} · 一个手写的 Agent(没有框架)
输入问题开始;命令:{_yellow("/reset")} 清空上下文  {_yellow("/history")} 看消息
{_yellow("/tools")} 看工具  {_yellow("/quit")} 退出
"""


def chat_once(agent: Agent, question: str) -> int:
    usage: Usage | None = None
    for event in agent.stream(question):
        render(event)
        if event.type == "usage":
            usage = event.usage
    print_usage(usage)
    print()
    return 0


def repl(agent: Agent) -> int:
    print(BANNER)
    while True:
        try:
            line = input(_green("\n你 > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            break
        if line == "/help":
            print(BANNER)
            continue
        if line == "/reset":
            agent.reset()
            print(_dim("上下文已清空。"))
            continue
        if line == "/history":
            print(agent.dump_messages())
            continue
        if line == "/tools":
            for schema in agent.tools.schemas():
                fn = schema["function"]
                print(f"- {_yellow(fn['name'])}: {fn['description']}")
            continue

        usage: Usage | None = None
        try:
            print(_dim("助手 > "), end="", flush=True)
            for event in agent.stream(line):
                render(event)
                if event.type == "usage":
                    usage = event.usage
        except KeyboardInterrupt:
            print(_red("\n[已中断]"))
            continue
        print_usage(usage)
        print()
    print(_dim("再见。"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="手写 Agent 命令行")
    parser.add_argument("--once", metavar="QUESTION", help="只问一次,不进入交互模式")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    agent = build_agent(settings)

    if args.once:
        return chat_once(agent, args.once)

    print(_dim(f"模型 {settings.model} @ {settings.base_url}"))
    print(_dim(f"沙箱 {settings.allowed_root}"))
    return repl(agent)


if __name__ == "__main__":
    raise SystemExit(main())
