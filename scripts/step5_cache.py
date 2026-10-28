"""Step 5:prompt caching —— 长前缀重复发,第二次开始按缓存价计费。

原理:服务端把输入按前缀切块缓存。只有**从头开始完全一致**的前缀才能命中。
所以稳定内容(system prompt、工具定义、长文档)放前面,变化内容放后面。

跑:
    uv run python scripts/step5_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab.config import Settings  # noqa: E402
from agentlab.llm import LLMClient  # noqa: E402

# 造一段足够长的稳定前缀(各家最小缓存长度不同:OpenAI 1024,DeepSeek 64 起)
PREFIX = (
    "你是 agent-lab 的知识库助手。下面是一段项目背景资料,请记住它,"
    "之后只根据资料回答问题。\n\n" + "agent-lab 是一个不使用任何 Agent 框架的教学项目。"
    "它用 httpx 直连 /chat/completions,手写 SSE 解析,"
    "用 while 循环执行工具调用,并用 FastAPI 暴露 SSE 接口。"
    "工具包括天气、文件读取和 AST 安全计算器。"
    "项目主张:Agent 的本质是带工具和状态的循环。\n\n"
) * 30


def ask(client: LLMClient, question: str, label: str) -> None:
    response = client.chat(
        [
            {"role": "system", "content": PREFIX},
            {"role": "user", "content": question},
        ],
        max_tokens=100,
    )
    usage = response.usage
    print(f"\n[{label}] 回答: {response.content[:80]}")
    print(
        f"  输入 {usage.prompt_tokens} tokens,"
        f"缓存命中 {usage.cached_tokens} tokens,"
        f"命中率 {usage.cache_hit_rate:.1%}"
    )


def main() -> None:
    settings = Settings.from_env()
    client = LLMClient(settings)

    print(f"稳定前缀长度约 {len(PREFIX)} 字符(约 {len(PREFIX)} tokens)")
    print("同一个前缀连发三次,观察 cached_tokens:")

    ask(client, "这个项目用什么库直连模型?", "第 1 次:冷启动")
    ask(client, "这个项目用什么库直连模型?", "第 2 次:应命中")
    ask(client, "Agent 的本质是什么?", "第 3 次:前缀不变,仍应命中")

    print(
        "\n注意:命中缓存的前提是前缀逐字节一致。"
        "常见破坏因素:动态时间戳、随机 ID、工具顺序变化、system 里塞了当前时间。"
    )
    print("Anthropic 需要显式 cache_control 断点;OpenAI/DeepSeek 是自动前缀缓存。")


if __name__ == "__main__":
    main()
