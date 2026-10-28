"""Step 3:手工走一遍 tool calling 的完整往返 —— 整个阶段最重要的一步。

你会看到 4 条消息怎样形成一个闭环:

    user        "北京天气?"
    assistant   tool_calls=[get_weather]     <- 模型说"我要用工具"
    tool        {"city":"北京",...}          <- 你执行完把结果回填
    assistant   "北京现在 18°C..."           <- 模型基于结果回答

跑:
    uv run python scripts/step3_tools.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab.config import Settings  # noqa: E402
from agentlab.llm import LLMClient  # noqa: E402
from agentlab.tools import default_registry  # noqa: E402


def show(title: str, messages: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(json.dumps(messages, ensure_ascii=False, indent=2))


def main() -> None:
    settings = Settings.from_env()
    client = LLMClient(settings)
    registry = default_registry(settings.allowed_root)

    messages: list[dict] = [
        {"role": "system", "content": "你必须用工具查天气,不要凭记忆回答。"},
        {"role": "user", "content": "北京现在天气怎么样?用一句话回答。"},
    ]

    print("发给模型的工具 schema(模型能看到的全部信息):")
    print(json.dumps(registry.schemas(), ensure_ascii=False, indent=2))

    # --- 第 1 次调用:模型决定要不要用工具 -------------------------------
    first = client.chat(messages, tools=registry.schemas())
    print("\n模型第一轮返回:")
    print("  content   =", repr(first.content))
    print("  tool_calls=", [(c.name, c.arguments) for c in first.tool_calls])
    print("  finish    =", first.finish_reason)

    if not first.tool_calls:
        print("\n模型这次没调工具。把 temperature 调低或把工具描述写得更强制一点再试。")
        return

    # 关键:assistant 这条消息要连同 tool_calls 原样存回历史
    messages.append(
        {
            "role": "assistant",
            "content": first.content or "",
            "tool_calls": [call.to_message() for call in first.tool_calls],
        }
    )

    # --- 执行工具,把结果作为 role=tool 回填 -----------------------------
    for call in first.tool_calls:
        result = registry.execute(call.name, call.arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,  # 必须对上 id
                "content": result,
            }
        )

    show("执行完工具后的完整 messages(注意最后两条)", messages)

    # --- 第 2 次调用:模型看到工具结果,给出最终回答 ---------------------
    second = client.chat(messages, tools=registry.schemas())
    print("\n" + "=" * 70)
    print("最终回答")
    print("=" * 70)
    print(second.content)
    print(f"\n[usage] 输入 {second.usage.prompt_tokens} / 输出 {second.usage.completion_tokens}")
    print("\n这就是 Agent loop 的全部秘密:上面这段逻辑循环起来而已。")
    print("打开 src/agentlab/loop.py 的 stream(),对照着看一遍。")


if __name__ == "__main__":
    main()
