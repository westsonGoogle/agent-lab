"""Step 1:最小的一次模型调用(不用 SDK,不用 LLMClient)。

跑:
    uv run python scripts/step1_hello.py
    uv run python scripts/step1_hello.py --json   # 看完整原始响应
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab.config import Settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="打印原始 JSON")
    args = parser.parse_args()

    settings = Settings.from_env()

    # 1) messages:对话就是一个数组,每条消息只有 role + content
    messages = [
        {"role": "system", "content": "你是一个简洁的助手,回答不超过 50 字。"},
        {"role": "user", "content": "用一句话解释:什么是 AI Agent?"},
    ]

    # 2) 请求体:model + messages 是必需品,其余都是可选项
    payload = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 200,
    }
    if settings.reasoning_effort:
        payload["reasoning_effort"] = settings.reasoning_effort

    print("=" * 60)
    print("请求体:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    # 3) 一次 HTTP POST 就完事,没有任何魔法
    resp = httpx.post(
        f"{settings.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            **settings.extra_headers,
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    print("=" * 60)
    print("助手回答:")
    print(data["choices"][0]["message"]["content"])

    print("=" * 60)
    print("usage(计费和缓存的依据):")
    print(json.dumps(data.get("usage"), ensure_ascii=False, indent=2))

    if args.json:
        print("=" * 60)
        print("完整响应:")
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
