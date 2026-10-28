"""Step 2:手写 SSE 解析,看清 streaming 到底是怎么回事。

服务端返回的不是一个 JSON,而是一串这样的文本:

    data: {"choices":[{"delta":{"content":"你"}}]}
    data: {"choices":[{"delta":{"content":"好"}}]}
    data: [DONE]

SDK 只是帮你把 data: 后面的 JSON 拼起来而已。

跑:
    uv run python scripts/step2_stream.py
    uv run python scripts/step2_stream.py --raw   # 同时打印原始行
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab.config import Settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", action="store_true", help="打印收到的原始 SSE 行")
    args = parser.parse_args()

    settings = Settings.from_env()
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "user", "content": "用三句话介绍杭州,每句话单独一行。"},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 300,
    }
    if settings.reasoning_effort:
        payload["reasoning_effort"] = settings.reasoning_effort

    print("助手 > ", end="", flush=True)
    started = time.perf_counter()
    first_token_at: float | None = None
    raw_lines = 0

    with httpx.stream(
        "POST",
        f"{settings.base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            **settings.extra_headers,
        },
        json=payload,
        timeout=120,
    ) as resp:
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line.strip():
                continue
            if args.raw and raw_lines < 8:
                print(f"\n[RAW] {line}", flush=True)
                raw_lines += 1

            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break

            chunk = json.loads(data)

            # usage 通常在最后一个 chunk(需要 stream_options.include_usage)
            if chunk.get("usage"):
                print(f"\n\n[usage] {json.dumps(chunk['usage'], ensure_ascii=False)}")
                continue

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            if text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                print(text, end="", flush=True)

    total = time.perf_counter() - started
    ttft = (first_token_at - started) if first_token_at else 0
    print(f"\n\n[性能] 首 token {ttft:.2f}s / 总耗时 {total:.2f}s")


if __name__ == "__main__":
    main()
