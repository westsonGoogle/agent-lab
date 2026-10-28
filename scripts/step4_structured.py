"""Step 4:structured output —— 让模型输出可被程序直接消费的 JSON。

两种做法:
A. 原生结构化输出:response_format={"type":"json_schema"} (推荐)
B. 兼容兜底:在 prompt 里给 schema,再手动 json.loads

跑:
    uv run python scripts/step4_structured.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab.config import Settings  # noqa: E402
from agentlab.llm import LLMClient, LLMError  # noqa: E402

SENTENCE = "我叫林舟,29 岁,在杭州做后端,主要写 Python 和 Go,最近在学 Agent。"

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "姓名"},
        "age": {"type": "integer", "description": "年龄"},
        "city": {"type": "string", "description": "城市"},
        "skills": {"type": "array", "items": {"type": "string"}, "description": "技能列表"},
    },
    "required": ["name", "age", "city", "skills"],
    "additionalProperties": False,
}


def main() -> None:
    settings = Settings.from_env()
    client = LLMClient(settings)
    messages = [
        {"role": "system", "content": "你是信息抽取器,只输出 JSON,不要解释。"},
        {"role": "user", "content": f"从这句话里抽取信息:{SENTENCE}"},
    ]

    print("=" * 70)
    print("方式 A:json_schema(服务端保证结构)")
    print("=" * 70)

    data: dict | None = None
    try:
        response = client.chat(
            messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "profile", "strict": True, "schema": SCHEMA},
            },
            temperature=0,
        )
        print(response.content)
        data = json.loads(response.content)
        print("\n解析成功:", data)
    except LLMError as exc:
        # 不少兼容端点不支持 json_schema,会返回 400
        print(f"该端点不支持 json_schema({str(exc)[:120]}...),走方案 B。")

    if data is None:
        print("\n" + "=" * 70)
        print("方式 B:prompt 约束 + 手动解析(兜底)")
        print("=" * 70)
        response = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是信息抽取器。只输出 JSON,必须符合下面的 schema,"
                        "不要 markdown 代码块:\n" + json.dumps(SCHEMA, ensure_ascii=False)
                    ),
                },
                {"role": "user", "content": f"从这句话里抽取信息:{SENTENCE}"},
            ],
            temperature=0,
        )
        print(response.content)
        text = (
            response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        )
        data = json.loads(text)
        print("\n解析成功:", data)

    print("\n结论:结构化输出让模型变成函数返回值,后面做 workflow 才不会痛。")


if __name__ == "__main__":
    main()
