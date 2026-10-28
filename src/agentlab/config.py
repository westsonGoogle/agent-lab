"""配置:所有可变参数都从环境变量来,代码里不写死任何密钥。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True, slots=True)
class Settings:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens: int = 1024
    max_steps: int = 8
    timeout: float = 60.0
    allowed_root: Path = field(default_factory=Path.cwd)
    # 有些网关要求额外的请求头(例如 x-opencode-session),用 JSON 传进来
    extra_headers: dict[str, str] = field(default_factory=dict)
    # 推理型模型可设 none/low/medium/high;留空则不发送该字段
    reasoning_effort: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        # 从当前目录向上找 .env,找到就加载(已存在的环境变量优先)
        load_dotenv(find_dotenv(usecwd=True))

        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "缺少 LLM_API_KEY。复制 .env.example 为 .env 并填入密钥,或 export LLM_API_KEY=..."
            )

        raw_headers = os.getenv("LLM_EXTRA_HEADERS", "").strip()
        extra_headers: dict[str, str] = {}
        if raw_headers:
            try:
                parsed = json.loads(raw_headers)
            except json.JSONDecodeError as exc:
                raise RuntimeError("LLM_EXTRA_HEADERS 必须是合法的 JSON 对象") from exc
            if not isinstance(parsed, dict):
                raise RuntimeError("LLM_EXTRA_HEADERS 必须是 JSON 对象")
            extra_headers = {str(key): str(value) for key, value in parsed.items()}

        return cls(
            base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            api_key=api_key,
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            allowed_root=Path(os.getenv("ALLOWED_ROOT", ".")).expanduser().resolve(),
            extra_headers=extra_headers,
            reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "").strip(),
        )
