"""直连模型 API:不用任何 SDK,只用 httpx 调 /chat/completions。

这一个文件里能看到 Agent 需要的全部协议细节:
- messages 结构(system / user / assistant / tool)
- tool calling 的请求与响应格式
- SSE streaming 的逐块解析(包括被切碎的工具参数)
- usage / prompt caching 的命中统计
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from .config import Settings

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """模型接口不可用 / 返回非法数据。"""


class ToolArgumentError(ValueError):
    """模型给的工具参数不是合法 JSON 对象。"""


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # 注意:这里是原始字符串,模型可能给出非法 JSON

    def parse_args(self) -> dict[str, Any]:
        try:
            args = json.loads(self.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolArgumentError(f"参数不是合法 JSON: {self.arguments!r}") from exc
        if not isinstance(args, dict):
            raise ToolArgumentError(f"参数必须是 JSON 对象,收到: {type(args).__name__}")
        return args

    def to_message(self) -> dict[str, Any]:
        """回填到 messages 里的形状必须和模型返回的一模一样。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0  # 命中 prompt cache 的输入 token 数
    reasoning_tokens: int = 0

    @classmethod
    def from_api(cls, data: dict[str, Any] | None) -> Usage:
        data = data or {}
        prompt_details = data.get("prompt_tokens_details") or {}
        completion_details = data.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
            # OpenAI 用 prompt_tokens_details.cached_tokens,DeepSeek 用 prompt_cache_hit_tokens
            cached_tokens=int(
                prompt_details.get("cached_tokens") or data.get("prompt_cache_hit_tokens") or 0
            ),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        )

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass(slots=True)
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    reasoning: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StreamEvent:
    """流式过程中的事件,最后一个是 final(带完整响应)。"""

    type: Literal["text", "reasoning", "usage", "final"]
    text: str = ""
    usage: Usage | None = None
    response: LLMResponse | None = None


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        max_retries: int = 3,
    ) -> None:
        self.settings = settings
        self.max_retries = max_retries
        self._client = client or httpx.Client(timeout=settings.timeout)
        self._owns_client = client is None

    # -- 生命周期 ---------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- 请求构造 ---------------------------------------------------------
    @property
    def url(self) -> str:
        return f"{self.settings.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "agent-lab/0.1.0",
        }
        headers.update(self.settings.extra_headers)
        return headers

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        stream: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.settings.reasoning_effort:
            payload["reasoning_effort"] = self.settings.reasoning_effort
        if response_format:
            payload["response_format"] = response_format
        if stream:
            payload["stream"] = True
            # 让服务端在流的最后补一个只含 usage 的 chunk
            payload["stream_options"] = {"include_usage": True}
        return payload

    # -- 非流式 -----------------------------------------------------------
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload = self._payload(messages, tools, stream=False, **kwargs)
        data = self._post(payload)
        return self._parse_response(data)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        delay = 0.8
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(self.url, headers=self._headers(), json=payload)
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    raise LLMError(f"网络错误: {exc}") from exc
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code >= 400:
                if resp.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                    retry_after = resp.headers.get("retry-after")
                    time.sleep(float(retry_after) if retry_after else delay)
                    delay *= 2
                    continue
                raise LLMError(f"模型接口报错 {resp.status_code}: {resp.text[:500]}")

            try:
                return resp.json()
            except ValueError as exc:
                raise LLMError(f"接口返回的不是 JSON: {resp.text[:300]}") from exc

        raise LLMError("重试次数用尽")  # pragma: no cover

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"响应里没有 choices: {data}")

        choice = choices[0]
        message = choice.get("message") or {}
        tool_calls = [
            ToolCall(
                id=item.get("id") or f"call_{index}",
                name=(item.get("function") or {}).get("name") or "",
                arguments=(item.get("function") or {}).get("arguments") or "",
            )
            for index, item in enumerate(message.get("tool_calls") or [])
        ]
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            usage=Usage.from_api(data.get("usage")),
            finish_reason=choice.get("finish_reason") or "",
            reasoning=message.get("reasoning_content") or "",
            raw=data,
        )

    # -- 流式 -------------------------------------------------------------
    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        payload = self._payload(messages, tools, stream=True, **kwargs)
        attempt = 0
        delay = 0.8

        while True:
            started = False
            try:
                with self._client.stream(
                    "POST", self.url, headers=self._headers(), json=payload
                ) as resp:
                    if resp.status_code >= 400:
                        body = resp.read().decode("utf-8", "replace")
                        # 有些兼容服务不认识 stream_options,去掉再试一次
                        if resp.status_code == 400 and "stream_options" in payload:
                            payload.pop("stream_options")
                            continue
                        if resp.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                            attempt += 1
                            time.sleep(delay)
                            delay *= 2
                            continue
                        raise LLMError(f"模型接口报错 {resp.status_code}: {body[:500]}")

                    for event in self._parse_sse(resp):
                        started = True
                        yield event
                    return
            except httpx.HTTPError as exc:
                if started or attempt >= self.max_retries:
                    raise LLMError(f"流式网络错误: {exc}") from exc
                attempt += 1
                time.sleep(delay)
                delay *= 2

    @staticmethod
    def _parse_sse(resp: httpx.Response) -> Iterator[StreamEvent]:
        content: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        usage = Usage()
        finish_reason = ""

        for line in resp.iter_lines():
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage"):
                usage = Usage.from_api(chunk["usage"])
                yield StreamEvent("usage", usage=usage)

            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}

            think = delta.get("reasoning_content") or delta.get("reasoning")
            if think:
                reasoning.append(think)
                yield StreamEvent("reasoning", text=think)

            text = delta.get("content")
            if text:
                content.append(text)
                yield StreamEvent("text", text=text)

            for fragment in delta.get("tool_calls") or []:
                index = int(fragment.get("index") or 0)
                slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if fragment.get("id"):
                    slot["id"] = fragment["id"]
                function = fragment.get("function") or {}
                # name 一般一次性给全;arguments 是一段一段给的,要拼起来
                if function.get("name") and not slot["name"]:
                    slot["name"] = function["name"]
                if function.get("arguments"):
                    slot["arguments"] += function["arguments"]

        tool_calls = [
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=slot["arguments"],
            )
            for index, slot in sorted(calls.items())
        ]
        yield StreamEvent(
            "final",
            response=LLMResponse(
                content="".join(content),
                tool_calls=tool_calls,
                usage=usage,
                finish_reason=finish_reason,
                reasoning="".join(reasoning),
            ),
        )
