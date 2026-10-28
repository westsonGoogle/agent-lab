"""FastAPI 服务:把同一个 Agent 暴露成 HTTP + SSE。

启动(用 --factory,避免 import 时就去读 .env):
    uv run uvicorn agentlab.server:create_app --factory --reload --port 8000

调用:
    curl -N -X POST localhost:8000/chat \
      -H 'Content-Type: application/json' \
      -d '{"message":"北京天气怎么样?","session_id":"demo"}'
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings
from .llm import LLMClient
from .loop import Agent
from .tools import default_registry


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(default=None, max_length=64)


AgentFactory = Callable[[], Agent]


def build_agent(settings: Settings) -> Agent:
    return Agent(
        llm=LLMClient(settings),
        tools=default_registry(settings.allowed_root),
        max_steps=settings.max_steps,
    )


def create_app(
    agent_factory: AgentFactory | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    factory = agent_factory or (lambda: build_agent(settings))
    sessions: dict[str, Agent] = {}

    app = FastAPI(title="agent-lab", version="0.1.0")
    app.state.settings = settings
    app.state.sessions = sessions

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": settings.model,
            "sessions": len(sessions),
        }

    @app.get("/tools")
    def tools() -> dict[str, Any]:
        return {"tools": default_registry(settings.allowed_root).schemas()}

    @app.post("/chat")
    def chat(req: ChatRequest, stream: bool = True) -> Any:
        if req.session_id:
            session_id = req.session_id
            agent = sessions.get(session_id)
            if agent is None:
                agent = sessions[session_id] = factory()
        else:
            session_id = uuid.uuid4().hex[:8]
            agent = sessions[session_id] = factory()

        if not stream:
            try:
                reply = agent.run(req.message)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            return {"session_id": session_id, "reply": reply, "turns": agent.turns}

        def event_stream() -> Iterator[str]:
            yield _sse({"type": "session", "session_id": session_id})
            for event in agent.stream(req.message):
                yield _sse(event.to_dict())
            yield _sse({"type": "end"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, bool]:
        return {"deleted": sessions.pop(session_id, None) is not None}

    return app


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
