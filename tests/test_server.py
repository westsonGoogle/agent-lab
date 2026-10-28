from __future__ import annotations

from collections.abc import Iterator

from fakes import FakeLLM, text_script
from fastapi.testclient import TestClient

from agentlab.config import Settings
from agentlab.loop import Agent
from agentlab.server import create_app
from agentlab.tools import ToolRegistry

SETTINGS = Settings(api_key="test", base_url="http://llm.test/v1", model="fake-model")


def make_client(*scripts: Iterator) -> TestClient:
    def factory() -> Agent:
        return Agent(llm=FakeLLM(list(scripts)), tools=ToolRegistry())

    return TestClient(create_app(agent_factory=factory, settings=SETTINGS))


def test_health_and_tools() -> None:
    client = make_client()
    assert client.get("/health").json()["model"] == "fake-model"

    tools = client.get("/tools").json()["tools"]
    names = {item["function"]["name"] for item in tools}
    assert {"calculator", "get_weather", "read_file", "list_dir"} <= names


def test_non_stream_chat_keeps_session() -> None:
    client = make_client(text_script("第一次"), text_script("第二次"))

    first = client.post("/chat?stream=false", json={"message": "你好"}).json()
    assert first["reply"] == "第一次"
    assert first["turns"] == 1
    session_id = first["session_id"]

    second = client.post(
        "/chat?stream=false",
        json={"message": "继续", "session_id": session_id},
    ).json()
    assert second["session_id"] == session_id
    assert second["reply"] == "第二次"
    assert second["turns"] == 2


def test_stream_chat_emits_sse_events() -> None:
    client = make_client(text_script("流式回答"))

    with client.stream("POST", "/chat", json={"message": "你好"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert '"type": "session"' in body
    assert '"type": "text"' in body
    assert '"type": "done"' in body
    assert '"type": "end"' in body
    assert "流式回答" in body


def test_delete_session() -> None:
    client = make_client(text_script("hi"))
    chat = client.post("/chat?stream=false", json={"message": "你好"}).json()
    assert client.delete(f"/sessions/{chat['session_id']}").json() == {"deleted": True}
    assert client.delete(f"/sessions/{chat['session_id']}").json() == {"deleted": False}


def test_empty_message_is_rejected() -> None:
    client = make_client()
    assert client.post("/chat", json={"message": ""}).status_code == 422
