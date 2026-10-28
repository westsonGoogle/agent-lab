from __future__ import annotations

from fakes import FakeLLM, text_script, tool_script

from agentlab.llm import ToolCall
from agentlab.loop import Agent
from agentlab.tools import Tool, ToolRegistry


def echo_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="echo",
                description="回显文本",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                func=lambda text: f"echo:{text}",
            )
        ]
    )


def make_agent(scripts: list, *, max_steps: int = 8) -> tuple[Agent, FakeLLM]:
    llm = FakeLLM(scripts)
    return Agent(llm=llm, tools=echo_registry(), max_steps=max_steps), llm


def test_tool_round_trip_builds_correct_messages() -> None:
    agent, llm = make_agent(
        [
            tool_script(ToolCall(id="call_1", name="echo", arguments='{"text": "hi"}')),
            text_script("完成"),
        ]
    )

    events = list(agent.stream("请回显 hi"))
    kinds = [event.type for event in events]

    assert kinds == ["tool_call", "tool_result", "text", "usage", "done"]
    assert events[0].arguments == {"text": "hi"}
    assert events[1].result == "echo:hi"
    assert events[-1].text == "完成"

    assert [message["role"] for message in agent.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert agent.messages[2]["tool_calls"][0]["id"] == "call_1"
    assert agent.messages[3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "echo:hi",
    }
    assert agent.messages[4]["content"] == "完成"

    # 第二次请求必须能看见工具结果,也仍然带着工具 schema
    second_call = llm.calls[1]
    assert second_call["messages"][-1]["role"] == "tool"
    assert second_call["tools"] is not None
    assert second_call["tools"][0]["function"]["name"] == "echo"


def test_multiple_tool_calls_in_one_turn() -> None:
    agent, _ = make_agent(
        [
            tool_script(
                ToolCall(id="call_1", name="echo", arguments='{"text": "a"}'),
                ToolCall(id="call_2", name="echo", arguments='{"text": "b"}'),
            ),
            text_script("都完成了"),
        ]
    )

    events = list(agent.stream("两个都做"))
    results = [event.result for event in events if event.type == "tool_result"]
    assert results == ["echo:a", "echo:b"]
    assert [m["role"] for m in agent.messages].count("tool") == 2


def test_invalid_tool_json_is_reported_back_to_model() -> None:
    agent, llm = make_agent(
        [
            tool_script(ToolCall(id="call_1", name="echo", arguments="not-json")),
            text_script("我改好了"),
        ]
    )

    list(agent.stream("坏参数"))
    tool_message = agent.messages[3]
    assert tool_message["role"] == "tool"
    assert "不是合法 JSON" in tool_message["content"]
    assert "不是合法 JSON" in llm.calls[1]["messages"][-1]["content"]


def test_unknown_tool_is_reported_back() -> None:
    agent, _ = make_agent(
        [
            tool_script(ToolCall(id="call_1", name="missing", arguments="{}")),
            text_script("换个工具"),
        ]
    )
    list(agent.stream("调错工具"))
    assert "没有这个工具" in agent.messages[3]["content"]


def test_max_steps_stops_runaway_loop() -> None:
    scripts = [
        tool_script(ToolCall(id=f"call_{i}", name="echo", arguments='{"text": "x"}'))
        for i in range(3)
    ]
    agent, _ = make_agent(scripts, max_steps=2)

    events = list(agent.stream("停不下来"))
    assert events[-1].type == "error"
    assert "超过最大步数 2" in events[-1].error
    assert not any(event.type == "done" for event in events)


def test_reset_clears_history() -> None:
    agent, _ = make_agent([text_script("hi")])
    agent.run("你好")
    assert agent.turns == 1
    agent.reset()
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    assert agent.turns == 0


def test_run_returns_final_text_only() -> None:
    agent, _ = make_agent(
        [
            tool_script(ToolCall(id="call_1", name="echo", arguments='{"text": "x"}')),
            text_script("最终答案"),
        ]
    )
    assert agent.run("问题") == "最终答案"


def test_event_to_dict_drops_empty_fields() -> None:
    agent, _ = make_agent([text_script("hi")])
    event = next(event for event in agent.stream("你好") if event.type == "text")
    data = event.to_dict()
    assert data["type"] == "text"
    assert data["text"] == "hi"
    assert "error" not in data
    assert "arguments" not in data
