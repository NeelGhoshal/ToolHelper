from anthropic.types import Message as AnthropicMessage
from conftest import FakeAnthropicClient

from toolhelper.llm.anthropic import AnthropicBackend, to_anthropic_messages
from toolhelper.llm.base import ChatOptions, Message, ToolCall


def sdk_message(content: list[dict]) -> AnthropicMessage:
    # Minimal SDK response object to exercise parsing. Not a recorded response:
    # no Anthropic key is available yet; replace with a recorded fixture when one is.
    return AnthropicMessage.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0},
    })


def test_system_and_grouped_tool_results():
    msgs = [
        Message(role="system", content="be brief"),
        Message(role="user", content="q"),
        Message(role="assistant", content="checking",
                tool_calls=[ToolCall(id="a", name="t1"), ToolCall(id="b", name="t2", arguments={"x": 1})]),
        Message(role="tool", tool_call_id="a", content="r1"),
        Message(role="tool", tool_call_id="b", content="boom", is_error=True),
    ]
    system, wire = to_anthropic_messages(msgs)
    assert system == "be brief"
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert wire[1]["content"] == [
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "a", "name": "t1", "input": {}},
        {"type": "tool_use", "id": "b", "name": "t2", "input": {"x": 1}},
    ]
    # All results for one assistant turn go back in a single user message.
    assert wire[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "a", "content": "r1"},
        {"type": "tool_result", "tool_use_id": "b", "content": "boom", "is_error": True},
    ]


def test_sampling_options_are_reported_ignored_and_not_sent(drug_tool):
    client = FakeAnthropicClient(sdk_message([
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "toolu_1", "name": "search-drugs", "input": {"query": "ibuprofen"}},
    ]))
    backend = AnthropicBackend("claude-opus-5-5", ChatOptions(temperature=0, seed=7, think=False,
                                                              effort="medium"), client=client)
    result = backend.chat([{"role": "user", "content": "find ibuprofen"}], [drug_tool])

    request = client.requests[0]
    assert "temperature" not in request and "seed" not in request
    assert request["output_config"] == {"effort": "medium"}
    assert request["max_tokens"] == 16000
    assert request["tools"][0]["input_schema"] == drug_tool.input_schema
    assert result.ignored_options == ["temperature", "seed", "think"]

    assert [(c.id, c.name) for c in result.tool_calls] == [("toolu_1", "search-drugs")]
    assert result.usage.measured is True
    assert result.usage.prompt == 150 and result.usage.cached_prompt == 50
    # Thinking blocks (with signatures) are replayed verbatim on the next turn.
    _, wire = to_anthropic_messages([Message(role="user", content="q"), result.message])
    assert wire[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig"}
