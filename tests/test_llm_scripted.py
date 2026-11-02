import pytest

from toolhelper.llm.base import ChatOptions, Message
from toolhelper.llm.scripted import ScriptedBackend, ScriptExhausted, UnknownToolError


def test_replays_steps_in_order_and_records_calls(drug_tool):
    backend = ScriptedBackend([
        {"role": "assistant", "tool_calls": [{"name": "search-drugs", "arguments": {"query": "ibuprofen"}}]},
        lambda msgs, tools: Message(role="assistant", content=f"saw {len(msgs)} messages"),
    ], defaults=ChatOptions(temperature=0))

    r1 = backend.chat([{"role": "user", "content": "find ibuprofen"}], [drug_tool])
    assert r1.tool_calls[0].name == "search-drugs"
    assert r1.tool_calls[0].id == "call_0"
    assert r1.stop_reason == "tool_calls"
    assert r1.usage.measured is False and r1.usage.prompt > 0

    history = [{"role": "user", "content": "find ibuprofen"}, r1.message,
               {"role": "tool", "tool_call_id": "call_0", "tool_name": "search-drugs", "content": "x"}]
    r2 = backend.chat(history, [drug_tool])
    assert r2.message.content == "saw 3 messages"
    assert r2.tool_calls == []
    assert len(backend.calls) == 2 and backend.remaining == 0
    assert backend.calls[0]["options"].temperature == 0
    assert r1.latency_ms >= 0


def test_exhausted_script_raises():
    backend = ScriptedBackend([{"role": "assistant", "content": "done"}])
    backend.chat([{"role": "user", "content": "hi"}])
    with pytest.raises(ScriptExhausted):
        backend.chat([{"role": "user", "content": "again"}])


def test_strict_tools_rejects_tools_not_offered(drug_tool):
    backend = ScriptedBackend([{"role": "assistant", "tool_calls": [{"name": "nope"}]}])
    with pytest.raises(UnknownToolError):
        backend.chat([{"role": "user", "content": "hi"}], [drug_tool])


def test_step_must_be_assistant():
    backend = ScriptedBackend([{"role": "user", "content": "wrong"}])
    with pytest.raises(ValueError):
        backend.chat([{"role": "user", "content": "hi"}])
