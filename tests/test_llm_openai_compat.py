from conftest import FakeOpenAIClient, load_fixture

from toolhelper.llm.base import ChatOptions, Message, ToolCall
from toolhelper.llm.openai_compat import OpenAICompatBackend, parse_arguments, to_openai_message

TOOL_CALL = load_fixture("llm/openai_compat_tool_call.json")


def test_tool_call_turn_matches_recorded_request(drug_tool):
    client = FakeOpenAIClient(TOOL_CALL["response"])
    backend = OpenAICompatBackend("qwen3:8b", ChatOptions(temperature=0, seed=7, num_ctx=8192, think=False),
                                  client=client)
    result = backend.chat([{"role": "user", "content": TOOL_CALL["request"]["messages"][0]["content"]}],
                          [drug_tool])

    assert client.requests[0] == TOOL_CALL["request"]
    recorded_call = TOOL_CALL["response"]["choices"][0]["message"]["tool_calls"][0]
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
        (recorded_call["id"], "search-drugs", {"query": "ibuprofen", "limit": 2})]
    assert result.usage.measured is True
    assert result.usage.prompt == TOOL_CALL["response"]["usage"]["prompt_tokens"]
    assert result.usage.completion == TOOL_CALL["response"]["usage"]["completion_tokens"]
    assert result.usage.cached_prompt == 0
    # Ollama /v1 returns qwen3's reasoning in a non-standard field; it is kept separately.
    assert result.message.thinking
    assert result.ignored_options == ["num_ctx", "think"]


def test_messages_convert_to_chat_completions_shape():
    assistant = Message(role="assistant", tool_calls=[ToolCall(id="c1", name="t", arguments={"q": 1})])
    assert to_openai_message(assistant)["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "t", "arguments": '{"q": 1}'}}]
    tool = Message(role="tool", tool_call_id="c1", tool_name="t", content="result")
    assert to_openai_message(tool) == {"role": "tool", "tool_call_id": "c1", "content": "result"}


def test_invalid_arguments_are_kept_raw():
    assert parse_arguments('{"q": 1}') == ({"q": 1}, None)
    assert parse_arguments("{not json") == ({}, "{not json")
    assert parse_arguments("[1]") == ({}, "[1]")
