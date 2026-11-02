from conftest import FakeOllamaClient, load_fixture

from toolhelper.llm.base import ChatOptions, Message
from toolhelper.llm.ollama import OllamaBackend

TOOL_CALL = load_fixture("llm/ollama_tool_call.json")
FINAL = load_fixture("llm/ollama_final_answer.json")
OPTS = ChatOptions(temperature=0, seed=7, num_ctx=8192, think=False)


def test_tool_call_turn_matches_recorded_request(drug_tool):
    client = FakeOllamaClient(TOOL_CALL["response"])
    backend = OllamaBackend("qwen3:8b", OPTS, client=client)
    result = backend.chat([{"role": "user", "content": TOOL_CALL["request"]["messages"][0]["content"]}],
                          [drug_tool])

    # The request we build is exactly the one that produced the recorded response.
    assert client.requests[0] == TOOL_CALL["request"]
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
        ("call_0", "search-drugs", {"query": "ibuprofen", "limit": 2})]
    assert result.usage.measured is True
    assert result.usage.prompt == TOOL_CALL["response"]["prompt_eval_count"]
    assert result.usage.completion == TOOL_CALL["response"]["eval_count"]
    assert result.applied_options["num_ctx"] == 8192  # logged, per the num_ctx gotcha
    assert result.ignored_options == []
    assert result.raw == TOOL_CALL["response"]


def test_tool_result_turn_round_trips(drug_tool):
    first = OllamaBackend("qwen3:8b", OPTS, client=FakeOllamaClient(TOOL_CALL["response"]))
    user = {"role": "user", "content": TOOL_CALL["request"]["messages"][0]["content"]}
    r1 = first.chat([user], [drug_tool])
    recorded_tool_msg = FINAL["request"]["messages"][2]

    client = FakeOllamaClient(FINAL["response"])
    backend = OllamaBackend("qwen3:8b", OPTS, client=client)
    history = [user, r1.message, Message(role="tool", tool_call_id="call_0", tool_name="search-drugs",
                                         content=recorded_tool_msg["content"])]
    r2 = backend.chat(history, [drug_tool])

    assert client.requests[0] == FINAL["request"]
    assert r2.tool_calls == []
    assert r2.message.content == FINAL["response"]["message"]["content"]
    assert r2.stop_reason == "stop"


def test_unset_options_are_not_sent(drug_tool):
    client = FakeOllamaClient(TOOL_CALL["response"])
    result = OllamaBackend("qwen3:8b", client=client).chat(
        [{"role": "user", "content": "x"}], [drug_tool], ChatOptions(effort="high"))
    assert client.requests[0]["options"] == {}
    assert "think" not in client.requests[0]
    assert result.ignored_options == ["effort"]
