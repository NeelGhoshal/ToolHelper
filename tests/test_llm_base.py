from toolhelper.llm.base import ChatOptions, Message, ToolCall, coerce_messages, estimate_tokens


def test_options_merge_overrides_only_set_fields():
    defaults = ChatOptions(temperature=0, seed=7, num_ctx=32768, extra={"a": 1})
    merged = defaults.merged(ChatOptions(seed=11, extra={"b": 2}))
    assert merged.temperature == 0
    assert merged.seed == 11
    assert merged.num_ctx == 32768
    assert merged.extra == {"a": 1, "b": 2}


def test_options_merge_none_copies_defaults():
    defaults = ChatOptions(temperature=0)
    merged = defaults.merged(None)
    assert merged == defaults and merged is not defaults


def test_coerce_messages_accepts_dicts():
    msgs = coerce_messages([
        {"role": "user", "content": "hi"},
        Message(role="assistant", tool_calls=[ToolCall(id="c", name="t")]),
    ])
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].tool_calls[0].arguments == {}


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
