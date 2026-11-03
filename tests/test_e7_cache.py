from datetime import UTC, datetime, timedelta

import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.efficiency.cache import ResultCache, ResultCacheStrategy
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend

pytestmark = pytest.mark.anyio
TASK = Task(id="t", prompt="test")


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 2, tzinfo=UTC)

    def __call__(self):
        return self.now


async def run(hub, strategy, *steps):
    async with hub:
        return await Agent(ScriptedBackend([*steps, final("done")]), hub, [strategy],
                           counter=estimate_tokens).run(TASK)


async def test_identical_call_is_served_from_cache_within_ttl(replay_hub, store):
    args = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")
    cache, clock = ResultCache(), Clock()
    first = await run(replay_hub, ResultCacheStrategy(cache, clock=clock), tool_call("omop__find_omop_concept", args))
    assert first.tool_calls[0].decision == "proceed" and first.transformations == []

    clock.now += timedelta(hours=1)
    second = await run(replay_hub, ResultCacheStrategy(cache, clock=clock), tool_call("omop__find_omop_concept", args))
    [tc] = second.tool_calls
    assert tc.cache_hit and tc.decision == "cached" and tc.raw_text == first.tool_calls[0].raw_text
    [t] = second.transformations
    assert t.evidence[0].ref == f"{first.run_id}:{first.tool_calls[0].call_id}"
    assert t.evidence[0].detail["age_seconds"] == 3600 and second.provenance_coverage == 1.0


async def test_expired_entries_and_errors_are_not_served(replay_hub, store):
    ok = recorded(store, "search-drugs", query="ibuprofen")
    err = {"limit": 3}  # recorded real validation error (is_error); Medical MCP's "not found" texts are not errors
    cache, clock = ResultCache(), Clock()
    await run(replay_hub, ResultCacheStrategy(cache, {"medical": 60}, clock=clock),
              tool_call("medical__search-drugs", ok), tool_call("medical__search-drugs", err))
    assert cache.get("medical__search-drugs", err) is None  # errors are never cached

    clock.now += timedelta(seconds=61)
    trace = await run(replay_hub, ResultCacheStrategy(cache, {"medical": 60}, clock=clock),
                      tool_call("medical__search-drugs", ok))
    assert trace.tool_calls[0].decision == "proceed" and trace.tool_calls[0].source == "replay"


def test_ttl_per_tool_overrides_server():
    s = ResultCacheStrategy(ResultCache(), {"medical__search-drugs": 5, "medical": 100})
    assert s.ttl_for("medical__search-drugs") == 5 and s.ttl_for("medical__get-drug-details") == 100
    assert s.ttl_for("local__tool") == 0
