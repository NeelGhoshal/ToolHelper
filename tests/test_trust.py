"""Trust components on real runs: scripted model, recorded responses, real strategies."""

from datetime import UTC, datetime, timedelta

import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import BaseStrategy, Task
from toolhelper.agent.loop import Agent
from toolhelper.efficiency.arg_guard import ArgumentGuard
from toolhelper.efficiency.cache import ResultCache, ResultCacheStrategy
from toolhelper.efficiency.compression import SchemaCompression
from toolhelper.efficiency.grounding import Grounding
from toolhelper.efficiency.shaping import EXPAND, ResultShaper, learn_policies
from toolhelper.efficiency.trust import load_policy, trust
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend

pytestmark = pytest.mark.anyio
TASK = Task(id="t", prompt="Map 'heart attack' to condition_occurrence.condition_concept_id.")


@pytest.fixture
def heart(store):
    return recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")


async def run(hub, strategies, *steps, answer="done", task=TASK):
    async with hub:
        return await Agent(ScriptedBackend([*steps, final(answer)], strict_tools=False), hub, strategies,
                           counter=estimate_tokens).run(task)


def test_default_policy_matches_the_readme():
    p = load_policy()
    assert p.weights == {"grounding": 0.35, "provenance": 0.25, "fidelity": 0.25, "freshness": 0.15}


async def test_grounding_names_ungrounded_ids(replay_hub, heart):
    trace = await run(replay_hub, [], tool_call("omop__find_omop_concept", heart),
                      answer="Myocardial infarction 4329847, or maybe 12345678.")
    g = trust(trace)["components"]["grounding"]
    assert g["value"] == 0.5
    assert g["reasons"] == ["12345678 is in no tool output or hint", "4329847 from omop__find_omop_concept"]


async def test_combined_score_renormalizes_over_applicable_components(replay_hub, heart):
    trace = await run(replay_hub, [], tool_call("omop__find_omop_concept", heart), answer="Myocardial infarction.")
    t = trust(trace)
    assert t["components"]["grounding"]["value"] is None and t["components"]["freshness"]["value"] is None
    assert t["weights"] == {"provenance": 0.5, "fidelity": 0.5} and t["trust"] == 1.0


async def test_unevidenced_change_lowers_provenance(replay_hub):
    class Sloppy(BaseStrategy):
        name = "sloppy"

        def transform_schemas(self, tools):  # changes tools without reporting it
            return tools[:-1]

    p = trust(await run(replay_hub, [Sloppy()]))["components"]["provenance"]
    assert p["value"] == 0.0 and p["reasons"] == ["sloppy.transform_schemas on tools: no evidence (change not reported)"]


async def test_fidelity_penalties(replay_hub, store, heart, dataset_index):
    shaper = ResultShaper("projection", learn_policies([await run(replay_hub, [], tool_call(
        "omop__find_omop_concept", heart), answer="Myocardial infarction 4329847 (SNOMED 22298006).")] * 3))
    expanded = await run(replay_hub, [shaper], tool_call("omop__find_omop_concept", heart), tool_call(EXPAND, {"handle": "r1"}))
    f = trust(expanded)["components"]["fidelity"]
    assert f["value"] == 0.9 and "re-expanded" in f["reasons"][0]

    drug = recorded(store, "search", entity="drug")
    compressed = await run(replay_hub, [SchemaCompression("aggressive"), ArgumentGuard("reject")],
                           tool_call("biomcp__search", drug))
    f = trust(compressed)["components"]["fidelity"]
    assert f["value"] == 0.9 and f["reasons"] == [
        "biomcp__search argument error (rejected) with a compressed schema (possible cause)"]

    # "HER-2" is itself an HGNC alias (an exact match); "HER 2" only matches after normalization.
    her2 = Task(id="h", prompt="Search BioMCP for the gene HER 2.")
    rewritten = await run(replay_hub, [Grounding(dataset_index, {"rewrite"})],
                          tool_call("biomcp__search", {"entity": "gene", "query": "HER 2", "limit": 3}), task=her2)
    assert rewritten.tool_calls[0].final_arguments["query"] == "ERBB2"
    f = trust(rewritten)["components"]["fidelity"]
    assert f["value"] == 0.85 and f["reasons"][0].startswith("E3 rewrote biomcp__search from a normalized match")


async def test_freshness_of_cache_and_datasets(replay_hub, heart, dataset_index):
    cache, now = ResultCache(), datetime(2026, 11, 2, tzinfo=UTC)
    await run(replay_hub, [ResultCacheStrategy(cache, clock=lambda: now)], tool_call("omop__find_omop_concept", heart))
    later = now + timedelta(hours=2)
    hit = await run(replay_hub, [ResultCacheStrategy(cache, {"omop": 3600 * 3}, clock=lambda: later),
                                 Grounding(dataset_index)], tool_call("omop__find_omop_concept", heart))
    started = datetime.fromisoformat(hit.started_at)
    fresh = trust(hit, lookup=lambda w: ((started - timedelta(days=1)).isoformat(), 180))["components"]["freshness"]
    assert fresh["value"] == 1.0 and any(r.startswith("cached result") for r in fresh["reasons"])
    stale = trust(hit, lookup=lambda w: ((started - timedelta(days=400)).isoformat(), 180))["components"]["freshness"]
    assert stale["value"] == 0.5
    assert [r for r in stale["reasons"] if r.startswith("STALE")] == [
        "STALE datasets/omop_eunomia@v1: fetched 400.0 days before the run, ttl 180 days"]


async def test_upstream_failures_on_compressed_tools_are_not_blamed_on_compression(replay_hub, store):
    missing = recorded(store, "get", entity="gene", id="NOTAGENE123")  # real "not found" error, not an argument error
    trace = await run(replay_hub, [SchemaCompression("aggressive")], tool_call("biomcp__get", missing))
    assert trace.tool_calls[0].is_error
    assert trust(trace)["components"]["fidelity"]["value"] == 1.0
