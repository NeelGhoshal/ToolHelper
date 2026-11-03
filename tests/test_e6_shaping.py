"""E6 result shaping over real recorded responses. Usage policies are learned from scripted
runs (the scripted answers decide what gets 'used'; every tool response is real)."""

import json

import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.efficiency.arg_guard import DEFAULT_CDM_FIELDS, CdmFields
from toolhelper.efficiency.shaping import EXPAND, ResultShaper, learn_policies, parse
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend

pytestmark = pytest.mark.anyio
TASK = Task(id="t", prompt="test")
ANSWER = "Myocardial infarction, concept_id 4329847 (SNOMED 22298006)."


@pytest.fixture
def heart(store):
    return recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")


@pytest.fixture
def cdm():
    if not DEFAULT_CDM_FIELDS.exists():
        pytest.skip("omop_mcp checkout not present")
    return CdmFields.load()


async def run(hub, strategies, *steps, answer="done"):
    async with hub:
        return await Agent(ScriptedBackend([*steps, final(answer)]), hub, strategies,
                           counter=estimate_tokens).run(TASK)


async def history(hub, n, *steps, answer):
    return [await run(hub, [], *steps, answer=answer) for _ in range(n)]


def test_parse_real_formats(store):
    def text(tool, **kw):
        rec = next(r for r in store if r.tool == tool and all(r.arguments.get(k) == v for k, v in kw.items()))
        return "\n".join(b["text"] for b in rec.result["content"])

    omop = parse("omop__find_omop_concept", text("find_omop_concept", keyword="heart attack",
                                                 omop_table="condition_occurrence"))
    assert omop.format == "omop" and omop.items[0].raw["concept_id"] == 4329847
    drugs = parse("medical__search-drugs", text("search-drugs", query="ibuprofen"))
    assert drugs.format == "list" and [i.rank for i in drugs.items] == [1, 2]
    assert {p.field for p in drugs.items[0].parts} >= {"Generic Name", "Manufacturer", "Last Updated"}
    gene = parse("biomcp__get", text("get", entity="gene", id="BRAF"))
    assert gene.format == "sections" and gene.items[0].parts[0].field.startswith("Summary")


async def test_learned_policy_drops_only_unused_fields(replay_hub, heart):
    traces = await history(replay_hub, 3, tool_call("omop__find_omop_concept", heart), answer=ANSWER)
    policy = learn_policies(traces)["omop__find_omop_concept"]
    assert policy.seen_runs == 3
    assert {"class", "concept", "validity", "domain"} <= set(policy.drop_fields)
    assert not {"concept_id", "name", "code", "vocab"} & set(policy.drop_fields)
    assert policy.drop_fields["class"] == sorted(t.run_id for t in traces)
    assert policy.runs_within(1) == sorted(t.run_id for t in traces)  # only the first candidate was used


async def test_shaping_is_evidenced_and_reversible(replay_hub, heart, cdm):
    policies = learn_policies(await history(replay_hub, 3, tool_call("omop__find_omop_concept", heart), answer=ANSWER))
    shaper = ResultShaper("projection", policies, cdm=cdm)
    trace = await run(replay_hub, [shaper], tool_call("omop__find_omop_concept", heart),
                      tool_call(EXPAND, {"handle": "r1"}), answer=ANSWER)
    shaped, expanded = trace.tool_calls
    assert shaped.shaped_result_tokens < shaped.raw_result_tokens
    data = json.loads(shaped.shaped_text.split("\n\n[shaped:")[0])
    assert [c["concept_id"] for c in data["candidates"]] == [4329847, 4085056]  # Condition domain only
    assert "class" not in data["candidates"][0] and "concept_id" in data["candidates"][0]
    assert f'{EXPAND}(handle="r1")' in shaped.shaped_text
    assert expanded.source == "local" and expanded.shaped_text == shaped.raw_text  # nothing is lost

    [t] = trace.transformations
    kinds = {e.kind for e in t.evidence}
    assert kinds == {"dataset_row", "run"} and trace.provenance_coverage == 1.0
    assert t.before_tokens > t.after_tokens and "handle r1" in t.note


async def test_top_n_and_summary_need_rank_evidence(replay_hub, store, cdm):
    zoloft = recorded(store, "search-drugs", query="ibuprofen")  # two real items
    traces = await history(replay_hub, 3, tool_call("medical__search-drugs", zoloft), answer="The Dye Free product from CVS Pharmacy.")
    policies = learn_policies(traces)
    assert policies["medical__search-drugs"].runs_within(1) == sorted(t.run_id for t in traces)

    top = await run(replay_hub, [ResultShaper("top_n", policies, top_n=1)], tool_call("medical__search-drugs", zoloft))
    shaped = parse("medical__search-drugs", top.tool_calls[0].shaped_text.split("\n\n[shaped:")[0])
    assert len(shaped.items) == 1 and "kept top 1" in top.transformations[0].note

    summary = await run(replay_hub, [ResultShaper("summary", policies, top_n=1)], tool_call("medical__search-drugs", zoloft))
    assert summary.tool_calls[0].shaped_text.count("\n") < top.tool_calls[0].shaped_text.count("\n")

    # Without learned evidence nothing is cut.
    bare = await run(replay_hub, [ResultShaper("top_n", {}, top_n=1)], tool_call("medical__search-drugs", zoloft))
    assert bare.transformations == [] and bare.tool_calls[0].shaped_text == bare.tool_calls[0].raw_text


async def test_unknown_handle_is_an_error(replay_hub):
    trace = await run(replay_hub, [ResultShaper()], tool_call(EXPAND, {"handle": "nope"}))
    assert trace.tool_calls[0].is_error and "no result with handle" in trace.tool_calls[0].shaped_text
