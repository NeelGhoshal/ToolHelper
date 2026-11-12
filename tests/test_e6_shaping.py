"""E6 result shaping over real recorded responses. Usage policies are learned from scripted
runs (the scripted answers decide what gets 'used'; every tool response is real)."""

import json

import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.efficiency.arg_guard import DEFAULT_CDM_FIELDS, CdmFields
from toolhelper.efficiency.shaping import EXPAND, ResultShaper, cut_to_budget, learn_policies, parse
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
    assert shaped.shaped_result_tokens <= 0.9 * shaped.raw_result_tokens  # handle note included

    # A cut that does not pay for its handle note is not made.
    strict = await run(replay_hub, [ResultShaper("projection", policies, cdm=cdm, min_reduction=0.95)],
                       tool_call("omop__find_omop_concept", heart), answer=ANSWER)
    assert strict.transformations == [] and strict.tool_calls[0].shaped_text == strict.tool_calls[0].raw_text
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


# --- budget mode: oversized results cut to whole leading units, the rest behind a handle -------

BUDGET = 600  # estimate_tokens (4 characters per token) in these tests


def recorded_text(store, tool, **kw):
    rec = next(r for r in store if r.tool == tool and all(r.arguments.get(k) == v for k, v in kw.items()))
    return "\n".join(b["text"] for b in rec.result["content"])


def test_budget_cut_keeps_whole_units_of_each_format(store):
    rxnorm = recorded_text(store, "search-drug-nomenclature", query="metformin")  # 134 products
    cut = cut_to_budget("medical__search-drug-nomenclature", rxnorm, BUDGET, estimate_tokens)
    kept, full = parse("medical__search-drug-nomenclature", cut.text), parse("medical__search-drug-nomenclature", rxnorm)
    assert estimate_tokens(cut.text) <= BUDGET and 1 <= len(kept.items) < len(full.items) == 134
    assert [i.parts for i in kept.items] == [i.parts for i in full.items[:len(kept.items)]]  # whole items only
    assert cut.note == f"kept items 1-{len(kept.items)} of 134"

    clinvar = recorded_text(store, "get", id="chr7:g.140453136A>T", sections=["clinvar"])  # one 49k-character section
    cut = cut_to_budget("biomcp__get", clinvar, BUDGET, estimate_tokens)
    assert estimate_tokens(cut.text) <= BUDGET and clinvar.startswith(cut.text.rstrip("\n"))
    assert "of section ClinVar" in cut.note

    articles = recorded_text(store, "variant_articles")  # JSON
    cut = cut_to_budget("biomcp__variant_articles", articles, 1000, estimate_tokens)
    data = json.loads(cut.text)  # still JSON: long lists keep their first items
    assert estimate_tokens(cut.text) <= 1000 and "more items]" in cut.text and data["items"][0]["request_id"] == "item-1"
    tight = cut_to_budget("biomcp__variant_articles", articles, BUDGET, estimate_tokens)  # too many keys to fit
    assert estimate_tokens(tight.text) <= BUDGET and tight.text.startswith('{\n  "items": [')
    assert tight.note.startswith("kept the first items of long lists") and "then the first" in tight.note

    assert cut_to_budget("biomcp__get", clinvar, 10**6, estimate_tokens) is None  # under budget: untouched


async def test_budget_evidence_is_runs_that_kept_what_they_used(replay_hub, store):
    rxnorm = recorded(store, "search-drug-nomenclature", query="metformin")
    first = await history(replay_hub, 2, tool_call("medical__search-drug-nomenclature", rxnorm),
                          answer="Kombiglyze, RxCUI 1043567.")  # the first product
    last = await history(replay_hub, 1, tool_call("medical__search-drug-nomenclature", rxnorm),
                         answer="metformin / pioglitazone, RxCUI 861822.")  # the 134th
    policy = learn_policies(first + last, budgets=[BUDGET], counter=estimate_tokens)["medical__search-drug-nomenclature"]
    assert policy.budget_runs[BUDGET] == sorted(t.run_id for t in first)
    assert policy.over_budget_runs[BUDGET] == sorted(t.run_id for t in first + last)


async def test_budget_shaping_is_evidenced_and_reversible(replay_hub, store):
    rxnorm = recorded(store, "search-drug-nomenclature", query="metformin")
    traces = await history(replay_hub, 2, tool_call("medical__search-drug-nomenclature", rxnorm),
                           answer="Kombiglyze, RxCUI 1043567.")
    policies = learn_policies(traces, budgets=[BUDGET], counter=estimate_tokens)
    trace = await run(replay_hub, [ResultShaper("budget", policies, budget=BUDGET)],
                      tool_call("medical__search-drug-nomenclature", rxnorm), tool_call(EXPAND, {"handle": "r1"}),
                      answer="RxCUI 1043567.")
    shaped, expanded = trace.tool_calls
    assert shaped.shaped_result_tokens < BUDGET + 60 < shaped.raw_result_tokens
    assert f'{EXPAND}(handle="r1")' in shaped.shaped_text and "of 134" in shaped.shaped_text
    [t] = trace.transformations
    assert {e.kind for e in t.evidence} == {"run"} and {e.ref for e in t.evidence} == {x.run_id for x in traces}
    assert trace.provenance_coverage == 1.0 and trace.problems() == []
    assert expanded.shaped_text == shaped.raw_text  # nothing is lost

    # Without runs that kept what they used, nothing is cut.
    bare = await run(replay_hub, [ResultShaper("budget", {}, budget=BUDGET)],
                     tool_call("medical__search-drug-nomenclature", rxnorm))
    assert bare.transformations == [] and bare.tool_calls[0].shaped_text == bare.tool_calls[0].raw_text


async def test_paged_expand_returns_what_was_cut_in_budget_sized_parts(replay_hub, store):
    rxnorm = recorded(store, "search-drug-nomenclature", query="metformin")
    traces = await history(replay_hub, 2, tool_call("medical__search-drug-nomenclature", rxnorm),
                           answer="Kombiglyze, RxCUI 1043567.")
    policies = learn_policies(traces, budgets=[BUDGET], counter=estimate_tokens)
    shaper = ResultShaper("budget", policies, budget=BUDGET, expand="page")
    trace = await run(replay_hub, [shaper], tool_call("medical__search-drug-nomenclature", rxnorm),
                      tool_call(EXPAND, {"handle": "r1"}), tool_call(EXPAND, {"handle": "r1", "page": 2}),
                      tool_call(EXPAND, {"handle": "r1", "page": 99}), answer="RxCUI 1043567.")
    shaped, one, two, beyond = trace.tool_calls
    assert f'More, one part at a time: {EXPAND}(handle="r1")' in shaped.shaped_text
    pages = shaper.pages["r1"]
    assert len(pages) > 2 and all(estimate_tokens(p) <= BUDGET for p in pages)
    assert one.shaped_text.startswith(pages[0]) and f'part 1 of {len(pages)} of what was cut; next: {EXPAND}(handle="r1", page=2)' in one.shaped_text
    assert two.shaped_text.startswith(pages[1]) and one.shaped_result_tokens < BUDGET + 60
    assert beyond.is_error and f"parts 1-{len(pages)}" in beyond.shaped_text
    # The part already shown plus the pages are the original: nothing is lost, nothing repeated.
    shown = shaped.shaped_text.split("\n\n[shaped:")[0]
    squash = lambda s: "".join(s.split())  # noqa: E731
    assert squash(shown + "".join(pages)) == squash(shaped.raw_text)
    assert trace.provenance_coverage == 1.0 and trace.problems() == []
    assert "page" in shaper.extra_tools()[0].tool.input_schema["properties"]
    assert "page" not in ResultShaper("budget", policies).extra_tools()[0].tool.input_schema["properties"]
