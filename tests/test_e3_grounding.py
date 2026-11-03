"""E3 dataset grounding over the real dataset excerpts and recorded responses."""

import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import CONTEXT_MARKER, Agent
from toolhelper.efficiency.grounding import RESOLVE, Grounding, uptake
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend

pytestmark = pytest.mark.anyio


async def run(hub, strategy, task, *steps):
    async with hub:
        return await Agent(ScriptedBackend([*steps, final("done")]), hub, [strategy],
                           counter=estimate_tokens).run(task)


@pytest.mark.parametrize("prompt, surface, expected", [
    ("Find pathways for HER2.", "HER2", ("gene", "ERBB2", "HGNC:3430")),
    ("Pull the FDA label details for Zoloft.", "Zoloft", ("drug_product", "sertraline hydrochloride", None)),
    ("What is the WHO life expectancy at birth in Japan?", "Japan", ("country", "Japan", "JPN")),
    ("Map 'heart attack' to condition_occurrence.condition_concept_id.", "heart attack",
     ("omop_condition", "Myocardial infarction", "4329847")),
    ("What is the dbSNP rsID of BRAF V600E?", "BRAF V600E", ("variant", "NM_004333.6(BRAF):c.1799T>A (p.Val600Glu)",
                                                             "rs113488022")),
])
def test_benchmark_entities_are_found(dataset_index, prompt, surface, expected):
    found = dict(Grounding(dataset_index).entities(prompt))
    best = found[surface][0]
    assert (best.type, best.canonical) == expected[:2]
    assert expected[2] is None or best.identifier == expected[2]


def test_plain_words_do_not_match_gene_aliases(dataset_index):
    # NEU is a real HGNC alias of ERBB2; as a lower-case word it must not become a gene,
    # while an upper-case or digit-bearing span ("NEU", "erbb2") still does.
    g = Grounding(dataset_index)
    assert dict(g.entities("the neu protein")) == {}
    assert dict(g.entities("the NEU protein"))["NEU"][0].canonical == "ERBB2"
    assert dict(g.entities("erbb2 amplification"))["erbb2"][0].canonical == "ERBB2"


async def test_hints_carry_dataset_rows_and_uptake_is_measured(replay_hub, store, dataset_index):
    task = Task(id="t", prompt="Map 'heart attack' to condition_occurrence.condition_concept_id.")
    heart = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")
    trace = await run(replay_hub, Grounding(dataset_index), task, tool_call("omop__find_omop_concept", heart))
    [item] = trace.context_items
    assert item.kind == "entity" and "Myocardial infarction [4329847]" in item.text
    assert item.evidence[0].ref == "datasets/omop_eunomia@v1#4329847"
    assert trace.messages[1].content.startswith(CONTEXT_MARKER) and trace.provenance_coverage == 1.0
    assert uptake(trace) == {"hints": 1, "taken": 0, "rate": 0.0, "taken_hints": []}  # keyword stayed "heart attack"


async def test_rewrite_mode_replaces_the_alias_in_arguments(replay_hub, dataset_index):
    task = Task(id="t", prompt="Search BioMCP for the gene HER2.")
    trace = await run(replay_hub, Grounding(dataset_index, {"rewrite"}), task,
                      tool_call("biomcp__search", {"entity": "gene", "query": "HER2", "limit": 3}))
    [tc] = trace.tool_calls
    assert tc.decision == "repaired" and tc.final_arguments["query"] == "ERBB2"
    assert tc.source == "replay" and not tc.is_error  # the recorded real ERBB2 search
    [t] = trace.transformations
    assert t.evidence[0].ref == "datasets/hgnc@v1#HGNC:3430" and trace.provenance_coverage == 1.0
    assert trace.context_items == []  # rewrite alone injects nothing


async def test_resolve_entity_tool(replay_hub, dataset_index):
    task = Task(id="t", prompt="Which country code is Japan?")
    trace = await run(replay_hub, Grounding(dataset_index, {"tool"}), task, tool_call(RESOLVE, {"text": "Japan"}))
    [tc] = trace.tool_calls
    assert tc.source == "local" and "country Japan [JPN]" in tc.shaped_text and "datasets/countries@v1" in tc.shaped_text
    assert RESOLVE in trace.tools_offered


def test_dataset_subset_limits_answers(dataset_index):
    found = dict(Grounding(dataset_index, datasets=["rxnorm"]).entities("Pull the FDA label details for Zoloft."))
    assert {m.type for m in found["Zoloft"]} == {"drug_brand"}
    assert dict(Grounding(dataset_index, datasets=["hgnc"]).entities("Zoloft")) == {}


def test_unknown_mode_is_rejected(dataset_index):
    with pytest.raises(ValueError, match="unknown E3 modes"):
        Grounding(dataset_index, {"inject"})
