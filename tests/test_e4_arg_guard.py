"""E4 argument guard. Rules are learned from the real recorded error responses in
tests/fixtures/replay; every repaired call must hit a real recording (strict replay)."""


import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.efficiency.arg_guard import (
    DEFAULT_CDM_FIELDS,
    ArgumentGuard,
    CdmFields,
    learn_rules,
    observations_from_recordings,
)
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend

pytestmark = pytest.mark.anyio
TASK = Task(id="t", prompt="test")


@pytest.fixture
def index(dataset_index):
    return dataset_index


@pytest.fixture
def rules(store):
    return learn_rules(observations_from_recordings(store))


@pytest.fixture
def cdm():
    if not DEFAULT_CDM_FIELDS.exists():
        pytest.skip("omop_mcp checkout not present")
    return CdmFields.load()


async def run(hub, guard, *steps):
    llm = ScriptedBackend([*steps, final("done")], strict_tools=False)
    async with hub:
        return await Agent(llm, hub, [guard], counter=estimate_tokens).run(TASK)


def test_rules_learned_from_real_errors(rules):
    by = {(r.kind, r.entity, r.value): r for r in rules}
    trial = by[("unknown_field", "trial", "keyword")]
    assert trial.tool == "biomcp__search" and trial.rename_to is None  # no recording moved the value
    assert all(e.startswith("recording:biomcp/search/") for e in trial.evidence)
    article = by[("unknown_field", "article", "query")]
    assert article.rename_to == "keyword" and len(article.rename_evidence) == 2
    assert ("invalid_section", "variant", "annotations") in by
    assert ("shell", None, None) in by


async def test_reject_mode_returns_a_hint_and_sends_nothing(replay_hub, store, rules):
    bad = recorded(store, "search", entity="trial", keyword=["BRAF V600E"])
    trace = await run(replay_hub, ArgumentGuard("reject", rules), tool_call("biomcp__search", bad))
    [tc] = trace.tool_calls
    assert tc.decision == "rejected" and tc.raw_result_tokens is None and tc.error_class == "rejected"
    assert "`keyword` is not a trial search field" in tc.shaped_text
    assert trace.provenance_coverage == 1.0 and trace.problems() == []
    [t] = trace.transformations
    assert t.strategy == "E4_arg_guard" and t.evidence[0].ref.startswith("recording:")


@pytest.mark.parametrize("bad, fixed", [
    # article `query` -> `keyword` (learned rename; the schema wants an array)
    ({"entity": "article", "query": "GLP-1 receptor agonists abdominal aortic aneurysm", "limit": 3},
     {"entity": "article", "keyword": ["GLP-1 receptor agonists abdominal aortic aneurysm"], "limit": 3}),
    # invalid variant section dropped
    ({"entity": "variant", "id": "chr7:g.140453136A>T", "sections": ["clinvar", "annotations"]},
     {"entity": "variant", "id": "chr7:g.140453136A>T", "sections": ["clinvar"]}),
])
async def test_repairs_land_on_real_successful_calls(replay_hub, rules, bad, fixed):
    name = "biomcp__get" if "id" in bad else "biomcp__search"
    trace = await run(replay_hub, ArgumentGuard("repair", rules), tool_call(name, bad))
    [tc] = trace.tool_calls
    assert tc.decision == "repaired" and tc.final_arguments == fixed
    assert tc.source == "replay" and not tc.is_error  # strict replay: this exact call was recorded
    assert trace.provenance_coverage == 1.0


async def test_shell_syntax_is_cut(replay_hub, rules):
    trace = await run(replay_hub, ArgumentGuard("repair", rules),
                      tool_call("biomcp__biomcp", {"command": "search drug imatinib | head -50"}))
    [tc] = trace.tool_calls
    assert tc.final_arguments == {"command": "search drug imatinib"} and not tc.is_error


async def test_schema_enum_is_rejected_with_the_valid_values(replay_hub, store):
    bad = recorded(store, "search", entity="drug")
    trace = await run(replay_hub, ArgumentGuard("repair"), tool_call("biomcp__search", bad))
    [tc] = trace.tool_calls
    assert tc.decision == "rejected" and "`entity` must be one of author, gene" in tc.shaped_text
    assert trace.transformations[0].evidence[0].kind == "schema_hash"


async def test_schema_bounds_are_clamped(replay_hub):
    guard = ArgumentGuard("repair")
    async with replay_hub:
        await Agent(ScriptedBackend([final("x")]), replay_hub, [guard], counter=estimate_tokens).run(TASK)
    [issue] = guard.check("biomcp__search", {"entity": "gene", "query": "TP53", "limit": 999})
    assert issue.rule == "schema:maximum" and issue.repair["limit"] == 25


async def test_cdm_table_is_repaired_from_omop_mcp_field_map(replay_hub, store, cdm):
    bad = recorded(store, "find_omop_concept", omop_table="condition")
    trace = await run(replay_hub, ArgumentGuard("repair", cdm=cdm), tool_call("omop__find_omop_concept", bad))
    [tc] = trace.tool_calls
    assert tc.final_arguments["omop_table"] == "condition_occurrence" and tc.source == "replay"
    assert "Myocardial infarction" in tc.raw_text
    [t] = trace.transformations
    assert t.evidence[0].ref.startswith("omop_mcp:data/omop_concept_id_fields.json@sha256:")


async def test_country_name_becomes_iso3_from_the_dataset(replay_hub, store, index):
    bad = recorded(store, "get-health-statistics", country="Japan")
    trace = await run(replay_hub, ArgumentGuard("repair", index=index), tool_call("medical__get-health-statistics", bad))
    [tc] = trace.tool_calls
    assert tc.final_arguments["country"] == "JPN" and tc.source == "replay"
    [t] = trace.transformations
    assert t.evidence[0].kind == "dataset_row" and t.evidence[0].ref == "datasets/countries@v1#JPN"

    reject = await run(replay_hub, ArgumentGuard("reject", index=index), tool_call("medical__get-health-statistics", bad))
    assert "use `JPN` for Japan" in reject.tool_calls[0].shaped_text


async def test_ndc_must_exist_in_the_fda_directory(replay_hub, index):
    guard = ArgumentGuard("repair", index=index)
    trace = await run(replay_hub, guard, tool_call("medical__get-drug-details", {"ndc": "0000-0000"}),
                      tool_call("medical__get-drug-details", {"ndc": "47781-951"}))
    absent, known = trace.tool_calls
    assert absent.decision == "rejected" and "not in the FDA NDC Directory" in absent.shaped_text
    assert absent.raw_result_tokens is None  # the unrelated product Medical MCP returns is never fetched
    assert known.decision == "proceed" and known.source == "replay"
    [issue] = guard.check("medical__get-drug-details", {"ndc": "Zoloft"})
    assert issue.rule == "ndc:name" and "58151-" in issue.hint
    assert trace.provenance_coverage == 1.0
