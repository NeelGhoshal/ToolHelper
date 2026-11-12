"""E1 tool retrieval over cards built from the real catalog snapshot and scripted runs."""

import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task, ToolDef
from toolhelper.agent.loop import Agent
from toolhelper.context.ingest import import_catalog
from toolhelper.context.local_store import LocalStore
from toolhelper.efficiency.retrieval import FIND_TOOLS, CardIndex, ToolRetrieval, build_cards, recall_at_k, write_cards
from toolhelper.experiments.scoring import score
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend

pytestmark = pytest.mark.anyio

OMOP_TASK = Task(id="omop-001", prompt="Map 'heart attack' to condition_occurrence.condition_concept_id.",
                 servers=["omop"], expected_paths=[["omop__find_omop_concept"]],
                 checks={"must_call_any": ["omop__find_omop_concept"], "answer_includes_any": ["4329847"]})


@pytest.fixture
def tools(catalog):
    return [ToolDef(name=t.namespaced, description=t.definition.get("description", ""),
                    input_schema=t.definition["inputSchema"], server=s, tool=t.name, schema_hash=t.schema_hash)
            for s, srv in catalog.servers.items() for t in srv.tools]


async def run(hub, strategies, task, *steps, answer="Myocardial infarction 4329847"):
    async with hub:
        return await Agent(ScriptedBackend([*steps, final(answer)], strict_tools=False), hub, strategies,
                           counter=estimate_tokens).run(task)


def test_cards_from_catalog_runs_and_rules(tools):
    class Run:  # the fields build_cards reads from a scored trace
        run_id, task_id, prompt, stop_reason, success = "run-1", "omop-001", OMOP_TASK.prompt, "final", True
        tool_calls = [type("C", (), {"name": "omop__find_omop_concept", "is_error": False})()]

    class Rule:
        kind, tool, entity, value, rename_to = "unknown_field", "biomcp__search", "trial", "keyword", None

    cards = {c.tool: c for c in build_cards(tools, [Run()], [Rule()])}
    assert len(cards) == 19
    omop = cards["omop__find_omop_concept"]
    assert omop.example_tasks == [("omop-001", OMOP_TASK.prompt)] and omop.evidence_runs == ["run-1"]
    assert "Required: keyword, omop_table, omop_field." in omop.usage_notes
    assert cards["biomcp__search"].pitfalls == ["trial search does not take `keyword`"]


def test_cards_are_written_as_proposed_toolcards(tools, catalog):
    store = LocalStore()
    import_catalog(store, catalog)
    card = build_cards(tools)[0]
    card.evidence_runs = ["run-1"]
    write_cards(store, [card])
    server, tool = card.tool.split("__", 1)
    thing = store.get(f"toolcards/{server}/{tool}")
    assert thing.data["status"] == "proposed" and thing.data["tool"].startswith(f"tools/{server}/{tool}@v")


def test_leave_one_out_hides_the_tasks_own_example(tools):
    class Run:
        run_id, task_id, stop_reason, success = "r", "x-1", "final", True
        prompt = "zzyzx quuxly frobnicate"
        tool_calls = [type("C", (), {"name": "medical__search-drugs", "is_error": False})()]

    index = CardIndex(build_cards(tools, [Run()]))
    assert index.search("zzyzx quuxly")[0][0] == "medical__search-drugs"
    assert index.search("zzyzx quuxly", exclude_task="x-1") == []


async def test_static_top_k_offers_the_right_tools(replay_hub, store, tools):
    heart = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")
    trace = await run(replay_hub, [ToolRetrieval(build_cards(tools), top_k=3)], OMOP_TASK,
                      tool_call("omop__find_omop_concept", heart))
    assert len(trace.tools_offered) == 3 and "omop__find_omop_concept" in trace.tools_offered
    assert recall_at_k(trace, OMOP_TASK) == 1.0 and score(trace, OMOP_TASK).success
    [t] = trace.transformations
    assert [e.kind for e in t.evidence] == ["schema_hash"] * 3 and t.evidence[0].detail["rank"] == 1
    assert trace.provenance_coverage == 1.0
    assert trace.llm_calls[0].est_breakdown["tool_defs"] < 1000  # vs ~3,600 for all 19


async def test_find_tools_adds_tools_mid_run(replay_hub, tools):
    task = Task(id="med-002", prompt="Map heart attack to an OMOP condition concept.")
    trace = await run(replay_hub, [ToolRetrieval(build_cards(tools), top_k=1, mode="find_tools")], task,
                      tool_call(FIND_TOOLS, {"query": "drug details by National Drug Code NDC"}),
                      tool_call("medical__get-drug-details", {"ndc": "47781-951"}), answer="done")
    first, second, *_ = trace.llm_calls
    assert "medical__get-drug-details" not in first.tools_offered and FIND_TOOLS in first.tools_offered
    assert "medical__get-drug-details" in second.tools_offered
    find, details = trace.tool_calls
    assert "medical__get-drug-details" in find.shaped_text
    assert details.source == "replay" and details.error_class is None  # callable once offered
    offer = [t for t in trace.transformations if t.hook == "offer"]
    assert offer and offer[0].strategy == "E1_retrieval" and trace.provenance_coverage == 1.0


async def test_server_menu_loads_the_servers_the_model_picks(replay_hub, tools, store):
    from toolhelper.efficiency.compression import SchemaCompression
    from toolhelper.efficiency.retrieval import LOAD_TOOLS, ServerMenu

    menu = ServerMenu(tools)
    args = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")
    trace = await run(replay_hub, [menu, SchemaCompression("minimal")], OMOP_TASK,
                      tool_call(LOAD_TOOLS, {"servers": ["omop"]}), tool_call("omop__find_omop_concept", args))
    first, second, *_ = trace.llm_calls
    assert first.tools_offered == [LOAD_TOOLS]
    assert sorted(second.tools_offered) == sorted([LOAD_TOOLS, "omop__batch_map_concepts_from_csv", "omop__find_omop_concept"])
    assert first.est_breakdown["tool_defs"] < 600 and second.est_breakdown["tool_defs"] < 900  # vs ~3,600 for all 19
    load, find = trace.tool_calls
    assert "Loaded 2 tools" in load.shaped_text and find.source == "replay" and find.error_class is None
    hidden, offer, *rest = [t for t in trace.transformations if t.strategy == "E1_server_menu"]
    assert (hidden.hook, len(hidden.evidence)) == ("select_tools", len(tools)) and offer.hook == "offer"
    assert {e.kind for e in hidden.evidence + offer.evidence} == {"schema_hash"} and len(offer.evidence) == 2
    # Tools loaded mid-run are compressed like the initial ones.
    assert any(t.strategy.startswith("E2") and t.turn == 1 and t.hook == "transform_schemas" for t in trace.transformations[2:])
    assert trace.provenance_coverage == 1.0 and trace.problems() == [] and score(trace, OMOP_TASK).success
    text = menu.extra_tools()[0].tool.description
    assert "- omop (2 tools): batch_map_concepts_from_csv: Process a CSV file" in text and "(entity: gene, variant, article" in text
    assert "search-drugs: Search for drug information using FDA database" in text and "- biomcp (7 tools)" in text


async def test_server_menu_refuses_unknown_servers_and_calls_before_loading(replay_hub, tools, store):
    from toolhelper.efficiency.retrieval import LOAD_TOOLS, ServerMenu

    args = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")
    trace = await run(replay_hub, [ServerMenu(tools)], OMOP_TASK, tool_call("omop__find_omop_concept", args),
                      tool_call(LOAD_TOOLS, {"servers": ["pubmed"]}), tool_call(LOAD_TOOLS, {"servers": "omop"}),
                      tool_call(LOAD_TOOLS, {"servers": ["omop"]}))
    early, unknown, loaded, again = trace.tool_calls
    assert early.error_class == "not_offered"
    assert unknown.is_error and "choose from ['biomcp', 'medical', 'omop']" in unknown.shaped_text
    assert "Loaded 2 tools" in loaded.shaped_text and "already loaded" in again.shaped_text and not again.is_error
    assert trace.provenance_coverage == 1.0
