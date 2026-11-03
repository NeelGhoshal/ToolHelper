import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.context.ingest import import_catalog
from toolhelper.context.local_store import LocalStore
from toolhelper.context.model import Revise
from toolhelper.efficiency.recipes import FROM_TASK, GET_RECIPE, Recipes, learn_recipe, run_paths, write_recipes
from toolhelper.experiments.scoring import score
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend

pytestmark = pytest.mark.anyio


def task(id, keyword="heart attack", category="condition_mapping"):
    return Task(id=id, category=category, prompt=f"Map '{keyword}' to condition_occurrence.condition_concept_id.",
                checks={"must_call_any": ["omop__find_omop_concept"], "answer_includes_any": ["4329847"]})


async def scored_run(hub, t, *steps, answer="4329847", strategies=()):
    async with hub:
        trace = await Agent(ScriptedBackend([*steps, final(answer)], strict_tools=False), hub, list(strategies),
                            counter=estimate_tokens).run(t)
    result = score(trace, t)
    trace.success, trace.score = result.success, result.model_dump()
    return trace


@pytest.fixture
def heart(store):
    return recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")


@pytest.fixture
async def history(replay_hub, store, heart):
    wrong = recorded(store, "find_omop_concept", omop_table="condition")
    direct = tool_call("omop__find_omop_concept", heart)
    detour = [tool_call("omop__find_omop_concept", wrong), direct]
    return [await scored_run(replay_hub, task("a"), direct), await scored_run(replay_hub, task("b"), direct),
            await scored_run(replay_hub, task("b"), *detour),  # same path once the error is skipped
            await scored_run(replay_hub, task("c"), direct, answer="no idea")]  # failed run


async def test_recipe_is_the_shortest_frequent_successful_path(history):
    paths = run_paths(history)
    recipe = learn_recipe(paths, "condition_mapping")
    assert recipe.steps == ["omop__find_omop_concept"] and len(recipe.evidence_runs) == 3
    assert recipe.success_rate == 0.75  # 3 of 4 runs in the category succeeded
    template = recipe.arg_templates[0]
    assert template["keyword"] == FROM_TASK and template["max_results"] == 5
    assert learn_recipe(paths, "other") is None


async def test_leave_one_out(history):
    paths = run_paths(history)
    assert learn_recipe(paths, "condition_mapping", exclude_task="b").evidence_runs == [history[0].run_id]
    assert learn_recipe([p for p in paths if p.task_id == "b"], "condition_mapping", exclude_task="b") is None


async def test_plan_is_injected_with_run_evidence(replay_hub, heart, history):
    strategy = Recipes(run_paths(history))
    trace = await scored_run(replay_hub, task("new", "myocardial infarction"), tool_call("omop__find_omop_concept", heart),
                             strategies=[strategy])
    [item] = trace.context_items
    assert item.kind == "recipe" and "omop__find_omop_concept" in item.text and FROM_TASK in item.text
    assert {e.ref for e in item.evidence} == {t.run_id for t in history[:3]} and trace.provenance_coverage == 1.0


async def test_category_falls_back_to_the_most_similar_prompt(history):
    strategy = Recipes(run_paths(history))
    assert strategy.category(Task(id="z", prompt="Map 'stroke' to condition_occurrence.condition_concept_id.")) == \
        "condition_mapping"


async def test_recipe_tool(replay_hub, history):
    trace = await scored_run(replay_hub, task("new"), tool_call(GET_RECIPE, {}), strategies=[Recipes(run_paths(history), "tool")])
    assert trace.context_items == [] and "omop__find_omop_concept" in trace.tool_calls[0].shaped_text


async def test_stored_recipe_goes_stale_when_its_schema_changes(history, catalog):
    store = LocalStore()
    import_catalog(store, catalog)
    write_recipes(store, [learn_recipe(run_paths(history), "condition_mapping")])
    thing = store.get("recipes/condition_mapping")
    assert thing.data["status"] == "proposed" and thing.data["depends_on"][0].startswith("tools/omop/find_omop_concept@v")
    schema = store.get("tools/omop/find_omop_concept")
    store.apply("schema change", [Revise(name=schema.name, data={**schema.data, "description": "changed"},
                                         expected_version=schema.version)])
    assert "recipes/condition_mapping" in {s.thing.name for s in store.stale()}
