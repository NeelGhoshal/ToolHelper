import pytest

from toolhelper.agent.hooks import BaseStrategy, CallDecision, ContextItem, ExtraTool, Task, ToolDef
from toolhelper.agent.loop import CONTEXT_MARKER, Agent, AgentConfig
from toolhelper.context.model import EvidenceRef
from toolhelper.llm.base import ChatOptions, estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend
from toolhelper.mcp.catalog import hub_catalog_hash
from toolhelper.mcp.hub import Hub, ToolResult
from toolhelper.monitoring.tracer import Trace

pytestmark = pytest.mark.anyio

TASK = Task(id="omop-001", servers=["omop"], prompt="Map 'heart attack' to condition_occurrence.condition_concept_id.")


@pytest.fixture
def heart_attack(store):
    """Args of the recorded real find_omop_concept call for 'heart attack'."""
    return next(r.arguments for r in store if r.tool == "find_omop_concept"
                and r.arguments["keyword"] == "heart attack" and r.arguments["omop_table"] == "condition_occurrence")


@pytest.fixture
def hub(server_configs, catalog, store):
    return Hub(server_configs, mode="replay", store=store, catalog=catalog, live_fallback=False)


def call(name, args):
    return {"role": "assistant", "tool_calls": [{"name": name, "arguments": args}]}


def answer(text):
    return {"role": "assistant", "content": text}


async def test_one_task_produces_a_complete_trace(hub, catalog, heart_attack, tmp_path):
    """Phase 4 done criterion."""
    llm = ScriptedBackend([call("omop__find_omop_concept", heart_attack),
                           answer("Myocardial infarction, concept_id 4329847 (SNOMED 22298006).")],
                          defaults=ChatOptions(temperature=0, seed=7))
    async with hub:
        assert hub_catalog_hash(hub) == catalog.catalog_hash
        trace = await Agent(llm, hub, counter=estimate_tokens).run(TASK)

    assert trace.problems() == []
    assert trace.stop_reason == "final" and "4329847" in trace.answer
    assert trace.catalog_version == catalog.catalog_hash and trace.hub_mode == "replay"
    assert trace.params == {"temperature": 0.0, "seed": 7}
    assert len(trace.tools_offered) == 19

    first, second = trace.llm_calls
    assert first.tool_calls == ["call_0"] and second.tool_calls == []
    assert first.measured is False  # scripted usage is estimated
    assert first.est_breakdown["tool_defs"] > 0 and first.est_breakdown["system"] > 0
    assert first.tool_results_in_context == {}
    assert list(second.tool_results_in_context) == ["call_0"]

    [tc] = trace.tool_calls
    assert (tc.server, tc.tool, tc.decision, tc.replayed, tc.is_error) == (
        "omop", "find_omop_concept", "proceed", True, False)
    assert tc.raw_result_tokens == tc.shaped_result_tokens > 0
    assert "Myocardial infarction" in tc.raw_text and tc.recorded_at
    assert [m.role for m in trace.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert trace.provenance_coverage == 1.0

    assert Trace.load(trace.save(tmp_path)) == trace


class Demo(BaseStrategy):
    """Exercises every hook against recorded real responses."""

    name = "demo"
    EV = [EvidenceRef(kind="dataset_row", ref="omop_concept_id_fields.json#condition_occurrence")]

    def __init__(self):
        self.seen: dict[str, ToolResult] = {}

    def prepare(self, task, ctx):
        return [ContextItem(kind="hint", text="Conditions map to condition_occurrence.condition_concept_id.",
                            evidence=self.EV)]

    def select_tools(self, task, tools):
        kept = [t for t in tools if t.server in task.servers or t.origin == self.name]
        self.run.transform("tools", before=len(tools), after=len(kept), evidence=[
            EvidenceRef(kind="run", ref="task.servers")], note=f"kept {len(kept)} of {len(tools)}")
        return kept

    def transform_schemas(self, tools):
        # Deliberately unreported: the loop must catch it.
        return [t.model_copy(update={"description": t.description[:40]}) for t in tools]

    def before_call(self, call):
        if call.arguments.get("omop_table") == "condition":
            return CallDecision(action="rejected", hint="omop_table must be condition_occurrence.",
                                evidence=self.EV)
        key = f"{call.name}{sorted(call.arguments.items())}"
        if key in self.seen:
            return CallDecision(action="cached", result=self.seen[key],
                                evidence=[EvidenceRef(kind="run", ref=f"same-run:{key}")])
        return CallDecision()

    def after_result(self, call, result):
        self.seen[f"{call.name}{sorted(call.arguments.items())}"] = result
        if result.source == "local":
            return result
        shaped = result.model_copy(update={"text": result.text[:200]})
        self.run.transform(call.name, before=result.text, after=shaped.text,
                           evidence=[EvidenceRef(kind="run", ref="test-projection")], note="first 200 chars")
        return shaped

    def extra_tools(self):
        async def echo(call):
            return ToolResult(name="demo__echo", server="local", tool="echo", arguments=call.arguments,
                              text=f"echo {call.arguments.get('text')}", is_error=False, source="local",
                              latency_ms=0)
        return [ExtraTool(tool=ToolDef(name="demo__echo", description="Echo text back.", input_schema={
            "type": "object", "properties": {"text": {"type": "string"}}}), handler=echo)]


async def test_every_hook_is_traced(hub, heart_attack):
    wrong = {**heart_attack, "omop_table": "condition"}
    llm = ScriptedBackend([
        call("omop__find_omop_concept", wrong),
        call("omop__find_omop_concept", heart_attack),
        {"role": "assistant", "tool_calls": [{"name": "omop__find_omop_concept", "arguments": heart_attack},
                                             {"name": "demo__echo", "arguments": {"text": "hi"}}]},
        call("medical__search-drugs", {"query": "ibuprofen"}),
        answer("concept_id 4329847"),
    ], strict_tools=False)
    async with hub:
        trace = await Agent(llm, hub, [Demo()], counter=estimate_tokens).run(TASK)

    assert trace.stop_reason == "final"
    assert sorted(trace.tools_offered) == ["demo__echo", "omop__batch_map_concepts_from_csv",
                                           "omop__find_omop_concept"]
    assert trace.messages[1].content.startswith(CONTEXT_MARKER)
    assert trace.llm_calls[0].est_breakdown["context"] > 0
    assert [c.text for c in trace.context_items] == [Demo().prepare(TASK, None)[0].text]

    rows = [(t.turn, t.decision, t.error_class, t.source) for t in trace.tool_calls]
    assert rows == [(1, "rejected", "rejected", None), (2, "proceed", None, "replay"),
                    (3, "cached", None, "replay"), (3, "proceed", None, "local"),
                    (4, "proceed", "not_offered", None)]
    assert trace.tool_calls[0].shaped_text == "omop_table must be condition_occurrence."
    assert trace.tool_calls[1].raw_result_tokens > trace.tool_calls[1].shaped_result_tokens
    assert trace.tool_calls[2].cache_hit and trace.tool_calls[3].shaped_text == "echo hi"
    assert len({t.call_id for t in trace.tool_calls}) == 5  # per-turn call ids made unique

    hooks = [(t.hook, t.reported, bool(t.evidence)) for t in trace.transformations]
    assert hooks == [("select_tools", True, True), ("transform_schemas", False, False),
                     ("before_call", True, True), ("after_result", True, True), ("before_call", True, True),
                     ("after_result", True, True)]  # cached results are shaped too
    assert trace.provenance_coverage == pytest.approx(6 / 7)
    [problem] = trace.problems()
    assert "demo.transform_schemas" in problem


async def test_max_turns_stops_the_run(hub, heart_attack):
    llm = ScriptedBackend([call("omop__find_omop_concept", heart_attack)] * 2)
    async with hub:
        trace = await Agent(llm, hub, config=AgentConfig(max_turns=2), counter=estimate_tokens).run(TASK)
    assert trace.stop_reason == "max_turns" and trace.answer is None
    assert len(trace.llm_calls) == 2 and trace.problems() == []
    assert [t.call_id for t in trace.tool_calls] == ["call_0", "t2_call_0"]


async def test_errors_end_the_run_with_a_trace(hub):
    llm = ScriptedBackend([call("omop__find_omop_concept", {"keyword": "never recorded",
                                                            "omop_table": "measurement",
                                                            "omop_field": "measurement_concept_id"})])
    async with hub:
        trace = await Agent(llm, hub, counter=estimate_tokens).run(TASK)
    assert trace.stop_reason == "error" and "ReplayMiss" in trace.error
    assert trace.finished_at and trace.problems()  # incomplete: the issued call has no record
