"""User-defined strategies (E10 and up): the definitions file, the registry, and each building
block run through the agent loop on recorded real responses."""

import pytest
import yaml
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent, AgentConfig
from toolhelper.efficiency import custom, registry
from toolhelper.efficiency.custom import Definition
from toolhelper.efficiency.registry import BUILTIN, ORDER, Inputs, build_strategies, check_spec
from toolhelper.experiments import design_form as form
from toolhelper.experiments.experiment import freeze_inputs, inputs_hash
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend

pytestmark = pytest.mark.anyio

TASK = Task(id="omop-001", servers=["omop"], prompt="Map 'heart attack' to condition_occurrence.condition_concept_id.",
            checks={"must_call_any": ["omop__find_omop_concept"], "answer_includes_any": ["4329847"]})
EARLY = Definition(name="Answer early", kind="instruction", text="Answer as soon as one result holds the ID.",
                   evidence_runs="max_turns")
OMOP_ONLY = Definition(name="OMOP only", kind="tool_filter", hide_servers=["medical"], hide_tools=["biomcp__search"])
CLOSING = Definition(name="Short close", kind="closing_prompt", text="Give the concept ID now.")


class Run:
    def __init__(self, run_id, stop_reason="final", errors=()):
        self.run_id, self.stop_reason = run_id, stop_reason
        self.tool_calls = [type("Call", (), {"is_error": e}) for e in errors]


RUNS = [Run("r-ok"), Run("r-limit", "max_turns"), Run("r-err", errors=[False, True])]


def one_call(store):
    return tool_call("omop__find_omop_concept",
                     recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence"))


def test_definitions_round_trip_and_extend_the_registry(tmp_path):
    assert custom.load() == {} and ORDER == list(BUILTIN) and custom.next_id({}) == "E10"
    custom.save({"E11": OMOP_ONLY, "E10": EARLY})
    defs = registry.install_custom()
    assert list(defs) == ["E10", "E11"] and defs["E10"] == EARLY and custom.next_id(defs) == "E12"
    assert ORDER == [*BUILTIN, "E10", "E11"] and check_spec("E10") == [] and check_spec("e11") == []
    assert check_spec("E10:x=1") == ["E10: unknown parameter 'x'; known: none"]
    built = build_strategies(["E11", "E9", "E10"], Inputs(traces=RUNS))
    assert [s.name for s in built] == ["E9_final_answer", "E10_answer_early", "E11_omop_only"]
    assert form.info("E10")["name"] == "Answer early" and form.needs_runs("E10", defs) and not form.needs_runs("E11", defs)
    custom.save({"E11": OMOP_ONLY})
    registry.install_custom()
    assert ORDER == [*BUILTIN, "E11"] and check_spec("E10") != []


def test_bad_definitions_are_refused(tmp_path):
    custom.FILE.write_text(yaml.safe_dump({"strategies": {"E4": EARLY.model_dump()}}), encoding="utf-8")
    with pytest.raises(ValueError, match="E10 or higher"):
        custom.load()
    assert Definition(name="", kind="instruction").problems() == ["give the strategy a name", "the text is empty"]
    assert Definition(name="x", kind="tool_filter").problems() == ["pick at least one server or tool to hide"]


def test_evidence_runs_follow_the_definition():
    assert custom.evidence_runs(RUNS, "max_turns") == ["r-limit"] and custom.evidence_runs(RUNS, "tool_errors") == ["r-err"]
    assert custom.evidence_runs(RUNS, "all") == ["r-err", "r-limit", "r-ok"]
    with pytest.raises(ValueError, match="E10 needs evidence"):
        custom.build("E10", EARLY, Inputs(traces=[Run("r-ok")]))
    with pytest.raises(ValueError, match="evidence"):
        custom.build("E12", CLOSING, Inputs(traces=[Run("r-ok")]))


async def test_an_instruction_reaches_the_prompt_with_its_evidence(replay_hub, store):
    llm = ScriptedBackend([one_call(store), final("concept_id 4329847")])
    async with replay_hub:
        trace = await Agent(llm, replay_hub, [custom.build("E10", EARLY, Inputs(traces=RUNS))], AgentConfig(max_turns=4),
                            counter=estimate_tokens).run(TASK)
    [item] = trace.context_items
    assert (item.strategy, item.kind, item.text) == ("E10_answer_early", "instruction", EARLY.text)
    assert [e.ref for e in item.evidence] == ["r-limit"] and item.evidence[0].detail["motivated_by"] == "max_turns"
    assert any(EARLY.text in (m.content or "") for m in trace.messages)
    assert trace.problems() == [] and trace.provenance_coverage == 1.0


async def test_a_tool_filter_hides_tools_and_cites_their_schema_hashes(replay_hub, store):
    llm = ScriptedBackend([one_call(store), final("concept_id 4329847")])
    async with replay_hub:
        trace = await Agent(llm, replay_hub, [custom.build("E11", OMOP_ONLY, Inputs())], AgentConfig(max_turns=4),
                            counter=estimate_tokens).run(TASK)
        hidden = [t for t in replay_hub.tools if t.server == "medical" or t.name == "biomcp__search"]
    offered = trace.llm_calls[0].tools_offered
    assert "omop__find_omop_concept" in offered and "biomcp__search" not in offered
    assert not [t for t in offered if t.startswith("medical__")] and [t for t in offered if t.startswith("biomcp__")]
    [t] = trace.transformations
    assert (t.strategy, t.hook, t.target) == ("E11_omop_only", "select_tools", "tools")
    assert len(t.evidence) == len(hidden) > 1 and {e.kind for e in t.evidence} == {"schema_hash"}
    assert trace.problems() == [] and trace.provenance_coverage == 1.0


async def test_hiding_every_server_is_an_error(replay_hub, store):
    everything = Definition(name="none", kind="tool_filter", hide_servers=["medical", "biomcp", "omop"])
    async with replay_hub:
        trace = await Agent(ScriptedBackend([final("x")]), replay_hub, [custom.build("E13", everything, Inputs())],
                            AgentConfig(max_turns=2), counter=estimate_tokens).run(TASK)
    assert trace.stop_reason == "error" and "E13 hides every server tool" in trace.error and trace.llm_calls == []


async def test_a_closing_prompt_is_asked_at_the_limit_in_the_users_words(replay_hub, store):
    llm = ScriptedBackend([one_call(store)] * 2 + [final("4329847")])
    async with replay_hub:
        trace = await Agent(llm, replay_hub, [custom.build("E12", CLOSING, Inputs(traces=RUNS))], AgentConfig(max_turns=2),
                            counter=estimate_tokens).run(TASK)
    assert trace.stop_reason == "final_at_limit" and trace.messages[-2].content == CLOSING.text
    [t] = trace.transformations
    assert t.strategy == "E12_short_close" and [e.ref for e in t.evidence] == ["r-limit"]
    assert trace.problems() == [] and trace.provenance_coverage == 1.0


def test_configs_name_the_definitions_they_use_and_freezing_stores_them(tmp_path):
    custom.save({"E10": EARLY, "E11": OMOP_ONLY})
    data = {"experiment": "x", "base": {"specs": ["E2:level=minimal"]}, "factors": {"A": {"specs": ["e11"]}}}
    assert custom.used_in(data) == {"E11": OMOP_ONLY.model_dump()}
    (tmp_path / "experiments-x.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    assert custom.configs_using("E11", tmp_path) == ["experiments-x.yaml"] and custom.configs_using("E10", tmp_path) == []

    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("{}\n", encoding="utf-8")
    common = dict(catalog_hash="c", tasks_file=tasks, datasets=[], llm_profile={}, store_dir=tmp_path, learn_from_runs=[])
    plain, frozen = freeze_inputs(**common), freeze_inputs(**common, custom_strategies=custom.used_in(data))
    assert "custom_strategies" not in plain and frozen["custom_strategies"]["E11"]["hide_servers"] == ["medical"]
    assert inputs_hash(plain) != inputs_hash(frozen)


def test_factors_of_another_config_join_as_factors_or_as_the_base():
    load = lambda n: yaml.safe_load((custom.CONFIG_DIR / f"experiments-{n}.yaml").read_text(encoding="utf-8"))  # noqa: E731
    p10b, p10c = load("p10b"), load("p10c")
    added = form.import_factors(p10c, p10b, ["A", "D"], into_base=False)
    assert list(added["factors"]) == ["H", "T", "R", "A", "D"] and added["factors"]["D"] == p10b["factors"]["D"]
    clash = form.import_factors(p10b, p10b, ["A"], into_base=False)
    assert list(clash["factors"]) == ["A", "B", "C", "D", "E"] and list(p10b["factors"]) == ["A", "B", "C", "D"]
    based = form.import_factors({**p10c, "base": {}}, p10b, ["A", "B", "C"], into_base=True)
    assert based["base"] == {"specs": ["E2:level=minimal", "E6:mode=budget,budget=2000", "E9"], "max_turns": 6}
    with pytest.raises(ValueError, match="conflicting"):
        form.import_factors({"base": {"specs": ["E2:level=light"]}, "factors": {}}, p10b, ["A"], into_base=True)
