import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent, AgentConfig
from toolhelper.efficiency.final_answer import PROMPT, FinalAnswerAtLimit
from toolhelper.experiments.scoring import score
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend
from toolhelper.monitoring.tracer import Trace

pytestmark = pytest.mark.anyio

TASK = Task(id="omop-001", servers=["omop"], prompt="Map 'heart attack' to condition_occurrence.condition_concept_id.",
            checks={"must_call_any": ["omop__find_omop_concept"], "answer_includes_any": ["4329847"]})


def looping(store, n):
    args = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")
    return [tool_call("omop__find_omop_concept", args)] * n


async def test_answer_is_requested_without_tools_at_the_limit(replay_hub, store, tmp_path):
    llm = ScriptedBackend(looping(store, 2) + [final("Myocardial infarction, concept_id 4329847.")])
    async with replay_hub:
        trace = await Agent(llm, replay_hub, [FinalAnswerAtLimit(["run-a", "run-b"])], AgentConfig(max_turns=2),
                            counter=estimate_tokens).run(TASK)

    assert trace.stop_reason == "final_at_limit" and "4329847" in trace.answer
    assert trace.problems() == [] and trace.provenance_coverage == 1.0
    closing = trace.llm_calls[-1]
    assert (closing.turn, closing.tools_offered, closing.tool_calls) == (3, [], [])
    assert trace.messages[-2].role == "user" and trace.messages[-2].content == PROMPT
    [t] = trace.transformations
    assert (t.strategy, t.hook, t.target) == ("E9_final_answer", "at_turn_limit", "history")
    assert [e.ref for e in t.evidence] == ["run-a", "run-b"] and t.after_tokens > 0
    assert score(trace, TASK).success
    assert Trace.load(trace.save(tmp_path)) == trace  # the new stop reason round-trips


async def test_tool_calls_in_the_closing_reply_are_dropped(replay_hub, store):
    llm = ScriptedBackend(looping(store, 2) + looping(store, 1), strict_tools=False)
    async with replay_hub:
        trace = await Agent(llm, replay_hub, [FinalAnswerAtLimit(["run-a"])], AgentConfig(max_turns=2),
                            counter=estimate_tokens).run(TASK)
    assert trace.stop_reason == "final_at_limit" and len(trace.tool_calls) == 2
    assert trace.problems() == []
    assert not score(trace, TASK).success  # an empty answer still fails its checks


async def test_no_change_when_the_run_answers_in_time(replay_hub, store):
    llm = ScriptedBackend(looping(store, 1) + [final("concept_id 4329847")])
    async with replay_hub:
        trace = await Agent(llm, replay_hub, [FinalAnswerAtLimit(["run-a"])], AgentConfig(max_turns=4),
                            counter=estimate_tokens).run(TASK)
    assert trace.stop_reason == "final" and trace.transformations == []


async def test_without_the_strategy_the_limit_is_a_failure(replay_hub, store):
    llm = ScriptedBackend(looping(store, 2))
    async with replay_hub:
        trace = await Agent(llm, replay_hub, [], AgentConfig(max_turns=2), counter=estimate_tokens).run(TASK)
    assert trace.stop_reason == "max_turns" and len(trace.llm_calls) == 2


def test_evidence_is_required():
    with pytest.raises(ValueError, match="evidence"):
        FinalAnswerAtLimit([])
