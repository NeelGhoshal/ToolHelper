import pytest
from conftest import final, recorded

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.efficiency.compaction import MARKER, RECALL, HistoryCompaction
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend

pytestmark = pytest.mark.anyio
TASK = Task(id="t", prompt="Map heart attack, then find ibuprofen products.")


def turn(name, args, content=""):
    return {"role": "assistant", "content": content, "tool_calls": [{"name": name, "arguments": args}]}


@pytest.fixture
def calls(store):
    return (recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence"),
            recorded(store, "search-drugs", query="ibuprofen"))


async def run(hub, strategy, *steps):
    async with hub:
        return await Agent(ScriptedBackend([*steps, final("Myocardial infarction 4329847.")]), hub, [strategy],
                           counter=estimate_tokens).run(TASK)


async def test_used_result_is_compacted_in_later_turns(replay_hub, calls):
    heart, ibu = calls
    trace = await run(replay_hub, HistoryCompaction("after_use"),
                      turn("omop__find_omop_concept", heart),
                      turn("medical__search-drugs", ibu, content="Heart attack is concept 4329847; now ibuprofen."))
    omop_id = trace.tool_calls[0].call_id
    carried = [c.tool_results_in_context.get(omop_id) for c in trace.llm_calls]
    assert carried[0] is None and carried[2] < carried[1]  # turn 3 carries the summary, not the result
    [t] = trace.transformations
    assert t.target == "omop__find_omop_concept" and t.after_tokens < t.before_tokens
    assert [e.detail.get("used") for e in t.evidence] == [None, True] and trace.provenance_coverage == 1.0
    compacted = next(m for m in trace.messages if m.role == "tool" and m.tool_call_id == omop_id)
    assert compacted.content.startswith(MARKER) and f'{RECALL}(handle="{omop_id}")' in compacted.content


async def test_unused_result_is_kept_by_after_use_but_not_after_turn(replay_hub, calls):
    heart, ibu = calls
    steps = (turn("omop__find_omop_concept", heart), turn("medical__search-drugs", ibu))
    kept = await run(replay_hub, HistoryCompaction("after_use"), *steps)
    assert kept.transformations == []
    dropped = await run(replay_hub, HistoryCompaction("after_turn"), *steps)
    assert [t.target for t in dropped.transformations] == ["omop__find_omop_concept"]


async def test_recall_returns_the_original(replay_hub, calls):
    heart, ibu = calls
    first = "call_0"
    trace = await run(replay_hub, HistoryCompaction("after_turn"), turn("omop__find_omop_concept", heart),
                      turn("medical__search-drugs", ibu), turn(RECALL, {"handle": first}))
    recall = trace.tool_calls[-1]
    assert recall.source == "local" and recall.shaped_text == trace.tool_calls[0].raw_text
