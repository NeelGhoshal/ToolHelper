from pathlib import Path

import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.efficiency.arg_guard import ArgumentGuard, learn_rules, observations_from_recordings
from toolhelper.efficiency.grounding import Grounding
from toolhelper.efficiency.trust import trust
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend
from toolhelper.monitoring.card import render_card, write_card

pytestmark = pytest.mark.anyio
REPLAY = Path(__file__).parent / "fixtures" / "replay"


async def test_card_shows_timeline_transformations_evidence_and_trust(replay_hub, store, dataset_index, tmp_path):
    task = Task(id="bio-x", prompt="Find recent articles on GLP-1 receptor agonists and aneurysm; HER2 too.")
    bad = {"entity": "article", "query": "GLP-1 receptor agonists abdominal aortic aneurysm", "limit": 3}
    guard = ArgumentGuard("repair", learn_rules(observations_from_recordings(store)))
    async with replay_hub:
        trace = await Agent(ScriptedBackend([tool_call("biomcp__search", bad), final("PMID 42957066 is relevant.")]),
                            replay_hub, [Grounding(dataset_index), guard], counter=estimate_tokens).run(task)
    trace.trust = trust(trace)
    html = render_card(trace, replay_dir=REPLAY)

    assert f"Run {trace.run_id}" in html and "Turn 1" in html and "Turn 2" in html
    assert 'id="call-call_0"' in html and "repaired" in html and "&quot;keyword&quot;" in html  # args before → after
    assert "E4_arg_guard" in html and "learned:unknown_field" in html
    assert "recording:biomcp/search/" in html and REPLAY.resolve().as_uri() in html  # evidence opens the recording
    assert "datasets/hgnc@v1#HGNC:3430" in html  # E3 hint evidence
    for component in ("grounding", "provenance", "fidelity", "freshness"):
        assert component in html
    assert "renamed an argument" in html  # fidelity reason for the learned rename
    path = write_card(trace, tmp_path, replay_dir=REPLAY)
    assert path.name == f"{trace.run_id}.html" and path.read_text(encoding="utf-8") == html


async def test_card_without_trust_or_changes(replay_hub, store):
    heart = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")
    async with replay_hub:
        trace = await Agent(ScriptedBackend([tool_call("omop__find_omop_concept", heart), final("4329847")]),
                            replay_hub, [], counter=estimate_tokens).run(Task(id="t", prompt="x"))
    html = render_card(trace)
    assert "Not scored for trust yet." in html and html.count("<p class=muted>None.</p>") == 2
