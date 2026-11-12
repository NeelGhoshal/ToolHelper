import json

import pytest

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.config import ROOT, TASKS_FILE
from toolhelper.experiments.runner import check_expectations, load_tasks, run_benchmark, validate_tasks
from toolhelper.experiments.scoring import score
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend
from toolhelper.mcp.hub import Hub
from toolhelper.monitoring.db import MonitorDB

pytestmark = pytest.mark.anyio


@pytest.fixture
def tasks():
    return {t.id: t for t in load_tasks(TASKS_FILE)}


@pytest.fixture
def heart_attack(store):
    return next(r.arguments for r in store if r.tool == "find_omop_concept"
                and r.arguments["keyword"] == "heart attack" and r.arguments["omop_table"] == "condition_occurrence")


@pytest.fixture
def hub(server_configs, catalog, store):
    return Hub(server_configs, mode="replay", store=store, catalog=catalog, live_fallback=False)


def steps(*items):
    out = []
    for item in items:
        out.append({"role": "assistant", "content": item} if isinstance(item, str) else
                   {"role": "assistant", "tool_calls": [{"name": item[0], "arguments": item[1]}]})
    return out


async def run(hub, task, *items):
    async with hub:
        return await Agent(ScriptedBackend(steps(*items), strict_tools=False), hub, counter=estimate_tokens).run(task)


def test_benchmark_file_is_valid(tasks, catalog):
    names = {t.namespaced for s in catalog.servers.values() for t in s.tools}
    result = validate_tasks(list(tasks.values()), names)
    assert result["problems"] == []
    assert 40 <= result["tasks"] <= 60
    assert result["by_server"]["cross"] == 10 and all(result["by_server"][s] >= 13 for s in ("medical", "biomcp", "omop"))
    assert 0.15 <= result["trap_share"] <= 0.30
    assert str(ROOT) in tasks["omop-011"].prompt  # {repo} expanded


def test_validation_catches_problems(tasks):
    bad = tasks["omop-001"].model_copy(update={"servers": [], "checks": {"must_call_any": ["omop__nope"]}})
    problems = validate_tasks([bad, bad], {"omop__find_omop_concept"})["problems"]
    assert any("duplicate" in p for p in problems) and any("unknown tool omop__nope" in p for p in problems)
    assert any("no servers" in p for p in problems)


async def test_successful_run_scores_and_is_grounded(hub, tasks, heart_attack):
    task = tasks["omop-001"]
    trace = await run(hub, task, ("omop__find_omop_concept", heart_attack), "concept_id 4329847 (SNOMED 22298006)")
    result = score(trace, task)
    assert result.success and all(result.checks.values())
    assert result.grounding == 1.0 and result.grounded == ["22298006", "4329847"]
    assert result.path_match is True


async def test_trap_and_ungrounded_ids(hub, tasks, heart_attack):
    task = tasks["omop-003"]  # 'condition table' trap: the call must use condition_occurrence
    wrong = {**heart_attack, "omop_table": "condition"}
    trace = await run(hub, task, ("omop__find_omop_concept", wrong), "concept_id 4329847, also 99999999")
    result = score(trace, task)
    assert result.checks == {"must_call_any": True, "called_with[0]": False, "answer_includes_any": True}
    assert not result.success
    assert result.grounding == 0.5 and result.ungrounded == ["99999999"]


async def test_error_results_do_not_satisfy_required_calls(hub, store):
    bad = next(r.arguments for r in store if r.tool == "search-drugs" and r.arguments == {"limit": 3})
    task = Task(id="t", prompt="Find drugs.", servers=["medical"], checks={"must_call_any": ["medical__search-drugs"]})
    trace = await run(hub, task, ("medical__search-drugs", bad), "No results.")
    assert score(trace, task).checks == {"must_call_any": False}


async def test_run_benchmark_saves_scores_and_ingests(hub, tasks, heart_attack, tmp_path):
    task = tasks["omop-001"]
    db = MonitorDB()
    async with hub:
        agent = Agent(ScriptedBackend(steps(("omop__find_omop_concept", heart_attack), "4329847")), hub,
                      counter=estimate_tokens, group_id="g1")
        [(trace, result)] = await run_benchmark([task], agent, traces_dir=tmp_path, db=db)
    assert trace.success and trace.score["success"] and (tmp_path / f"{trace.run_id}.json").exists()
    [row] = db.rows("SELECT group_id, success, score_json FROM runs")
    assert row["group_id"] == "g1" and row["success"] == 1 and json.loads(row["score_json"])["grounding"] == 1.0
    assert check_expectations([task], [trace]) == []
    wrong = task.model_copy(update={"checks": {**task.checks, "answer_includes_any": ["123456789"]}})
    [issue] = check_expectations([wrong], [trace])
    assert "123456789" in issue


def test_db_migrates_runs_without_score_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY)")
    con.commit()
    con.close()
    db = MonitorDB(path)
    assert "score_json" in {r[1] for r in db.db.execute("PRAGMA table_info(runs)")}


async def test_answer_grounded_check(hub, heart_attack):
    task = Task(id="g", prompt="Map heart attack.", servers=["omop"],
                checks={"must_call_any": ["omop__find_omop_concept"], "answer_grounded": True})
    good = await run(hub, task, ("omop__find_omop_concept", heart_attack), "concept_id 4329847")
    assert score(good, task).checks["answer_grounded"] is True
    bad = await run(hub, task, ("omop__find_omop_concept", heart_attack), "concept_id 4329847 or 12345678")
    assert score(bad, task).checks["answer_grounded"] is False


async def test_rescore_uses_current_tasks_and_latest_run(hub, tasks, heart_attack):
    from toolhelper.experiments.runner import latest_per_task, rescore

    task = tasks["omop-001"]
    first = await run(hub, task, ("omop__find_omop_concept", heart_attack), "no idea")
    second = await run(hub, task, ("omop__find_omop_concept", heart_attack), "4329847")
    assert latest_per_task([second, first]) == [second]
    stricter = task.model_copy(update={"checks": {**task.checks, "answer_includes_all": ["SNOMED"]}})
    [(trace, result)] = rescore([second], [stricter])
    assert not result.success and trace.success is False and trace.score["checks"]["answer_includes_all"] is False


class FlakyBackend(ScriptedBackend):
    """Scripted, except that the first `fail` LLM calls raise the OpenAI SDK's real timeout error."""

    def __init__(self, items, fail):
        super().__init__(items, strict_tools=False)
        self.fail = fail

    def _chat(self, messages, tools, opts):
        if self.fail:
            import httpx
            import openai

            self.fail -= 1
            raise openai.APITimeoutError(request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1"))
        return super()._chat(messages, tools, opts)


@pytest.mark.parametrize("fail, retries, runs, passed", [(1, 1, 2, True), (2, 1, 2, False), (1, 0, 1, False)])
async def test_timeouts_are_retried(hub, tasks, heart_attack, tmp_path, fail, retries, runs, passed):
    from toolhelper.experiments.runner import is_transient

    task = tasks["omop-001"]
    llm = FlakyBackend(steps(("omop__find_omop_concept", heart_attack), "concept_id 4329847 (SNOMED)"), fail)
    db = MonitorDB(tmp_path / "m.sqlite")
    async with hub:
        results = await run_benchmark([task], Agent(llm, hub, counter=estimate_tokens, group_id="g"),
                                      traces_dir=tmp_path, db=db, retry_timeouts=retries)
    [(trace, result)] = results  # one final result per task
    assert result.success is passed
    saved = db.rows("SELECT run_id FROM runs WHERE group_id = 'g'")
    assert len(saved) == runs  # the timed-out run is kept, not overwritten
    if runs == 2:
        assert trace.retry_of and trace.retry_of != trace.run_id and trace.retry_of in {r["run_id"] for r in saved}
    assert is_transient(trace) is (not passed)


def test_only_api_failures_count_as_transient():
    from toolhelper.experiments.runner import is_transient

    class T:
        def __init__(self, stop, error):
            self.stop_reason, self.error = stop, error

    assert is_transient(T("error", "APITimeoutError: Request timed out."))
    assert is_transient(T("error", "APIConnectionError: Connection error."))
    assert is_transient(T("error", "NotFoundError: Error code: 404"))  # the model endpoint was down
    assert not is_transient(T("error", "NotFoundError: wref target 'x' not found"))
    assert not is_transient(T("error", "ReplayMiss: no recording for ..."))
    assert not is_transient(T("max_turns", None))
