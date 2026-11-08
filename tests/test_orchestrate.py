"""The Phase 10 orchestrator end to end, with passes run in-process (scripted model, recorded
responses) instead of as `toolhelper bench run` subprocesses."""

import anyio
import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.context.local_store import LocalStore
from toolhelper.efficiency.registry import Inputs, build_strategies
from toolhelper.experiments.decision import curate, write_decision
from toolhelper.experiments.experiment import ExperimentConfig, write_experiment
from toolhelper.experiments.orchestrate import Orchestrator, command, needs_cross_fit
from toolhelper.experiments.runner import run_benchmark
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend
from toolhelper.monitoring.db import MonitorDB

TASKS = [Task(id=f"omop-{i:03d}", servers=["omop"], category="condition_mapping",
              prompt="Map 'heart attack' to condition_occurrence.condition_concept_id.",
              checks={"must_call_any": ["omop__find_omop_concept"], "answer_includes_any": ["4329847"]})
         for i in range(1, 5)]


def config(**kw):
    return ExperimentConfig(experiment="t10", llm="scripted", parallel=2, learn_from="learn", baseline_repeats=2,
                            close_call_repeats=2, finalist_repeats=2,
                            arms={"E2": ["E2:level=aggressive"], "E9": ["E9"]},
                            e3_grid={"datasets": [], "modes": []}, gate={"max_success_drop": 0.02, "min_trust": 0.9},
                            greedy={"min_gain": 0.03}, bootstrap={"samples": 200, "seed": 7}, **kw)


class LimitRun:  # what E9 learns from: a learning-group run that hit max_turns
    run_id, stop_reason, tool_calls = "learn-1", "max_turns", []


class Done:
    returncode = 0

    def poll(self):
        return 0


def test_command_line():
    cfg = config()
    cmd = command(cfg, __import__("toolhelper.experiments.orchestrate", fromlist=["Pass"]).Pass(
        group="g", specs=["E4:mode=repair"], tasks=["a"], cross_fit=True), ["a", "b"], "A")
    assert cmd[:5] == ["uv", "run", "toolhelper", "bench", "run"] and cmd[-2:] == ["--fold", "A"]
    assert "--strategy" in cmd and "E4:mode=repair" in cmd and "--only" in cmd and "a,b" in cmd
    assert needs_cross_fit(["E4"]) and not needs_cross_fit(["E2", "E9"])


def test_orchestrator_runs_the_plan(server_configs, catalog, store, tmp_path):
    from toolhelper.mcp.hub import Hub

    heart = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")
    db = MonitorDB(tmp_path / "m.sqlite")
    launched = []

    def launch(cmd, log):
        group = cmd[cmd.index("--group") + 1]
        only = cmd[cmd.index("--only") + 1].split(",")
        specs = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--strategy"]
        launched.append((group, tuple(specs), len(only)))

        async def go():
            hub = Hub(server_configs, mode="replay", store=store, catalog=catalog, live_fallback=False)
            async with hub:
                strategies = build_strategies(specs, Inputs(traces=[LimitRun()]))
                for t in [t for t in TASKS if t.id in only]:
                    llm = ScriptedBackend([tool_call("omop__find_omop_concept", heart),
                                           final("Myocardial infarction 4329847.")])
                    agent = Agent(llm, hub, strategies, counter=estimate_tokens, group_id=group)
                    await run_benchmark([t], agent, traces_dir=tmp_path / "traces", db=db)

        anyio.run(go)
        return Done()

    ctx = LocalStore()
    cfg = config()
    write_experiment(ctx, cfg, {"catalog_hash": "x"}, {"A": [], "B": []})
    orch = Orchestrator(cfg, TASKS, db, tmp_path / "work", launch=launch, poll_s=0, log=lambda m: None)
    orch.step_baseline()
    assert [g for g, _, _ in launched] == ["t10-baseline-r1", "t10-baseline-r2"]
    orch.step_baseline()  # resumable: complete passes are not run again
    assert len(launched) == 2

    orch.step_singles()
    e2 = orch.state["singles"]["E2"]
    assert e2["gate"]["passed"] and e2["vs_baseline"]["delta_success"] == 0
    assert e2["vs_baseline"]["delta_tokens_per_success"] < -0.05  # aggressive compression shrinks every prompt

    orch.step_greedy()
    assert orch.state["greedy"][0]["add"] == "E2" and orch.state["chosen"]["strategies"][0] == "E2"
    orch.step_finalists()
    assert len(orch.state["chosen"]["groups"]) == 2

    wref = write_decision(ctx, orch)
    decision = ctx.get(wref)
    assert decision.data["status"] == "proposed" and decision.data["evidence"]
    assert all(e.startswith("runsummaries/t10-") for e in decision.data["evidence"])
    assert ctx.get("runsummaries/t10-baseline-r1").data["runs"] == 4
    assert "accepted by tester" in curate(ctx, "decisions/t10", "tester")
    assert ctx.get("decisions/t10").data["status"] == "accepted"
    assert any("E2 alone" in line for line in __import__("json").loads(decision.data["chosen_config_json"])["decisions"])
