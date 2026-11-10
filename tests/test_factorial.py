"""Phase 10, second iteration: the fractional factorial design, its effect estimates and the
plan end to end. Effect tests use small constructed outcome tables (analysis inputs, not server
data); the orchestrator test runs real recorded responses through the scripted backend."""

import json
import math

import anyio
import pytest
from conftest import final, recorded, tool_call

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent, AgentConfig
from toolhelper.context.local_store import LocalStore
from toolhelper.efficiency.registry import Inputs, build_strategies
from toolhelper.experiments.experiment import write_experiment
from toolhelper.experiments.factorial import (
    Factor,
    FactorialConfig,
    FactorialOrchestrator,
    RunMetrics,
    alias_pairs,
    cells,
    choose,
    factor_effects,
    factorial_loop,
    grounded_tasks,
    load_config,
)
from toolhelper.experiments.runner import run_benchmark
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend
from toolhelper.monitoring.db import MonitorDB

GATE = {"max_success_drop": 0.02, "min_trust": 0.9}


def config(factors, generators=None, **kw):
    return FactorialConfig(**{"experiment": "t10b", "llm": "scripted", "parallel": 2, "learn_from": ["learn"],
                              "factors": factors, "generators": generators or {}, "confirm_repeats": 2, "gate": GATE,
                              "bootstrap": {"samples": 300, "seed": 7}, **kw})


FOUR = {"A": Factor(specs=["E2:level=minimal"]), "B": Factor(specs=["E6:mode=budget,budget=2000"]),
        "C": Factor(specs=["E9"], max_turns=6), "D": Factor(specs=["E4:mode=repair"])}


def test_half_fraction_is_balanced_orthogonal_and_aliased_in_pairs():
    cfg = config(FOUR, {"D": "ABC"})
    cs = cells(cfg)
    assert len(cs) == 8 and len({c.name for c in cs}) == 8
    assert {c.name for c in cs} == {"none", "AD", "BD", "AB", "CD", "AC", "BC", "ABCD"}
    for f in "ABCD":
        assert sum(c.on[f] for c in cs) == 4
    for f, g in (("A", "B"), ("A", "D"), ("C", "D")):
        assert sorted((c.on[f], c.on[g]) for c in cs).count((True, True)) == 2  # orthogonal
    assert alias_pairs(cfg) == [("AB", "CD"), ("AC", "BD"), ("AD", "BC")]
    abcd = next(c for c in cs if c.name == "ABCD")
    assert abcd.specs == ["E2:level=minimal", "E6:mode=budget,budget=2000", "E9", "E4:mode=repair"]
    assert abcd.max_turns == 6 and next(c for c in cs if c.name == "AB").max_turns is None
    full = cells(config({"A": Factor(specs=["E2"]), "B": Factor(specs=["E9"])}))
    assert [c.name for c in full] == ["none", "B", "A", "AB"]


TASKS = [f"t{i:02d}" for i in range(24)]


def table(cfg, tokens, success=lambda cell, i: True, trust=lambda cell: 1.0):
    """cell name -> task -> RunMetrics, from per-cell rules."""
    out = {}
    for c in cells(cfg):
        out[c.name] = {t: RunMetrics(run_id=f"{c.name}-{t}", task_id=t, success=success(c, i),
                                     tokens=tokens(c, i), trust=trust(c), turns=3, first_prompt=5000 - 1000 * c.on["A"],
                                     carried=20_000 * (0.4 if c.on["B"] else 1.0), errors=0, expands=0)
                       for i, t in enumerate(TASKS)}
    return out


def test_effects_recover_known_contrasts_and_the_choice_rule_applies_the_gate():
    cfg = config(FOUR, {"D": "ABC"})
    base = [10_000 + 4_000 * (i % 5) for i in range(len(TASKS))]

    def tokens(c, i):  # additive savings: A 1,000 tokens per run, B 3,000, D 2,000; C none
        return base[i] - 1000 * c.on["A"] - 3000 * c.on["B"] - 2000 * c.on["D"]

    runs = table(cfg, tokens, success=lambda c, i: not (c.on["D"] and i % 4 == 0),  # D loses a quarter of tasks
                 trust=lambda c: 0.85 if c.on["C"] else 0.97)
    eff = factor_effects(runs, cells(cfg), TASKS, samples=300, seed=7)
    mean = sum(base) / len(base)  # the other factors average out: each is on in half of either side
    assert math.isclose(eff["A"]["tokens_rel"], -1000 / (mean - 2500), rel_tol=1e-9)
    assert math.isclose(eff["B"]["tokens_rel"], -3000 / (mean - 1500), rel_tol=1e-9)
    assert eff["C"]["tokens_rel"] == 0 and eff["C"]["tokens_rel_ci"] == [0.0, 0.0]
    assert eff["B"]["tokens_rel_ci"][1] < 0 and eff["B"]["tokens_gm"] < 0
    assert eff["D"]["success_pts"] == -0.25 and eff["A"]["success_pts"] == 0
    assert eff["A"]["mediators"]["first_prompt"] == -1000 and eff["B"]["mediators"]["carried"] == -12_000
    assert eff["C"]["trust_on"] == 0.85 and eff["A"]["tasks"] == 24

    chosen, reasons = choose(eff, GATE)
    assert chosen == ["A", "B"] and reasons["A"].startswith("on:")
    assert reasons["C"].startswith("off: tokens/success") and "includes 0" in reasons["C"]
    assert reasons["D"].startswith("off:") and "success -25.0 points" in reasons["D"]


def test_runs_lost_to_api_failures_are_left_out():
    cfg = config({"A": Factor(specs=["E2"]), "B": Factor(specs=["E9"])})
    runs = table(cfg, lambda c, i: 10_000 if c.on["A"] else 20_000)
    runs["A"]["t00"] = runs["A"]["t00"].model_copy(update={"transient": True, "tokens": 999_999})
    del runs["AB"]["t01"]  # a run that never happened
    eff = factor_effects(runs, cells(cfg), TASKS, samples=100, seed=1)
    assert math.isclose(eff["A"]["tokens_rel"], -0.5) and eff["A"]["tasks"] == 24


def test_config_file_is_the_preregistered_design():
    cfg = load_config("configs/experiments-p10b.yaml")
    assert cfg.design == "factorial" and cfg.generators == {"D": "ABC"} and len(cells(cfg)) == 8
    assert cfg.learn_from[0] == "bench-baseline-record" and cfg.confirm_repeats == 3
    assert cfg.base.specs == [] and cfg.report_subsets == []


E3_MODES = {"H": Factor(specs=["E3:modes=hints"]), "T": Factor(specs=["E3:modes=tool"]),
            "R": Factor(specs=["E3:modes=rewrite"])}
ABC = Factor(specs=["E2:level=minimal", "E6:mode=budget,budget=2000", "E9"], max_turns=6)


def test_base_config_runs_in_every_cell_and_modes_of_one_strategy_merge():
    cs = {c.name: c for c in cells(config(E3_MODES, base=ABC))}
    assert len(cs) == 8
    assert cs["none"].specs == ABC.specs and cs["none"].max_turns == 6
    assert cs["H"].specs == [*ABC.specs, "E3:modes=hints"]
    assert cs["HTR"].specs == [*ABC.specs, "E3:modes=hints+tool+rewrite"] and cs["HTR"].max_turns == 6
    assert cs["TR"].specs[-1] == "E3:modes=tool+rewrite"
    for c in cs.values():
        built = build_strategies(c.specs, Inputs(index=object(), traces=[LimitRun()]))
        assert len(built) == 3 + ("E3" in " ".join(c.specs))
    clash = config({"A": Factor(specs=["E2:level=light"]), "B": Factor(specs=["E2:level=minimal"])})
    with pytest.raises(ValueError, match="E2"):
        cells(clash)


def test_e3_config_file_puts_the_modes_on_top_of_the_accepted_config():
    cfg = load_config("configs/experiments-p10c.yaml")
    assert cfg.base.specs == ["E2:level=minimal", "E6:mode=budget,budget=2000", "E9"] and cfg.base.max_turns == 6
    assert set(cfg.factors) == {"H", "T", "R"} and cfg.generators == {} and len(cells(cfg)) == 8
    assert cfg.learn_from == load_config("configs/experiments-p10b.yaml").learn_from  # ABC behaves as in p10b
    assert cfg.report_subsets == ["grounded"]


def test_grounded_tasks_are_those_whose_prompt_names_a_dataset_entity(dataset_index):
    tasks = [Task(id="g", prompt="Pull the FDA label details for Zoloft."), Task(id="n", prompt="List the tools.")]
    assert grounded_tasks(tasks, dataset_index) == ["g"]


# --- the plan end to end ------------------------------------------------------------------------


class LimitRun:  # what E9 learns from: a learning-group run that hit max_turns
    run_id, stop_reason, tool_calls = "learn-1", "max_turns", []


class Done:
    returncode = 0

    def poll(self):
        return 0


PLAN_TASKS = [Task(id=f"omop-{i:03d}", servers=["omop"], category="condition_mapping",
                   prompt="Map 'heart attack' to condition_occurrence.condition_concept_id.",
                   checks={"must_call_any": ["omop__find_omop_concept"], "answer_includes_any": ["4329847"]})
              for i in range(1, 7)]


def scripted_launcher(server_configs, catalog, store, db, tmp_path, launched):
    """A `launch` that runs each pass in-process: the scripted backend on a recorded real call."""
    from toolhelper.mcp.hub import Hub

    heart = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")

    def launch(cmd, log):
        group = cmd[cmd.index("--group") + 1]
        only = cmd[cmd.index("--only") + 1].split(",")
        specs = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--strategy"]
        max_turns = int(cmd[cmd.index("--max-turns") + 1])
        launched.append((group, tuple(specs), max_turns, cmd[cmd.index("--mode") + 1], len(only)))
        assert cmd[cmd.index("--learn-from") + 1] == "learn,learn2"

        async def go():
            hub = Hub(server_configs, mode="replay", store=store, catalog=catalog, live_fallback=False)
            async with hub:
                strategies = build_strategies(specs, Inputs(traces=[LimitRun()]))
                for t in [t for t in PLAN_TASKS if t.id in only]:
                    llm = ScriptedBackend([tool_call("omop__find_omop_concept", heart),
                                           final("Myocardial infarction 4329847.")])
                    agent = Agent(llm, hub, strategies, config=AgentConfig(max_turns=max_turns),
                                  counter=estimate_tokens, group_id=group)
                    await run_benchmark([t], agent, traces_dir=tmp_path / "traces", db=db)

        anyio.run(go)
        return Done()

    return launch


@pytest.mark.parametrize("with_base", [False, True])
def test_factorial_plan_end_to_end(server_configs, catalog, store, tmp_path, with_base):
    db, launched = MonitorDB(tmp_path / "m.sqlite"), []
    launch = scripted_launcher(server_configs, catalog, store, db, tmp_path, launched)
    base = Factor(specs=["E7"], max_turns=5) if with_base else Factor()
    cfg = config({"A": Factor(specs=["E2:level=aggressive"]), "B": Factor(specs=["E9"], max_turns=4)},
                 learn_from=["learn", "learn2"], base=base)
    ctx = LocalStore()
    write_experiment(ctx, cfg, {"catalog_hash": "x"}, {"A": [], "B": []})
    subsets = {"grounded": [t.id for t in PLAN_TASKS[:4]]} if with_base else None
    orch = FactorialOrchestrator(cfg, PLAN_TASKS, db, tmp_path / "work", launch=launch, poll_s=0, log=lambda m: None,
                                 subsets=subsets)

    orch.step_factorial()
    assert sorted(g for g, *_ in launched) == ["t10b-f-A-r1", "t10b-f-AB-r1", "t10b-f-B-r1", "t10b-f-none-r1"]
    off_limit = 5 if with_base else 8
    assert {(g, t) for g, _, t, _, _ in launched} >= {("t10b-f-B-r1", 4), ("t10b-f-none-r1", off_limit)}
    assert all(specs[:1] == ("E7",) for _, specs, *_ in launched) or not with_base
    orch.step_factorial()  # resumable
    assert len(launched) == 4

    orch.step_effects()
    eff = orch.state["effects"]
    assert set(orch.state["effects_by_kind"]) == {"omop"} and orch.state["effects_by_kind"]["omop"]["A"]["tasks"] == 6
    if with_base:
        assert orch.state["effects_by_subset"]["grounded"]["A"]["tasks"] == 4
    else:
        assert "effects_by_subset" not in orch.state
    assert eff["A"]["tokens_rel"] < -0.05 and eff["A"]["success_pts"] == 0  # compression shrinks every prompt
    assert orch.state["chosen"]["factors"] == ["A"]  # B (E9) changes nothing in runs that finish
    assert orch.state["chosen"]["specs"] == ([*base.specs, "E2:level=aggressive"])

    orch.step_confirm()
    assert sorted(g for g, *_ in launched[4:]) == ["t10b-f-A-r2", "t10b-f-none-r2"]
    confirm = orch.state["confirm"]
    assert confirm["gate"]["passed"] and confirm["vs_baseline"]["delta_tokens_per_success"] < 0
    assert confirm["arms"] == {"chosen": ["t10b-f-A-r1", "t10b-f-A-r2"], "baseline": ["t10b-f-none-r1", "t10b-f-none-r2"]}

    wref = orch.step_decision(ctx)
    decision = ctx.get(wref)
    assert decision.name == "decisions/t10b" and decision.data["status"] == "proposed"
    assert all(e.startswith("runsummaries/t10b-f-") for e in decision.data["evidence"])
    lines = json.loads(decision.data["chosen_config_json"])["decisions"]
    assert any(line.startswith("A (E2:level=aggressive) on") for line in lines)
    assert any(r["config"] == "B" for r in json.loads(decision.data["rejected_json"]))

    orch.step_live()
    live = [x for x in launched if x[3] == "live"]
    assert sorted(g for g, *_ in live) == ["t10b-live-baseline", "t10b-live-chosen"]
    live_base = next(x for x in live if x[0] == "t10b-live-baseline")
    assert live_base[1:3] == (tuple(base.specs), off_limit)  # the live baseline is the base config
    assert set(orch.state["live"]["live"]) == {"t10b-live-baseline", "t10b-live-chosen"}


@pytest.mark.parametrize("factor, saves", [("A", True), ("B", False)])
def test_confirm_only_plan_tests_a_preregistered_lead_on_fresh_runs(server_configs, catalog, store, tmp_path,
                                                                    factor, saves):
    db, launched = MonitorDB(tmp_path / "m.sqlite"), []
    launch = scripted_launcher(server_configs, catalog, store, db, tmp_path, launched)
    cfg = config({"A": Factor(specs=["E2:level=aggressive"]), "B": Factor(specs=["E9"], max_turns=4)},
                 learn_from=["learn", "learn2"], confirm_only=[factor], require_saving=True)
    ctx = LocalStore()
    write_experiment(ctx, cfg, {"catalog_hash": "x"}, {"A": [], "B": []})
    orch = FactorialOrchestrator(cfg, PLAN_TASKS, db, tmp_path / "work", launch=launch, poll_s=0, log=lambda m: None)

    factorial_loop(orch, ctx)
    replay = sorted(g for g, _, _, mode, _ in launched if mode != "live")
    assert replay == [f"t10b-f-{factor}-r1", f"t10b-f-{factor}-r2", "t10b-f-none-r1", "t10b-f-none-r2"]  # no factorial
    gate = orch.state["confirm"]["gate"]
    assert gate["passed"] is saves
    assert orch.state["chosen"]["factors"] == ([factor] if saves else [])
    if not saves:  # E9 changes nothing in runs that finish: tokens/success CI includes 0
        assert any("tokens/success" in r for r in gate["reasons"])
    live = sorted(g for g, _, _, mode, _ in launched if mode == "live")
    assert live == (["t10b-live-baseline", "t10b-live-chosen"] if saves else ["t10b-live-baseline"])
    assert ctx.get("decisions/t10b").data["status"] == "proposed"
    with pytest.raises(ValueError, match="confirm_only"):
        FactorialOrchestrator(config({"A": Factor(specs=["E2"])}, confirm_only=["Z"]), PLAN_TASKS, db,
                              tmp_path / "w2", launch=launch, poll_s=0, log=lambda m: None)
