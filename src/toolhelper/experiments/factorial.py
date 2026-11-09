"""Phase 10, second iteration (p10b): a two-level fractional factorial over strategy variants
aimed at the measured cost centres, with task-paired inference and a confirmation stage.

Why the first iteration (p10) could not show an effect, from its own runs (2026-11-08):

- Tokens went to tool results carried across turns (51%) and to tool definitions re-sent on
  every turn (44%); the variants it tested barely touched either. E2 `light` saved exactly 635
  measured tokens per turn (turn-1 prompts are deterministic), about 6% of a run; E6
  `projection` dropped unused fields of three tools; E8 `after_use` hid facts still needed;
  E1 dropped needed tools.
- One run per arm could not see effects that small: the per-task run-to-run spread is 0.29 in
  log tokens, so with 54 tasks the smallest change detectable with 80% power is ~14% with one
  run per arm and ~8.5% with three. Greedy steps compared single noisy passes; the -14%
  single-arm E2 result was noise (its mechanism caps it at ~6%).

Design: factors are strategy specs plus an optional turn limit; `generators` define the
fraction (D = ABC gives the resolution IV half of 2^4: main effects are clear of two-factor
interactions, which are aliased in pairs). Each cell runs every task once, so each main effect
compares half of all runs with the other half, paired by task: as precise as four repeats per
arm in a one-at-a-time design. Passes run in a seeded random order, `parallel` at a time.

Effects per factor, runs with it on vs off (per task, the mean over the cells on each side):
relative change in mean tokens per run (ratio of sums), the geometric-mean ratio (per-task log
ratios, so heavy tasks do not dominate), success points, tokens per success, mean trust of the
runs with it on, and its mediators (first-turn prompt tokens, i.e. definitions; carried result
tokens; turns; tool errors; expands). 95% CIs come from a paired bootstrap over tasks.

Choice (pre-registered): a factor is on when its tokens-per-success CI lies below zero, its
success effect is >= -max_success_drop and runs with it on have mean trust >= min_trust. The
chosen config is confirmed against the baseline with `confirm_repeats` paired repeats each (the
p10 gate), with one fallback (drop the chosen factor costing most success) if it fails, and
checked live on one fold.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from toolhelper.config import ROOT
from toolhelper.experiments.analysis import TRANSIENT, _pct, summarize
from toolhelper.experiments.orchestrate import Orchestrator, Pass, needs_cross_fit

MEDIATORS = ("first_prompt", "carried", "turns", "errors", "expands")
EXPAND = "toolhelper__expand_result"


class Factor(BaseModel):
    specs: list[str] = Field(default_factory=list)
    max_turns: int | None = None  # a turn limit the factor sets (E9's limit variant)
    note: str = ""


class FactorialConfig(BaseModel):
    design: Literal["factorial"] = "factorial"
    experiment: str
    llm: str
    mode: str = "replay"
    max_turns: int = 8
    parallel: int = 3
    learn_from: list[str]
    base: Factor = Field(default_factory=Factor)  # on in every cell, the baseline and the live baseline
    factors: dict[str, Factor]
    generators: dict[str, str] = Field(default_factory=dict)
    confirm_repeats: int = 3
    live_fold: str = "A"
    order_seed: int = 7
    gate: dict[str, float]
    bootstrap: dict[str, int]
    report_subsets: list[Literal["grounded"]] = Field(default_factory=list)  # task subsets with their own effects


def load_config(path: Path | str) -> FactorialConfig:
    path = Path(path)
    return FactorialConfig.model_validate(yaml.safe_load((path if path.is_absolute() else ROOT / path)
                                                        .read_text(encoding="utf-8")))


# --- the design -----------------------------------------------------------------------------


class Cell(BaseModel):
    name: str  # the factors that are on, e.g. "AD", or "none"
    on: dict[str, bool]
    specs: list[str]
    max_turns: int | None = None


def merge_specs(specs: list[str]) -> list[str]:
    """One spec per strategy: specs of the same strategy merge their parameters, and `modes`
    values join with `+` (`E3:modes=hints` + `E3:modes=tool` -> `E3:modes=hints+tool`)."""
    merged: dict[str, dict[str, str]] = {}
    for spec in specs:
        sid, _, rest = spec.partition(":")
        params = merged.setdefault(sid.strip().upper(), {})
        for pair in filter(None, rest.split(",")):
            key, _, value = pair.partition("=")
            key, value = key.strip(), value.strip()
            if key == "modes" and key in params:
                params[key] += "+" + value
            elif params.get(key, value) != value:
                raise ValueError(f"{sid}: conflicting {key} ({params[key]} vs {value})")
            else:
                params[key] = value
    return [sid + (":" + ",".join(f"{k}={v}" for k, v in p.items()) if p else "") for sid, p in merged.items()]


def make_cell(cfg: FactorialConfig, on: dict[str, bool]) -> Cell:
    specs = merge_specs([*cfg.base.specs, *(s for f, v in on.items() if v for s in cfg.factors[f].specs)])
    limits = [lim for lim in [cfg.base.max_turns, *(cfg.factors[f].max_turns for f, v in on.items() if v)] if lim]
    return Cell(name="".join(f for f, v in on.items() if v) or "none", on=on, specs=specs,
                max_turns=min(limits) if limits else None)


def grounded_tasks(tasks, index) -> list[str]:
    """Tasks whose prompt names an entity E3 can resolve in any dataset (the tasks it can act on)."""
    from toolhelper.efficiency.grounding import Grounding

    g = Grounding(index)
    return [t.id for t in tasks if g.entities(t.prompt)]


def cells(cfg: FactorialConfig) -> list[Cell]:
    """Full factorial over the non-generated factors; each generated factor is on where the
    product of its word's signs (+1 on, -1 off) is +1."""
    letters = list(cfg.factors)
    base = [f for f in letters if f not in cfg.generators]
    out = []
    for bits in itertools.product([False, True], repeat=len(base)):
        on = dict(zip(base, bits, strict=True))
        for f, word in cfg.generators.items():
            on[f] = math.prod(1 if on[g] else -1 for g in word) > 0
        out.append(make_cell(cfg, {f: on[f] for f in letters}))
    return out


def interaction_groups(cfg: FactorialConfig) -> list[tuple[str, ...]]:
    """Two-factor interactions grouped by identical contrast columns (aliases share a group)."""
    cs = cells(cfg)
    cols: dict[tuple[bool, ...], list[str]] = {}
    for i, j in itertools.combinations(cfg.factors, 2):
        cols.setdefault(tuple(c.on[i] == c.on[j] for c in cs), []).append(i + j)
    return sorted(tuple(v) for v in cols.values())


def alias_pairs(cfg: FactorialConfig) -> list[tuple[str, ...]]:
    return [g for g in interaction_groups(cfg) if len(g) > 1]


# --- per-run metrics ------------------------------------------------------------------------


class RunMetrics(BaseModel):
    run_id: str
    task_id: str
    success: bool
    tokens: int  # prompt + completion, measured
    trust: float | None = None
    transient: bool = False
    turns: int = 0
    first_prompt: int = 0  # measured prompt tokens of the first LLM call: system, task, definitions
    carried: float = 0  # estimated result tokens summed over the later calls that re-send them
    errors: int = 0  # failed or rejected hub calls
    expands: int = 0


def load_metrics(db, group: str) -> dict[str, RunMetrics]:
    """Latest run per task in `group`, with its mediators."""
    import json

    latest = {r["task_id"]: r for r in db.rows(
        "SELECT run_id, task_id, success, prompt_tokens, completion_tokens, trust_json, error, turns FROM runs "
        "WHERE group_id = ? ORDER BY started_at", (group,))}
    out = {}
    for task, r in latest.items():
        first = db.rows("SELECT prompt_tokens FROM llm_calls WHERE run_id = ? ORDER BY turn LIMIT 1", (r["run_id"],))
        calls = db.rows("SELECT name, server, is_error, carried_tokens FROM tool_calls WHERE run_id = ?", (r["run_id"],))
        out[task] = RunMetrics(
            run_id=r["run_id"], task_id=task, success=bool(r["success"]),
            tokens=(r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0),
            trust=(json.loads(r["trust_json"]) or {}).get("trust") if r["trust_json"] else None,
            transient=(r["error"] or "").startswith(TRANSIENT), turns=r["turns"] or 0,
            first_prompt=first[0]["prompt_tokens"] if first else 0,
            carried=sum(c["carried_tokens"] or 0 for c in calls),
            errors=sum(bool(c["is_error"]) for c in calls if c["server"] and c["server"] != "toolhelper"),
            expands=sum(c["name"] == EXPAND for c in calls))
    return out


# --- effects --------------------------------------------------------------------------------

Runs = dict[str, dict[str, RunMetrics]]  # cell name -> task -> run


def _side(runs: Runs, cs: list[Cell], tasks: list[str], pick: Callable[[Cell], bool]) -> dict[str, dict[str, float]]:
    """task -> mean of each metric over the picked cells' runs (API failures left out)."""
    out = {}
    for t in tasks:
        rs = [r for c in cs if pick(c) and (r := runs.get(c.name, {}).get(t)) is not None and not r.transient]
        if rs:
            out[t] = {"tokens": statistics.fmean(r.tokens for r in rs),
                      "success": statistics.fmean(r.success for r in rs),
                      **{m: statistics.fmean(getattr(r, m) for r in rs) for m in MEDIATORS}}
    return out


def _ci(values: list[float]) -> list[float] | None:
    return [round(_pct(values, 0.025), 4), round(_pct(values, 0.975), 4)] if values else None


def _contrast(on: dict[str, dict[str, float]], off: dict[str, dict[str, float]], tasks: list[str],
              samples: int, seed: int) -> dict[str, Any]:
    paired = [t for t in tasks if t in on and t in off]
    if not paired:
        return {"tasks": 0}

    def stats(ids: list[str]) -> tuple[float | None, float | None, float, float | None]:
        ton, toff = sum(on[t]["tokens"] for t in ids), sum(off[t]["tokens"] for t in ids)
        son, soff = sum(on[t]["success"] for t in ids), sum(off[t]["success"] for t in ids)
        logs = [math.log(on[t]["tokens"] / off[t]["tokens"]) for t in ids if on[t]["tokens"] and off[t]["tokens"]]
        return (ton / toff - 1 if toff else None, math.exp(statistics.fmean(logs)) - 1 if logs else None,
                (son - soff) / len(ids), (ton / son) / (toff / soff) - 1 if son and soff and toff else None)

    rel, gm, pts, tps = stats(paired)
    rng = random.Random(seed)
    boot: list[list[float]] = [[], [], [], []]
    for _ in range(samples):
        for k, v in enumerate(stats([rng.choice(paired) for _ in paired])):
            if v is not None:
                boot[k].append(v)
    return {"tasks": len(paired), "tokens_rel": rel, "tokens_rel_ci": _ci(boot[0]), "tokens_gm": gm,
            "tokens_gm_ci": _ci(boot[1]), "success_pts": pts, "success_pts_ci": _ci(boot[2]), "tps_rel": tps,
            "tps_rel_ci": _ci(boot[3]),
            "mediators": {m: round(statistics.fmean(on[t][m] - off[t][m] for t in paired), 3) for m in MEDIATORS}}


def factor_effects(runs: Runs, cs: list[Cell], tasks: list[str], *, samples: int = 2000,
                   seed: int = 7) -> dict[str, dict[str, Any]]:
    out = {}
    for f in cs[0].on:
        e = _contrast(_side(runs, cs, tasks, lambda c, f=f: c.on[f]),
                      _side(runs, cs, tasks, lambda c, f=f: not c.on[f]), tasks, samples, seed)
        trusts = [r.trust for c in cs if c.on[f] for r in runs.get(c.name, {}).values()
                  if r.task_id in tasks and not r.transient and r.trust is not None]
        out[f] = {**e, "trust_on": statistics.mean(trusts) if trusts else None}
    return out


def interaction_effects(runs: Runs, cs: list[Cell], groups: list[tuple[str, ...]], tasks: list[str], *,
                        samples: int = 2000, seed: int = 7) -> dict[str, dict[str, Any]]:
    """Each two-factor interaction (or alias group): cells where both factors agree vs differ."""
    out = {}
    for g in groups:
        i, j = g[0]
        out["=".join(g)] = _contrast(_side(runs, cs, tasks, lambda c: c.on[i] == c.on[j]),
                                     _side(runs, cs, tasks, lambda c: c.on[i] != c.on[j]), tasks, samples, seed)
    return out


def _pc(x: float | None, scale: float = 1.0, fmt: str = "+.1%") -> str:
    return "n/a" if x is None else format(x * scale, fmt)


def describe(e: dict[str, Any], *, mediators: bool = True) -> str:
    """One line with an effect's numbers and CIs."""
    if not e.get("tasks"):
        return "no paired tasks"
    line = (f"tokens/run {_pc(e['tokens_rel'])} {e['tokens_rel_ci']}, geomean {_pc(e['tokens_gm'])} "
            f"{e['tokens_gm_ci']}, success {_pc(e['success_pts'], 100, '+.1f')} points {e['success_pts_ci']}, "
            f"tokens/success {_pc(e['tps_rel'])} {e['tps_rel_ci']}")
    if mediators:
        line += f", trust {e.get('trust_on')}; per run: {e['mediators']}"
    return line


def choose(effects: dict[str, dict[str, Any]], gate: dict[str, float]) -> tuple[list[str], dict[str, str]]:
    """The pre-registered rule: on when tokens per success falls (CI below zero), success holds
    within the gate and trust stays above it. Returns the factors on and a reason per factor."""
    chosen, reasons = [], {}
    for f, e in effects.items():
        ci, why = e.get("tps_rel_ci"), []
        if e.get("tps_rel") is None or ci is None or ci[1] >= 0:
            where = "is above" if ci and ci[0] > 0 else "includes"
            why.append(f"tokens/success {(e.get('tps_rel') or 0):+.1%} (CI {ci}) {where} 0")
        if e.get("success_pts") is not None and e["success_pts"] < -gate["max_success_drop"]:
            why.append(f"success {e['success_pts'] * 100:+.1f} points (limit -{gate['max_success_drop'] * 100:.0f})")
        if e.get("trust_on") is not None and e["trust_on"] < gate["min_trust"]:
            why.append(f"trust {e['trust_on']:.3f} < {gate['min_trust']}")
        if why:
            reasons[f] = "off: " + "; ".join(why)
        else:
            chosen.append(f)
            reasons[f] = (f"on: tokens/success {e['tps_rel']:+.1%} (CI {ci}), success {e['success_pts'] * 100:+.1f} "
                          f"points, trust {e['trust_on']:.3f}" if e.get("trust_on") is not None else
                          f"on: tokens/success {e['tps_rel']:+.1%} (CI {ci}), success {e['success_pts'] * 100:+.1f} points")
    return chosen, reasons


# --- the plan -------------------------------------------------------------------------------


class FactorialOrchestrator(Orchestrator):
    """Runs the factorial, picks the config, confirms it, writes the Decision, checks it live.
    Resumable like the p10 orchestrator (state and pass logs in the work directory)."""

    def __init__(self, cfg: FactorialConfig, tasks, db, workdir: Path, *,
                 subsets: dict[str, list[str]] | None = None, **kw) -> None:
        super().__init__(cfg, tasks, db, workdir, **kw)
        self.cells = cells(cfg)
        self.baseline = make_cell(cfg, {f: False for f in cfg.factors})
        self.subsets = subsets or {}  # name -> task ids, effects reported separately (e.g. `grounded`)

    def group(self, cell: str, repeat: int = 1) -> str:
        return f"{self.cfg.experiment}-f-{cell}-r{repeat}"

    def baseline_groups(self) -> list[str]:
        return [self.group("none", r) for r in range(1, self.cfg.confirm_repeats + 1)]

    def cell_pass(self, cell: Cell, repeat: int = 1, mode: str | None = None) -> Pass:
        return Pass(group=self.group(cell.name, repeat), specs=cell.specs, tasks=list(self.tasks),
                    mode=mode or self.cfg.mode, cross_fit=needs_cross_fit(cell.specs), max_turns=cell.max_turns)

    def runs(self) -> Runs:
        return {c.name: load_metrics(self.db, self.group(c.name)) for c in self.cells}

    def step_factorial(self) -> None:
        passes = [self.cell_pass(c) for c in self.cells]
        random.Random(self.cfg.order_seed).shuffle(passes)
        self.run(passes)
        self.note(f"factorial: {len(self.cells)} cells x {len(self.tasks)} tasks done")

    def step_effects(self) -> None:
        runs, tasks, b = self.runs(), list(self.tasks), self.cfg.bootstrap
        eff = factor_effects(runs, self.cells, tasks, samples=b["samples"], seed=b["seed"])
        inter = interaction_effects(runs, self.cells, interaction_groups(self.cfg), tasks, samples=b["samples"],
                                    seed=b["seed"])
        chosen, reasons = choose(eff, self.cfg.gate)
        cell = make_cell(self.cfg, {f: f in chosen for f in self.cfg.factors})
        self.state["cells"] = {c.name: summarize([runs[c.name]], tasks) for c in self.cells}
        self.state["effects"], self.state["interactions"] = eff, inter
        self.state["effects_by_kind"] = {
            k: factor_effects(runs, self.cells, [t for t in tasks if self.kinds[t] == k], samples=b["samples"] // 4,
                              seed=b["seed"]) for k in sorted(set(self.kinds.values()))}
        self.state["chosen"] = {"factors": chosen, "cell": cell.name, "specs": cell.specs, "max_turns": cell.max_turns,
                                "reasons": reasons}
        for f, e in eff.items():
            self.note(f"{f} {self.cfg.factors[f].specs}: {describe(e)} -> {reasons[f]}")
        for g, e in inter.items():
            self.note(f"interaction {g}: {describe(e, mediators=False)}")
        if self.subsets:
            self.state["effects_by_subset"] = {
                name: factor_effects(runs, self.cells, [t for t in tasks if t in set(ids)], samples=b["samples"],
                                     seed=b["seed"]) for name, ids in self.subsets.items()}
        for k, effs in self.state["effects_by_kind"].items():
            self.note(f"{k}: " + "; ".join(f"{f} tokens/run {_pc(e.get('tokens_rel'))} {e.get('tokens_rel_ci')} success "
                                           f"{_pc(e.get('success_pts'), 100, '+.1f')} pts" for f, e in effs.items()))
        for name, effs in self.state.get("effects_by_subset", {}).items():
            for f, e in effs.items():
                self.note(f"{name} tasks only, {f}: {describe(e)}")
        self.note(f"chosen: {chosen or 'nothing (the baseline stays)'}")

    def step_confirm(self) -> None:
        chosen = self.state["chosen"]
        factors = list(chosen["factors"])
        while factors:
            cell = make_cell(self.cfg, {f: f in factors for f in self.cfg.factors})
            repeats = range(1, self.cfg.confirm_repeats + 1)
            arms = {"chosen": [self.group(cell.name, r) for r in repeats], "baseline": self.baseline_groups()}
            self.run([p for r in repeats for p in (self.cell_pass(cell, r), self.cell_pass(self.baseline, r))])
            result = self.evaluate(arms["chosen"])
            self.state["confirm"] = {"factors": factors, "cell": cell.name, "arms": arms, **result}
            self.note(f"confirm {cell.name} ({self.cfg.confirm_repeats} repeats vs baseline): {result['summary']}; "
                      f"{result['vs_baseline']}; gate {'pass' if result['gate']['passed'] else result['gate']['reasons']}")
            if result["gate"]["passed"]:
                break
            worst = min(factors, key=lambda f: self.state["effects"][f]["success_pts"])
            factors = [f for f in factors if f != worst] if len(factors) > 1 else []
            self.note(f"confirmation failed the gate; dropping {worst}")
        cell = make_cell(self.cfg, {f: f in factors for f in self.cfg.factors})
        chosen.update({"factors": factors, "cell": cell.name, "specs": cell.specs, "max_turns": cell.max_turns,
                       "groups": self.state.get("confirm", {}).get("arms", {}).get("chosen", []) if factors else []})
        self.save()

    def step_decision(self, store) -> str:
        from toolhelper.experiments.decision import write_factorial_decision

        wref = write_factorial_decision(store, self)
        self.state["decision"] = wref
        self.note(f"decision written: {wref} (proposed; approve with `toolhelper curate {wref.split('@')[0]}`)")
        return wref

    def step_live(self) -> None:
        fold, chosen = self.folds[self.cfg.live_fold], self.state["chosen"]
        b = self.baseline
        passes = [Pass(group=f"{self.cfg.experiment}-live-baseline", specs=b.specs, tasks=fold, mode="live",
                       cross_fit=needs_cross_fit(b.specs), max_turns=b.max_turns)]
        if chosen["factors"]:
            passes.append(Pass(group=f"{self.cfg.experiment}-live-chosen", specs=chosen["specs"], tasks=fold,
                               mode="live", cross_fit=needs_cross_fit(chosen["specs"]), max_turns=chosen["max_turns"]))
        self.run(passes)
        live = {p.group: summarize(self.arm([p.group]), fold) for p in passes}
        replay = {"baseline": summarize(self.arm(self.baseline_groups()), fold)}
        if chosen["factors"]:
            replay["chosen"] = summarize(self.arm(chosen["groups"]), fold)
        self.state["live"] = {"live": live, "replay_same_tasks": replay}
        self.note(f"live confirmation on fold {self.cfg.live_fold}: {live}; replay on the same tasks: {replay}")


def factorial_loop(orch: FactorialOrchestrator, store) -> None:
    orch.step_factorial()
    orch.step_effects()
    orch.step_confirm()
    orch.step_decision(store)
    orch.step_live()
    orch.note("experiment finished")
