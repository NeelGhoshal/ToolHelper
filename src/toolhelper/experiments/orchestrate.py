"""Runs the trimmed Phase 10 plan unattended and resumably.

Each step is a set of passes (one benchmark run per group, or two cross-fitted halves for
strategies that learn from runs). A pass is complete when every task has a run that did not
end in a transient API failure; on restart, complete passes are skipped and incomplete ones
re-run only their missing tasks. Up to `parallel` passes run at once.

Steps: baseline repeats -> each strategy alone -> close-call repeats -> E3 dataset x mode
grid -> greedy forward selection -> finalist repeats -> RunSummary + Decision things ->
live confirmation on fold A. Decisions and their numbers go to state.json and the report.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from toolhelper.agent.hooks import Task
from toolhelper.experiments.analysis import Arm, by_kind, compare, gate, group_runs, summarize
from toolhelper.experiments.experiment import ExperimentConfig, folds, kind

LEARNING = {"E1", "E4", "E5", "E6"}  # strategies whose behaviour is learned from earlier runs


class Pass(BaseModel):
    group: str
    specs: list[str] = Field(default_factory=list)
    tasks: list[str]
    mode: str = "replay"
    cross_fit: bool = False


def needs_cross_fit(specs: list[str]) -> bool:
    return any(s.split(":", 1)[0].upper() in LEARNING for s in specs)


def command(cfg: ExperimentConfig, p: Pass, tasks: list[str], fold: str | None) -> list[str]:
    cmd = ["uv", "run", "toolhelper", "bench", "run", "--llm", cfg.llm, "--mode", p.mode, "--group", p.group,
           "--retry-timeouts", "2", "--max-turns", str(cfg.max_turns), "--learn-from", cfg.learn_from,
           "--only", ",".join(tasks)]
    for s in p.specs:
        cmd += ["--strategy", s]
    if fold:
        cmd += ["--fold", fold]
    return cmd


class Orchestrator:
    def __init__(self, cfg: ExperimentConfig, tasks: list[Task], db, workdir: Path, *,
                 launch: Callable[[list[str], Path], Any] | None = None, poll_s: float = 30,
                 log: Callable[[str], None] = print) -> None:
        self.cfg, self.tasks, self.db, self.workdir = cfg, {t.id: t for t in tasks}, db, workdir
        self.folds = folds(tasks)
        self.kinds = {t.id: kind(t) for t in tasks}
        self.launch = launch or self._popen
        self.poll_s, self.log = poll_s, log
        workdir.mkdir(parents=True, exist_ok=True)
        self.state_path = workdir / "state.json"
        self.state: dict[str, Any] = json.loads(self.state_path.read_text(encoding="utf-8")) \
            if self.state_path.exists() else {}

    # --- running passes -----------------------------------------------------------------

    def _popen(self, cmd: list[str], log_path: Path):
        out = open(log_path, "ab")
        return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[3])

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def note(self, msg: str) -> None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        self.state.setdefault("log", []).append(f"{stamp} {msg}")
        self.save()
        self.log(msg)

    def missing(self, p: Pass) -> list[str]:
        done = group_runs(self.db, p.group, self.kinds)
        return [t for t in p.tasks if t not in done or done[t].transient]

    def run(self, passes: list[Pass], max_attempts: int = 3) -> None:
        """Runs passes to completion, `parallel` processes at a time."""
        queue: list[tuple[Pass, str | None, list[str], int]] = []
        for p in passes:
            todo = self.missing(p)
            if not todo:
                continue
            if p.cross_fit:
                for f in "AB":
                    part = [t for t in todo if t in set(self.folds[f])]
                    if part:
                        queue.append((p, f, part, 1))
            else:
                queue.append((p, None, todo, 1))
        running: list[tuple[Any, Pass, str | None, int]] = []
        while queue or running:
            while queue and len(running) < self.cfg.parallel:
                p, f, part, attempt = queue.pop(0)
                log = self.workdir / f"{p.group}{'-' + f if f else ''}.log"
                self.note(f"start {p.group}{' fold ' + f if f else ''}: {len(part)} tasks, {p.specs or 'no strategies'}"
                          f"{' (attempt %d)' % attempt if attempt > 1 else ''}")
                running.append((self.launch(command(self.cfg, p, part, f), log), p, f, attempt))
            time.sleep(self.poll_s)
            for item in list(running):
                proc, p, f, attempt = item
                if proc.poll() is None:
                    continue
                running.remove(item)
                left = [t for t in self.missing(p) if f is None or t in set(self.folds[f])]
                if left and attempt < max_attempts:
                    queue.append((p, f, left, attempt + 1))
                self.note(f"finished {p.group}{' fold ' + f if f else ''} (exit {proc.returncode}); "
                          f"{len(left)} tasks still missing")

    # --- analysis helpers -----------------------------------------------------------------

    def arm(self, groups: list[str]) -> Arm:
        return [group_runs(self.db, g, self.kinds) for g in groups]

    def baseline_groups(self) -> list[str]:
        return [f"{self.cfg.experiment}-baseline-r{i}" for i in range(1, self.cfg.baseline_repeats + 1)]

    def evaluate(self, groups: list[str], tasks: list[str] | None = None) -> dict[str, Any]:
        tasks = tasks or list(self.tasks)
        arm, base = self.arm(groups), self.arm(self.baseline_groups())
        summary = summarize(arm, tasks)
        cmp = compare(arm, base, tasks, samples=self.cfg.bootstrap["samples"], seed=self.cfg.bootstrap["seed"])
        return {"groups": groups, "summary": summary, "by_kind": by_kind(arm, tasks), "vs_baseline": cmp,
                "gate": gate(cmp, summary, max_success_drop=self.cfg.gate["max_success_drop"],
                             min_trust=self.cfg.gate["min_trust"])}

    # --- the plan ---------------------------------------------------------------------------

    def step_baseline(self) -> None:
        all_tasks = list(self.tasks)
        self.run([Pass(group=g, tasks=all_tasks, mode=self.cfg.mode) for g in self.baseline_groups()])
        self.state["baseline"] = {"summary": summarize(self.arm(self.baseline_groups()), all_tasks),
                                  "by_kind": by_kind(self.arm(self.baseline_groups()), all_tasks)}
        self.note(f"baseline: {self.state['baseline']['summary']}")

    def single_groups(self, sid: str) -> list[str]:
        repeats = self.state.get("repeats", {}).get(sid, 1)
        return [f"{self.cfg.experiment}-{sid}-r{i}" for i in range(1, repeats + 1)]

    def step_singles(self) -> None:
        passes = [Pass(group=f"{self.cfg.experiment}-{sid}-r1", specs=specs, tasks=list(self.tasks), mode=self.cfg.mode,
                       cross_fit=needs_cross_fit(specs)) for sid, specs in self.cfg.arms.items()]
        self.run(passes)
        close = []
        for sid in self.cfg.arms:
            result = self.evaluate(self.single_groups(sid))
            if result["gate"]["close_call"] and self.state.get("repeats", {}).get(sid, 1) < self.cfg.close_call_repeats:
                close.append(sid)
        if close:
            self.note(f"close calls get a second repeat: {close}")
            self.state.setdefault("repeats", {}).update({sid: self.cfg.close_call_repeats for sid in close})
            self.run([Pass(group=f"{self.cfg.experiment}-{sid}-r{r}", specs=self.cfg.arms[sid], tasks=list(self.tasks),
                           mode=self.cfg.mode, cross_fit=needs_cross_fit(self.cfg.arms[sid]))
                      for sid in close for r in range(2, self.cfg.close_call_repeats + 1)])
        self.state["singles"] = {sid: self.evaluate(self.single_groups(sid)) for sid in self.cfg.arms}
        for sid, r in self.state["singles"].items():
            self.note(f"{sid}: success {r['vs_baseline'].get('delta_success')} tokens/success "
                      f"{r['vs_baseline'].get('delta_tokens_per_success')} trust {r['summary'].get('trust_mean')} "
                      f"gate {'pass' if r['gate']['passed'] else 'FAIL ' + '; '.join(r['gate']['reasons'])}")

    def e3_tasks(self, dataset: str, index) -> list[str]:
        from toolhelper.efficiency.grounding import Grounding

        g = Grounding(index, datasets=[dataset])
        return [t.id for t in self.tasks.values() if g.entities(t.prompt)]

    def step_e3_grid(self, index) -> None:
        passes, subsets = [], {}
        for d in self.cfg.e3_grid["datasets"]:
            subsets[d] = self.e3_tasks(d, index)
            if not subsets[d]:
                continue
            for m in self.cfg.e3_grid["modes"]:
                passes.append(Pass(group=f"{self.cfg.experiment}-E3-{d}-{m}-r1",
                                   specs=[f"E3:modes={m},datasets={d}"], tasks=subsets[d], mode=self.cfg.mode))
        self.note(f"E3 grid task subsets: { {d: len(v) for d, v in subsets.items()} }")
        self.run(passes)
        self.state["e3_grid"] = {p.group: {"dataset": p.specs[0].split("datasets=")[1], "mode": p.specs[0].split("modes=")[1].split(",")[0],
                                           **self.evaluate([p.group], p.tasks)} for p in passes}
        self.save()

    def step_greedy(self) -> None:
        singles = self.state["singles"]
        gated = [sid for sid, r in singles.items() if r["gate"]["passed"]
                 and r["vs_baseline"].get("delta_tokens_per_success") is not None]
        order = sorted(gated, key=lambda sid: singles[sid]["vs_baseline"]["delta_tokens_per_success"])
        if not order:
            self.state["chosen"] = {"strategies": [], "specs": [], "groups": self.baseline_groups()}
            self.note("greedy: no strategy passed the gate; the baseline stays")
            return
        chosen, specs = [order[0]], list(self.cfg.arms[order[0]])
        current = singles[order[0]]
        steps = [{"step": 0, "add": order[0], "result": current, "accepted": True, "reason": "best single strategy"}]
        for k, sid in enumerate(order[1:], start=1):
            group = f"{self.cfg.experiment}-greedy-{k}-r1"
            trial = specs + self.cfg.arms[sid]
            self.run([Pass(group=group, specs=trial, tasks=list(self.tasks), mode=self.cfg.mode,
                           cross_fit=needs_cross_fit(trial))])
            result = self.evaluate([group])
            cur_tps = current["summary"].get("tokens_per_success")
            new_tps = result["summary"].get("tokens_per_success")
            gain = (1 - new_tps / cur_tps) if cur_tps and new_tps else 0.0
            ok = result["gate"]["passed"] and gain >= self.cfg.greedy["min_gain"]
            reason = (f"tokens/success {gain:+.1%} vs current" + ("" if result["gate"]["passed"]
                                                                    else "; gate: " + "; ".join(result["gate"]["reasons"])))
            steps.append({"step": k, "add": sid, "group": group, "specs": trial, "result": result,
                          "accepted": ok, "gain": gain, "reason": reason})
            self.note(f"greedy {k}: + {sid} -> {'accepted' if ok else 'rejected'} ({reason})")
            if ok:
                chosen.append(sid)
                specs, current = trial, result
        last_group = next((s["group"] for s in reversed(steps) if s["accepted"] and s.get("group")),
                          self.single_groups(order[0])[0])
        self.state["greedy"] = steps
        self.state["chosen"] = {"strategies": chosen, "specs": specs, "groups": [last_group]}
        self.save()

    def step_finalists(self) -> None:
        chosen = self.state["chosen"]
        if not chosen["specs"]:
            return
        first = chosen["groups"][0]
        base = first.rsplit("-r", 1)[0]
        groups = [first] + [f"{base}-final-r{i}" for i in range(2, self.cfg.finalist_repeats + 1)]
        self.run([Pass(group=g, specs=chosen["specs"], tasks=list(self.tasks), mode=self.cfg.mode,
                       cross_fit=needs_cross_fit(chosen["specs"])) for g in groups[1:]])
        chosen["groups"] = groups
        chosen["result"] = self.evaluate(groups)
        self.note(f"chosen {chosen['strategies']}: {chosen['result']['vs_baseline']}")

    def step_live(self) -> None:
        fold_a = self.folds["A"]
        chosen = self.state["chosen"]
        passes = [Pass(group=f"{self.cfg.experiment}-live-baseline", tasks=fold_a, mode="live")]
        if chosen["specs"]:
            passes.append(Pass(group=f"{self.cfg.experiment}-live-chosen", specs=chosen["specs"], tasks=fold_a,
                               mode="live", cross_fit=needs_cross_fit(chosen["specs"])))
        self.run(passes)
        live = {p.group: summarize(self.arm([p.group]), fold_a) for p in passes}
        replay = {"baseline": summarize(self.arm(self.baseline_groups()), fold_a)}
        if chosen["specs"]:
            replay["chosen"] = summarize(self.arm(chosen["groups"]), fold_a)
        self.state["live"] = {"live": live, "replay_same_tasks": replay}
        self.note(f"live confirmation on fold A: {live}; replay on the same tasks: {replay}")


def main_loop(orch: Orchestrator, index, store) -> None:
    from toolhelper.experiments.decision import write_decision

    orch.step_baseline()
    orch.step_singles()
    orch.step_e3_grid(index)
    orch.step_greedy()
    orch.step_finalists()
    wref = write_decision(store, orch)
    orch.state["decision"] = wref
    orch.note(f"decision written: {wref} (proposed; approve with `toolhelper curate {wref.split('@')[0]}`)")
    orch.step_live()
    orch.note("experiment finished")


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use `toolhelper experiment auto`")
