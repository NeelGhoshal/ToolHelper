"""Phase 10 outputs: RunSummary things, the E3 per-tool routing table and the Decision.

The Decision (`decisions/<experiment>`) records the chosen config, each key decision with its
numbers, the rejected alternatives and evidence links (pinned RunSummary versions, which list
their run ids). It stays `proposed` until approved with `toolhelper curate`.

E3 routing, per (tool, dataset, mode) against the baseline on the same tasks: hint uptake,
change in that tool's argument errors and calls per run, hint tokens added vs total tokens
saved, and the wrong-hint rate (runs that used a hinted value and failed a task the baseline
passed).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from toolhelper.context.ingest import _upsert
from toolhelper.context.model import ContextStore, EvidenceRef, Revise
from toolhelper.mcp.canonical import canonical_json
from toolhelper.monitoring.tracer import Trace

ARG_ERRORS = ("tool_error", "protocol_error", "rejected")


def _traces(db, group: str) -> dict[str, Trace]:
    out = {}
    for r in db.rows("SELECT trace_path FROM runs WHERE group_id = ? ORDER BY started_at", (group,)):
        if r["trace_path"] and Path(r["trace_path"]).exists():
            t = Trace.load(r["trace_path"])
            out[t.task_id] = t
    return out


def e3_routing(db, grid: dict[str, dict[str, Any]], baseline_groups: list[str]) -> list[dict[str, Any]]:
    from toolhelper.efficiency.grounding import uptake

    base = [_traces(db, g) for g in baseline_groups]
    rows = []
    for group, info in grid.items():
        runs = _traces(db, group)
        per_tool: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        hint_tokens = taken = hints = wrong = 0
        for task_id, t in runs.items():
            u = uptake(t)
            hints, taken = hints + u["hints"], taken + u["taken"]
            hint_tokens += sum(i.tokens for i in t.context_items if i.strategy.startswith("E3"))
            base_pass = any(b.get(task_id) is not None and b[task_id].success for b in base)
            wrong += bool(u["taken"]) and not t.success and base_pass
            for c in t.tool_calls:
                if c.server:
                    per_tool[c.name]["calls"] += 1
                    per_tool[c.name]["errors"] += c.error_class in ARG_ERRORS
            for b in base:
                if task_id in b:
                    for c in b[task_id].tool_calls:
                        if c.server:
                            per_tool[c.name]["base_calls"] += 1 / len(base)
                            per_tool[c.name]["base_errors"] += (c.error_class in ARG_ERRORS) / len(base)
        n = max(1, len(runs))
        for tool, v in sorted(per_tool.items()):
            rows.append({"tool": tool, "dataset": info["dataset"], "mode": info["mode"], "tasks": len(runs),
                         "uptake": taken / hints if hints else None,
                         "delta_arg_errors_per_run": (v["errors"] - v["base_errors"]) / n,
                         "delta_calls_per_run": (v["calls"] - v["base_calls"]) / n,
                         "hint_tokens_per_run": hint_tokens / n,
                         "delta_tokens_per_success": info["vs_baseline"].get("delta_tokens_per_success"),
                         "delta_success": info["vs_baseline"].get("delta_success"),
                         "wrong_hint_rate": wrong / n})
    return rows


def write_summaries(store: ContextStore, experiment_wref: str, orch, groups: dict[str, str]) -> dict[str, str]:
    """RunSummary per group; returns group -> pinned wref."""
    ops = []
    for group, arm_name in groups.items():
        runs = orch.arm([group])[0]
        s = orch.evaluate([group])["summary"] if runs else {"runs": 0}
        repeat = int(group.rsplit("-r", 1)[1]) if "-r" in group and group.rsplit("-r", 1)[1].isdigit() else 1
        data = {"experiment": experiment_wref, "arm": arm_name, "group": group, "repeat": repeat,
                "runs": s.get("runs", 0), "success_rate": s.get("success_rate") or 0.0,
                "tokens_per_success": s.get("tokens_per_success") or 0.0, "trust_mean": s.get("trust_mean") or 0.0,
                "run_ids": sorted(r.run_id for r in runs.values())}
        ops.append(_upsert(store, f"runsummaries/{group}", "RunSummary", data,
                           evidence=[EvidenceRef(kind="run", ref=r) for r in data["run_ids"][:50]]))
    store.apply("phase 10 run summaries", ops)
    return {g: f"runsummaries/{g}@v{store.get(f'runsummaries/{g}').version}" for g in groups}


def write_decision(store: ContextStore, orch) -> str:
    cfg, st = orch.cfg, orch.state
    experiment = store.get(f"experiments/{cfg.experiment}")
    exp_wref = f"{experiment.name}@v{experiment.version}"
    groups = {g: "baseline" for g in orch.baseline_groups()}
    for sid in cfg.arms:
        groups.update({g: sid for g in orch.single_groups(sid)})
    groups.update({g: f"E3:{v['dataset']}:{v['mode']}" for g, v in st.get("e3_grid", {}).items()})
    groups.update({s["group"]: f"greedy+{s['add']}" for s in st.get("greedy", []) if s.get("group")})
    groups.update({g: "chosen" for g in st.get("chosen", {}).get("groups", [])})
    pins = write_summaries(store, exp_wref, orch, groups)

    decisions, rejected = [], []
    for sid, r in st.get("singles", {}).items():
        v = r["vs_baseline"]
        line = (f"{sid} alone: success {v.get('delta_success', 0) * 100:+.1f} points {v.get('delta_success_ci')}, "
                f"tokens/success {(v.get('delta_tokens_per_success') or 0):+.1%} {v.get('delta_tokens_per_success_ci')}, "
                f"trust {r['summary'].get('trust_mean')}, McNemar p={v.get('mcnemar', {}).get('p')}")
        if not r["gate"]["passed"]:
            rejected.append({"config": sid, "why": "gate: " + "; ".join(r["gate"]["reasons"]), "numbers": line})
        decisions.append(line)
    for s in st.get("greedy", [])[1:]:
        (decisions if s["accepted"] else rejected).append(
            f"greedy + {s['add']}: {s['reason']}" if s["accepted"] else
            {"config": f"+{s['add']}", "why": s["reason"], "numbers": s["result"]["vs_baseline"]})
    chosen = st.get("chosen", {})
    per_kind = {}
    for k in ("medical", "biomcp", "omop", "cross"):
        best = None
        for sid, r in st.get("singles", {}).items():
            kb, base = r["by_kind"].get(k, {}), st.get("baseline", {}).get("by_kind", {}).get(k, {})
            if kb.get("tokens_per_success") and base.get("success_rate") is not None and \
                    kb.get("success_rate", 0) >= base["success_rate"] - cfg.gate["max_success_drop"] and \
                    (kb.get("trust_mean") or 0) >= cfg.gate["min_trust"]:
                if best is None or kb["tokens_per_success"] < best[1]:
                    best = (sid, kb["tokens_per_success"])
        per_kind[k] = best[0] if best else "baseline"
    routing = e3_routing(orch.db, st.get("e3_grid", {}), orch.baseline_groups()) if st.get("e3_grid") else []
    st["e3_routing"] = routing
    orch.save()

    data = {"title": f"Phase 10 {cfg.experiment}: chosen efficiency config",
            "objective": "minimize tokens per successful task, subject to success >= baseline - 2 points and trust >= 0.9",
            "chosen_config_json": canonical_json({"strategies": chosen.get("strategies", []), "specs": chosen.get("specs", []),
                                                  "result": chosen.get("result", {}).get("vs_baseline"),
                                                  "per_server_best_single": per_kind, "decisions": decisions}),
            "constraints_json": canonical_json({**cfg.gate, **cfg.greedy, "folds": "two-fold cross-fitting"}),
            "evidence": sorted(pins.values()), "rejected_json": canonical_json(rejected), "status": "proposed"}
    name = f"decisions/{cfg.experiment}"
    store.apply(f"decision {cfg.experiment}", [_upsert(store, name, "Decision", data)])
    return f"{name}@v{store.get(name).version}"


def curate(store: ContextStore, name: str, approved_by: str) -> str:
    """Approves a proposed Decision (records who approved it)."""
    thing = store.get(name)
    if thing.data.get("status") == "accepted":
        return f"{thing.name}@v{thing.version} is already accepted"
    data = {**thing.data, "status": "accepted", "approved_by": approved_by}
    store.apply(f"approve {name}", [Revise(name=thing.name, data=data, expected_version=thing.version)])
    new = store.get(name)
    return f"{new.name}@v{new.version} accepted by {approved_by}"
