"""Per-tool and per-run metrics from the monitoring database."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from toolhelper.monitoring.db import MonitorDB


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def _where(group: str | None) -> tuple[str, tuple]:
    return ("WHERE r.group_id = ?", (group,)) if group else ("", ())


def tool_metrics(db: MonitorDB, group: str | None = None) -> list[dict[str, Any]]:
    where, params = _where(group)
    rows = db.rows(f"SELECT tc.* FROM tool_calls tc JOIN runs r USING (run_id) {where}", params)
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tool[row["name"]].append(row)
    totals = {name: sum(r["args_tokens"] + r["shaped_result_tokens"] + r["carried_tokens"] for r in calls)
              for name, calls in by_tool.items()}
    grand = sum(totals.values()) or 1
    out = []
    for name, calls in sorted(by_tool.items(), key=lambda kv: -totals[kv[0]]):
        raw = [r["raw_result_tokens"] for r in calls if r["raw_result_tokens"] is not None]
        decided = [r["used"] for r in calls if r["used"] is not None]
        live = [r["latency_ms"] for r in calls if r["source"] == "live"]
        out.append({
            "tool": name, "calls": len(calls),
            "errors": sum(r["is_error"] for r in calls),
            "error_rate": sum(r["is_error"] for r in calls) / len(calls),
            "args_tokens_mean": sum(r["args_tokens"] for r in calls) / len(calls),
            "raw_result_tokens_mean": sum(raw) / len(raw) if raw else None,
            "shaped_result_tokens_mean": sum(r["shaped_result_tokens"] for r in calls) / len(calls),
            "carried_tokens": sum(r["carried_tokens"] for r in calls),
            "latency_p50_ms": percentile([r["latency_ms"] for r in calls], 50),
            "latency_p95_ms": percentile([r["latency_ms"] for r in calls], 95),
            "live_latency_p50_ms": percentile(live, 50), "live_latency_p95_ms": percentile(live, 95),
            "replayed_share": sum(r["replayed"] for r in calls) / len(calls),
            "used_rate": sum(decided) / len(decided) if decided else None,
            "token_share": totals[name] / grand,
        })
    return out


def run_metrics(db: MonitorDB, group: str | None = None) -> list[dict[str, Any]]:
    where, params = _where(group)
    out = []
    for r in db.rows(f"SELECT * FROM runs r {where} ORDER BY started_at", params):
        out.append({
            "run_id": r["run_id"], "task_id": r["task_id"], "model": r["model"], "hub_mode": r["hub_mode"],
            "turns": r["turns"], "tool_calls": r["n_tool_calls"], "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"], "measured": bool(r["measured"]),
            "tokens_by_component": json.loads(r["tokens_by_component_json"]),
            "success": None if r["success"] is None else bool(r["success"]),
            "trust": json.loads(r["trust_json"]) if r["trust_json"] else None,
            "provenance": r["provenance"], "complete": bool(r["complete"]), "stop_reason": r["stop_reason"],
            "error": r["error"],
        })
    return out


def warnings(db: MonitorDB, group: str | None = None) -> list[dict[str, Any]]:
    where, params = _where(group)
    out = []
    for r in db.rows(f"SELECT l.run_id, l.turn, l.warnings_json FROM llm_calls l JOIN runs r USING (run_id) "
                     f"{where}", params):
        out += [{"run_id": r["run_id"], "turn": r["turn"], "warning": w} for w in json.loads(r["warnings_json"])]
    for r in db.rows(f"SELECT run_id, problems_json FROM runs r {where}", params):
        out += [{"run_id": r["run_id"], "turn": None, "warning": p} for p in json.loads(r["problems_json"])]
    return out
