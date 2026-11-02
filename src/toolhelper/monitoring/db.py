"""Monitoring database: the README Phase 5 tables, filled from traces.

Ingesting a run replaces any earlier rows for the same run_id. Token columns follow
the trace's labels: `prompt_tokens`/`completion_tokens` are measured when `measured`
is 1; every other token column (args, results, carried, breakdowns) is estimated
with the trace's tokenizer, and `calibrated_json` scales the breakdown to the
measured total when there is one.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from toolhelper.context.model import parse_wref
from toolhelper.monitoring.pathways import carried_tokens, context_used, result_used, run_pathway
from toolhelper.monitoring.tokens import calibrate, measured_tool_defs
from toolhelper.monitoring.tracer import Trace

NEAR_NUM_CTX = 0.9
TRUNCATED = {"length", "max_tokens"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, group_id TEXT, task_id TEXT, backend TEXT, model TEXT, params_json TEXT,
    config_hash TEXT, catalog_version TEXT, success INTEGER, trust_json TEXT, answer TEXT,
    stop_reason TEXT, error TEXT, hub_mode TEXT, tokenizer TEXT, started_at TEXT, turns INTEGER,
    n_tool_calls INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, measured INTEGER,
    tokens_by_component_json TEXT, provenance REAL, complete INTEGER, problems_json TEXT,
    pathway_json TEXT, trace_path TEXT, score_json TEXT);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT, turn INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER, measured INTEGER,
    est_breakdown_json TEXT, calibrated_json TEXT, latency_ms REAL, done_reason TEXT,
    tools_offered INTEGER, tools_offered_json TEXT, num_ctx INTEGER, warnings_json TEXT);
CREATE TABLE IF NOT EXISTS tool_calls (
    run_id TEXT, turn INTEGER, call_id TEXT, name TEXT, server TEXT, tool TEXT, args_json TEXT,
    args_tokens INTEGER, raw_result_tokens INTEGER, shaped_result_tokens INTEGER, carried_tokens INTEGER,
    latency_ms REAL, is_error INTEGER, error_class TEXT, decision TEXT, cache_hit INTEGER,
    replayed INTEGER, source TEXT, used INTEGER);
CREATE TABLE IF NOT EXISTS transformations (
    run_id TEXT, turn INTEGER, strategy TEXT, hook TEXT, target TEXT, before_tokens INTEGER,
    after_tokens INTEGER, evidence_json TEXT, note TEXT, reported INTEGER);
CREATE TABLE IF NOT EXISTS context_items (
    run_id TEXT, strategy TEXT, kind TEXT, wref TEXT, version INTEGER, tokens INTEGER, used INTEGER);
"""
TABLES = ["runs", "llm_calls", "tool_calls", "transformations", "context_items"]


def _bool(value: bool | None) -> int | None:
    return None if value is None else int(value)


def llm_call_warnings(trace: Trace) -> list[list[str]]:
    out, previous = [], None
    for call in trace.llm_calls:
        warnings = []
        num_ctx = call.applied_options.get("num_ctx")
        if num_ctx and call.prompt_tokens >= NEAR_NUM_CTX * num_ctx:
            warnings.append(f"prompt {call.prompt_tokens} tokens is near num_ctx {num_ctx}")
        if call.done_reason in TRUNCATED:
            warnings.append(f"output truncated (done_reason={call.done_reason})")
        if call.measured and previous and previous.measured and call.messages_sent > previous.messages_sent \
                and call.prompt_tokens < previous.prompt_tokens:
            warnings.append("measured prompt shrank while the prompt grew: prompt cache suspected")
        out.append(warnings)
        previous = call
    return out


class MonitorDB:
    def __init__(self, path: Path | str = ":memory:", tool_costs: dict[str, Any] | None = None) -> None:
        """`tool_costs` is a `toolhelper tokens tools` measurement; when it matches a run's model
        and catalog, the tool-definition share of each prompt is taken from it, not estimated."""
        self.tool_costs = tool_costs
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(runs)")}
        if "score_json" not in columns:  # databases created before Phase 6
            self.db.execute("ALTER TABLE runs ADD COLUMN score_json TEXT")

    def close(self) -> None:
        self.db.close()

    def ingest(self, trace: Trace, trace_path: Path | str | None = None) -> None:
        used = result_used(trace)
        carried = carried_tokens(trace)
        warnings = llm_call_warnings(trace)
        measured = bool(trace.llm_calls) and all(c.measured for c in trace.llm_calls)
        components: dict[str, int] = {}
        calibrated_rows = []
        for call in trace.llm_calls:
            tool_defs = measured_tool_defs(self.tool_costs, trace.model, trace.catalog_version,
                                           call.tools_offered) if call.measured else None
            anchored = {"tool_defs": tool_defs} if tool_defs is not None else None
            parts = calibrate(call.est_breakdown, call.prompt_tokens if call.measured else None, anchored)
            basis = "estimated" if not call.measured else (
                "measured, tool_defs anchored" if anchored else "measured")
            calibrated_rows.append({"basis": basis, "parts": parts})
            for k, v in parts.items():
                components[k] = components.get(k, 0) + v
        components["completion"] = sum(c.completion_tokens for c in trace.llm_calls)
        problems = trace.problems()
        with self.db:
            for table in TABLES:
                self.db.execute(f"DELETE FROM {table} WHERE run_id = ?", (trace.run_id,))
            self.db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?)", (
                    trace.run_id, trace.group_id, trace.task_id, trace.backend, trace.model,
                    json.dumps(trace.params), trace.config_hash, trace.catalog_version, _bool(trace.success),
                    json.dumps(trace.trust) if trace.trust is not None else None, trace.answer,
                    trace.stop_reason, trace.error, trace.hub_mode, trace.tokenizer, trace.started_at,
                    len(trace.llm_calls), len(trace.tool_calls),
                    sum(c.prompt_tokens for c in trace.llm_calls), components["completion"], int(measured),
                    json.dumps({"basis": "measured" if measured else "estimated", "parts": components}),
                    trace.provenance_coverage, int(not problems), json.dumps(problems),
                    json.dumps(run_pathway(trace)), str(trace_path) if trace_path else None,
                    json.dumps(trace.score) if trace.score is not None else None))
            for call, cal, warn in zip(trace.llm_calls, calibrated_rows, warnings):
                self.db.execute("INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    trace.run_id, call.turn, call.prompt_tokens, call.completion_tokens, int(call.measured),
                    json.dumps(call.est_breakdown), json.dumps(cal), call.latency_ms, call.done_reason,
                    len(call.tools_offered), json.dumps(call.tools_offered), call.applied_options.get("num_ctx"),
                    json.dumps(warn)))
            for t in trace.tool_calls:
                self.db.execute(
                    "INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                        trace.run_id, t.turn, t.call_id, t.name, t.server, t.tool, json.dumps(t.final_arguments),
                        t.args_tokens, t.raw_result_tokens, t.shaped_result_tokens, carried.get(t.call_id, 0),
                        t.latency_ms, int(t.is_error), t.error_class, t.decision, int(t.cache_hit),
                        int(t.replayed), t.source, _bool(used.get(t.call_id))))
            for t in trace.transformations:
                self.db.execute("INSERT INTO transformations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    trace.run_id, t.turn, t.strategy, t.hook, t.target, t.before_tokens, t.after_tokens,
                    json.dumps([e.model_dump() for e in t.evidence]), t.note, int(t.reported)))
            for item in trace.context_items:
                version = parse_wref(item.wref)[1] if item.wref else None
                self.db.execute("INSERT INTO context_items VALUES (?, ?, ?, ?, ?, ?, ?)", (
                    trace.run_id, item.strategy, item.kind, item.wref, version, item.tokens,
                    _bool(context_used(item, trace))))

    def ingest_dir(self, directory: Path | str) -> int:
        paths = sorted(Path(directory).glob("*.json"))
        for path in paths:
            self.ingest(Trace.load(path), path)
        return len(paths)

    def rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute(sql, params).fetchall()]
