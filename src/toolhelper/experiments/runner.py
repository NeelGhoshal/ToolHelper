"""Benchmark tasks: loading, validation and runs (each trace scored, saved and ingested)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.experiments.scoring import Score, score
from toolhelper.monitoring.db import MonitorDB
from toolhelper.monitoring.tracer import Trace

TRAP_SHARE = (0.15, 0.30)


def _expand(value: Any, repo: str) -> Any:
    if isinstance(value, str):
        return value.replace("{repo}", repo)
    if isinstance(value, list):
        return [_expand(v, repo) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, repo) for k, v in value.items()}
    return value


def load_tasks(path: Path | str, repo: Path | str | None = None) -> list[Task]:
    """Reads a JSONL task file; `{repo}` in any string becomes the repo root."""
    from toolhelper.config import ROOT

    root = str(repo or ROOT)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Task.model_validate(_expand(json.loads(line), root)) for line in lines if line.strip()]


def validate_tasks(tasks: list[Task], tool_names: set[str]) -> dict[str, Any]:
    """Structural checks against the catalog's namespaced tool names. Returns counts and problems."""
    problems = []
    ids = Counter(t.id for t in tasks)
    problems += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    servers = {name.split("__", 1)[0] for name in tool_names}
    for t in tasks:
        referenced = {tool for path in t.expected_paths for tool in path}
        referenced |= set(t.checks.get("must_call_any", [])) | set(t.checks.get("must_call_all", []))
        referenced |= {w["tool"] for w in t.checks.get("called_with", [])}
        problems += [f"{t.id}: unknown tool {name}" for name in sorted(referenced - tool_names)]
        problems += [f"{t.id}: unknown server {s}" for s in t.servers if s not in servers]
        if not t.servers:
            problems.append(f"{t.id}: no servers")
        if not (t.checks.get("must_call_any") or t.checks.get("must_call_all")):
            problems.append(f"{t.id}: no required tool check")
        undeclared = {name.split("__", 1)[0] for name in referenced} - set(t.servers)
        problems += [f"{t.id}: uses {s} but does not list it in servers" for s in sorted(undeclared)]
    kinds = Counter("cross" if len(t.servers) > 1 else t.servers[0] for t in tasks if t.servers)
    traps = sum(bool(t.traps) for t in tasks)
    share = traps / len(tasks) if tasks else 0
    if tasks and not TRAP_SHARE[0] <= share <= TRAP_SHARE[1]:
        problems.append(f"trap share {share:.0%} outside {TRAP_SHARE[0]:.0%}-{TRAP_SHARE[1]:.0%}")
    return {"tasks": len(tasks), "by_server": dict(kinds), "traps": traps, "trap_share": share,
            "trap_kinds": dict(Counter(k for t in tasks for k in t.traps)), "problems": problems}


async def run_benchmark(tasks: list[Task], agent: Agent, *, traces_dir: Path | str, db: MonitorDB,
                        progress: Callable[[Trace, Score], None] | None = None) -> list[tuple[Trace, Score]]:
    out = []
    for task in tasks:
        trace = await agent.run(task)
        result = score(trace, task)
        trace.success, trace.score = result.success, result.model_dump()
        path = trace.save(traces_dir)
        db.ingest(trace, path)
        out.append((trace, result))
        if progress:
            progress(trace, result)
    return out


def check_expectations(tasks: list[Task], traces: list[Trace]) -> list[str]:
    """Expected answer strings that appear in none of the task's real tool outputs: either the
    model never reached the data, or the expectation is wrong and should be fixed from real data."""
    by_task: dict[str, list[Trace]] = {}
    for t in traces:
        by_task.setdefault(t.task_id, []).append(t)
    out = []
    for task in tasks:
        outputs = " ".join((c.raw_text or "") for t in by_task.get(task.id, []) for c in t.tool_calls).lower()
        for key in ("answer_includes_any", "answer_includes_all"):
            wanted = task.checks.get(key, [])
            found = [s for s in wanted if s.lower() in outputs]
            if wanted and (not found if key.endswith("any") else len(found) < len(wanted)):
                missing = [s for s in wanted if s not in found]
                out.append(f"{task.id}: {key} {missing} not in any tool output"
                           + ("" if task.id in by_task else " (no runs)"))
    return out
