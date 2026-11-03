"""Scoring a run against its task: hard checks, grounding and expected-path match.

Checks (all optional; a run succeeds when it reached an answer, `final` or E9's
`final_at_limit`, and every defined check passes):
- `must_call_any` / `must_call_all`: tools called without an error result.
- `called_with`: `[{"tool": name, "args": {...}}]`, a call whose arguments include these
  values (strings compared case-insensitively); used to test trap handling.
- `answer_includes_any` / `answer_includes_all`: case-insensitive substrings of the answer.
- `answer_grounded`: the answer states identifiers and every one appears in tool outputs.

Grounding is the share of identifiers in the answer (tokens containing a digit, not
already in the prompt) that appear in this run's tool outputs or evidenced context.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from toolhelper.agent.hooks import Task
from toolhelper.monitoring.pathways import identifiers
from toolhelper.monitoring.tracer import Trace

# Stop reasons that end with an answer; `final_at_limit` is E9's answer after `max_turns`.
ANSWERED = ("final", "final_at_limit")


class Score(BaseModel):
    task_id: str
    run_id: str
    success: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    grounding: float | None = None  # None when the answer has no identifiers
    grounded: list[str] = Field(default_factory=list)
    ungrounded: list[str] = Field(default_factory=list)
    path_match: bool | None = None  # None when the task has no expected paths


def _ok_calls(trace: Trace) -> list[Any]:
    return [c for c in trace.tool_calls if not c.is_error]


def _matches(value: Any, want: Any) -> bool:
    if isinstance(want, str) and isinstance(value, str):
        return value.strip().lower() == want.strip().lower()
    return value == want


def _is_subsequence(path: list[str], sequence: list[str]) -> bool:
    it = iter(sequence)
    return all(step in it for step in path)


def score(trace: Trace, task: Task) -> Score:
    checks: dict[str, bool] = {}
    c = task.checks
    called = {t.name for t in _ok_calls(trace)}
    answer = (trace.answer or "").lower()
    if c.get("must_call_any"):
        checks["must_call_any"] = bool(called & set(c["must_call_any"]))
    if c.get("must_call_all"):
        checks["must_call_all"] = set(c["must_call_all"]) <= called
    for i, want in enumerate(c.get("called_with", [])):
        checks[f"called_with[{i}]"] = any(
            t.name == want["tool"] and all(_matches(t.final_arguments.get(k), v) for k, v in want["args"].items())
            for t in trace.tool_calls)
    if c.get("answer_includes_any"):
        checks["answer_includes_any"] = any(s.lower() in answer for s in c["answer_includes_any"])
    if c.get("answer_includes_all"):
        checks["answer_includes_all"] = all(s.lower() in answer for s in c["answer_includes_all"])

    ids = sorted(identifiers(trace.answer or "") - identifiers(task.prompt))
    sources = " ".join(t.raw_text or "" for t in trace.tool_calls) + " " + " ".join(
        t.shaped_text for t in trace.tool_calls) + " " + " ".join(i.text for i in trace.context_items if i.evidence)
    known = identifiers(sources)
    grounded = [i for i in ids if i in known]
    if c.get("answer_grounded"):
        checks["answer_grounded"] = bool(ids) and len(grounded) == len(ids)
    path_match = None
    if task.expected_paths:
        sequence = [t.name for t in _ok_calls(trace)]
        path_match = any(_is_subsequence(p, sequence) for p in task.expected_paths)
    return Score(
        task_id=task.id, run_id=trace.run_id, checks=checks,
        success=trace.stop_reason in ANSWERED and all(checks.values()),
        grounding=len(grounded) / len(ids) if ids else None, grounded=grounded,
        ungrounded=[i for i in ids if i not in known], path_match=path_match,
    )
