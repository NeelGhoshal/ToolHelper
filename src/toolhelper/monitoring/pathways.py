"""How the LLM gets to its tools: per-run pathway facts and cross-run aggregates.

"Used" is a heuristic: a result counts as used when an identifier it introduced
(a token containing a digit, e.g. 4329847, rs113488022, ERBB2, NM_004333.6) shows up
in a later tool call's arguments or in the answer. Results that introduce no new
identifiers are `None` (undecidable), not dead ends.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from toolhelper.mcp.canonical import canonical_json
from toolhelper.monitoring.tracer import ContextItemRecord, ToolCallRecord, Trace

ID_RE = re.compile(r"[A-Za-z0-9_.:>-]*\d[A-Za-z0-9_.:>-]*")
SUGGESTION_RE = re.compile(r"`(?:biomcp )?((?:get|search|list)\s[^`]+)`")


def identifiers(text: str) -> set[str]:
    return {m.strip(".:-") for m in ID_RE.findall(text or "") if len(m.strip(".:-")) >= 3}


def result_used(trace: Trace) -> dict[str, bool | None]:
    """call_id -> whether its result was used later (None: no new identifiers to follow)."""
    calls = trace.tool_calls
    answer = trace.answer or ""
    out: dict[str, bool | None] = {}
    for i, call in enumerate(calls):
        if call.is_error:
            out[call.call_id] = False
            continue
        known = identifiers(trace.prompt) | identifiers(canonical_json(call.final_arguments))
        new = identifiers(call.shaped_text) - known
        if not new:
            out[call.call_id] = None
            continue
        later = " ".join(canonical_json(c.final_arguments) for c in calls[i + 1:]) + " " + answer
        out[call.call_id] = bool(new & identifiers(later))
    return out


def context_used(item: ContextItemRecord, trace: Trace) -> bool | None:
    ids = identifiers(item.text) - identifiers(trace.prompt)
    if not ids:
        return None
    later = " ".join(canonical_json(c.final_arguments) for c in trace.tool_calls) + " " + (trace.answer or "")
    return bool(ids & identifiers(later))


def carried_tokens(trace: Trace) -> dict[str, int]:
    """call_id -> its result's tokens summed over every later LLM call that included it (estimated)."""
    out: Counter[str] = Counter()
    for llm_call in trace.llm_calls:
        out.update(llm_call.tool_results_in_context)
    return dict(out)


def as_command(call: ToolCallRecord) -> str | None:
    """A BioMCP call as the CLI command it is equivalent to."""
    args = call.final_arguments
    if call.server != "biomcp":
        return None
    if call.tool == "biomcp":
        return str(args.get("command", "")).removeprefix("biomcp ")
    if call.tool == "get":
        return " ".join(["get", str(args.get("entity", "")), str(args.get("id", "")), *args.get("sections", [])])
    if call.tool == "search":
        return f"search {args.get('entity', '')}"
    return None


def suggestions(text: str) -> list[str]:
    return [s.strip() for s in SUGGESTION_RE.findall(text or "")]


def follows(command: str, suggestion: str) -> bool:
    """Whether a command matches a suggestion: same words up to the first flag; <x> matches anything."""
    want = [w for w in suggestion.split()]
    cut = next((i for i, w in enumerate(want) if w.startswith("-")), len(want))
    want = want[:cut]
    have = command.split()
    return len(want) >= 2 and len(have) >= len(want) and all(
        w.startswith("<") or w.lower() == h.lower() for w, h in zip(want, have))


def expected_tools(trace: Trace) -> set[str]:
    return {t for path in trace.task.get("expected_paths", []) for t in path}


def run_pathway(trace: Trace) -> dict[str, Any]:
    used = result_used(trace)
    calls = trace.tool_calls
    names = [c.name for c in calls]
    expected = trace.task.get("expected_paths", [])
    first_ok = None
    if calls and expected:
        first_ok = any(path and path[0] == calls[0].name for path in expected)
    seen: set[str] = set()
    redundant = retries = 0
    for i, c in enumerate(calls):
        key = c.name + canonical_json(c.final_arguments)
        redundant += key in seen
        seen.add(key)
        retries += i > 0 and calls[i - 1].is_error and calls[i - 1].name == c.name
    useful_turns = [c.turn for c in calls if used.get(c.call_id)]
    followed = offered = 0
    for i, c in enumerate(calls[:-1]):
        hints = suggestions(c.shaped_text) if c.server == "biomcp" and not c.is_error else []
        if hints:
            offered += 1
            nxt = as_command(calls[i + 1])
            followed += bool(nxt and any(follows(nxt, h) for h in hints))
    allowed = expected_tools(trace)
    return {
        "run_id": trace.run_id, "task_id": trace.task_id, "success": trace.success,
        "sequence": names,
        "tools_offered_per_turn": [len(c.tools_offered) for c in trace.llm_calls],
        "first_tool": names[0] if names else None, "first_tool_correct": first_ok,
        "turns_to_first_useful_call": useful_turns[0] if useful_turns else None,
        "retries": retries, "redundant_calls": redundant,
        "dead_ends": [c.call_id for c in calls if used.get(c.call_id) is False and not c.is_error],
        "detours": [c.call_id for c in calls if allowed and c.name not in allowed],
        "biomcp_suggestions": {"offered": offered, "followed": followed},
    }


def aggregate(pathways: list[dict[str, Any]]) -> dict[str, Any]:
    transitions: Counter[tuple[str, str]] = Counter()
    sequences: dict[str, Counter[str]] = {"success": Counter(), "failure": Counter(), "unscored": Counter()}
    offered = followed = 0
    for p in pathways:
        path = ["START", *p["sequence"], "END"]
        transitions.update(zip(path, path[1:]))
        outcome = "unscored" if p["success"] is None else ("success" if p["success"] else "failure")
        sequences[outcome][" > ".join(p["sequence"]) or "(no tools)"] += 1
        offered += p["biomcp_suggestions"]["offered"]
        followed += p["biomcp_suggestions"]["followed"]
    firsts = [p["first_tool_correct"] for p in pathways if p["first_tool_correct"] is not None]
    return {
        "transitions": [{"from": a, "to": b, "count": n} for (a, b), n in transitions.most_common()],
        "common_sequences": {k: v.most_common(10) for k, v in sequences.items()},
        "first_tool_correct_rate": sum(firsts) / len(firsts) if firsts else None,
        "biomcp_suggestion_following": {"offered": offered, "followed": followed,
                                        "rate": followed / offered if offered else None},
        "retries": sum(p["retries"] for p in pathways),
        "redundant_calls": sum(p["redundant_calls"] for p in pathways),
        "dead_ends": sum(len(p["dead_ends"]) for p in pathways),
        "detours": sum(len(p["detours"]) for p in pathways),
    }
