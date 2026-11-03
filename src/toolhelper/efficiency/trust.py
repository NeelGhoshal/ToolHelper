"""Trust (README Phase 9): four components, each 0-1 with the reasons behind it, and a
weighted combination (weights in configs/policy.yaml).

- grounding: identifiers in the answer (tokens with a digit, not in the prompt) that appear
  in this run's tool outputs (raw or shaped) or in evidenced context items;
- provenance: share of transformations and context items carrying evidence (1.0, or a bug);
- fidelity: 1.0 minus penalties for expand/recall calls, low-confidence rewrites and
  argument errors on tools whose schemas E2 compressed (a possible cause only: whether
  compression raises error rates is measured against uncompressed runs in Phase 10);
- freshness: share of cached results and datasets used as evidence that were within their
  TTL when the run started.

A component that does not apply (no identifiers, nothing cached or looked up) is None and
left out of the combined score, whose weights are renormalized over the rest.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from toolhelper.config import ROOT
from toolhelper.monitoring.pathways import identifiers
from toolhelper.monitoring.tracer import Trace

POLICY = ROOT / "configs" / "policy.yaml"
EXPAND_TOOLS = ("toolhelper__expand_result", "toolhelper__recall_result")
ARG_ERRORS = ("tool_error", "protocol_error", "rejected")
# Error texts that blame the arguments (as opposed to upstream failures like capture_corrupt or "not found").
ARG_ERROR_TEXT = re.compile(r"invalid|unknown|validation|must |required|not valid|read-only commands|"
                            r"outside its supported bounds|mutually exclusive|Call not sent", re.I)

# Dataset wref (`datasets/hgnc@v1`) -> (fetched_at ISO, ttl_days), or None if unknown.
DatasetLookup = Callable[[str], tuple[str, float] | None]


class TrustPolicy(BaseModel):
    weights: dict[str, float]
    fidelity_penalties: dict[str, float]


def load_policy(path: Path | str = POLICY) -> TrustPolicy:
    return TrustPolicy.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["trust"])


def store_lookup(store) -> DatasetLookup:
    """Dataset ages from the context store's Dataset things."""
    from toolhelper.context.model import NotFoundError

    def lookup(wref: str) -> tuple[str, float] | None:
        try:
            data = store.get(wref).data
        except NotFoundError:
            return None
        return data["fetched_at"], float(data["ttl_days"])

    return lookup


def _component(value: float | None, reasons: list[str]) -> dict[str, Any]:
    return {"value": None if value is None else round(value, 4), "reasons": reasons}


def grounding(trace: Trace) -> dict[str, Any]:
    ids = sorted(identifiers(trace.answer or "") - identifiers(trace.prompt))
    if not ids:
        return _component(None, ["the answer states no identifiers"])
    sources: dict[str, list[str]] = {}
    for c in trace.tool_calls:
        for i in identifiers((c.raw_text or "") + " " + (c.shaped_text or "")):
            sources.setdefault(i, []).append(c.name)
    for item in trace.context_items:
        if item.evidence:
            text = item.text + " " + " ".join(str(v) for e in item.evidence for v in (e.detail or {}).values())
            for i in identifiers(text):
                sources.setdefault(i, []).append(f"{item.kind} hint")
    reasons = [f"{i} from {sorted(set(sources[i]))[0]}" if i in sources else f"{i} is in no tool output or hint"
               for i in ids]
    return _component(sum(i in sources for i in ids) / len(ids), reasons)


def provenance(trace: Trace) -> dict[str, Any]:
    items = [*trace.transformations, *trace.context_items]
    if not items:
        return _component(1.0, ["no transformations or injected items"])
    missing = [f"{t.strategy}.{t.hook} on {t.target}: no evidence" + ("" if t.reported else " (change not reported)")
               for t in trace.transformations if not t.evidence]
    missing += [f"{i.strategy} {i.kind} item: no evidence" for i in trace.context_items if not i.evidence]
    reasons = missing or [f"all {len(items)} transformations and injected items carry evidence"]
    return _component(trace.provenance_coverage, reasons)


def _low_confidence(t) -> str | None:
    if t.hook != "before_call":
        return None
    if t.strategy.startswith("E3") and any((e.detail or {}).get("method") not in (None, "exact") for e in t.evidence):
        return f"E3 rewrote {t.target} from a normalized match ({t.note})"
    if t.strategy.startswith("E4") and "learned:unknown_field" in t.note and \
            any("rename" in (e.detail or {}) for e in t.evidence):
        return f"E4 renamed an argument of {t.target} from a rule learned from runs ({t.note})"
    return None


def fidelity(trace: Trace, penalties: dict[str, float]) -> dict[str, Any]:
    value, reasons = 1.0, []
    for c in trace.tool_calls:
        if c.name in EXPAND_TOOLS:
            value -= penalties["expand"]
            reasons.append(f"{c.name} {c.final_arguments}: a shaped or compacted result had to be re-expanded")
    for t in trace.transformations:
        if why := _low_confidence(t):
            value -= penalties["low_confidence_rewrite"]
            reasons.append(why)
    # One run cannot show the compression caused an error (the baseline makes some of the same
    # ones with full schemas), so this flags a possible cause; Phase 10 compares error rates.
    compressed = {t.target for t in trace.transformations if t.strategy.startswith("E2")}
    for c in trace.tool_calls:
        if c.name in compressed and c.error_class in ARG_ERRORS and ARG_ERROR_TEXT.search(c.shaped_text or ""):
            value -= penalties["compressed_schema_arg_error"]
            reasons.append(f"{c.name} argument error ({c.error_class}) with a compressed schema (possible cause)")
    return _component(max(0.0, value), reasons or ["no expansions, low-confidence rewrites or "
                                                   "argument errors on compressed schemas"])


def freshness(trace: Trace, lookup: DatasetLookup | None) -> dict[str, Any]:
    started = datetime.fromisoformat(trace.started_at)
    checks: dict[str, tuple[bool, str]] = {}
    for t in trace.transformations:
        for e in t.evidence:
            d = e.detail or {}
            if t.strategy.startswith("E7") and "age_seconds" in d:
                ok = d["age_seconds"] <= d["ttl_seconds"]
                checks[f"cache:{e.ref}"] = (ok, f"cached result {e.ref}: age {d['age_seconds']:.0f}s of ttl {d['ttl_seconds']:.0f}s")
    refs = {e.ref.split("#", 1)[0] for item in [*trace.transformations, *trace.context_items] for e in item.evidence
            if e.kind == "dataset_row" and e.ref.startswith("datasets/")}
    for wref in sorted(refs):
        info = lookup(wref) if lookup else None
        if info is None:
            checks[wref] = (False, f"{wref}: version not found in the context store")
            continue
        fetched, ttl_days = info
        age = (started - datetime.fromisoformat(fetched)).total_seconds() / 86_400
        checks[wref] = (age <= ttl_days, f"{wref}: fetched {age:.1f} days before the run, ttl {ttl_days:g} days")
    if not checks:
        return _component(None, ["no cached results or datasets used"])
    return _component(sum(ok for ok, _ in checks.values()) / len(checks),
                      [("" if ok else "STALE ") + why for ok, why in checks.values()])


def trust(trace: Trace, policy: TrustPolicy | None = None, lookup: DatasetLookup | None = None) -> dict[str, Any]:
    policy = policy or load_policy()
    components = {"grounding": grounding(trace), "provenance": provenance(trace),
                  "fidelity": fidelity(trace, policy.fidelity_penalties), "freshness": freshness(trace, lookup)}
    used = {k: w for k, w in policy.weights.items() if components[k]["value"] is not None}
    total = sum(used.values())
    combined = sum(components[k]["value"] * w for k, w in used.items()) / total if total else None
    return {"trust": None if combined is None else round(combined, 4), "components": components,
            "weights": {k: round(w / total, 4) for k, w in used.items()} if total else {}}
