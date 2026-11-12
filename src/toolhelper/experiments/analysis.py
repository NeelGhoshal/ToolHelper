"""Phase 10 statistics: arms vs the baseline, paired by task.

An arm is a list of repeat groups; each group contributes its latest run per task. Per task
the repeats are averaged (success share, tokens), and comparisons resample tasks:

- success rate and tokens per successful task (total tokens / successes), with median and
  p90 tokens per run, mean trust, and the same per server kind;
- paired bootstrap 95% CIs on the change in success (points) and the relative change in
  tokens per success;
- McNemar's exact test on success flips, pairing repeat i of the arm with repeat i of the
  baseline (pooled over pairs);
- the gate (success drop <= max_success_drop, mean trust >= min_trust) and close calls.

Runs that still ended in a transient API failure (after retries) are left out of a task's
comparison rather than counted as model failures.
"""

from __future__ import annotations

import json
import random
import statistics
from math import comb
from typing import Any

from pydantic import BaseModel

# The model endpoint answering 404 is an outage too (NVIDIA, 2026-11-11 23:49-23:55 local: 116 p10e runs ended
# before the first turn); the full message keeps the context store's own NotFoundError out.
TRANSIENT = ("APITimeoutError", "APIConnectionError", "NotFoundError: Error code: 404")


class RunRow(BaseModel):
    run_id: str
    group: str
    task_id: str
    kind: str
    success: bool
    tokens: int  # prompt + completion, measured when the backend reports usage
    trust: float | None
    transient: bool = False


def group_runs(db, group: str, kinds: dict[str, str]) -> dict[str, RunRow]:
    """Latest run per task in `group`."""
    rows = db.rows("SELECT run_id, task_id, success, prompt_tokens, completion_tokens, trust_json, error, started_at "
                   "FROM runs WHERE group_id = ? ORDER BY started_at", (group,))
    out = {}
    for r in rows:
        trust = (json.loads(r["trust_json"]) or {}).get("trust") if r["trust_json"] else None
        out[r["task_id"]] = RunRow(run_id=r["run_id"], group=group, task_id=r["task_id"],
                                   kind=kinds.get(r["task_id"], "?"), success=bool(r["success"]),
                                   tokens=(r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0), trust=trust,
                                   transient=(r["error"] or "").startswith(TRANSIENT))
    return out


Arm = list[dict[str, RunRow]]  # one dict per repeat group


def _runs(arm: Arm, tasks: list[str]) -> list[RunRow]:
    return [g[t] for g in arm for t in tasks if t in g and not g[t].transient]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize(arm: Arm, tasks: list[str]) -> dict[str, Any]:
    runs = _runs(arm, tasks)
    if not runs:
        return {"runs": 0}
    successes = sum(r.success for r in runs)
    tokens = [r.tokens for r in runs]
    trusts = [r.trust for r in runs if r.trust is not None]
    out = {"runs": len(runs), "repeats": len(arm), "success_rate": successes / len(runs),
           "tokens_per_success": sum(tokens) / successes if successes else None,
           "median_tokens": statistics.median(tokens), "p90_tokens": _pct(tokens, 0.9),
           "trust_mean": statistics.mean(trusts) if trusts else None,
           "transient_excluded": sum(1 for g in arm for t in tasks if t in g and g[t].transient)}
    return out


def by_kind(arm: Arm, tasks: list[str]) -> dict[str, dict[str, Any]]:
    kinds = sorted({r.kind for g in arm for r in g.values()})
    return {k: summarize(arm, [t for t in tasks if any(t in g and g[t].kind == k for g in arm)]) for k in kinds}


def _task_stats(arm: Arm, tasks: list[str]) -> dict[str, tuple[float, float]]:
    """task -> (mean success, mean tokens) over the arm's repeats."""
    out = {}
    for t in tasks:
        runs = [g[t] for g in arm if t in g and not g[t].transient]
        if runs:
            out[t] = (sum(r.success for r in runs) / len(runs), sum(r.tokens for r in runs) / len(runs))
    return out


def mcnemar(arm: Arm, base: Arm, tasks: list[str]) -> dict[str, Any]:
    b = c = 0  # b: baseline passed, arm failed; c: baseline failed, arm passed
    for i, g in enumerate(arm):
        bg = base[min(i, len(base) - 1)]
        for t in tasks:
            if t in g and t in bg and not (g[t].transient or bg[t].transient):
                b += bg[t].success and not g[t].success
                c += g[t].success and not bg[t].success
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return {"lost": b, "gained": c, "p": round(p, 4)}


def compare(arm: Arm, base: Arm, tasks: list[str], *, samples: int = 2000, seed: int = 7) -> dict[str, Any]:
    a, b = _task_stats(arm, tasks), _task_stats(base, tasks)
    paired = sorted(set(a) & set(b))
    if not paired:
        return {"tasks": 0}

    def metrics(ids: list[str]) -> tuple[float, float | None]:
        ds = sum(a[t][0] - b[t][0] for t in ids) / len(ids)
        sa, sb = sum(a[t][0] for t in ids), sum(b[t][0] for t in ids)
        if not sa or not sb:
            return ds, None
        tps_a, tps_b = sum(a[t][1] for t in ids) / sa, sum(b[t][1] for t in ids) / sb
        return ds, tps_a / tps_b - 1

    point_s, point_t = metrics(paired)
    rng = random.Random(seed)
    boot_s, boot_t = [], []
    for _ in range(samples):
        ids = [rng.choice(paired) for _ in paired]
        s, t = metrics(ids)
        boot_s.append(s)
        if t is not None:
            boot_t.append(t)

    def ci(values: list[float]) -> list[float] | None:
        return [round(_pct(values, 0.025), 4), round(_pct(values, 0.975), 4)] if values else None

    return {"tasks": len(paired), "delta_success": round(point_s, 4), "delta_success_ci": ci(boot_s),
            "delta_tokens_per_success": None if point_t is None else round(point_t, 4),
            "delta_tokens_per_success_ci": ci(boot_t), "mcnemar": mcnemar(arm, base, paired)}


def gate(cmp: dict[str, Any], summary: dict[str, Any], *, max_success_drop: float, min_trust: float,
         margin: float = 0.02) -> dict[str, Any]:
    ds, trust = cmp.get("delta_success"), summary.get("trust_mean")
    reasons = []
    if ds is not None and ds < -max_success_drop:
        reasons.append(f"success {ds * 100:+.1f} points (limit -{max_success_drop * 100:.0f})")
    if trust is not None and trust < min_trust:
        reasons.append(f"trust {trust:.3f} < {min_trust}")
    close = (ds is not None and abs(ds + max_success_drop) <= margin) or \
            (trust is not None and abs(trust - min_trust) <= margin)
    return {"passed": not reasons, "reasons": reasons, "close_call": close}
