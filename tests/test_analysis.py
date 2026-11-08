"""Phase 10 statistics on small constructed outcome tables (analysis inputs, not server data)."""

from toolhelper.experiments.analysis import RunRow, compare, gate, mcnemar, summarize


def arm(outcomes: dict[str, tuple[bool, int]], group="g", trust=1.0, transient=()):
    return {t: RunRow(run_id=f"{group}-{t}", group=group, task_id=t, kind="omop" if t < "m" else "medical",
                      success=s, tokens=tok, trust=trust, transient=t in transient) for t, (s, tok) in outcomes.items()}


TASKS = [f"t{i:02d}" for i in range(20)]
BASE = [arm({t: (i % 4 != 0, 10_000) for i, t in enumerate(TASKS)}, "b1")]  # 15/20 pass, 10k tokens each


def test_summary_metrics():
    s = summarize(BASE, TASKS)
    assert s["runs"] == 20 and s["success_rate"] == 0.75
    assert s["tokens_per_success"] == 200_000 / 15 and s["median_tokens"] == 10_000 and s["p90_tokens"] == 10_000


def test_cheaper_arm_with_same_success():
    cheaper = [arm({t: (i % 4 != 0, 6_000) for i, t in enumerate(TASKS)}, "a1")]
    c = compare(cheaper, BASE, TASKS, samples=500)
    assert c["delta_success"] == 0 and c["delta_tokens_per_success"] == -0.4
    lo, hi = c["delta_tokens_per_success_ci"]
    assert lo <= -0.4 <= hi and c["mcnemar"] == {"lost": 0, "gained": 0, "p": 1.0}
    assert gate(c, summarize(cheaper, TASKS), max_success_drop=0.02, min_trust=0.9)["passed"]


def test_success_drop_fails_the_gate_and_mcnemar_counts_flips():
    worse = [arm({t: (i % 2 == 1, 8_000) for i, t in enumerate(TASKS)}, "a1")]  # 10/20
    c = compare(worse, BASE, TASKS, samples=500)
    assert c["delta_success"] == -0.25
    m = mcnemar(worse, BASE, TASKS)
    assert (m["lost"], m["gained"]) == (5, 0) and m["p"] == 0.0625  # 2 * 0.5**5
    g = gate(c, summarize(worse, TASKS), max_success_drop=0.02, min_trust=0.9)
    assert not g["passed"] and g["reasons"] == ["success -25.0 points (limit -2)"]


def test_low_trust_fails_and_near_misses_are_close_calls():
    lowtrust = [arm({t: (i % 4 != 0, 9_000) for i, t in enumerate(TASKS)}, "a1", trust=0.89)]
    g = gate(compare(lowtrust, BASE, TASKS, samples=100), summarize(lowtrust, TASKS), max_success_drop=0.02,
             min_trust=0.9)
    assert not g["passed"] and g["close_call"] and g["reasons"] == ["trust 0.890 < 0.9"]


def test_transient_failures_are_excluded_not_counted():
    flaky = [arm({t: (i % 4 != 0, 10_000) for i, t in enumerate(TASKS)}, "a1", transient={"t01", "t02"})]
    s, c = summarize(flaky, TASKS), compare(flaky, BASE, TASKS, samples=100)
    assert s["runs"] == 18 and s["transient_excluded"] == 2 and c["tasks"] == 18 and c["delta_success"] == 0
