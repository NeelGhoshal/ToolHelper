import re

import pytest

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.llm.base import ChatOptions, ChatResult, Message, ToolSpec, Usage, estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend
from toolhelper.mcp.hub import Hub
from toolhelper.monitoring.db import MonitorDB
from toolhelper.monitoring.pathways import follows, identifiers, run_pathway, suggestions
from toolhelper.monitoring.report import build_report, render_html, write_report
from toolhelper.monitoring.stats import percentile, tool_metrics
from toolhelper.monitoring.tokens import calibrate, measure_tool_costs

pytestmark = pytest.mark.anyio


def call(name, args):
    return {"role": "assistant", "tool_calls": [{"name": name, "arguments": args}]}


def answer_with_id_from_last_result(messages, tools):
    last = next(m for m in reversed(messages) if m.role == "tool")
    return Message(role="assistant", content=f"Found {max(sorted(identifiers(last.content)), key=len)}.")


def recorded(store, tool, **match):
    return next(r.arguments for r in store if r.tool == tool
                and all(r.arguments.get(k) == v for k, v in match.items()))


@pytest.fixture
async def traces(server_configs, catalog, store):
    """Three runs over recorded real responses."""
    hub = Hub(server_configs, mode="replay", store=store, catalog=catalog, live_fallback=False)
    variant = recorded(store, "search", entity="variant")
    omop = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")
    bad = next(r.arguments for r in store if r.tool == "search-drugs" and r.arguments == {"limit": 3})
    good = recorded(store, "search-drugs", query="ibuprofen")
    runs = [
        (Task(id="bio-1", prompt="What is known about BRAF V600E variants?", servers=["biomcp"],
              expected_paths=[["biomcp__get", "biomcp__search"]]),
         [call("biomcp__get", {"entity": "gene", "id": "BRAF"}), call("biomcp__search", variant),
          answer_with_id_from_last_result], None),
        (Task(id="omop-1", prompt="Map heart attack to condition_occurrence.", servers=["omop"],
              expected_paths=[["omop__find_omop_concept"]]),
         [call("omop__find_omop_concept", omop), answer_with_id_from_last_result], ChatOptions(num_ctx=600)),
        (Task(id="med-1", prompt="Find ibuprofen products.", servers=["medical"],
              expected_paths=[["medical__search-drugs"]]),
         [call("medical__search-drugs", bad), call("medical__search-drugs", good),
          call("medical__search-drugs", good), call("medical__search-drug-nomenclature", {"query": "metformin"}),
          {"role": "assistant", "content": "Ibuprofen products are listed above."}], None),
    ]
    out = []
    async with hub:
        for task, steps, options in runs:
            trace = await Agent(ScriptedBackend(steps, defaults=options), hub, counter=estimate_tokens).run(task)
            trace.success = task.id != "med-1"
            out.append(trace)
    return out


def test_calibrate_sums_to_measured():
    parts = {"system": 70, "tool_defs": 3600, "history": 30}
    assert sum(calibrate(parts, 2791).values()) == 2791
    assert abs(calibrate(parts, 2791)["tool_defs"] - 3600 * 2791 / 3700) < 1
    assert calibrate(parts, None) == parts and calibrate({"a": 0}, 5) == {"a": 0}


class MeasuringFake:
    """A backend whose measured prompt cost is known: 100 base, 10 header when tools are present."""

    kind, model, defaults = "fake", "fake-1", ChatOptions(temperature=0)
    COST = {"a": 30, "b": 50, "c": 70}

    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None, options=None):
        self.calls.append((messages, options))
        tools = tools or []
        prompt = 100 + (10 if tools else 0) + sum(self.COST[t.name] for t in tools)
        return ChatResult(message=Message(role="assistant"), tool_calls=[], latency_ms=0, raw={}, backend="fake",
                          model="fake-1", usage=Usage(prompt=prompt, completion=1, measured=True))


def test_measure_tool_costs_with_and_without_tools():
    fake = MeasuringFake()
    costs = measure_tool_costs(fake, [ToolSpec(name=n) for n in "abc"], "system prompt")
    assert costs["all_tools"] == 160 and costs["base_prompt"] == 100
    assert costs["tools"] == {"a": {"alone": 40, "marginal": 30}, "b": {"alone": 60, "marginal": 50},
                              "c": {"alone": 80, "marginal": 70}}
    nonces = [re.match(r"\[(\d{12})\]", m[0].content).group(1) for m, _ in fake.calls]
    assert len(set(nonces)) == len(nonces) == 2 + 2 * 3
    assert all(o.max_tokens == 1 for _, o in fake.calls)


def test_biomcp_suggestions():
    hints = suggestions("## Next commands\n- `biomcp get gene BRAF pathways`\n- `biomcp search variant -g BRAF`")
    assert hints == ["get gene BRAF pathways", "search variant -g BRAF"]
    assert follows("search variant", "search variant -g BRAF")
    assert follows("get gene BRAF pathways", "get gene BRAF pathways")
    assert not follows("get gene BRAF", "get gene BRAF pathways")
    assert follows("get drug imatinib", "get drug <name>")


async def test_pathways(traces):
    bio, omop, med = (run_pathway(t) for t in traces)
    assert bio["sequence"] == ["biomcp__get", "biomcp__search"]
    assert bio["first_tool_correct"] is True
    assert bio["biomcp_suggestions"] == {"offered": 1, "followed": 1}  # followed `search variant -g BRAF`
    assert omop["turns_to_first_useful_call"] == 1 and omop["dead_ends"] == []
    assert med["retries"] == 1 and med["redundant_calls"] == 1
    assert len(med["detours"]) == 1  # search-drug-nomenclature is not on an expected path
    assert med["dead_ends"]  # results never referenced by the answer


async def test_ingest_report_and_metrics(traces, tmp_path):
    db = MonitorDB()
    for t in traces:
        db.ingest(t)
    db.ingest(traces[0])  # re-ingesting replaces, not duplicates
    assert db.rows("SELECT COUNT(*) AS n FROM runs")[0]["n"] == 3
    assert db.rows("SELECT COUNT(*) AS n FROM tool_calls")[0]["n"] == 2 + 1 + 4

    bio = traces[0]
    first = bio.tool_calls[0]
    [row] = db.rows("SELECT * FROM tool_calls WHERE call_id = ? AND run_id = ?", (first.call_id, bio.run_id))
    later = [c.tool_results_in_context.get(first.call_id, 0) for c in bio.llm_calls]
    assert row["carried_tokens"] == sum(later) == 2 * first.shaped_result_tokens  # seen by turns 2 and 3

    near = [w for w in build_report(db)["warnings"] if "num_ctx" in w["warning"]]
    assert near and {w["run_id"] for w in near} == {traces[1].run_id}

    tools = {t["tool"]: t for t in tool_metrics(db)}
    assert tools["medical__search-drugs"]["calls"] == 3
    assert tools["medical__search-drugs"]["error_rate"] == pytest.approx(1 / 3)
    assert sum(t["token_share"] for t in tools.values()) == pytest.approx(1.0)
    assert all(t["replayed_share"] == 1.0 for t in tools.values())

    report = build_report(db)
    assert report["overview"]["runs"] == 3 and report["overview"]["success_rate"] == pytest.approx(2 / 3)
    parts = report["overview"]["tokens_by_component"]
    assert parts["tool_defs"] > 0 and parts["tool_results"] > 0
    starts = {t["to"]: t["count"] for t in report["pathways"]["transitions"] if t["from"] == "START"}
    assert starts == {"biomcp__get": 1, "omop__find_omop_concept": 1, "medical__search-drugs": 1}
    html = render_html(report)
    for section in ("Overview", "Tools", "Tool-definition cost", "Pathways", "Runs", "Warnings"):
        assert f"<h2>{section}" in html
    json_path, html_path = write_report(report, tmp_path)
    assert json_path.exists() and html_path.exists()


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([5, 1, 3], 50) == 3 and percentile(list(range(1, 101)), 95) == 95


def test_calibrate_anchors_measured_tool_defs():
    parts = {"system": 70, "tool_defs": 3628, "history": 24}
    out = calibrate(parts, 2791, {"tool_defs": 2674})
    assert out["tool_defs"] == 2674 and sum(out.values()) == 2791
    assert calibrate(parts, 2000, {"tool_defs": 2674}) == calibrate(parts, 2000)  # inconsistent: ignored


def test_tool_defs_are_not_anchored_to_schemas_a_strategy_rewrote():
    from types import SimpleNamespace

    from toolhelper.monitoring.db import tool_defs_anchor

    costs = {"model": "m", "catalog_version": "c", "tools": {"a": {"alone": 40, "marginal": 30},
                                                             "b": {"alone": 60, "marginal": 50}}}
    call = SimpleNamespace(measured=True, tools_offered=["a", "b"])
    trace = SimpleNamespace(model="m", catalog_version="c", transformations=[])
    assert tool_defs_anchor(costs, trace, call) == 90
    trace.transformations = [SimpleNamespace(hook="transform_schemas", target="b")]  # E2 compressed b
    assert tool_defs_anchor(costs, trace, call) is None
    trace.transformations = [SimpleNamespace(hook="after_result", target="b")]
    assert tool_defs_anchor(costs, trace, call) == 90
    assert tool_defs_anchor(costs, trace, SimpleNamespace(measured=False, tools_offered=["a"])) is None


def test_measured_tool_defs():
    from toolhelper.monitoring.tokens import measured_tool_defs

    costs = {"model": "m", "catalog_version": "c", "tools": {"a": {"alone": 40, "marginal": 30},
                                                             "b": {"alone": 60, "marginal": 50}}}
    assert measured_tool_defs(costs, "m", "c", ["a", "b"]) == 10 + 30 + 50
    assert measured_tool_defs(costs, "m", "c", ["a"]) == 40
    assert measured_tool_defs(costs, "other", "c", ["a"]) is None
    assert measured_tool_defs(costs, "m", "c", ["a", "local__tool"]) is None
