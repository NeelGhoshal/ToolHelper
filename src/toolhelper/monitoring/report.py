"""`toolhelper report`: JSON and a self-contained HTML page from the monitoring database."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from toolhelper.monitoring.db import MonitorDB
from toolhelper.monitoring.pathways import aggregate
from toolhelper.monitoring.stats import run_metrics, tool_metrics, warnings

COMPONENTS = ["system", "tool_defs", "context", "history", "tool_results", "completion"]


def latest_tool_costs(directory: Path | str) -> dict[str, Any] | None:
    paths = sorted(Path(directory).glob("tool-costs-*.json"))
    return json.loads(paths[-1].read_text(encoding="utf-8")) if paths else None


def build_report(db: MonitorDB, *, group: str | None = None, tool_costs: dict[str, Any] | None = None) -> dict:
    runs = run_metrics(db, group)
    where, params = ("WHERE group_id = ?", (group,)) if group else ("", ())
    pathways = [json.loads(r["pathway_json"]) for r in db.rows(f"SELECT pathway_json FROM runs {where}", params)]
    scored = [r["success"] for r in runs if r["success"] is not None]
    totals = {k: sum(r["tokens_by_component"]["parts"].get(k, 0) for r in runs) for k in COMPONENTS}
    return {
        "generated_at": datetime.now(UTC).isoformat(), "group": group,
        "labels": {
            "prompt_tokens": "measured when the run's backend reports usage (see runs[].measured)",
            "tokens_by_component": "estimated per component, calibrated to the measured prompt total when measured",
            "tool token columns": "estimated (tokenizer in each trace)",
            "used": "heuristic: identifiers from the result reappear in later arguments or the answer",
        },
        "overview": {
            "runs": len(runs), "scored": len(scored),
            "success_rate": sum(scored) / len(scored) if scored else None,
            "complete_traces": sum(r["complete"] for r in runs),
            "all_measured": all(r["measured"] for r in runs) if runs else None,
            "mean_turns": sum(r["turns"] for r in runs) / len(runs) if runs else None,
            "tokens_by_component": totals,
        },
        "tools": tool_metrics(db, group), "runs": runs, "pathways": aggregate(pathways),
        "run_pathways": pathways, "warnings": warnings(db, group), "tool_costs": tool_costs,
    }


PERCENT = {"trust_score", "error_rate", "replayed_share", "used_rate", "token_share", "success_rate", "provenance",
           "first_tool_correct_rate"}


def _fmt(value: Any, key: str = "") -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.0%}" if key in PERCENT else f"{value:,.1f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return escape(str(value))


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p class=muted>None.</p>"
    head = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(r.get(key), key)}</td>" for key, _ in columns) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _bar(parts: dict[str, int]) -> str:
    total = sum(parts.values()) or 1
    segs = "".join(f'<span class="seg c{i}" style="width:{100 * parts.get(k, 0) / total:.2f}%" '
                   f'title="{k}: {parts.get(k, 0):,}"></span>' for i, k in enumerate(COMPONENTS))
    legend = " ".join(f'<span class="key c{i}"></span>{k} {parts.get(k, 0):,}' for i, k in enumerate(COMPONENTS))
    return f'<div class="bar">{segs}</div><p class="legend">{legend}</p>'


def render_html(report: dict[str, Any]) -> str:
    o, p = report["overview"], report["pathways"]
    tool_cols = [("tool", "Tool"), ("calls", "Calls"), ("error_rate", "Errors"), ("args_tokens_mean", "Args"),
                 ("raw_result_tokens_mean", "Raw result"), ("shaped_result_tokens_mean", "Shaped result"),
                 ("carried_tokens", "Carried"), ("latency_p50_ms", "p50 ms"), ("latency_p95_ms", "p95 ms"),
                 ("replayed_share", "Replayed"), ("used_rate", "Used"), ("token_share", "Share")]
    run_cols = [("run_id", "Run"), ("task_id", "Task"), ("model", "Model"), ("turns", "Turns"),
                ("tool_calls", "Calls"), ("prompt_tokens", "Prompt"), ("completion_tokens", "Completion"),
                ("measured", "Measured"), ("success", "Success"), ("trust_score", "Trust"), ("provenance", "Provenance"),
                ("complete", "Complete"), ("stop_reason", "Stop")]
    nodes = sorted({t["from"] for t in p["transitions"]} | {t["to"] for t in p["transitions"]},
                   key=lambda n: (n != "START", n == "END", n))
    counts = {(t["from"], t["to"]): t["count"] for t in p["transitions"]}
    matrix = "<table class=matrix><thead><tr><th>from \\ to</th>" + "".join(
        f"<th>{escape(n)}</th>" for n in nodes if n != "START") + "</tr></thead><tbody>" + "".join(
        "<tr><th>" + escape(a) + "</th>" + "".join(f"<td>{counts.get((a, b), '')}</td>" for b in nodes
                                                   if b != "START") + "</tr>" for a in nodes if a != "END"
    ) + "</tbody></table>" if nodes else "<p class=muted>No tool calls.</p>"
    seqs = "".join(f"<h3>{escape(k)}</h3>" + _table([{"sequence": s, "runs": n} for s, n in v],
                                                     [("sequence", "Sequence"), ("runs", "Runs")])
                   for k, v in p["common_sequences"].items() if v)
    costs = report.get("tool_costs")
    cost_html = "<p class=muted>Not measured yet: run <code>toolhelper tokens tools --llm &lt;profile&gt;</code>.</p>"
    if costs:
        rows = [{"tool": k, **v} for k, v in sorted(costs["tools"].items(), key=lambda kv: -kv[1]["marginal"])]
        cost_html = (f"<p>{escape(costs['backend'])}:{escape(costs['model'])}, measured: all tools add "
                     f"<b>{costs['all_tools']:,}</b> prompt tokens (base prompt {costs['base_prompt']:,}).</p>"
                     + _table(rows, [("tool", "Tool"), ("alone", "Alone"), ("marginal", "Marginal")]))
    sugg = p["biomcp_suggestion_following"]
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>ToolHelper report</title>
<style>
:root{{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--c0:#4c6ef5;--c1:#e8590c;--c2:#2f9e44;
--c3:#ae3ec9;--c4:#1098ad;--c5:#868e96}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ecebe6;--muted:#9a9993;--line:#2e2d2a}}}}
body{{margin:0 auto;max-width:1100px;padding:24px 16px;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,sans-serif}} h1{{font-size:22px}} h2{{margin-top:32px;font-size:17px}}
table{{border-collapse:collapse;width:100%;margin:8px 0;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} .muted,.legend{{color:var(--muted)}}
.bar{{display:flex;height:14px;border-radius:3px;overflow:hidden;background:var(--line)}}
.seg{{display:block}} .key{{display:inline-block;width:10px;height:10px;margin:0 4px 0 10px}}
.c0{{background:var(--c0)}}.c1{{background:var(--c1)}}.c2{{background:var(--c2)}}.c3{{background:var(--c3)}}
.c4{{background:var(--c4)}}.c5{{background:var(--c5)}} .matrix td{{text-align:center}}
</style></head><body>
<h1>ToolHelper report</h1>
<p class=muted>Generated {escape(report['generated_at'])}{' · group ' + escape(report['group']) if report['group'] else ''}.
Prompt totals are measured when the backend reports usage; component splits and all tool token
columns are estimated, with splits calibrated to measured totals.</p>
<h2>Overview</h2>
<p>{o['runs']} runs, {o['complete_traces']} complete traces, success {_fmt(o['success_rate'], 'success_rate')}
({o['scored']} scored), mean turns {_fmt(o['mean_turns'])}.</p>
{_bar(o['tokens_by_component'])}
<h2>Tools</h2>{_table(report['tools'], tool_cols)}
<h2>Tool-definition cost (measured)</h2>{cost_html}
<h2>Pathways</h2>
<p>First tool correct: {_fmt(p['first_tool_correct_rate'], 'first_tool_correct_rate')} · retries {p['retries']} · redundant calls
{p['redundant_calls']} · dead ends {p['dead_ends']} · detours {p['detours']} · BioMCP suggestions followed
{sugg['followed']}/{sugg['offered']}.</p>
<h3>Transitions</h3>{matrix}{seqs}
<h2>Runs</h2>{_table(report['runs'], run_cols)}
<h2>Warnings</h2>{_table(report['warnings'], [('run_id', 'Run'), ('turn', 'Turn'), ('warning', 'Warning')])}
</body></html>"""


def write_report(report: dict[str, Any], directory: Path | str) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, html_path = directory / "report.json", directory / "report.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path
