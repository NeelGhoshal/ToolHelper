"""Trace cards (README Phase 9): one self-contained HTML page per run with the timeline,
tokens per step, every transformation with before/after previews and evidence links, the
injected context items and the trust components with their reasons.

Evidence links: `<run_id>:<call_id>` jumps to that call (here, or on the other run's card in
the same folder); `recording:<server>/<tool>/<key>` opens the recorded response file;
dataset rows, schema hashes and file refs are shown as text.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from toolhelper.monitoring.tracer import Trace

STYLE = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--box:#f1f0ec;--ok:#2f9e44;--warn:#e8590c;--bad:#c92a2a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#9a9993;--line:#2e2d2a;--box:#22211f}}
body{margin:0 auto;max-width:1100px;padding:24px 16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
h1{font-size:20px;margin-bottom:4px} h2{margin-top:28px;font-size:16px} .muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:6px 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
code,pre{font:12px/1.45 ui-monospace,Consolas,monospace} pre{white-space:pre-wrap;word-break:break-word;
background:var(--box);padding:8px;border-radius:4px;max-height:320px;overflow:auto;margin:4px 0}
.turn{border-left:3px solid var(--line);padding:2px 0 2px 12px;margin:12px 0}
.score{display:inline-block;min-width:44px;font-weight:600} .ok{color:var(--ok)} .warn{color:var(--warn)}
.bad{color:var(--bad)} details summary{cursor:pointer;color:var(--muted)} a{color:inherit}
"""


def _score_class(v: float | None) -> str:
    return "muted" if v is None else "ok" if v >= 0.9 else "warn" if v >= 0.6 else "bad"


def _num(v: Any) -> str:
    return "–" if v is None else f"{v:,}" if isinstance(v, int) else f"{v:.2f}" if isinstance(v, float) else escape(str(v))


def _evidence(e, run_id: str, replay_dir: Path | None) -> str:
    ref = escape(e.ref)
    detail = f' <span class="muted">{escape(json.dumps(e.detail, ensure_ascii=False))[:240]}</span>' if e.detail else ""
    if e.kind == "run" and e.ref.startswith("recording:") and replay_dir:
        server_tool_key = e.ref.split(":", 1)[1]
        path = (replay_dir / f"{server_tool_key}.json").resolve()
        return f'<a href="{path.as_uri()}">{ref}</a>{detail}'
    if e.kind == "run" and ":" in e.ref:
        other, call = e.ref.split(":", 1)
        href = f"#call-{escape(call)}" if other == run_id else f"{escape(other)}.html#call-{escape(call)}"
        return f'<a href="{href}">{ref}</a>{detail}'
    return f"<code>{escape(e.kind)}</code> {ref}{detail}"


def _pre(text: str | None, label: str) -> str:
    return f"<details><summary>{label}</summary><pre>{escape(text)}</pre></details>" if text else ""


def render_card(trace: Trace, *, replay_dir: Path | None = None) -> str:
    t = trace
    trust = t.trust or {}
    comps = trust.get("components", {})
    prompt = sum(c.prompt_tokens for c in t.llm_calls)
    completion = sum(c.completion_tokens for c in t.llm_calls)
    measured = all(c.measured for c in t.llm_calls) if t.llm_calls else False

    trust_rows = "".join(
        f'<tr><td>{escape(name)}</td><td><span class="score {_score_class(c["value"])}">{_num(c["value"])}</span></td>'
        f'<td>{_num(trust.get("weights", {}).get(name))}</td>'
        f'<td>{"<br>".join(escape(r) for r in c["reasons"])}</td></tr>' for name, c in comps.items())
    trust_html = (f'<p>Trust <span class="score {_score_class(trust.get("trust"))}">{_num(trust.get("trust"))}</span> '
                  f'<span class="muted">(weights renormalized over the components that apply)</span></p>'
                  f"<table><thead><tr><th>Component</th><th>Value</th><th>Weight</th><th>Reasons</th></tr></thead>"
                  f"<tbody>{trust_rows}</tbody></table>") if comps else "<p class=muted>Not scored for trust yet.</p>"

    calls_by_turn: dict[int, list] = {}
    for c in t.tool_calls:
        calls_by_turn.setdefault(c.turn, []).append(c)
    turns = []
    for llm in t.llm_calls:
        parts = ", ".join(f"{k} {v:,}" for k, v in sorted(llm.est_breakdown.items(), key=lambda kv: -kv[1]) if v)
        calls = []
        for c in calls_by_turn.get(llm.turn, []):
            changed = c.final_arguments != c.arguments
            status = (f'<span class="bad">{escape(c.error_class)}</span>' if c.is_error else '<span class="ok">ok</span>')
            calls.append(
                f'<div id="call-{escape(c.call_id)}"><b>{escape(c.name)}</b> <span class="muted">{escape(c.call_id)} · '
                f'{escape(c.decision)} · {escape(c.source or "-")}</span> {status} · result {_num(c.raw_result_tokens)} '
                f'→ {_num(c.shaped_result_tokens)} tokens · {c.latency_ms:,.0f} ms'
                f'<pre>{escape(json.dumps(c.arguments, ensure_ascii=False))}</pre>'
                + (f'<pre>→ {escape(json.dumps(c.final_arguments, ensure_ascii=False))}</pre>' if changed else "")
                + _pre(c.shaped_text, "result as the model saw it")
                + (_pre(c.raw_text, "raw server result") if c.raw_text != c.shaped_text else "") + "</div>")
        turns.append(
            f'<div class="turn"><b>Turn {llm.turn}</b> <span class="muted">LLM: prompt {llm.prompt_tokens:,} · '
            f'completion {llm.completion_tokens:,}' + (f" · thinking {llm.thinking_tokens:,}" if llm.thinking_tokens else "")
            + f' · {"measured" if llm.measured else "estimated"} · {llm.latency_ms / 1000:,.1f} s · '
            f'{len(llm.tools_offered)} tools offered</span><br><span class="muted">estimated split: {escape(parts)}</span>'
            + "".join(calls) + "</div>")

    trans = "".join(
        f"<tr><td>{x.turn}</td><td>{escape(x.strategy)}<br><span class=muted>{escape(x.hook)}</span></td>"
        f"<td>{escape(x.target)}</td><td>{x.before_tokens:,} → {x.after_tokens:,}</td>"
        f"<td>{escape(x.note)}{'' if x.reported else ' <span class=bad>(not reported)</span>'}"
        f"{_pre(x.before_text, 'before')}{_pre(x.after_text, 'after')}</td>"
        f"<td>{'<br>'.join(_evidence(e, t.run_id, replay_dir) for e in x.evidence) or '<span class=bad>none</span>'}</td></tr>"
        for x in t.transformations)
    trans_html = (f"<table><thead><tr><th>Turn</th><th>Strategy</th><th>Target</th><th>Tokens</th><th>Change</th>"
                  f"<th>Evidence</th></tr></thead><tbody>{trans}</tbody></table>") if trans else "<p class=muted>None.</p>"
    items = "".join(
        f"<tr><td>{escape(i.strategy)}</td><td>{escape(i.kind)}</td><td>{escape(i.text)}</td><td>{i.tokens:,}</td>"
        f"<td>{'<br>'.join(_evidence(e, t.run_id, replay_dir) for e in i.evidence) or '<span class=bad>none</span>'}</td></tr>"
        for i in t.context_items)
    items_html = (f"<table><thead><tr><th>Strategy</th><th>Kind</th><th>Text</th><th>Tokens</th><th>Evidence</th></tr>"
                  f"</thead><tbody>{items}</tbody></table>") if items else "<p class=muted>None.</p>"
    strategies = ", ".join(s["name"] for s in t.config.get("strategies", [])) or "none"
    outcome = ("pass" if t.success else "fail") if t.success is not None else "not scored"

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>Trace {escape(t.run_id)}</title>
<style>{STYLE}</style></head><body>
<h1>{escape(t.task_id)} · {escape(outcome)} · {escape(t.stop_reason or "?")}</h1>
<p class=muted>Run {escape(t.run_id)} · group {escape(t.group_id or "-")} · {escape(t.backend)}:{escape(t.model)} ·
hub {escape(t.hub_mode)} · strategies: {escape(strategies)} · started {escape(t.started_at)}</p>
<p>{escape(t.prompt)}</p>
<p><b>Answer:</b> {escape(t.answer or "(none)")}</p>
{f'<p class=bad>Error: {escape(t.error)}</p>' if t.error else ''}
{f'<p class=muted>Retry of {escape(t.retry_of)}</p>' if t.retry_of else ''}
<p>{len(t.llm_calls)} LLM turns · {len(t.tool_calls)} tool calls · prompt {prompt:,} · completion {completion:,} tokens
({"measured" if measured else "estimated"}) · provenance {t.provenance_coverage:.2f}</p>
<h2>Trust</h2>{trust_html}
<h2>Timeline</h2>{''.join(turns) or '<p class=muted>No LLM calls.</p>'}
<h2>Transformations</h2>{trans_html}
<h2>Injected context</h2>{items_html}
</body></html>"""


def write_card(trace: Trace, directory: Path | str, *, replay_dir: Path | None = None) -> Path:
    path = Path(directory) / f"{trace.run_id}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_card(trace, replay_dir=replay_dir), encoding="utf-8")
    return path
