"""ToolHelper command line. Every command that talks to a model takes `--llm <profile>`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import anyio
import typer
import yaml
from rich.console import Console
from rich.table import Table

from toolhelper.config import (
    CATALOG_DIR,
    CONTEXT_DB,
    REPLAY_DIR,
    MONITOR_DB,
    REPORTS_DIR,
    ROOT,
    TASKS_FILE,
    TRACES_DIR,
    load_llm_config,
    load_servers_config,
    make_backend,
)
from toolhelper.context.ingest import import_catalog
from toolhelper.context.local_store import LocalStore
from toolhelper.context.model import NotFoundError
from toolhelper.llm.base import Message, ToolSpec
from toolhelper.mcp.catalog import (
    CatalogSnapshot,
    build_snapshot,
    hub_catalog_hash,
    diff_snapshots,
    latest_snapshot,
    load_snapshot,
    save_snapshot,
    tiktoken_counter,
)
from toolhelper.mcp.hub import Hub, ToolCallTimeout
from toolhelper.mcp.recorder import RecordingStore

app = typer.Typer(no_args_is_help=True)
llm_app = typer.Typer(no_args_is_help=True, help="LLM backends and profiles.")
app.add_typer(llm_app, name="llm")
context_app = typer.Typer(no_args_is_help=True, help="Versioned context store.")
app.add_typer(context_app, name="context")
trace_app = typer.Typer(no_args_is_help=True, help="Run traces.")
app.add_typer(trace_app, name="trace")
monitor_app = typer.Typer(no_args_is_help=True, help="Monitoring database.")
app.add_typer(monitor_app, name="monitor")
tokens_app = typer.Typer(no_args_is_help=True, help="Token measurements.")
app.add_typer(tokens_app, name="tokens")
bench_app = typer.Typer(no_args_is_help=True, help="Benchmark tasks.")
app.add_typer(bench_app, name="bench")
datasets_app = typer.Typer(no_args_is_help=True, help="Grounding datasets and the entity index.")
app.add_typer(datasets_app, name="datasets")
experiment_app = typer.Typer(no_args_is_help=True, help="Phase 10 experiments.")
app.add_typer(experiment_app, name="experiment")
console = Console()

LLMOption = Annotated[str | None, typer.Option("--llm", help="LLM profile from configs/llm.yaml.")]

PING_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


@llm_app.command("profiles")
def profiles() -> None:
    """List the configured LLM profiles."""
    cfg = load_llm_config()
    table = Table("profile", "backend", "model", "options")
    for name, p in cfg.profiles.items():
        label = f"{name} (default)" if name == cfg.default else name
        table.add_row(label, p.backend, p.model, json.dumps(p.options.model_dump(exclude_none=True)))
    console.print(table)


@llm_app.command("ping")
def ping(llm: LLMOption = None) -> None:
    """Send one tool-calling test turn (live) and show the result."""
    name, profile = load_llm_config().get(llm)
    backend = make_backend(profile)
    result = backend.chat([Message(role="user", content="What's the weather in Paris?")], [PING_TOOL])
    tokens = "measured" if result.usage.measured else "estimated"
    console.print(f"[bold]{name}[/] {result.backend}:{result.model}  {result.latency_ms:.0f} ms")
    console.print(f"usage ({tokens}): prompt={result.usage.prompt} completion={result.usage.completion}"
                  f" thinking={result.usage.thinking} cached={result.usage.cached_prompt}")
    console.print(f"applied={result.applied_options} ignored={result.ignored_options}"
                  f" stop={result.stop_reason}")
    for call in result.tool_calls:
        console.print(f"tool call: {call.name}({json.dumps(call.arguments)})")
    if result.message.content:
        console.print(f"content: {result.message.content}")
    if not result.tool_calls:
        raise typer.Exit(1)


ModeOption = Annotated[str, typer.Option("--mode", help="live, record or replay.")]
ServersOption = Annotated[str | None, typer.Option("--servers", help="Comma-separated subset of servers.")]
StoreOption = Annotated[Path, typer.Option("--store", help="Recording store directory.")]
CatalogDirOption = Annotated[Path, typer.Option("--catalog-dir", help="Catalog snapshot directory.")]
ContextOption = Annotated[Path, typer.Option("--db", help="Local context store (SQLite).")]


def make_hub(mode: str, servers: str | None, store: Path, catalog_dir: Path, *,
             live_fallback: bool = True) -> Hub:
    configs = load_servers_config()
    if servers:
        names = [s.strip() for s in servers.split(",")]
        configs = {n: configs[n] for n in names}
    catalog = latest_snapshot(catalog_dir) if mode == "replay" else None
    if mode == "replay" and catalog is None:
        raise typer.BadParameter(f"no catalog snapshot in {catalog_dir}; run `toolhelper catalog` first")
    return Hub(configs, mode=mode, store=RecordingStore(store), catalog=catalog, live_fallback=live_fallback)


@app.command("catalog")
def catalog_cmd(mode: ModeOption = "live", servers: ServersOption = None,
                store: StoreOption = REPLAY_DIR, catalog_dir: CatalogDirOption = CATALOG_DIR,
                save: Annotated[bool, typer.Option(help="Save a new snapshot if the catalog changed.")] = True,
                store_context: Annotated[bool, typer.Option(
                    "--store-context/--no-store-context", help="Write ToolSchema versions.")] = True,
                context_db: ContextOption = CONTEXT_DB,
                ) -> None:
    """Snapshot every tool schema, hash it, count its tokens and diff against the last snapshot."""
    previous = latest_snapshot(catalog_dir)

    async def run() -> CatalogSnapshot:
        async with make_hub(mode, servers, store, catalog_dir) as hub:
            return build_snapshot(hub, tiktoken_counter())

    snap = anyio.run(run)
    table = Table("tool", "tokens", "schema hash")
    for s in snap.servers.values():
        for t in s.tools:
            table.add_row(t.namespaced, str(t.tokens), t.schema_hash[:12])
        table.add_row(f"[bold]{s.name}[/] ({len(s.tools)} tools, {s.version})", f"[bold]{s.tokens}[/]",
                      s.server_hash[:12], end_section=True)
    console.print(table)
    console.print(f"{snap.tool_count} tools, {snap.tokens} tokens (estimated, {snap.tokenizer}), "
                  f"source={snap.source}, catalog hash {snap.catalog_hash[:16]}")

    if previous is None:
        diff = None
    else:
        if servers:
            previous = previous.model_copy(update={
                "servers": {n: s for n, s in previous.servers.items() if n in snap.servers}})
        diff = diff_snapshots(previous, snap)
        same = previous.catalog_hash == snap.catalog_hash if not servers else not (
            diff.added or diff.removed or diff.changed)
        console.print(f"vs snapshot of {previous.captured_at} ({previous.source}): "
                      + ("[green]tool hashes identical[/]" if same else "[yellow]changed[/]"))
        for name in diff.added:
            console.print(f"  + {name}")
        for name in diff.removed:
            console.print(f"  - {name}")
        for change in diff.changed:
            console.print(f"  ~ {change.name}: {', '.join(change.fields)}")
        for name, change in diff.server_info.items():
            console.print(f"  server {name}: {change['old']} -> {change['new']}")
    if save and mode != "replay" and not servers and (diff is None or not diff.empty):
        console.print(f"saved {save_snapshot(snap, catalog_dir)}")
    if store_context and mode != "replay" and not servers:
        store_snapshot(snap, context_db)


def store_snapshot(snap: CatalogSnapshot, context_db: Path) -> None:
    configs = load_servers_config()
    launch = {n: " ".join([c.command, *c.args]) for n, c in configs.items()}
    store = LocalStore(context_db)
    result = import_catalog(store, snap, launch)
    console.print(f"context store: {len(result.written)} versions written, "
                  f"{len(result.unchanged)} unchanged")
    for s in result.stale:
        console.print(f"  [yellow]stale[/] {s.thing.wref}: "
                      + ", ".join(f"{l.target}@v{l.target_version} (head v{l.target_head})" for l in s.links))
    store.close()


@app.command("call")
def call_cmd(name: str, arguments: Annotated[str, typer.Argument(help="JSON object.")] = "{}",
             mode: ModeOption = "live", store: StoreOption = REPLAY_DIR,
             catalog_dir: CatalogDirOption = CATALOG_DIR) -> None:
    """Call one namespaced tool, e.g. `toolhelper call medical__search-drugs '{"query": "ibuprofen"}'`."""
    server = name.split("__", 1)[0]

    async def run():
        async with make_hub(mode, server, store, catalog_dir) as hub:
            return await hub.call(name, json.loads(arguments))

    result = anyio.run(run)
    console.print(f"[bold]{result.name}[/] source={result.source} is_error={result.is_error} "
                  f"{result.latency_ms:.0f} ms")
    console.print(result.text, markup=False)


@app.command("record")
def record_cmd(calls: Annotated[Path, typer.Argument(help="YAML file with a `calls` list.")],
               store: StoreOption = REPLAY_DIR, catalog_dir: CatalogDirOption = CATALOG_DIR,
               only_missing: Annotated[bool, typer.Option(help="Skip calls already recorded.")] = False,
               ) -> None:
    """Record real responses for a list of calls (live). `{repo}` in string args is the repo root."""
    entries = yaml.safe_load(calls.read_text(encoding="utf-8"))["calls"]
    servers = ",".join(dict.fromkeys(e["server"] for e in entries))

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("{repo}", str(ROOT))
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value

    async def run() -> None:
        async with make_hub("record", servers, store, catalog_dir) as hub:
            for e in entries:
                name = f"{e['server']}__{e['tool']}"
                args = expand(e.get("args", {}))
                if only_missing and hub.store.get(e["server"], e["tool"], args):
                    continue
                try:
                    r = await hub.call(name, args)
                except ToolCallTimeout as exc:
                    console.print(f"[red]timeout[/] {name}: {exc}")
                    continue
                kind = "protocol_error" if r.protocol_error else ("is_error" if r.is_error else "ok")
                console.print(f"{kind:>14}  {name} {json.dumps(r.arguments)[:80]}  "
                              f"{len(r.text)} chars, {r.latency_ms:.0f} ms")

    anyio.run(run)


def _show(thing) -> None:
    status = "" if thing.active else f" [red]retracted[/] ({thing.reason or 'no reason'})"
    console.print(f"[bold]{thing.wref}[/] {','.join(thing.shapes)}{status}  {thing.committed_at or ''}"
                  f"  {thing.message or ''}")


@context_app.command("import-catalog")
def context_import(snapshot: Annotated[Path | None, typer.Argument(help="Snapshot file (default: latest).")] = None,
                   catalog_dir: CatalogDirOption = CATALOG_DIR, db: ContextOption = CONTEXT_DB) -> None:
    """Store a catalog snapshot as McpServer and ToolSchema versions."""
    snap = load_snapshot(snapshot) if snapshot else latest_snapshot(catalog_dir)
    if snap is None:
        raise typer.BadParameter(f"no catalog snapshot in {catalog_dir}")
    store_snapshot(snap, db)


@context_app.command("head")
def context_head(shape: Annotated[str | None, typer.Option(help="Only this shape.")] = None,
                 match: Annotated[str | None, typer.Option(help="Name glob, e.g. 'tools/omop/*'.")] = None,
                 retracted: Annotated[bool, typer.Option(help="Include retracted things.")] = False,
                 db: ContextOption = CONTEXT_DB) -> None:
    """List current versions."""
    for thing in LocalStore(db).head(shape=shape, match=match, include_retracted=retracted):
        _show(thing)


@context_app.command("show")
def context_show(wref: str, db: ContextOption = CONTEXT_DB) -> None:
    """Show one thing (name, name@HEAD or name@vN) with its data and links."""
    store = LocalStore(db)
    try:
        thing = store.get(wref, include_retracted=True)
    except NotFoundError as e:
        raise typer.BadParameter(str(e)) from None
    _show(thing)
    console.print_json(json.dumps(thing.data))
    for link in store.refs(thing.name, direction="outbound"):
        console.print(f"  -> {link.field}: {link.target}@v{link.target_version}"
                      + (" [yellow]stale[/]" if link.stale else ""))
    for link in store.refs(thing.name):
        console.print(f"  <- {link.source}@v{link.source_version} ({link.field})"
                      + (" [yellow]stale[/]" if link.stale else ""))


@context_app.command("history")
def context_history(name: str, db: ContextOption = CONTEXT_DB) -> None:
    """List every version of a thing."""
    for thing in LocalStore(db).history(name):
        _show(thing)


@context_app.command("stale")
def context_stale(shape: Annotated[str | None, typer.Option(help="Only this shape.")] = None,
                  db: ContextOption = CONTEXT_DB) -> None:
    """List things pinned to versions that have since been revised or retracted."""
    stale = LocalStore(db).stale(shape=shape)
    for s in stale:
        console.print(f"[yellow]{s.thing.wref}[/] " + ", ".join(
            f"{l.field} -> {l.target}@v{l.target_version} (head v{l.target_head}"
            + ("" if l.target_active else ", retracted") + ")" for l in s.links))
    if not stale:
        console.print("nothing stale")


@app.command("publish")
def publish_cmd(repo: Annotated[str, typer.Option(help="WarmHub repository, org/name.")],
                shapes: Annotated[str | None, typer.Option(help="Comma-separated shapes to publish.")] = None,
                dry_run: Annotated[bool, typer.Option(help="Show the plan without writing.")] = False,
                db: ContextOption = CONTEXT_DB) -> None:
    """Push curated context (accepted, not stale, with evidence) to WarmHub. Needs WH_TOKEN."""
    from toolhelper.context.shapes import SHAPES
    from toolhelper.context.warmhub_store import WarmHubStore, publish

    remote = WarmHubStore(repo)
    if not dry_run:
        created = remote.ensure_shapes(list(SHAPES))
        if created:
            console.print(f"created shapes: {', '.join(created)}")
    plan = publish(LocalStore(db), remote, shapes=shapes.split(",") if shapes else None, dry_run=dry_run)
    for wref in plan.written:
        console.print(f"{'would write' if dry_run else 'wrote'} {wref}")
    for name, why in plan.skipped.items():
        console.print(f"[dim]skipped {name}: {why}[/]")
    console.print(f"{len(plan.written)} written, {len(plan.unchanged)} unchanged, {len(plan.skipped)} skipped")


TracesOption = Annotated[Path, typer.Option("--traces", help="Trace directory.")]
MonitorOption = Annotated[Path, typer.Option("--monitor-db", help="Monitoring database (SQLite).")]


@app.command("run")
def run_cmd(prompt: Annotated[str, typer.Option(help="The task prompt.")],
            task_id: Annotated[str, typer.Option(help="Task id recorded in the trace.")] = "adhoc",
            llm: LLMOption = None, mode: ModeOption = "replay", servers: ServersOption = None,
            store: StoreOption = REPLAY_DIR, catalog_dir: CatalogDirOption = CATALOG_DIR,
            max_turns: Annotated[int, typer.Option(help="Maximum LLM turns.")] = 8,
            traces: TracesOption = TRACES_DIR, db: MonitorOption = MONITOR_DB) -> None:
    """Run the agent on one task, save its trace and add it to the monitoring database.
    Replay mode records misses from the live servers."""
    from toolhelper.agent.hooks import Task
    from toolhelper.agent.loop import Agent, AgentConfig

    _, profile = load_llm_config().get(llm)
    task = Task(id=task_id, prompt=prompt, servers=servers.split(",") if servers else [])

    async def go():
        async with make_hub(mode, servers, store, catalog_dir) as hub:
            return await Agent(make_backend(profile), hub, config=AgentConfig(max_turns=max_turns)).run(task)

    trace = anyio.run(go)
    path = trace.save(traces)
    show_trace(trace)
    from toolhelper.monitoring.db import MonitorDB

    from toolhelper.monitoring.report import latest_tool_costs

    MonitorDB(db, latest_tool_costs(catalog_dir)).ingest(trace, path)
    console.print(f"saved {path}")


@trace_app.command("card")
def trace_card(run_id: Annotated[str | None, typer.Argument(help="Run id or trace file.")] = None,
               group: Annotated[str | None, typer.Option(help="Write cards for every run in this group.")] = None,
               out: Annotated[Path | None, typer.Option(help="Output folder (default data/reports/cards).")] = None,
               traces: TracesOption = TRACES_DIR, db: MonitorOption = MONITOR_DB,
               store: StoreOption = REPLAY_DIR) -> None:
    """Write HTML trace cards: timeline, tokens per step, transformations with evidence, trust."""
    from toolhelper.monitoring.card import write_card
    from toolhelper.monitoring.db import MonitorDB
    from toolhelper.monitoring.tracer import Trace

    if group:
        rows = MonitorDB(db).rows("SELECT trace_path FROM runs WHERE group_id = ?", (group,))
        paths = [Path(r["trace_path"]) for r in rows if r["trace_path"]]
    elif run_id:
        paths = [Path(run_id) if Path(run_id).exists() else traces / f"{run_id}.json"]
    else:
        raise typer.BadParameter("give a run id or --group")
    trust_fn, out = _trust_fn(), out or REPORTS_DIR / "cards"
    for path in paths:
        trace = Trace.load(path)
        if trace.trust is None:
            trace.trust = trust_fn(trace)
        console.print(f"wrote {write_card(trace, out, replay_dir=store)}")


@trace_app.command("show")
def trace_show(run_id: Annotated[str, typer.Argument(help="Run id or trace file.")],
               traces: TracesOption = TRACES_DIR) -> None:
    """Print a run's timeline: LLM calls, tool calls, transformations and the answer."""
    from toolhelper.monitoring.tracer import Trace

    path = Path(run_id) if Path(run_id).exists() else traces / f"{run_id}.json"
    show_trace(Trace.load(path))


def show_trace(trace) -> None:
    console.print(f"[bold]{trace.run_id}[/] task={trace.task_id} {trace.backend}:{trace.model} "
                  f"hub={trace.hub_mode} catalog={trace.catalog_version[:12]} config={trace.config_hash[:12]}")
    console.print(f"params={trace.params}  tools offered={len(trace.tools_offered)}")
    for item in trace.context_items:
        console.print(f"  context [{item.kind}] from {item.strategy}: {item.tokens} tokens (estimated)")
    calls = {}
    for t in trace.tool_calls:
        calls.setdefault(t.turn, []).append(t)
    transforms = {}
    for t in trace.transformations:
        transforms.setdefault(t.turn, []).append(t)
    for t in transforms.get(0, []):
        console.print(f"  [magenta]{t.strategy}.{t.hook}[/] {t.target}: {t.before_tokens} -> {t.after_tokens}"
                      + ("" if t.evidence else " [red]no evidence[/]"))
    for c in trace.llm_calls:
        label = "measured" if c.measured else "estimated"
        parts = " ".join(f"{k}={v}" for k, v in c.est_breakdown.items())
        console.print(f"turn {c.turn}: prompt={c.prompt_tokens} completion={c.completion_tokens} ({label}), "
                      f"{c.latency_ms:.0f} ms, done={c.done_reason}")
        console.print(f"  estimated breakdown: {parts}", style="dim")
        for t in calls.get(c.turn, []):
            status = f"[red]{t.error_class}[/]" if t.is_error else "ok"
            raw = "-" if t.raw_result_tokens is None else t.raw_result_tokens
            console.print(f"  {t.call_id} {t.name} {canonical(t.final_arguments)} -> {status}, {t.decision}, "
                          f"{t.source or '-'}, result tokens {raw} -> {t.shaped_result_tokens}", markup=True)
        for t in transforms.get(c.turn, []):
            console.print(f"  [magenta]{t.strategy}.{t.hook}[/] {t.target}: {t.before_tokens} -> "
                          f"{t.after_tokens}" + ("" if t.evidence else " [red]no evidence[/]"))
    console.print(f"stop={trace.stop_reason} provenance={trace.provenance_coverage:.2f}")
    if trace.error:
        console.print(f"[red]error[/] {trace.error}")
    if trace.answer is not None:
        console.print("answer:", style="bold")
        console.print(trace.answer, markup=False)
    problems = trace.problems()
    console.print("[green]trace complete[/]" if not problems else f"[red]incomplete[/]: {problems}")


def canonical(value) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 80 else text[:77] + "..."


@monitor_app.command("ingest")
def monitor_ingest(traces: TracesOption = TRACES_DIR, db: MonitorOption = MONITOR_DB,
                   catalog_dir: CatalogDirOption = CATALOG_DIR) -> None:
    """(Re)load every trace in a directory into the monitoring database."""
    from toolhelper.monitoring.db import MonitorDB
    from toolhelper.monitoring.report import latest_tool_costs

    count = MonitorDB(db, latest_tool_costs(catalog_dir)).ingest_dir(traces)
    console.print(f"ingested {count} traces into {db}")


@app.command("report")
def report_cmd(group: Annotated[str | None, typer.Option(help="Only runs in this group.")] = None,
               out: Annotated[Path | None, typer.Option(help="Output directory.")] = None,
               db: MonitorOption = MONITOR_DB, catalog_dir: CatalogDirOption = CATALOG_DIR) -> None:
    """Write report.json and report.html: tools, runs, tokens by component, pathways, warnings."""
    from datetime import UTC, datetime

    from toolhelper.monitoring.db import MonitorDB
    from toolhelper.monitoring.report import build_report, latest_tool_costs, write_report

    report = build_report(MonitorDB(db), group=group, tool_costs=latest_tool_costs(catalog_dir))
    out = out or REPORTS_DIR / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    json_path, html_path = write_report(report, out)
    o = report["overview"]
    console.print(f"{o['runs']} runs, {len(report['tools'])} tools, {len(report['warnings'])} warnings")
    console.print(f"wrote {json_path}\nwrote {html_path}")


@tokens_app.command("tools")
def tokens_tools(llm: LLMOption = None, servers: ServersOption = None, mode: ModeOption = "replay",
                 store: StoreOption = REPLAY_DIR, catalog_dir: CatalogDirOption = CATALOG_DIR) -> None:
    """Measure what the tool definitions cost the model itself (with vs without tools, 1 output token).
    Makes 2 + 2 x tools LLM calls; no tool is called."""
    from datetime import UTC, datetime

    from toolhelper.agent.loop import DEFAULT_SYSTEM_PROMPT
    from toolhelper.monitoring.tokens import measure_tool_costs

    name, profile = load_llm_config().get(llm)

    async def specs():
        async with make_hub(mode, servers, store, catalog_dir) as hub:
            return hub.tool_specs(), hub_catalog_hash(hub)

    tools, catalog_version = anyio.run(specs)
    costs = measure_tool_costs(make_backend(profile), tools, DEFAULT_SYSTEM_PROMPT)
    costs.update(profile=name, catalog_version=catalog_version, measured_at=datetime.now(UTC).isoformat())
    table = Table("tool", "alone", "marginal")
    for tool, c in sorted(costs["tools"].items(), key=lambda kv: -kv[1]["marginal"]):
        table.add_row(tool, str(c["alone"]), str(c["marginal"]))
    console.print(table)
    console.print(f"all {len(tools)} tools: {costs['all_tools']} prompt tokens (measured, {costs['model']}); "
                  f"base prompt {costs['base_prompt']}")
    slug = costs["model"].replace(":", "-").replace("/", "-")
    path = catalog_dir / f"tool-costs-{slug}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(costs, indent=2), encoding="utf-8")
    console.print(f"saved {path}")


TasksOption = Annotated[Path, typer.Option("--tasks", help="Benchmark JSONL file.")]


def _select(tasks, only: str | None, limit: int | None):
    if only:
        wanted = only.split(",")
        tasks = [t for t in tasks if t.id in wanted or any(t.id.startswith(w.rstrip("*")) for w in wanted
                                                           if w.endswith("*"))]
    return tasks[:limit] if limit else tasks


@bench_app.command("validate")
def bench_validate(tasks: TasksOption = TASKS_FILE, catalog_dir: CatalogDirOption = CATALOG_DIR) -> None:
    """Check task structure against the catalog: tool names, servers, counts and trap share."""
    from toolhelper.experiments.runner import load_tasks, validate_tasks

    snap = latest_snapshot(catalog_dir)
    names = {t.namespaced for s in snap.servers.values() for t in s.tools}
    result = validate_tasks(load_tasks(tasks), names)
    console.print(f"{result['tasks']} tasks {result['by_server']}, traps {result['traps']} "
                  f"({result['trap_share']:.0%}) {result['trap_kinds']}")
    for problem in result["problems"]:
        console.print(f"[red]{problem}[/]")
    if result["problems"]:
        raise typer.Exit(1)


@bench_app.command("run")
def bench_run(tasks: TasksOption = TASKS_FILE, llm: LLMOption = None, mode: ModeOption = "replay",
              only: Annotated[str | None, typer.Option(help="Task ids, comma-separated; 'omop-*' prefixes.")] = None,
              limit: Annotated[int | None, typer.Option(help="Run only the first N tasks.")] = None,
              group: Annotated[str | None, typer.Option(help="Group id (default: bench-<time>-<mode>).")] = None,
              strict_replay: Annotated[bool, typer.Option(help="Replay without live fallback.")] = False,
              max_turns: Annotated[int, typer.Option(help="Maximum LLM turns.")] = 8,
              strategy: Annotated[list[str] | None, typer.Option(
                  "--strategy", help="Efficiency strategy spec, repeatable: E4:mode=repair, E1:top_k=8, E9.")] = None,
              learn_from: Annotated[str, typer.Option(help="Groups whose traces strategies learn from, comma-separated.")]
              = "bench-baseline-record",
              retry_timeouts: Annotated[int, typer.Option(
                  help="Re-run tasks that ended in an LLM API timeout or connection error, up to N times.")] = 1,
              fold: Annotated[str | None, typer.Option(
                  help="Cross-fitting: run only this fold's tasks (A or B); strategies learn from the other fold.")]
              = None,
              store: StoreOption = REPLAY_DIR, catalog_dir: CatalogDirOption = CATALOG_DIR,
              traces: TracesOption = TRACES_DIR, db: MonitorOption = MONITOR_DB) -> None:
    """Run the agent over benchmark tasks; every trace is scored, saved and ingested under one group."""
    from datetime import UTC, datetime

    from toolhelper.agent.loop import Agent, AgentConfig
    from toolhelper.experiments.runner import load_tasks, run_benchmark
    from toolhelper.monitoring.db import MonitorDB
    from toolhelper.monitoring.report import latest_tool_costs

    all_tasks = load_tasks(tasks)
    selected = _select(all_tasks, only, limit)
    learn_tasks = None
    if fold:
        from toolhelper.experiments.experiment import folds, other_fold

        fold_map = folds(all_tasks)
        if fold not in fold_map:
            raise typer.BadParameter("fold must be A or B")
        selected = [t for t in selected if t.id in set(fold_map[fold])]
        learn_tasks = set(fold_map[other_fold(fold)])
    _, profile = load_llm_config().get(llm)
    group = group or f"bench-{datetime.now(UTC):%Y%m%dT%H%M%S}-{mode}"
    monitor = MonitorDB(db, latest_tool_costs(catalog_dir))

    def progress(trace, result) -> None:
        live = sum(c.source == "live" for c in trace.tool_calls)
        mark = "[green]pass[/]" if result.success else "[red]fail[/]"
        failed = [k for k, v in result.checks.items() if not v]
        retry = f" (retry of {trace.retry_of})" if trace.retry_of else ""
        trust_value = (trace.trust or {}).get("trust")
        console.print(f"{mark} {trace.task_id:9}{retry} turns={len(trace.llm_calls)} calls={len(trace.tool_calls)} "
                      f"trust={'-' if trust_value is None else f'{trust_value:.2f}'} "
                      f"live={live} prompt={sum(c.prompt_tokens for c in trace.llm_calls)} "
                      f"grounding={'-' if result.grounding is None else f'{result.grounding:.2f}'} "
                      + (f"failed={failed}" if failed else "") + (f" [red]{trace.error}[/]" if trace.error else ""))

    async def go():
        async with make_hub(mode, None, store, catalog_dir, live_fallback=not strict_replay) as hub:
            strategies = _strategies(strategy or [], hub, learn_from, monitor, store, learn_tasks)
            if strategies:
                console.print("strategies: " + ", ".join(f"{s.name} {s.config()}" for s in strategies))
            agent = Agent(make_backend(profile), hub, strategies, config=AgentConfig(max_turns=max_turns),
                          group_id=group)
            return await run_benchmark(selected, agent, traces_dir=traces, db=monitor, progress=progress,
                                       retry_timeouts=retry_timeouts, trust_fn=_trust_fn())

    results = anyio.run(go)
    passed = sum(r.success for _, r in results)
    live = sum(c.source == "live" for t, _ in results for c in t.tool_calls)
    calls = sum(len(t.tool_calls) for t, _ in results)
    prompt = sum(c.prompt_tokens for t, _ in results for c in t.llm_calls)
    console.print(f"[bold]{group}[/]: {passed}/{len(results)} passed, {calls} tool calls ({live} live), "
                  f"{prompt:,} prompt tokens")


def _trust_fn(context_db: Path = CONTEXT_DB):
    """Trust with dataset ages from the context store (Phase 9)."""
    from toolhelper.efficiency.trust import load_policy, store_lookup, trust

    policy, lookup = load_policy(), store_lookup(LocalStore(context_db))
    return lambda trace: trust(trace, policy, lookup)


def _strategies(specs: list[str], hub: Hub, learn_from: str, monitor, store: Path,
                learn_tasks: set[str] | None = None) -> list:
    """Strategies with inputs learned from the traces of `learn_from` (comma-separated groups; only
    `learn_tasks`' runs when cross-fitting), the hub's tools, the recordings, the dataset index and
    omop_mcp's CDM map."""
    if not specs:
        return []
    from toolhelper.agent.hooks import ToolDef
    from toolhelper.datasets.base import INDEX_PATH
    from toolhelper.datasets.index import EntityIndex
    from toolhelper.efficiency.arg_guard import DEFAULT_CDM_FIELDS
    from toolhelper.efficiency.registry import Inputs, build_strategies
    from toolhelper.mcp.canonical import tool_hash
    from toolhelper.monitoring.tracer import Trace

    groups = [g.strip() for g in learn_from.split(",") if g.strip()]
    rows = [r for g in groups for r in monitor.rows("SELECT trace_path FROM runs WHERE group_id = ?", (g,))]
    learned = [Trace.load(r["trace_path"]) for r in rows if r["trace_path"] and Path(r["trace_path"]).exists()]
    if learn_tasks is not None:
        learned = [t for t in learned if t.task_id in learn_tasks]
    if not learned:
        raise typer.BadParameter(f"no traces in group {learn_from!r} to learn from")
    tools = [ToolDef(name=t.name, description=t.tool.description or "", input_schema=t.tool.input_schema,
                     server=t.server, tool=t.tool.name, schema_hash=tool_hash(t.tool)) for t in hub.tools]
    inputs = Inputs(traces=learned, tools=tools, recordings=list(RecordingStore(store)),
                    index=EntityIndex(INDEX_PATH) if INDEX_PATH.exists() else None, cdm_path=DEFAULT_CDM_FIELDS)
    console.print(f"learning from {len(learned)} runs in {learn_from}"
                  + (f" ({len(learn_tasks)} other-fold tasks)" if learn_tasks is not None else ""))
    return build_strategies(specs, inputs)


@bench_app.command("check")
def bench_check(group: Annotated[str, typer.Option(help="Benchmark group to check.")],
                tasks: TasksOption = TASKS_FILE, db: MonitorOption = MONITOR_DB) -> None:
    """List expected answer strings that appear in none of a group's real tool outputs."""
    from toolhelper.experiments.runner import check_expectations, load_tasks
    from toolhelper.monitoring.db import MonitorDB
    from toolhelper.monitoring.tracer import Trace

    rows = MonitorDB(db).rows("SELECT trace_path FROM runs WHERE group_id = ?", (group,))
    traces = [Trace.load(r["trace_path"]) for r in rows if r["trace_path"]]
    issues = check_expectations(load_tasks(tasks), traces)
    for issue in issues:
        console.print(issue)
    console.print(f"{len(issues)} unverified expectations across {len(traces)} runs")


@bench_app.command("rescore")
def bench_rescore(group: Annotated[str, typer.Option(help="Benchmark group to rescore.")],
                  tasks: TasksOption = TASKS_FILE, db: MonitorOption = MONITOR_DB,
                  catalog_dir: CatalogDirOption = CATALOG_DIR) -> None:
    """Rescore a group's saved traces against the current tasks; updates trace files and the database."""
    from toolhelper.experiments.runner import latest_per_task, load_tasks, rescore
    from toolhelper.monitoring.db import MonitorDB
    from toolhelper.monitoring.report import latest_tool_costs
    from toolhelper.monitoring.tracer import Trace

    monitor = MonitorDB(db, latest_tool_costs(catalog_dir))
    rows = monitor.rows("SELECT trace_path FROM runs WHERE group_id = ?", (group,))
    results = rescore([Trace.load(r["trace_path"]) for r in rows if r["trace_path"]], load_tasks(tasks), _trust_fn())
    for trace, _ in results:
        path = next(r["trace_path"] for r in rows if r["trace_path"].endswith(f"{trace.run_id}.json"))
        trace.save(Path(path).parent)
        monitor.ingest(trace, path)
    latest = {t.run_id for t in latest_per_task([t for t, _ in results])}
    final = [(t, r) for t, r in results if t.run_id in latest]
    by_kind: dict[str, list[bool]] = {}
    for trace, result in final:
        kind = trace.task_id.split("-")[0]
        by_kind.setdefault(kind, []).append(result.success)
    passed = sum(r.success for _, r in final)
    console.print(f"{group}: {passed}/{len(final)} passed (latest run per task) "
                  + " ".join(f"{k}={sum(v)}/{len(v)}" for k, v in by_kind.items()))
    for trace, result in sorted(final, key=lambda tr: tr[0].task_id):
        if not result.success:
            failed = [k for k, v in result.checks.items() if not v] or [trace.stop_reason or "?"]
            console.print(f"  [red]fail[/] {trace.task_id:9} {failed}")


DatasetsOnly = Annotated[str | None, typer.Option("--only", help="Dataset names, comma-separated.")]


def _datasets(only: str | None):
    from toolhelper.datasets.base import load_config

    cfg = load_config()
    if not only:
        return list(cfg.values())
    unknown = [n for n in only.split(",") if n not in cfg]
    if unknown:
        raise typer.BadParameter(f"unknown datasets {unknown}; known: {sorted(cfg)}")
    return [cfg[n] for n in only.split(",")]


@datasets_app.command("fetch")
def datasets_fetch(only: DatasetsOnly = None) -> None:
    """Download dataset files to <datasets dir>/raw/<name>/ with URL, date and sha256 (hand-downloaded
    files are only checksummed). A failed source is reported and the others still run."""
    from toolhelper.datasets.base import RAW_DIR
    from toolhelper.datasets.build import fetch_dataset

    failed = 0
    for cfg in _datasets(only):
        try:
            meta = fetch_dataset(cfg, RAW_DIR / cfg.name)
        except Exception as e:  # one source being down must not stop the rest
            failed += 1
            console.print(f"[red]{cfg.name}: {type(e).__name__}: {e}[/]")
            continue
        for file, m in meta["files"].items():
            console.print(f"{cfg.name}: {file} {m['bytes']:,} bytes sha256 {m['sha256'][:12]} "
                          f"last-modified {m['last_modified'] or '-'}")
    if failed:
        raise typer.Exit(1)


@datasets_app.command("build")
def datasets_build(only: DatasetsOnly = None, db: ContextOption = CONTEXT_DB) -> None:
    """Load fetched files into <datasets dir>/index.sqlite and version them as datasets/<name>."""
    from toolhelper.datasets.base import INDEX_PATH, RAW_DIR
    from toolhelper.datasets.build import build_dataset
    from toolhelper.datasets.index import EntityIndex

    index, store, failed = EntityIndex(INDEX_PATH), LocalStore(db), 0
    for cfg in _datasets(only):
        try:
            r = build_dataset(cfg, RAW_DIR / cfg.name, index, store)
        except Exception as e:
            failed += 1
            console.print(f"[red]{cfg.name}: {type(e).__name__}: {e}[/]")
            continue
        console.print(f"{r.wref}: {r.rows:,} rows, sha256 {r.sha256[:12]}"
                      + ("" if r.reindexed else " (unchanged)"))
    if failed:
        raise typer.Exit(1)


@datasets_app.command("list")
def datasets_list(db: ContextOption = CONTEXT_DB) -> None:
    """Indexed datasets with their version, rows and age against the TTL."""
    from datetime import UTC, datetime

    from toolhelper.datasets.base import INDEX_PATH, load_config
    from toolhelper.datasets.index import EntityIndex

    cfg, store = load_config(), LocalStore(db)
    indexed = {d["name"]: d for d in EntityIndex(INDEX_PATH).datasets()}
    table = Table("dataset", "indexed", "source version", "rows", "fetched", "ttl")
    for name, c in cfg.items():
        d = indexed.get(name)
        try:
            thing = store.get(f"datasets/{name}").data
        except NotFoundError:
            thing = None
        if not d or not thing:
            table.add_row(name, "-", "-", "-", "-", f"{c.ttl_days:g}d")
            continue
        age = (datetime.now(UTC) - datetime.fromisoformat(thing["fetched_at"])).days
        stale = " [red]stale[/]" if age > c.ttl_days else ""
        table.add_row(name, d["wref"], thing["version"], f"{d['rows']:,}", f"{age}d ago{stale}",
                      f"{c.ttl_days:g}d")
    console.print(table)


@datasets_app.command("resolve")
def datasets_resolve(surface: str, type: Annotated[str | None, typer.Option(help="Entity type filter.")] = None,
                     rows: Annotated[bool, typer.Option(help="Print each match's evidence row.")] = True) -> None:
    """Resolve a surface form (e.g. HER2, Zoloft, Japan) to canonical IDs with dataset-row evidence."""
    from toolhelper.datasets.base import INDEX_PATH
    from toolhelper.datasets.index import EntityIndex

    res = EntityIndex(INDEX_PATH).resolve(surface, type=type)
    console.print(f"[bold]{res.query}[/]: {res.status}")
    for m in res.matches:
        console.print(f"  {m.type:16} {m.canonical} [{m.identifier}]  via {m.source} '{m.surface}' "
                      f"({m.method}, {m.score:g}; {m.rows} row{'s' if m.rows != 1 else ''})  {m.dataset} row {m.row_key}")
        if rows:
            console.print(f"    {json.dumps(m.row, ensure_ascii=False)[:400]}")


ExperimentConfigOption = Annotated[Path, typer.Option("--config", help="Experiment plan (configs/experiments*.yaml).")]


def _experiment_config(path: Path):
    """The p10 plan or, for `design: factorial`, the fractional factorial (p10b)."""
    import yaml

    from toolhelper.experiments.experiment import load_experiment_config
    from toolhelper.experiments.factorial import load_config

    path = path if path.is_absolute() else ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return load_config(path) if raw.get("design") == "factorial" else load_experiment_config(path)


@experiment_app.command("freeze")
def experiment_freeze(tasks: TasksOption = TASKS_FILE, catalog_dir: CatalogDirOption = CATALOG_DIR,
                      store: StoreOption = REPLAY_DIR, db: MonitorOption = MONITOR_DB,
                      context_db: ContextOption = CONTEXT_DB,
                      config: ExperimentConfigOption = Path("configs/experiments.yaml")) -> None:
    """Freeze the experiment's inputs and folds as experiments/<name> in the context store."""
    from toolhelper.datasets.base import INDEX_PATH
    from toolhelper.datasets.index import EntityIndex
    from toolhelper.experiments.experiment import folds, freeze_inputs, inputs_hash, write_experiment
    from toolhelper.experiments.runner import load_tasks
    from toolhelper.monitoring.db import MonitorDB

    cfg = _experiment_config(config)
    _, profile = load_llm_config().get(cfg.llm)
    groups = [cfg.learn_from] if isinstance(cfg.learn_from, str) else cfg.learn_from
    runs = [r["run_id"] for g in groups for r in MonitorDB(db).rows("SELECT run_id FROM runs WHERE group_id = ?", (g,))]
    inputs = freeze_inputs(catalog_hash=latest_snapshot(catalog_dir).catalog_hash, tasks_file=tasks,
                           datasets=EntityIndex(INDEX_PATH).datasets(), llm_profile={"name": cfg.llm, **profile.model_dump()},
                           store_dir=store, learn_from_runs=runs)
    fold_map = folds(load_tasks(tasks))
    thing = write_experiment(LocalStore(context_db), cfg, inputs, fold_map)
    console.print(f"{thing.name}@v{thing.version}: inputs {inputs_hash(inputs)[:12]}, folds A={len(fold_map['A'])} "
                  f"B={len(fold_map['B'])}, {inputs['replay_set_at_start']['recordings']} recordings, "
                  f"learning from {len(runs)} runs of {', '.join(groups)}")


@experiment_app.command("auto")
def experiment_auto(tasks: TasksOption = TASKS_FILE, db: MonitorOption = MONITOR_DB,
                    context_db: ContextOption = CONTEXT_DB,
                    config: ExperimentConfigOption = Path("configs/experiments.yaml")) -> None:
    """Run a whole Phase 10 plan unattended (resumable): the trimmed p10 plan, or the fractional
    factorial for a `design: factorial` config (configs/experiments-p10b.yaml)."""
    from toolhelper.experiments.runner import load_tasks
    from toolhelper.monitoring.db import MonitorDB

    cfg = _experiment_config(config)
    workdir, log = ROOT / "data" / "experiments" / cfg.experiment, (lambda m: console.print(m, markup=False))
    if getattr(cfg, "design", None) == "factorial":
        from toolhelper.experiments.factorial import FactorialOrchestrator, factorial_loop, grounded_tasks

        task_list, subsets = load_tasks(tasks), {}
        if "grounded" in cfg.report_subsets:
            from toolhelper.datasets.base import INDEX_PATH
            from toolhelper.datasets.index import EntityIndex

            subsets["grounded"] = grounded_tasks(task_list, EntityIndex(INDEX_PATH))
        factorial_loop(FactorialOrchestrator(cfg, task_list, MonitorDB(db), workdir, log=log, subsets=subsets),
                       LocalStore(context_db))
        return
    from toolhelper.datasets.base import INDEX_PATH
    from toolhelper.datasets.index import EntityIndex
    from toolhelper.experiments.orchestrate import Orchestrator, main_loop

    orch = Orchestrator(cfg, load_tasks(tasks), MonitorDB(db), workdir, log=log)
    main_loop(orch, EntityIndex(INDEX_PATH), LocalStore(context_db))


@experiment_app.command("status")
def experiment_status(db: MonitorOption = MONITOR_DB,
                      config: ExperimentConfigOption = Path("configs/experiments.yaml")) -> None:
    """Progress of each experiment group: tasks done, passes, transient failures."""
    cfg = _experiment_config(config)
    rows = _monitor_rows(db, "SELECT group_id, COUNT(DISTINCT task_id) AS tasks, SUM(success) AS passed, "
                              "SUM(error LIKE 'APITimeout%') AS timeouts FROM runs WHERE group_id LIKE ? "
                              "GROUP BY group_id ORDER BY MIN(started_at)", (f"{cfg.experiment}-%",))
    table = Table("group", "tasks", "passed", "timeouts")
    for r in rows:
        table.add_row(r["group_id"], str(r["tasks"]), str(r["passed"]), str(r["timeouts"]))
    console.print(table)


@experiment_app.command("app")
def experiment_app_cmd(port: Annotated[int, typer.Option(help="Port to serve on.")] = 8501,
                       browser: Annotated[bool, typer.Option(help="Open the page in a browser.")] = True) -> None:
    """Open the Streamlit app to edit, launch, pause and read experiments (needs the `dashboard` extra)."""
    import importlib.util
    import subprocess
    import sys
    import threading
    import webbrowser

    if importlib.util.find_spec("streamlit") is None:
        raise typer.BadParameter("streamlit is not installed; run `uv run --extra dashboard toolhelper experiment app`")
    script = Path(__file__).parent / "app" / "experiments_app.py"
    url = f"http://localhost:{port}"
    console.print(f"serving {url} (Ctrl+C stops the app; running experiments keep going)")
    if browser:
        threading.Timer(3, webbrowser.open, [url]).start()
    # Headless skips Streamlit's first-run email prompt, which blocks a detached start.
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port),
                                      "--server.headless", "true", "--browser.gatherUsageStats", "false",
                                      "--server.address", "localhost"], cwd=ROOT))  # it launches runs with your keys


def _monitor_rows(db: Path, sql: str, params: tuple) -> list[dict]:
    from toolhelper.monitoring.db import MonitorDB

    return MonitorDB(db).rows(sql, params)


@app.command("curate")
def curate_cmd(name: Annotated[str, typer.Argument(help="Decision to approve, e.g. decisions/p10.")],
               by: Annotated[str, typer.Option(help="Who approves it.")],
               context_db: ContextOption = CONTEXT_DB) -> None:
    """Approve a proposed Decision (Phase 10)."""
    from toolhelper.experiments.decision import curate

    console.print(curate(LocalStore(context_db), name, by))
