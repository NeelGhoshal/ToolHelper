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
    ROOT,
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


def make_hub(mode: str, servers: str | None, store: Path, catalog_dir: Path) -> Hub:
    configs = load_servers_config()
    if servers:
        names = [s.strip() for s in servers.split(",")]
        configs = {n: configs[n] for n in names}
    catalog = latest_snapshot(catalog_dir) if mode == "replay" else None
    if mode == "replay" and catalog is None:
        raise typer.BadParameter(f"no catalog snapshot in {catalog_dir}; run `toolhelper catalog` first")
    return Hub(configs, mode=mode, store=RecordingStore(store), catalog=catalog)


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
