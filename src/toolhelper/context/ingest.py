"""Imports catalog snapshots into the context store as McpServer and ToolSchema versions.

Names follow WarmHub's naming-as-navigation: `servers/<server>` and
`tools/<server>/<tool>`. A tool gets a new ToolSchema version only when its schema,
description or token count changes; `captured_at` alone does not revise it. Each
new version carries `schema_hash` evidence, including how its tokens were counted.
"""

from __future__ import annotations

from toolhelper.context.model import Add, ContextStore, EvidenceRef, NotFoundError, Operation, Revise
from toolhelper.mcp.canonical import canonical_json
from toolhelper.mcp.catalog import CatalogSnapshot


def server_name(server: str) -> str:
    return f"servers/{server}"


def tool_name(server: str, tool: str) -> str:
    return f"tools/{server}/{tool}"


def _upsert(store: ContextStore, name: str, shape: str, data: dict, *, keep: tuple[str, ...] = (),
            evidence: list[EvidenceRef] | None = None) -> Operation:
    try:
        current = store.get(name)
    except NotFoundError:
        return Add(name=name, shapes=[shape], data=data, evidence=evidence or [])
    # Fields in `keep` (e.g. captured_at) don't count as a change on their own.
    data = {**data, **{k: current.data[k] for k in keep if k in current.data}}
    return Revise(name=name, shapes=[shape], data=data, expected_version=current.version,
                  evidence=evidence or [])


def import_catalog(store: ContextStore, snap: CatalogSnapshot, launch: dict[str, str] | None = None):
    """`launch` maps server to its (unexpanded) launch command. Returns the CommitResult."""
    ops: list[Operation] = []
    for name, server in snap.servers.items():
        ops.append(_upsert(store, server_name(name), "McpServer", {
            "name": name, "package": server.package, "version": server.version,
            "launch": (launch or {}).get(name, ""),
        }))
    for name, server in snap.servers.items():
        for tool in server.tools:
            data = {
                "server": server_name(name), "tool": tool.name,
                "description": tool.definition.get("description", ""),
                "input_schema_json": canonical_json(tool.definition.get("inputSchema", {})),
                "schema_hash": tool.schema_hash, "catalog_tokens": tool.tokens,
                "captured_at": snap.captured_at,
            }
            evidence = [EvidenceRef(kind="schema_hash", ref=tool.schema_hash, detail={
                "catalog_hash": snap.catalog_hash, "source": snap.source, "tokenizer": snap.tokenizer,
                "tokens_measured": snap.tokens_measured, "server_info": server.server_info,
            })]
            ops.append(_upsert(store, tool_name(name, tool.name), "ToolSchema", data, keep=("captured_at",),
                               evidence=evidence))
    # One commit: server revisions land first, so tool schemas pin the new server version.
    return store.apply(f"catalog snapshot {snap.captured_at}", ops)
