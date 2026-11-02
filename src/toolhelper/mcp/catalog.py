"""Catalog snapshots: every tool schema as canonical JSON, hashed and token-counted.

Tokens are counted on the tool as an LLM sees it (OpenAI/Ollama function format,
namespaced name) with a tokenizer that is not the experiment model's, so they are
labelled estimated. Snapshots are JSON files in `data/catalog/`, diffed against the
previous one; Phase 3 imports them into the context store as ToolSchema versions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from toolhelper.mcp.canonical import sha256_hex, tool_hash
from toolhelper.monitoring.tokens import DEFAULT_TOKENIZER, TokenCounter, llm_view, tiktoken_counter

__all__ = ["DEFAULT_TOKENIZER", "TokenCounter", "llm_view", "tiktoken_counter"]


def hub_catalog_hash(hub: Any) -> str:
    """The catalog hash of a connected hub's tool list (same as a snapshot's)."""
    servers = {}
    for name in hub.servers:
        hashes = sorted((t.tool.name, tool_hash(t.tool)) for t in hub.tools if t.server == name)
        servers[name] = sha256_hex([h for _, h in hashes])
    return sha256_hex(servers)


class ToolSnapshot(BaseModel):
    name: str
    namespaced: str
    definition: dict[str, Any]
    schema_hash: str
    tokens: int


class ServerSnapshot(BaseModel):
    name: str
    package: str
    version: str
    server_info: dict[str, Any]
    protocol_version: str
    tools: list[ToolSnapshot]
    server_hash: str
    tokens: int


class CatalogSnapshot(BaseModel):
    captured_at: str
    source: str  # live or replay
    tokenizer: str
    tokens_measured: bool = False
    servers: dict[str, ServerSnapshot]
    catalog_hash: str

    @property
    def tool_count(self) -> int:
        return sum(len(s.tools) for s in self.servers.values())

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.servers.values())


def build_snapshot(hub: Any, counter: TokenCounter, tokenizer: str = DEFAULT_TOKENIZER) -> CatalogSnapshot:
    servers = {}
    for name, cfg in hub.servers.items():
        tools = []
        for t in sorted((t for t in hub.tools if t.server == name), key=lambda t: t.tool.name):
            definition = t.tool.model_dump(mode="json", by_alias=True, exclude_none=True)
            tools.append(ToolSnapshot(name=t.tool.name, namespaced=t.name, definition=definition,
                                      schema_hash=tool_hash(definition),
                                      tokens=counter(llm_view(t.name, definition))))
        servers[name] = ServerSnapshot(
            name=name, package=cfg.package, version=cfg.version, server_info=hub.server_info[name],
            protocol_version=hub.protocol_version[name], tools=tools,
            server_hash=sha256_hex([t.schema_hash for t in tools]), tokens=sum(t.tokens for t in tools),
        )
    return CatalogSnapshot(
        captured_at=datetime.now(UTC).isoformat(), source="replay" if hub.mode == "replay" else "live",
        tokenizer=tokenizer, servers=servers,
        catalog_hash=sha256_hex({n: s.server_hash for n, s in servers.items()}),
    )


class ToolChange(BaseModel):
    name: str
    old_hash: str
    new_hash: str
    fields: list[str]


class CatalogDiff(BaseModel):
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    changed: list[ToolChange] = Field(default_factory=list)
    server_info: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed or self.server_info)


def _tools(snap: CatalogSnapshot) -> dict[str, ToolSnapshot]:
    return {t.namespaced: t for s in snap.servers.values() for t in s.tools}


def diff_snapshots(old: CatalogSnapshot, new: CatalogSnapshot) -> CatalogDiff:
    before, after = _tools(old), _tools(new)
    diff = CatalogDiff(added=sorted(after.keys() - before.keys()),
                       removed=sorted(before.keys() - after.keys()))
    for name in sorted(before.keys() & after.keys()):
        a, b = before[name], after[name]
        if a.schema_hash != b.schema_hash:
            keys = a.definition.keys() | b.definition.keys()
            fields = sorted(k for k in keys if a.definition.get(k) != b.definition.get(k))
            diff.changed.append(ToolChange(name=name, old_hash=a.schema_hash, new_hash=b.schema_hash,
                                           fields=fields))
    for name in old.servers.keys() & new.servers.keys():
        o, n = old.servers[name], new.servers[name]
        if (o.server_info, o.version) != (n.server_info, n.version):
            diff.server_info[name] = {"old": {"version": o.version, **o.server_info},
                                      "new": {"version": n.version, **n.server_info}}
    return diff


def save_snapshot(snap: CatalogSnapshot, directory: Path | str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(snap.captured_at).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"catalog-{stamp}.json"
    path.write_text(snap.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_snapshot(path: Path | str) -> CatalogSnapshot:
    return CatalogSnapshot.model_validate_json(Path(path).read_text(encoding="utf-8"))


def latest_snapshot(directory: Path | str) -> CatalogSnapshot | None:
    paths = sorted(Path(directory).glob("catalog-*.json"))
    return load_snapshot(paths[-1]) if paths else None
