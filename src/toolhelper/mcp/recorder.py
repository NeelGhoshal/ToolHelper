"""Recordings of real MCP server responses, keyed by server, tool and canonical args.

One JSON file per recording at `<root>/<server>/<tool>/<args_key[:16]>.json`, so
fixture sets are readable and diffable. Re-recording the same call overwrites it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from toolhelper.mcp.canonical import canonical_json, sha256_hex


def args_key(arguments: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(arguments))


class Recording(BaseModel):
    server: str
    tool: str
    arguments: dict[str, Any]
    args_key: str
    recorded_at: str
    package: str
    server_version: str  # configured package version or commit
    server_info: dict[str, Any]  # serverInfo as the server reported it
    schema_hash: str
    latency_ms: float
    # Exactly one of these is set: the CallToolResult (wire form) or a JSON-RPC error.
    result: dict[str, Any] | None = None
    protocol_error: dict[str, Any] | None = None


class RecordingStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def path(self, server: str, tool: str, key: str) -> Path:
        return self.root / server / tool / f"{key[:16]}.json"

    def get(self, server: str, tool: str, arguments: dict[str, Any]) -> Recording | None:
        key = args_key(arguments)
        path = self.path(server, tool, key)
        if not path.exists():
            return None
        rec = Recording.model_validate_json(path.read_text(encoding="utf-8"))
        return rec if rec.args_key == key else None

    def save(self, rec: Recording) -> Path:
        path = self.path(rec.server, rec.tool, rec.args_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rec.model_dump_json(indent=2), encoding="utf-8")
        return path

    def __iter__(self) -> Iterator[Recording]:
        for path in sorted(self.root.glob("*/*/*.json")):
            yield Recording.model_validate_json(path.read_text(encoding="utf-8"))
