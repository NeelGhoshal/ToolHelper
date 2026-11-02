"""Canonical JSON, hashing and tool naming shared by the hub, catalog and recorder."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import mcp_types as types

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: Any) -> str:
    text = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tool_definition(tool: types.Tool) -> dict[str, Any]:
    """The tool exactly as the server lists it, in wire (camelCase) form."""
    return tool.model_dump(mode="json", by_alias=True, exclude_none=True)


def tool_hash(tool: types.Tool | dict[str, Any]) -> str:
    return sha256_hex(tool if isinstance(tool, dict) else tool_definition(tool))


def namespaced(server: str, tool: str) -> str:
    name = f"{server}__{tool}"
    if not NAME_RE.match(name):
        raise ValueError(f"tool name {name!r} must match {NAME_RE.pattern}")
    return name
