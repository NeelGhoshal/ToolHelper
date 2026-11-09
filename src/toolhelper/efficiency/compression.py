"""E2: schema compression. Shorter tool definitions; validation-relevant keywords stay.

- `light`: drops `title`, `default`, `$schema`, `examples` and `$comment`; tool
  descriptions keep their first paragraph (<= 300 chars), property descriptions their
  first sentence (<= 100 chars).
- `aggressive`: as light, but property descriptions go entirely and the tool description
  keeps one sentence (<= 100 chars).
- `minimal`: as light, and also drops the bounds that only reject values without telling the
  model what to send: string lengths, `minItems`, `uniqueItems` and `format`. Most of BioMCP's
  schema tokens are these (an array of strings repeats them per property), which is why
  `light` saved only 635 of 4,804 measured tool-definition tokens per turn in Phase 10.

Types, enums, required, numeric bounds, `maxItems`, patterns and combinators are never
touched, and nothing is added, so any call valid under the original schema is valid under the
compressed one (E4 validates against the originals either way; under `minimal` the server and
E4 still enforce the dropped bounds). Evidence per tool: the original schema hash, with the
compressed hash in its detail. `arg_error_rate(traces)` gives the later error rate.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from toolhelper.agent.hooks import BaseStrategy, ToolDef
from toolhelper.context.model import EvidenceRef
from toolhelper.mcp.canonical import sha256_hex
from toolhelper.monitoring.tokens import spec_view

Level = Literal["light", "aggressive", "minimal"]
DROP = {"title", "default", "$schema", "examples", "$comment"}
BOUNDS_ONLY = {"minLength", "maxLength", "minItems", "uniqueItems", "format"}  # dropped by `minimal`
SENTENCE = re.compile(r"(?<=[.!?])\s")


def _shorten(text: str, limit: int, *, sentence: bool) -> str:
    text = text.strip()
    text = (SENTENCE.split(text, maxsplit=1)[0] if sentence else text.split("\n\n", 1)[0]).strip()
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def compress_schema(schema: Any, level: Level, *, in_properties: bool = False) -> Any:
    if isinstance(schema, list):
        return [compress_schema(s, level) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if (key in DROP or (level == "minimal" and key in BOUNDS_ONLY)) and not in_properties:
            continue
        if key == "description" and not in_properties and isinstance(value, str):
            if level == "aggressive":
                continue
            out[key] = _shorten(value, 100, sentence=True)
        elif key in ("properties", "patternProperties", "$defs", "definitions") and isinstance(value, dict):
            out[key] = {name: compress_schema(sub, level) for name, sub in value.items()}
        else:
            out[key] = compress_schema(value, level)
    return out


def compress_tool(tool: ToolDef, level: Level) -> ToolDef:
    limit, sentence = (100, True) if level == "aggressive" else (300, False)
    return tool.model_copy(update={"description": _shorten(tool.description or "", limit, sentence=sentence),
                                   "input_schema": compress_schema(tool.input_schema, level)})


class SchemaCompression(BaseStrategy):
    name = "E2_compression"

    def __init__(self, level: Level = "light") -> None:
        self.level = level

    def config(self) -> dict[str, Any]:
        return {"level": self.level}

    def transform_schemas(self, tools: list[ToolDef]) -> list[ToolDef]:
        out = []
        for t in tools:
            if not t.server:  # local strategy tools are already short
                out.append(t)
                continue
            c = compress_tool(t, self.level)
            if c.description == t.description and c.input_schema == t.input_schema:
                out.append(t)
                continue
            self.run.transform(t.name, before=spec_view(t), after=spec_view(c), note=f"{self.level} compression",
                               evidence=[EvidenceRef(kind="schema_hash", ref=t.schema_hash or sha256_hex(spec_view(t)),
                                                     detail={"compressed_sha256": sha256_hex(spec_view(c)),
                                                             "level": self.level})])
            out.append(c)
        return out


def arg_error_rate(traces) -> dict[str, Any]:
    """Share of hub calls rejected by a guard or failing at the server, for comparing levels."""
    calls = [c for t in traces for c in t.tool_calls if c.server]
    errors = [c for c in calls if c.is_error and c.error_class in ("tool_error", "protocol_error", "rejected")]
    return {"calls": len(calls), "errors": len(errors), "rate": len(errors) / len(calls) if calls else None}
