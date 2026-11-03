"""E7: result cache. Reuses an identical canonical call within its TTL instead of calling again.

The key is the tool plus canonical arguments (the recorder's `args_key`), so it sits on
the final, possibly repaired, arguments: list it after E4 and before E6 so it stores
unshaped results. Only successful hub results are stored. A hit is evidenced by the run
and call that fetched the original and the result's age; the record/replay layer stays
below this cache, so hit rates are ToolHelper's own.

Variant: TTL per source (server, optionally per tool).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from toolhelper.agent.hooks import BaseStrategy, CallDecision, PendingCall
from toolhelper.context.model import EvidenceRef
from toolhelper.mcp.hub import ToolResult
from toolhelper.mcp.recorder import args_key

DEFAULT_TTL = {"medical": 86_400, "biomcp": 86_400, "omop": 7 * 86_400}  # seconds

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (tool TEXT, key TEXT, result_json TEXT, cached_at TEXT, run_id TEXT,
                                    call_id TEXT, PRIMARY KEY (tool, key));
"""


class ResultCache:
    def __init__(self, path: Path | str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.executescript(SCHEMA)

    def get(self, tool: str, arguments: dict[str, Any]) -> tuple[ToolResult, str, str, str] | None:
        row = self.db.execute("SELECT result_json, cached_at, run_id, call_id FROM results WHERE tool = ? AND key = ?",
                              (tool, args_key(arguments))).fetchone()
        return (ToolResult.model_validate_json(row[0]), row[1], row[2], row[3]) if row else None

    def put(self, result: ToolResult, run_id: str, call_id: str, now: datetime) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?)",
                            (f"{result.server}__{result.tool}", args_key(result.arguments), result.model_dump_json(),
                             now.isoformat(), run_id, call_id))


class ResultCacheStrategy(BaseStrategy):
    name = "E7_cache"

    def __init__(self, cache: ResultCache, ttl: dict[str, float] | None = None, *, clock=lambda: datetime.now(UTC)) -> None:
        self.cache = cache
        self.ttl = {**DEFAULT_TTL, **(ttl or {})}
        self.clock = clock
        self.served: set[str] = set()

    def config(self) -> dict[str, Any]:
        return {"ttl": self.ttl}

    def prepare(self, task, ctx) -> list:
        self.served = set()  # call ids repeat across runs (call_0, ...); the cache itself persists
        return []

    def ttl_for(self, tool: str) -> float:
        return self.ttl.get(tool, self.ttl.get(tool.split("__", 1)[0], 0))

    def before_call(self, call: PendingCall) -> CallDecision:
        hit = self.cache.get(call.name, call.arguments)
        if hit is None:
            return CallDecision()
        result, cached_at, run_id, call_id = hit
        age = (self.clock() - datetime.fromisoformat(cached_at)).total_seconds()
        ttl = self.ttl_for(call.name)
        if age > ttl:
            return CallDecision()
        self.served.add(call.id)
        return CallDecision(action="cached", result=result, note=f"cache hit, age {age:.0f}s of ttl {ttl:.0f}s",
                            evidence=[EvidenceRef(kind="run", ref=f"{run_id}:{call_id}",
                                                  detail={"cached_at": cached_at, "age_seconds": round(age, 1),
                                                          "ttl_seconds": ttl})])

    def after_result(self, call: PendingCall, result: ToolResult) -> ToolResult:
        if call.id not in self.served and result.source != "local" and not result.is_error \
                and self.ttl_for(call.name) > 0:
            self.cache.put(result, self.run.run_id, call.id, self.clock())
        return result
