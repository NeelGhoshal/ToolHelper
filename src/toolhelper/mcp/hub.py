"""MCP hub: connects the servers, namespaces their tools and routes calls.

Modes (record/replay sits below ToolHelper's own cache):
- `live`: call the live servers only.
- `record`: call the live servers and save every response.
- `replay`: serve recorded responses through the in-process replay server; a miss
  falls through to live and is recorded, unless `live_fallback=False`.

Each connection runs in its own task inside the hub's task group, so live
connections can be opened lazily (on a replay miss) from any task.
"""

from __future__ import annotations

import time
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Any, Literal

import anyio
import mcp_types as types
from mcp import Client
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel

from toolhelper.config import ServerConfig
from toolhelper.llm.base import ToolSpec
from toolhelper.mcp.canonical import namespaced, tool_hash
from toolhelper.mcp.recorder import Recording, RecordingStore, args_key
from toolhelper.mcp.replay_server import REPLAY_META, REPLAY_MISS, build_replay_server

Mode = Literal["live", "record", "replay"]


class ReplayMiss(LookupError):
    pass


class ToolCallTimeout(TimeoutError):
    pass


class HubTool(BaseModel):
    name: str  # namespaced
    server: str
    tool: types.Tool

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.tool.description or "",
                        input_schema=self.tool.input_schema)


class ToolResult(BaseModel):
    name: str
    server: str
    tool: str
    arguments: dict[str, Any]
    text: str
    structured: dict[str, Any] | None = None
    is_error: bool
    protocol_error: dict[str, Any] | None = None
    source: Literal["live", "replay", "local"]  # local: a strategy-provided tool
    latency_ms: float
    recorded_at: str | None = None
    # CallToolResult in wire form, exactly as the server returned it (None on protocol error).
    raw: dict[str, Any] | None = None


def content_text(block: dict[str, Any]) -> str:
    match block.get("type"):
        case "text":
            return block["text"]
        case "resource":
            res = block.get("resource", {})
            return res.get("text") or f"[resource {res.get('uri')}]"
        case "resource_link":
            return f"[resource {block.get('uri')}]"
        case kind:
            return f"[{kind} {block.get('mimeType', '')}]".rstrip()


class Hub:
    def __init__(self, servers: dict[str, ServerConfig], *, mode: Mode = "live",
                 store: RecordingStore | None = None, catalog: Any = None,
                 live_fallback: bool = True, live_targets: dict[str, Any] | None = None) -> None:
        """`catalog` is a CatalogSnapshot (required for replay). `live_targets` overrides
        what a server's live connection points at (tests use in-process servers)."""
        if mode in ("record", "replay") and store is None:
            raise ValueError(f"mode {mode!r} needs a recording store")
        if mode == "replay" and catalog is None:
            raise ValueError("replay mode needs a catalog snapshot")
        self.servers = servers
        self.mode = mode
        self.store = store
        self.catalog = catalog
        self.live_fallback = live_fallback
        self.live_targets = live_targets or {}
        self.server_info: dict[str, dict[str, Any]] = {}
        self.protocol_version: dict[str, str] = {}
        self._registry: dict[str, HubTool] = {}
        self._replay: dict[str, Client] = {}
        self._live: dict[str, Client] = {}
        self._live_info: dict[str, dict[str, Any]] = {}
        self._live_lock = anyio.Lock()
        self._rate: dict[str, tuple[anyio.Lock, list[float]]] = {
            name: (anyio.Lock(), [0.0]) for name in servers}
        self._stack: AsyncExitStack | None = None
        self._closing = anyio.Event()

    async def __aenter__(self) -> Hub:
        # Fresh connection state, so a hub can be entered again after it was closed.
        self._closing = anyio.Event()
        self._registry, self._replay, self._live, self._live_info = {}, {}, {}, {}
        self._stack = AsyncExitStack()
        self._tg = await self._stack.enter_async_context(anyio.create_task_group())
        try:
            for name in self.servers:
                if self.mode == "replay":
                    snap = self.catalog.servers[name]
                    client = await self._open(build_replay_server(name, snap, self.store))
                    self._replay[name] = client
                    self.server_info[name] = snap.server_info
                    self.protocol_version[name] = snap.protocol_version
                else:
                    client = await self._live_client(name)
                    self.server_info[name] = self._live_info[name]
                    self.protocol_version[name] = client.protocol_version
                for tool in (await client.list_tools()).tools:
                    hub_tool = HubTool(name=namespaced(name, tool.name), server=name, tool=tool)
                    self._registry[hub_tool.name] = hub_tool
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._closing.set()
        if self._stack is not None:
            stack, self._stack = self._stack, None
            await stack.aclose()

    async def _open(self, target: Any) -> Client:
        async def run(*, task_status=anyio.TASK_STATUS_IGNORED) -> None:
            # Always the handshake protocol: BioMCP exits on the discover probe, and on the
            # 2026-07-28 protocol the SDK drops tool fields (`execution`) and stamps results.
            async with Client(target, mode="legacy") as client:
                task_status.started(client)
                await self._closing.wait()

        return await self._tg.start(run)

    async def _live_client(self, name: str) -> Client:
        async with self._live_lock:
            if name not in self._live:
                target = self.live_targets.get(name) or self.servers[name].stdio_params(name)
                client = await self._open(target)
                self._live[name] = client
                info = client.server_info
                self._live_info[name] = info.model_dump(mode="json", by_alias=True, exclude_none=True) \
                    if info else {}
            return self._live[name]

    @property
    def tools(self) -> list[HubTool]:
        return list(self._registry.values())

    def tool_specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._registry.values()]

    def get_tool(self, name: str) -> HubTool:
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(f"unknown tool {name!r}") from None

    async def call(self, name: str, arguments: dict[str, Any] | None = None, *,
                   timeout: float | None = None) -> ToolResult:
        tool = self.get_tool(name)
        args = arguments or {}
        timeout = timeout or self.servers[tool.server].timeout_s
        if self.mode == "replay":
            try:
                return await self._invoke(tool, args, timeout, live=False)
            except ReplayMiss:
                if not self.live_fallback:
                    raise
        result = await self._invoke(tool, args, timeout, live=True)
        if self.mode in ("record", "replay"):
            self._save(tool, result)
        return result

    async def _invoke(self, tool: HubTool, args: dict[str, Any], timeout: float, *, live: bool) -> ToolResult:
        if live:
            client = await self._live_client(tool.server)
            await self._throttle(tool.server)
        else:
            client = self._replay[tool.server]
        start = time.perf_counter()
        raw, protocol_error = None, None
        try:
            with anyio.fail_after(timeout):
                result = await client.call_tool(tool.tool.name, args)
            raw = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        except MCPError as e:
            if not live and e.code == REPLAY_MISS:
                raise ReplayMiss(f"no recording for {tool.name} {args}") from None
            protocol_error = {"code": e.code, "message": e.message, "data": getattr(e, "data", None)}
        except TimeoutError:
            raise ToolCallTimeout(f"{tool.name} timed out after {timeout}s") from None
        latency_ms = (time.perf_counter() - start) * 1000

        recorded_at = None
        if raw is not None and REPLAY_META in (raw.get("_meta") or {}):
            provenance = raw["_meta"].pop(REPLAY_META)
            recorded_at = None if live else provenance.get("recorded_at")
            if not raw["_meta"]:
                del raw["_meta"]
        if raw is not None:
            text = "\n".join(content_text(b) for b in raw.get("content", []))
            is_error = bool(raw.get("isError"))
        else:
            text, is_error = protocol_error["message"], True
        return ToolResult(
            name=tool.name, server=tool.server, tool=tool.tool.name, arguments=args, text=text,
            structured=(raw or {}).get("structuredContent"), is_error=is_error,
            protocol_error=protocol_error, source="live" if live else "replay",
            latency_ms=latency_ms, recorded_at=recorded_at, raw=raw,
        )

    async def _throttle(self, server: str) -> None:
        interval = self.servers[server].min_interval_s
        if interval <= 0:
            return
        lock, last = self._rate[server]
        async with lock:
            wait = last[0] + interval - time.monotonic()
            if wait > 0:
                await anyio.sleep(wait)
            last[0] = time.monotonic()

    def _save(self, tool: HubTool, result: ToolResult) -> None:
        cfg = self.servers[tool.server]
        self.store.save(Recording(
            server=tool.server, tool=result.tool, arguments=result.arguments,
            args_key=args_key(result.arguments), recorded_at=datetime.now(UTC).isoformat(),
            package=cfg.package, server_version=cfg.version,
            server_info=self._live_info.get(tool.server, {}), schema_hash=tool_hash(tool.tool),
            latency_ms=result.latency_ms, result=result.raw, protocol_error=result.protocol_error,
        ))

