"""In-process MCP server that serves recorded real responses.

Tools are listed exactly as captured in the catalog snapshot; calls are answered
from the recording store. A call with no recording fails with `REPLAY_MISS`.
"""

from __future__ import annotations

from typing import Any

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError

from toolhelper.mcp.recorder import RecordingStore

REPLAY_MISS = -32099
REPLAY_META = "toolhelper/replay"


def build_replay_server(server: str, snapshot: Any, store: RecordingStore, *,
                        miss_code: int = REPLAY_MISS) -> Server:
    """`snapshot` is the catalog's ServerSnapshot for `server`."""
    tools = [types.Tool.model_validate(t.definition) for t in snapshot.tools]
    info = snapshot.server_info

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        arguments = params.arguments or {}
        rec = store.get(server, params.name, arguments)
        if rec is None:
            raise MCPError(miss_code, f"no recording for {server}/{params.name}",
                           {"server": server, "tool": params.name, "arguments": arguments})
        if rec.protocol_error is not None:
            err = rec.protocol_error
            raise MCPError(err["code"], err["message"], err.get("data"))
        result = types.CallToolResult.model_validate(rec.result)
        provenance = {"recorded_at": rec.recorded_at, "args_key": rec.args_key}
        result.meta = {**(result.meta or {}), REPLAY_META: provenance}
        return result

    return Server(
        info.get("name", server), version=info.get("version", ""), title=info.get("title"),
        on_list_tools=list_tools, on_call_tool=call_tool,
    )
