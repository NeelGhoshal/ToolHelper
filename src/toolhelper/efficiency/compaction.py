"""E8: history compaction. Replaces consumed tool results in later turns with a short
summary plus a handle; `recall_result(handle)` returns the original, so nothing is lost.

Variants:
- `after_turn`: compact every result carried past the turn it arrived in;
- `after_use`: compact a result only once its identifiers (tokens containing a digit, the
  Phase 5 heuristic) appear in a later assistant message or call, i.e. it was consumed.

Results shorter than `min_tokens`, errors and already compacted results are left alone.
Evidence is the call that produced the result and, for `after_use`, the later call or
message that used it. Carried-token savings show up in each LLM call's
`tool_results_in_context`; recalls are calls to `recall_result`.
"""

from __future__ import annotations

from typing import Any, Literal

from toolhelper.agent.hooks import BaseStrategy, ExtraTool, PendingCall, Task, ToolDef
from toolhelper.context.model import EvidenceRef
from toolhelper.llm.base import Message
from toolhelper.mcp.canonical import canonical_json
from toolhelper.mcp.hub import ToolResult
from toolhelper.monitoring.pathways import identifiers

Mode = Literal["after_turn", "after_use"]
RECALL = "toolhelper__recall_result"
MARKER = "[compacted:"


def summary(text: str, limit: int = 240) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    out = ""
    for line in lines:
        if len(out) + len(line) > limit:
            break
        out = f"{out} | {line}" if out else line
    return out or text[:limit]


class HistoryCompaction(BaseStrategy):
    name = "E8_compaction"

    def __init__(self, mode: Mode = "after_use", *, min_tokens: int = 150) -> None:
        self.mode = mode
        self.min_tokens = min_tokens

    def config(self) -> dict[str, Any]:
        return {"mode": self.mode, "min_tokens": self.min_tokens}

    def prepare(self, task: Task, ctx) -> list:
        self.arrived: dict[str, int] = {}
        self.originals: dict[str, Message] = {}
        return []

    def _use(self, msg: Message, later: list[Message]) -> str | None:
        """Id of the first later call (or 'answer') that reuses an identifier from `msg`."""
        ids = identifiers(msg.content)
        if not ids:
            return None
        for m in later:
            if m.role != "assistant":
                continue
            for c in m.tool_calls:
                if ids & identifiers(canonical_json(c.arguments)):
                    return c.id
            if ids & identifiers(m.content or ""):
                return "assistant-message"
        return None

    def compact_history(self, messages: list[Message]) -> list[Message]:
        turn = self.run.turn
        out = list(messages)
        for i, m in enumerate(out):
            if m.role != "tool" or not m.tool_call_id:
                continue
            self.arrived.setdefault(m.tool_call_id, turn)
            if (m.is_error or m.content.startswith(MARKER) or self.arrived[m.tool_call_id] >= turn
                    or self.run.tokens(m.content) < self.min_tokens):
                continue
            evidence = [EvidenceRef(kind="run", ref=f"{self.run.run_id}:{m.tool_call_id}",
                                    detail={"arrived_turn": self.arrived[m.tool_call_id]})]
            if self.mode == "after_use":
                used_by = self._use(m, out[i + 1:])
                if used_by is None:
                    continue
                evidence.append(EvidenceRef(kind="run", ref=f"{self.run.run_id}:{used_by}", detail={"used": True}))
            handle = m.tool_call_id
            self.originals[handle] = m
            text = f"{MARKER} {summary(m.content)} ... recall with {RECALL}(handle=\"{handle}\")]"
            self.run.transform(m.tool_name or "history", before=m.content, after=text, evidence=evidence,
                               note=f"{self.mode}: result {handle} compacted")
            out[i] = m.model_copy(update={"content": text})
        return out

    def extra_tools(self) -> list[ExtraTool]:
        async def recall(call: PendingCall) -> ToolResult:
            handle = str(call.arguments.get("handle", ""))
            original = self.originals.get(handle)
            text = original.content if original else f"Error: no compacted result with handle {handle!r}."
            return ToolResult(name=RECALL, server="toolhelper", tool="recall_result", arguments=call.arguments,
                              text=text, is_error=original is None, source="local", latency_ms=0)

        return [ExtraTool(tool=ToolDef(name=RECALL, description=(
            "Return the full text of an earlier tool result that was compacted in the history."),
            input_schema={"type": "object", "properties": {"handle": {"type": "string"}}, "required": ["handle"]}),
            handler=recall)]
