"""Strategy hook interface for the agent loop (README Phase 4).

Strategies report each change they make with `run.transform(...)`, attaching
evidence. The loop also diffs every hook's input and output: a change a strategy
did not report is recorded without evidence, which shows up as provenance
coverage below 1.0.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from toolhelper.context.model import EvidenceRef
from toolhelper.llm.base import Message, ToolSpec
from toolhelper.mcp.hub import ToolResult

if TYPE_CHECKING:
    from toolhelper.agent.loop import RunContext


class Task(BaseModel):
    """One benchmark task (Phase 6 format); only `id` and `prompt` are required."""

    id: str
    prompt: str
    servers: list[str] = Field(default_factory=list)
    category: str | None = None
    expected_paths: list[list[str]] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)
    traps: list[str] = Field(default_factory=list)


class ContextItem(BaseModel):
    """Something injected into the prompt: a hint, recipe or card, with its evidence."""

    kind: str
    text: str
    wref: str | None = None  # pinned context-store version, when it came from there
    evidence: list[EvidenceRef] = Field(default_factory=list)


class ToolDef(ToolSpec):
    """A tool offered to the LLM. `server`/`tool` locate hub tools; `origin` names the
    strategy that provided a local tool."""

    server: str | None = None
    tool: str | None = None
    schema_hash: str | None = None
    origin: str = "hub"


class PendingCall(BaseModel):
    id: str
    turn: int
    name: str
    arguments: dict[str, Any]


class CallDecision(BaseModel):
    action: Literal["proceed", "repaired", "rejected", "cached"] = "proceed"
    arguments: dict[str, Any] | None = None  # repaired arguments
    hint: str | None = None  # returned to the model when rejected
    result: ToolResult | None = None  # served when cached
    evidence: list[EvidenceRef] = Field(default_factory=list)
    note: str = ""


ToolHandler = Callable[[PendingCall], Awaitable[ToolResult]]


class ExtraTool(BaseModel):
    """A local tool a strategy adds (find_tools, resolve_entity, expand_result)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: ToolDef
    handler: ToolHandler


class Strategy(Protocol):
    name: str

    def bind(self, run: RunContext) -> None: ...
    def config(self) -> dict[str, Any]: ...
    def prepare(self, task: Task, ctx: RunContext) -> list[ContextItem]: ...
    def select_tools(self, task: Task, tools: list[ToolDef]) -> list[ToolDef]: ...
    def transform_schemas(self, tools: list[ToolDef]) -> list[ToolDef]: ...
    def before_call(self, call: PendingCall) -> CallDecision: ...
    def after_result(self, call: PendingCall, result: ToolResult) -> ToolResult: ...
    def compact_history(self, messages: list[Message]) -> list[Message]: ...
    def extra_tools(self) -> list[ExtraTool]: ...
    def at_turn_limit(self, task: Task, messages: list[Message]) -> Message | None: ...


class BaseStrategy:
    """No-op defaults; subclasses override the hooks they need."""

    name = "base"

    def bind(self, run: RunContext) -> None:
        self.run = run

    def config(self) -> dict[str, Any]:
        return {}

    def prepare(self, task: Task, ctx: RunContext) -> list[ContextItem]:
        return []

    def select_tools(self, task: Task, tools: list[ToolDef]) -> list[ToolDef]:
        return tools

    def transform_schemas(self, tools: list[ToolDef]) -> list[ToolDef]:
        return tools

    def before_call(self, call: PendingCall) -> CallDecision:
        return CallDecision()

    def after_result(self, call: PendingCall, result: ToolResult) -> ToolResult:
        return result

    def compact_history(self, messages: list[Message]) -> list[Message]:
        return messages

    def extra_tools(self) -> list[ExtraTool]:
        return []

    def at_turn_limit(self, task: Task, messages: list[Message]) -> Message | None:
        """Called once when `max_turns` runs out. A returned message is sent with no tools
        offered, and the reply becomes the answer (stop reason `final_at_limit`)."""
        return None
