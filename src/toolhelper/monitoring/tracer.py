"""Run traces: everything one agent run did, shaped for the Phase 5 monitoring tables.

A trace holds the run row, one record per LLM call and per tool call, every
transformation and context item, and the final transcript. Prompt tokens are kept
measured (from the backend) and estimated per component side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from toolhelper.context.model import EvidenceRef
from toolhelper.llm.base import Message


class LLMCallRecord(BaseModel):
    turn: int
    prompt_tokens: int
    completion_tokens: int
    measured: bool
    thinking_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    est_breakdown: dict[str, int]  # estimated prompt tokens per component
    latency_ms: float
    done_reason: str | None
    tools_offered: list[str]
    messages_sent: int  # prefix length of the transcript sent on this call
    # Tool results present in this call's prompt (call id -> estimated tokens), for carried tokens.
    tool_results_in_context: dict[str, int] = Field(default_factory=dict)
    applied_options: dict[str, Any] = Field(default_factory=dict)
    ignored_options: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)  # call ids the model issued


class ToolCallRecord(BaseModel):
    turn: int
    call_id: str
    name: str
    server: str | None
    tool: str | None
    arguments: dict[str, Any]
    final_arguments: dict[str, Any]  # after any repair
    args_tokens: int
    decision: Literal["proceed", "repaired", "rejected", "cached"]
    raw_result_tokens: int | None  # None when no call reached a server or local tool
    shaped_result_tokens: int
    latency_ms: float
    is_error: bool
    error_class: str | None = None
    cache_hit: bool = False
    replayed: bool = False
    source: str | None = None  # live, replay or local
    recorded_at: str | None = None
    raw_text: str | None = None
    shaped_text: str


class TransformationRecord(BaseModel):
    turn: int
    strategy: str
    hook: str
    target: str
    before_tokens: int
    after_tokens: int
    evidence: list[EvidenceRef] = Field(default_factory=list)
    note: str = ""
    reported: bool = True  # False: detected by the loop, not reported by the strategy


class ContextItemRecord(BaseModel):
    strategy: str
    kind: str
    wref: str | None
    tokens: int
    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class Trace(BaseModel):
    run_id: str
    group_id: str | None = None
    task_id: str
    prompt: str
    task: dict[str, Any] = Field(default_factory=dict)  # full Task, incl. expected_paths
    backend: str
    model: str
    params: dict[str, Any]
    config: dict[str, Any]
    config_hash: str
    catalog_version: str
    hub_mode: str
    tokenizer: str
    started_at: str
    finished_at: str | None = None
    stop_reason: Literal["final", "max_turns", "error"] | None = None
    error: str | None = None
    answer: str | None = None
    success: bool | None = None  # set by scoring (Phase 6)
    trust: dict[str, Any] | None = None  # set by the trust layer (Phase 9)
    tools_offered: list[str] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    transformations: list[TransformationRecord] = Field(default_factory=list)
    context_items: list[ContextItemRecord] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)

    @property
    def provenance_coverage(self) -> float:
        """Share of transformations and context items that carry evidence (should be 1.0)."""
        items = [*self.transformations, *self.context_items]
        return sum(bool(i.evidence) for i in items) / len(items) if items else 1.0

    def problems(self) -> list[str]:
        """Why this trace is incomplete; empty when complete."""
        out = []
        if self.stop_reason is None or self.finished_at is None:
            out.append("run did not finish")
        if not self.llm_calls:
            out.append("no LLM calls")
        if self.stop_reason == "final" and self.answer is None:
            out.append("final stop without an answer")
        issued = [c for call in self.llm_calls for c in call.tool_calls]
        recorded = [t.call_id for t in self.tool_calls]
        if issued != recorded:
            out.append(f"tool calls issued {issued} != recorded {recorded}")
        for call in self.llm_calls:
            if not call.est_breakdown:
                out.append(f"turn {call.turn}: no token breakdown")
            if call.messages_sent > len(self.messages):
                out.append(f"turn {call.turn}: transcript shorter than what was sent")
        if self.provenance_coverage < 1.0:
            missing = [f"{t.strategy}.{t.hook}" for t in self.transformations if not t.evidence]
            missing += [f"{c.strategy}.prepare ({c.kind})" for c in self.context_items if not c.evidence]
            out.append(f"provenance coverage {self.provenance_coverage:.2f}: no evidence for {missing}")
        return out

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> Trace:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
