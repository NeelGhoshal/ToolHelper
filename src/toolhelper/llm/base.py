"""Backend-neutral chat interface shared by every LLM backend.

`chat(messages, tools, options) -> ChatResult(message, tool_calls, usage, latency_ms, raw)`.
Messages and tools use one neutral shape; each backend converts to and from its own
wire format. Token usage is always labelled measured (reported by the backend) or
estimated.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolSpec(BaseModel):
    """A tool as MCP describes it: name, description, JSON Schema input."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Set when the backend returned arguments that were not a valid JSON object.
    raw_arguments: str | None = None


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # For role="tool": which call this answers.
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    thinking: str | None = None
    # Backend-native blocks to send back verbatim to the same backend kind
    # (e.g. Anthropic thinking blocks with signatures), keyed by backend kind.
    provider_blocks: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class Usage(BaseModel):
    prompt: int
    completion: int
    measured: bool
    # Portion of `prompt` served from a prompt cache, when the backend reports it.
    cached_prompt: int | None = None
    # Thinking/reasoning tokens, when reported separately (included in `completion`).
    thinking: int | None = None

    @property
    def total(self) -> int:
        return self.prompt + self.completion


class ChatOptions(BaseModel):
    """Generation options. `None` means "not set"; backends skip unset options."""

    temperature: float | None = None
    seed: int | None = None
    num_ctx: int | None = None
    max_tokens: int | None = None
    think: bool | None = None
    effort: str | None = None
    # Extra backend-specific request fields, passed through as-is.
    extra: dict[str, Any] = Field(default_factory=dict)

    def merged(self, override: ChatOptions | None) -> ChatOptions:
        if override is None:
            return self.model_copy(deep=True)
        data = self.model_dump()
        for key, value in override.model_dump(exclude_unset=True).items():
            if key == "extra":
                data["extra"] = {**data["extra"], **value}
            elif value is not None:
                data[key] = value
        return ChatOptions(**data)


class ChatResult(BaseModel):
    message: Message
    tool_calls: list[ToolCall]
    usage: Usage
    latency_ms: float
    raw: dict[str, Any]
    backend: str
    model: str
    stop_reason: str | None = None
    # Options actually sent to the backend, and those the backend cannot honour.
    applied_options: dict[str, Any] = Field(default_factory=dict)
    ignored_options: list[str] = Field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Rough offline token estimate (~4 characters per token). Always `measured=False`."""
    return (len(text) + 3) // 4 if text else 0


def coerce_messages(messages: list[Message | dict[str, Any]]) -> list[Message]:
    return [m if isinstance(m, Message) else Message.model_validate(m) for m in messages]


def coerce_tools(tools: list[ToolSpec | dict[str, Any]] | None) -> list[ToolSpec]:
    return [t if isinstance(t, ToolSpec) else ToolSpec.model_validate(t) for t in tools or []]


class LLMBackend(ABC):
    """Base class: merges profile defaults with per-call options and times the call."""

    kind: str = "base"

    def __init__(self, model: str, defaults: ChatOptions | None = None) -> None:
        self.model = model
        self.defaults = defaults or ChatOptions()

    def chat(
        self,
        messages: list[Message | dict[str, Any]],
        tools: list[ToolSpec | dict[str, Any]] | None = None,
        options: ChatOptions | None = None,
    ) -> ChatResult:
        opts = self.defaults.merged(options)
        start = time.perf_counter()
        result = self._chat(coerce_messages(messages), coerce_tools(tools), opts)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    @abstractmethod
    def _chat(self, messages: list[Message], tools: list[ToolSpec], opts: ChatOptions) -> ChatResult:
        """Run one chat turn. Implementations leave `latency_ms` at 0; `chat` fills it."""
