"""Scripted backend: deterministic assistant turns for tests.

Each step is an assistant `Message` (or dict) or a callable
`(messages, tools) -> Message | dict`, consumed one per `chat` call. The script
decides only what the model says or which tool it calls; tool results always come
from recorded real server responses. Usage is estimated, never measured.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from toolhelper.llm.base import ChatOptions, ChatResult, LLMBackend, Message, ToolSpec, Usage, estimate_tokens

Step = Message | dict[str, Any] | Callable[[list[Message], list[ToolSpec]], Message | dict[str, Any]]


class ScriptExhausted(RuntimeError):
    pass


class UnknownToolError(ValueError):
    pass


class ScriptedBackend(LLMBackend):
    kind = "scripted"

    def __init__(self, steps: list[Step] | None = None, model: str = "scripted",
                 defaults: ChatOptions | None = None, *, strict_tools: bool = True) -> None:
        super().__init__(model, defaults)
        self.steps = list(steps or [])
        self.strict_tools = strict_tools
        self.calls: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return len(self.steps) - len(self.calls)

    def _chat(self, messages: list[Message], tools: list[ToolSpec], opts: ChatOptions) -> ChatResult:
        index = len(self.calls)
        if index >= len(self.steps):
            raise ScriptExhausted(f"script has {len(self.steps)} steps; call {index + 1} has none")
        self.calls.append({"messages": messages, "tools": tools, "options": opts})
        step = self.steps[index]
        reply = step(messages, tools) if callable(step) else step
        message = reply if isinstance(reply, Message) else Message.model_validate(reply)
        if message.role != "assistant":
            raise ValueError(f"step {index} must be an assistant message, got {message.role!r}")
        message = message.model_copy(update={
            "tool_calls": [c.model_copy(update={"id": c.id or f"call_{i}"})
                           for i, c in enumerate(message.tool_calls)],
        })
        if self.strict_tools:
            offered = {t.name for t in tools}
            unknown = [c.name for c in message.tool_calls if c.name not in offered]
            if unknown:
                raise UnknownToolError(f"step {index} calls tools not offered: {unknown}")
        prompt_text = "".join(m.content for m in messages) + "".join(
            t.name + t.description + str(t.input_schema) for t in tools)
        completion_text = message.content + "".join(c.name + str(c.arguments) for c in message.tool_calls)
        usage = Usage(prompt=estimate_tokens(prompt_text), completion=estimate_tokens(completion_text),
                      measured=False)
        return ChatResult(
            message=message, tool_calls=message.tool_calls, usage=usage, latency_ms=0,
            raw={"step": index}, backend=self.kind, model=self.model,
            stop_reason="tool_calls" if message.tool_calls else "stop",
            applied_options=opts.model_dump(exclude_none=True, exclude={"extra"}),
        )
