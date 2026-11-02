"""Ollama native backend (`/api/chat`) via the ollama Python client."""

from __future__ import annotations

from typing import Any

from toolhelper.llm.base import (
    ChatOptions,
    ChatResult,
    LLMBackend,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    estimate_tokens,
)


def to_ollama_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.input_schema},
    }


def to_ollama_message(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        out["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls]
    if m.role == "tool":
        out["tool_name"] = m.tool_name
    return out


class OllamaBackend(LLMBackend):
    kind = "ollama"

    def __init__(self, model: str, defaults: ChatOptions | None = None, *,
                 host: str = "http://localhost:11434", client: Any = None) -> None:
        super().__init__(model, defaults)
        if client is None:
            import ollama

            client = ollama.Client(host=host)
        self.client = client

    def _chat(self, messages: list[Message], tools: list[ToolSpec], opts: ChatOptions) -> ChatResult:
        options: dict[str, Any] = {}
        if opts.num_ctx is not None:
            options["num_ctx"] = opts.num_ctx
        if opts.temperature is not None:
            options["temperature"] = opts.temperature
        if opts.seed is not None:
            options["seed"] = opts.seed
        if opts.max_tokens is not None:
            options["num_predict"] = opts.max_tokens
        options.update(opts.extra)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [to_ollama_message(m) for m in messages],
            "options": options,
        }
        if tools:
            request["tools"] = [to_ollama_tool(t) for t in tools]
        if opts.think is not None:
            request["think"] = opts.think

        resp = self.client.chat(**request)
        msg = resp.message
        # Ollama returns no call ids; number them within the turn.
        calls = [
            ToolCall(id=f"call_{i}", name=c.function.name, arguments=dict(c.function.arguments or {}))
            for i, c in enumerate(msg.tool_calls or [])
        ]
        message = Message(role="assistant", content=msg.content or "", tool_calls=calls, thinking=msg.thinking)
        if resp.prompt_eval_count is not None and resp.eval_count is not None:
            # Note: a reused prompt cache can make prompt_eval_count smaller than the logical prompt.
            usage = Usage(prompt=resp.prompt_eval_count, completion=resp.eval_count, measured=True)
        else:
            usage = Usage(
                prompt=sum(estimate_tokens(m.content) for m in messages),
                completion=estimate_tokens(message.content),
                measured=False,
            )
        applied = {**options, **({"think": opts.think} if opts.think is not None else {})}
        ignored = [k for k in ("effort",) if getattr(opts, k) is not None]
        return ChatResult(
            message=message, tool_calls=calls, usage=usage, latency_ms=0,
            raw=resp.model_dump(mode="json"), backend=self.kind, model=self.model,
            stop_reason=resp.done_reason, applied_options=applied, ignored_options=ignored,
        )
