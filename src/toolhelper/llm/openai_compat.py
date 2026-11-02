"""OpenAI-compatible Chat Completions backend.

Covers OpenAI, vLLM, LM Studio and Ollama's own `/v1` endpoint. `num_ctx` and
`think` have no Chat Completions equivalent and are reported as ignored; pass
server-specific fields (e.g. `reasoning_effort`) through `options.extra`.
"""

from __future__ import annotations

import json
import os
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


def to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.input_schema},
    }


def to_openai_message(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}}
            for c in m.tool_calls
        ]
    return out


def parse_arguments(text: str | None) -> tuple[dict[str, Any], str | None]:
    """Returns (arguments, raw_arguments); raw is kept only when parsing fails."""
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}, text
    return (value, None) if isinstance(value, dict) else ({}, text)


class OpenAICompatBackend(LLMBackend):
    kind = "openai_compat"

    def __init__(self, model: str, defaults: ChatOptions | None = None, *,
                 base_url: str | None = None, api_key: str | None = None,
                 api_key_env: str | None = None, client: Any = None) -> None:
        super().__init__(model, defaults)
        if client is None:
            import openai

            key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
            if key is None and base_url and not os.environ.get("OPENAI_API_KEY"):
                key = "unused"  # local servers (Ollama, vLLM, LM Studio) accept any key
            client = openai.OpenAI(base_url=base_url, api_key=key)
        self.client = client

    def _chat(self, messages: list[Message], tools: list[ToolSpec], opts: ChatOptions) -> ChatResult:
        applied: dict[str, Any] = {}
        if opts.temperature is not None:
            applied["temperature"] = opts.temperature
        if opts.seed is not None:
            applied["seed"] = opts.seed
        if opts.max_tokens is not None:
            applied["max_tokens"] = opts.max_tokens
        applied.update(opts.extra)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [to_openai_message(m) for m in messages],
            **applied,
        }
        if tools:
            request["tools"] = [to_openai_tool(t) for t in tools]

        resp = self.client.chat.completions.create(**request)
        choice = resp.choices[0]
        msg = choice.message
        calls = []
        for c in msg.tool_calls or []:
            args, raw_args = parse_arguments(c.function.arguments)
            calls.append(ToolCall(id=c.id, name=c.function.name, arguments=args, raw_arguments=raw_args))
        extra = msg.model_extra or {}
        thinking = extra.get("reasoning") or extra.get("reasoning_content")
        message = Message(role="assistant", content=msg.content or "", tool_calls=calls, thinking=thinking)

        if resp.usage is not None:
            u = resp.usage
            details = u.prompt_tokens_details
            reasoning = u.completion_tokens_details
            usage = Usage(
                prompt=u.prompt_tokens, completion=u.completion_tokens, measured=True,
                cached_prompt=details.cached_tokens if details else None,
                thinking=reasoning.reasoning_tokens if reasoning else None,
            )
        else:
            usage = Usage(
                prompt=sum(estimate_tokens(m.content) for m in messages),
                completion=estimate_tokens(message.content),
                measured=False,
            )
        ignored = [k for k in ("num_ctx", "think", "effort") if getattr(opts, k) is not None]
        return ChatResult(
            message=message, tool_calls=calls, usage=usage, latency_ms=0,
            raw=resp.model_dump(mode="json"), backend=self.kind, model=self.model,
            stop_reason=choice.finish_reason, applied_options=applied, ignored_options=ignored,
        )
