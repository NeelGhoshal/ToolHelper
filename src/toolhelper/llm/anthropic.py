"""Anthropic Messages API backend.

Current Claude models take no sampling parameters (`temperature`, `seed`) and
thinking cannot be disabled on Opus 5.5; depth is set with `effort`. Those options
are reported as ignored rather than silently dropped. Thinking blocks are kept in
`Message.provider_blocks["anthropic"]` and sent back unchanged on later turns.
"""

from __future__ import annotations

import os
from typing import Any

from toolhelper.llm.base import ChatOptions, ChatResult, LLMBackend, Message, ToolCall, ToolSpec, Usage

DEFAULT_MAX_TOKENS = 16000


def to_anthropic_tool(tool: ToolSpec) -> dict[str, Any]:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}


def block_to_param(block: Any) -> dict[str, Any] | None:
    """Converts a response content block to the request form for replay."""
    match block.type:
        case "text":
            return {"type": "text", "text": block.text}
        case "tool_use":
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        case "thinking":
            return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
        case "redacted_thinking":
            return {"type": "redacted_thinking", "data": block.data}
    return None


def to_anthropic_messages(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Returns (system, messages). Leading system messages become the top-level system
    prompt; later ones stay in place as mid-conversation system messages. Consecutive
    tool results are grouped into a single user message."""
    i = 0
    system_parts = []
    while i < len(messages) and messages[i].role == "system":
        system_parts.append(messages[i].content)
        i += 1
    out: list[dict[str, Any]] = []
    for m in messages[i:]:
        if m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            if m.is_error:
                block["is_error"] = True
            prev = out[-1] if out else None
            if prev and prev["role"] == "user" and isinstance(prev["content"], list) \
                    and all(b.get("type") == "tool_result" for b in prev["content"]):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == "assistant":
            blocks = m.provider_blocks.get("anthropic")
            if blocks is None:
                blocks = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in m.tool_calls]
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": m.role, "content": m.content})
    return ("\n\n".join(system_parts) or None), out


class AnthropicBackend(LLMBackend):
    kind = "anthropic"

    def __init__(self, model: str, defaults: ChatOptions | None = None, *,
                 api_key_env: str | None = "ANTHROPIC_API_KEY", client: Any = None) -> None:
        super().__init__(model, defaults)
        if client is None:
            import anthropic

            key = os.environ.get(api_key_env) if api_key_env else None
            # Without a key the SDK resolves credentials itself (e.g. an `ant auth login` profile).
            client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self.client = client

    def _chat(self, messages: list[Message], tools: list[ToolSpec], opts: ChatOptions) -> ChatResult:
        system, wire = to_anthropic_messages(messages)
        applied: dict[str, Any] = {"max_tokens": opts.max_tokens or DEFAULT_MAX_TOKENS}
        if opts.effort is not None:
            applied["output_config"] = {"effort": opts.effort}
        applied.update(opts.extra)
        request: dict[str, Any] = {"model": self.model, "messages": wire, **applied}
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [to_anthropic_tool(t) for t in tools]

        resp = self.client.messages.create(**request)
        texts, calls, thinking, blocks = [], [], [], []
        for b in resp.content:
            if b.type == "text":
                texts.append(b.text)
            elif b.type == "tool_use":
                calls.append(ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {})))
            elif b.type == "thinking" and b.thinking:
                thinking.append(b.thinking)
            param = block_to_param(b)
            if param is not None:
                blocks.append(param)
        message = Message(
            role="assistant", content="".join(texts), tool_calls=calls,
            thinking="\n".join(thinking) or None, provider_blocks={"anthropic": blocks},
        )
        u = resp.usage
        cache_read = u.cache_read_input_tokens or 0
        usage = Usage(
            prompt=u.input_tokens + cache_read + (u.cache_creation_input_tokens or 0),
            completion=u.output_tokens, measured=True, cached_prompt=cache_read,
        )
        ignored = [k for k in ("temperature", "seed", "num_ctx", "think") if getattr(opts, k) is not None]
        return ChatResult(
            message=message, tool_calls=calls, usage=usage, latency_ms=0,
            raw=resp.model_dump(mode="json"), backend=self.kind, model=self.model,
            stop_reason=resp.stop_reason, applied_options=applied, ignored_options=ignored,
        )
