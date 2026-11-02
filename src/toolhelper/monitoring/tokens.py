"""Token counting shared by the catalog and the tracer.

Counts here come from a local tokenizer (cl100k_base by default), not the model's,
so they are always estimates. Measured counts come only from LLM backends.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from toolhelper.mcp.canonical import canonical_json

TokenCounter = Callable[[str], int]
DEFAULT_TOKENIZER = "cl100k_base"


@lru_cache(maxsize=4)
def tiktoken_counter(encoding: str = DEFAULT_TOKENIZER) -> TokenCounter:
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text))


def llm_view(name: str, definition: dict[str, Any]) -> str:
    """A tool as sent to an LLM: function format under its namespaced name."""
    return canonical_json({"type": "function", "function": {
        "name": name, "description": definition.get("description", ""),
        "parameters": definition.get("inputSchema", {})}})


def spec_view(spec: Any) -> str:
    """`llm_view` for a ToolSpec-like object (name, description, input_schema)."""
    return llm_view(spec.name, {"description": spec.description, "inputSchema": spec.input_schema})


def message_text(message: Any) -> str:
    """The text of a Message that reaches the model: content plus tool-call names and arguments."""
    calls = "".join(c.name + canonical_json(c.arguments) for c in message.tool_calls)
    return message.content + calls


def breakdown(messages: list[Any], tools: list[Any], counter: TokenCounter,
              context_marker: str) -> dict[str, int]:
    """Estimated prompt tokens per component of one LLM call. System messages that start
    with `context_marker` count as injected context."""
    parts = {"system": 0, "tool_defs": 0, "context": 0, "history": 0, "tool_results": 0}
    parts["tool_defs"] = sum(counter(spec_view(t)) for t in tools)
    for m in messages:
        n = counter(message_text(m))
        if m.role == "system":
            parts["context" if m.content.startswith(context_marker) else "system"] += n
        elif m.role == "tool":
            parts["tool_results"] += n
        else:
            parts["history"] += n
    return parts


def calibrate(parts: dict[str, int], measured_total: int | None,
              anchored: dict[str, int] | None = None) -> dict[str, int]:
    """Scales estimated parts so they sum exactly to the measured total (largest remainder).
    `anchored` parts are already measured; only the rest is scaled to what remains."""
    if measured_total is None:
        return dict(parts)
    anchored = {k: v for k, v in (anchored or {}).items() if k in parts}
    if sum(anchored.values()) > measured_total:
        anchored = {}  # inconsistent with this call's total: fall back to plain scaling
    rest = {k: v for k, v in parts.items() if k not in anchored}
    target = measured_total - sum(anchored.values())
    estimated = sum(rest.values())
    if estimated == 0:
        return {**parts, **anchored}
    exact = {k: v * target / estimated for k, v in rest.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True)[:target - sum(out.values())]:
        out[k] += 1
    return {k: (anchored[k] if k in anchored else out[k]) for k in parts}


def measured_tool_defs(costs: dict[str, Any] | None, model: str, catalog_version: str,
                       offered: list[str]) -> int | None:
    """Measured prompt cost of an offered tool set: the tools header plus each tool's marginal
    cost, when a measurement exists for this model and catalog and covers every offered tool."""
    if not costs or not offered or costs.get("model") != model or costs.get("catalog_version") != catalog_version:
        return None
    tools = costs["tools"]
    if any(name not in tools for name in offered):
        return None
    header = min(t["alone"] - t["marginal"] for t in tools.values())
    return header + sum(tools[name]["marginal"] for name in offered)


def measure_tool_costs(backend: Any, tools: list[Any], system_prompt: str, user: str = "Hello.") -> dict[str, Any]:
    """Exact tool-definition cost as the model sees it: the same prompt with and without tools,
    one output token, and a fresh nonce at the start of the system prompt so no prompt cache
    is reused. `alone` = tool on its own minus no tools; `marginal` = all tools minus all but it."""
    import secrets

    from toolhelper.llm.base import ChatOptions, Message

    def prompt_tokens(subset: list[Any]) -> int:
        # Fixed-length digits tokenize to a constant count, unlike hex (which would add noise).
        nonce = "".join(secrets.choice("0123456789") for _ in range(12))
        messages = [Message(role="system", content=f"[{nonce}]\n{system_prompt}"),
                    Message(role="user", content=user)]
        result = backend.chat(messages, subset, ChatOptions(max_tokens=1))
        if not result.usage.measured:
            raise ValueError(f"{backend.kind} does not report measured prompt tokens")
        return result.usage.prompt

    base, full = prompt_tokens([]), prompt_tokens(tools)
    per_tool = {}
    for i, tool in enumerate(tools):
        per_tool[tool.name] = {
            "alone": prompt_tokens([tool]) - base,
            "marginal": full - prompt_tokens(tools[:i] + tools[i + 1:]),
        }
    return {"backend": backend.kind, "model": backend.model,
            "options": backend.defaults.merged(ChatOptions(max_tokens=1)).model_dump(exclude_none=True),
            "measured": True, "base_prompt": base, "all_tools_prompt": full, "all_tools": full - base,
            "tools": per_tool}
