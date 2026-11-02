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
