"""Record real LLM responses used as offline test fixtures for the LLM backends.

Uses the raw SDKs (not ToolHelper's adapters) so fixtures are exactly what the
backends return. The tool definition and the tool result both come from the live
Medical MCP server. Run once, by hand:

    uv run python -X utf8 scripts/record_llm_fixtures.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import anyio
import httpx
import ollama
import openai
from mcp import Client, StdioServerParameters

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3:8b"
OPTIONS = {"num_ctx": 8192, "temperature": 0, "seed": 7}
MEDICAL = StdioServerParameters(
    command="node",
    args=[r"C:\Users\neelg\src\medical-mcp\node_modules\medical-mcp\build\index.js"],
)
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "llm"
PROMPT = "Find drugs matching 'ibuprofen'. Return at most 2 results."


async def medical(tool: str | None = None, args: dict | None = None):
    async with Client(MEDICAL, mode="legacy") as c:
        if tool is None:
            return next(t for t in (await c.list_tools()).tools if t.name == "search-drugs")
        r = await c.call_tool(tool, args or {})
        return "\n".join(b.text for b in r.content if hasattr(b, "text"))


def save(name: str, request: dict, response: dict, **meta) -> None:
    doc = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "ollama_version": httpx.get(f"{OLLAMA_HOST}/api/version").json()["version"],
        **meta,
        "request": request,
        "response": response,
    }
    (OUT / f"{name}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tool = anyio.run(medical)
    spec = {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
    ollama_tools = [{"type": "function", "function": {
        "name": spec["name"], "description": spec["description"], "parameters": spec["input_schema"]}}]
    msgs = [{"role": "user", "content": PROMPT}]

    llm = ollama.Client(host=OLLAMA_HOST)
    req1 = {"model": MODEL, "messages": msgs, "tools": ollama_tools, "think": False, "options": OPTIONS}
    r1 = llm.chat(**req1)
    save("ollama_tool_call", req1, r1.model_dump(mode="json"), tool_spec=spec)

    call = r1.message.tool_calls[0]
    result = anyio.run(medical, call.function.name, dict(call.function.arguments))
    msgs2 = msgs + [
        {"role": "assistant", "content": r1.message.content or "", "tool_calls": [
            {"function": {"name": call.function.name, "arguments": dict(call.function.arguments)}}]},
        {"role": "tool", "tool_name": call.function.name, "content": result},
    ]
    req2 = {**req1, "messages": msgs2}
    r2 = llm.chat(**req2)
    save("ollama_final_answer", req2, r2.model_dump(mode="json"), tool_spec=spec,
         tool_result_source="medical-mcp npm 1.0.8, live call")

    oa = openai.OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")
    req3 = {"model": MODEL, "messages": msgs, "temperature": 0, "seed": 7, "tools": ollama_tools}
    r3 = oa.chat.completions.create(**req3)
    save("openai_compat_tool_call", req3, r3.model_dump(mode="json"), tool_spec=spec,
         endpoint="ollama /v1")


if __name__ == "__main__":
    main()
