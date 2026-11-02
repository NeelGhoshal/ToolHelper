"""ToolHelper command line. Every command that talks to a model takes `--llm <profile>`."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from toolhelper.config import load_llm_config, make_backend
from toolhelper.llm.base import Message, ToolSpec

app = typer.Typer(no_args_is_help=True)
llm_app = typer.Typer(no_args_is_help=True, help="LLM backends and profiles.")
app.add_typer(llm_app, name="llm")
console = Console()

LLMOption = Annotated[str | None, typer.Option("--llm", help="LLM profile from configs/llm.yaml.")]

PING_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


@llm_app.command("profiles")
def profiles() -> None:
    """List the configured LLM profiles."""
    cfg = load_llm_config()
    table = Table("profile", "backend", "model", "options")
    for name, p in cfg.profiles.items():
        label = f"{name} (default)" if name == cfg.default else name
        table.add_row(label, p.backend, p.model, json.dumps(p.options.model_dump(exclude_none=True)))
    console.print(table)


@llm_app.command("ping")
def ping(llm: LLMOption = None) -> None:
    """Send one tool-calling test turn (live) and show the result."""
    name, profile = load_llm_config().get(llm)
    backend = make_backend(profile)
    result = backend.chat([Message(role="user", content="What's the weather in Paris?")], [PING_TOOL])
    tokens = "measured" if result.usage.measured else "estimated"
    console.print(f"[bold]{name}[/] {result.backend}:{result.model}  {result.latency_ms:.0f} ms")
    console.print(f"usage ({tokens}): prompt={result.usage.prompt} completion={result.usage.completion}"
                  f" thinking={result.usage.thinking} cached={result.usage.cached_prompt}")
    console.print(f"applied={result.applied_options} ignored={result.ignored_options}"
                  f" stop={result.stop_reason}")
    for call in result.tool_calls:
        console.print(f"tool call: {call.name}({json.dumps(call.arguments)})")
    if result.message.content:
        console.print(f"content: {result.message.content}")
    if not result.tool_calls:
        raise typer.Exit(1)
