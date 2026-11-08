"""Config loading. LLM profiles live in `configs/llm.yaml`, MCP servers in `configs/servers.yaml`."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from mcp import StdioServerParameters
from pydantic import BaseModel, Field

from toolhelper.llm.base import ChatOptions, LLMBackend

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
LLM_CONFIG = CONFIG_DIR / "llm.yaml"
SERVERS_CONFIG = CONFIG_DIR / "servers.yaml"
CATALOG_DIR = ROOT / "data" / "catalog"
REPLAY_DIR = ROOT / "data" / "replay"
CONTEXT_DB = ROOT / "data" / "context.sqlite"
TRACES_DIR = ROOT / "data" / "traces"
MONITOR_DB = ROOT / "data" / "monitoring.sqlite"
REPORTS_DIR = ROOT / "data" / "reports"
TASKS_FILE = ROOT / "data" / "tasks" / "benchmark.jsonl"


class LLMProfile(BaseModel):
    backend: Literal["ollama", "openai_compat", "anthropic", "scripted"]
    model: str
    host: str | None = None
    base_url: str | None = None
    # Name of the environment variable holding the API key; keys are never stored in config.
    api_key_env: str | None = None
    # Hosted endpoints (openai_compat only): client-side request spacing, SDK retries on 429/5xx/timeouts,
    # and a per-request timeout.
    requests_per_minute: float | None = None
    max_retries: int | None = None
    timeout_s: float | None = None
    max_wait_s: float | None = None  # keep retrying a stalled request this long
    options: ChatOptions = Field(default_factory=ChatOptions)


class LLMConfig(BaseModel):
    default: str
    profiles: dict[str, LLMProfile]

    def get(self, name: str | None) -> tuple[str, LLMProfile]:
        name = name or self.default
        if name not in self.profiles:
            raise KeyError(f"unknown LLM profile {name!r}; known: {sorted(self.profiles)}")
        return name, self.profiles[name]


def load_llm_config(path: Path | str = LLM_CONFIG) -> LLMConfig:
    return LLMConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def make_backend(profile: LLMProfile, *, client: Any = None) -> LLMBackend:
    """Builds the backend for a profile. `client` injects an SDK client (for tests)."""
    opts = profile.options
    match profile.backend:
        case "ollama":
            from toolhelper.llm.ollama import OllamaBackend

            return OllamaBackend(profile.model, opts, host=profile.host or "http://localhost:11434",
                                 client=client)
        case "openai_compat":
            from toolhelper.llm.openai_compat import OpenAICompatBackend

            return OpenAICompatBackend(profile.model, opts, base_url=profile.base_url,
                                       api_key_env=profile.api_key_env, client=client,
                                       requests_per_minute=profile.requests_per_minute,
                                       max_retries=profile.max_retries,
                                       timeout_s=profile.timeout_s, max_wait_s=profile.max_wait_s)
        case "anthropic":
            from toolhelper.llm.anthropic import AnthropicBackend

            return AnthropicBackend(profile.model, opts, api_key_env=profile.api_key_env, client=client)
        case "scripted":
            from toolhelper.llm.scripted import ScriptedBackend

            return ScriptedBackend([], model=profile.model, defaults=opts)


def get_backend(name: str | None = None, path: Path | str = LLM_CONFIG) -> LLMBackend:
    _, profile = load_llm_config(path).get(name)
    return make_backend(profile)


class MissingSecretError(RuntimeError):
    pass


def read_env(name: str) -> str | None:
    """Reads an environment variable, falling back to the Windows user environment
    (shells started before a variable was set don't see it)."""
    value = os.environ.get(name)
    if value or sys.platform != "win32":
        return value
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return winreg.QueryValueEx(key, name)[0] or None
    except OSError:
        return None


class ServerConfig(BaseModel):
    package: str
    version: str
    command: str
    args: list[str] = Field(default_factory=list)
    env_vars: list[str] = Field(default_factory=list)
    timeout_s: float = 60
    # Minimum seconds between live calls (upstream rate limits).
    min_interval_s: float = 0

    def stdio_params(self, name: str) -> StdioServerParameters:
        env = {}
        for var in self.env_vars:
            value = read_env(var)
            if not value:
                raise MissingSecretError(f"server {name!r} needs environment variable {var}")
            env[var] = value
        return StdioServerParameters(
            command=os.path.expandvars(self.command),
            args=[os.path.expandvars(a) for a in self.args],
            env=env or None,
        )


def load_servers_config(path: Path | str = SERVERS_CONFIG) -> dict[str, ServerConfig]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {name: ServerConfig.model_validate(cfg) for name, cfg in data["servers"].items()}
