"""Config loading. LLM profiles live in `configs/llm.yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from toolhelper.llm.base import ChatOptions, LLMBackend

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
LLM_CONFIG = CONFIG_DIR / "llm.yaml"


class LLMProfile(BaseModel):
    backend: Literal["ollama", "openai_compat", "anthropic", "scripted"]
    model: str
    host: str | None = None
    base_url: str | None = None
    # Name of the environment variable holding the API key; keys are never stored in config.
    api_key_env: str | None = None
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
                                       api_key_env=profile.api_key_env, client=client)
        case "anthropic":
            from toolhelper.llm.anthropic import AnthropicBackend

            return AnthropicBackend(profile.model, opts, api_key_env=profile.api_key_env, client=client)
        case "scripted":
            from toolhelper.llm.scripted import ScriptedBackend

            return ScriptedBackend([], model=profile.model, defaults=opts)


def get_backend(name: str | None = None, path: Path | str = LLM_CONFIG) -> LLMBackend:
    _, profile = load_llm_config(path).get(name)
    return make_backend(profile)
