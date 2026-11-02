import pytest
from typer.testing import CliRunner

from toolhelper.cli import app
from toolhelper.config import LLM_CONFIG, load_llm_config, make_backend
from toolhelper.llm.anthropic import AnthropicBackend
from toolhelper.llm.ollama import OllamaBackend
from toolhelper.llm.openai_compat import OpenAICompatBackend
from toolhelper.llm.scripted import ScriptedBackend


def test_shipped_profiles_load_and_build():
    cfg = load_llm_config(LLM_CONFIG)
    assert cfg.default in cfg.profiles
    expected = {"ollama": OllamaBackend, "openai_compat": OpenAICompatBackend,
                "anthropic": AnthropicBackend, "scripted": ScriptedBackend}
    for profile in cfg.profiles.values():
        # Inject a dummy client so no SDK client (or credential lookup) is created.
        backend = make_backend(profile, client=object())
        assert isinstance(backend, expected[profile.backend])
        assert backend.model == profile.model


def test_default_ollama_profile_follows_experiment_defaults():
    _, p = load_llm_config().get(None)
    assert p.backend == "ollama"
    assert p.options.temperature == 0 and p.options.seed is not None
    assert p.options.num_ctx and p.options.think is False


def test_unknown_profile():
    with pytest.raises(KeyError):
        load_llm_config().get("nope")


def test_cli_lists_profiles():
    result = CliRunner().invoke(app, ["llm", "profiles"])
    assert result.exit_code == 0
    assert "ollama-qwen3" in result.output
