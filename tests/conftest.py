import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeOllamaClient:
    """Stands in for `ollama.Client`: records requests, returns recorded responses."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        import ollama

        self.responses = [ollama.ChatResponse.model_validate(r) for r in responses]
        self.requests: list[dict[str, Any]] = []

    def chat(self, **request: Any):
        self.requests.append(request)
        return self.responses[len(self.requests) - 1]


class FakeOpenAIClient:
    def __init__(self, *responses: dict[str, Any]) -> None:
        from openai.types.chat import ChatCompletion

        self.responses = [ChatCompletion.model_validate(r) for r in responses]
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: Any):
        self.requests.append(request)
        return self.responses[len(self.requests) - 1]


class FakeAnthropicClient:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request: Any):
        self.requests.append(request)
        return self.responses[len(self.requests) - 1]


@pytest.fixture
def drug_tool():
    """The real `search-drugs` tool spec, captured from the live Medical MCP server."""
    from toolhelper.llm.base import ToolSpec

    return ToolSpec.model_validate(load_fixture("llm/ollama_tool_call.json")["tool_spec"])


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def catalog():
    """Catalog snapshot captured from the live servers."""
    from toolhelper.mcp.catalog import load_snapshot

    return load_snapshot(FIXTURES / "catalog.json")


@pytest.fixture
def store():
    """Real responses recorded from the live servers (see fixtures/replay_calls.yaml)."""
    from toolhelper.mcp.recorder import RecordingStore

    return RecordingStore(FIXTURES / "replay")


@pytest.fixture
def server_configs():
    from toolhelper.config import load_servers_config

    return load_servers_config()


@pytest.fixture
def replay_hub(server_configs, catalog, store):
    """Strict replay over the recorded real responses: a call that was never recorded fails."""
    from toolhelper.mcp.hub import Hub

    return Hub(server_configs, mode="replay", store=store, catalog=catalog, live_fallback=False)


def tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """A scripted assistant turn calling one tool."""
    return {"role": "assistant", "tool_calls": [{"name": name, "arguments": args}]}


def final(text: str) -> dict[str, Any]:
    """A scripted assistant turn that answers."""
    return {"role": "assistant", "content": text}


def recorded(store, tool: str, **match: Any) -> dict[str, Any]:
    """Arguments of a recorded real call whose arguments include `match`."""
    return next(r.arguments for r in store if r.tool == tool and all(r.arguments.get(k) == v for k, v in match.items()))


@pytest.fixture(scope="session")
def dataset_index(tmp_path_factory):
    """Entity index built from the real dataset excerpts in fixtures/datasets/."""
    import shutil

    from toolhelper.context.local_store import LocalStore
    from toolhelper.datasets.base import load_config
    from toolhelper.datasets.build import build_dataset
    from toolhelper.datasets.index import EntityIndex

    tmp = tmp_path_factory.mktemp("dataset_index")
    shutil.copytree(FIXTURES / "datasets", tmp / "raw")
    index, store, cfg = EntityIndex(tmp / "index.sqlite"), LocalStore(), load_config()
    for name in ("ndc", "rxnorm", "hgnc", "countries", "clinvar", "omop_eunomia"):
        build_dataset(cfg[name], tmp / "raw" / name, index, store)
    return index
