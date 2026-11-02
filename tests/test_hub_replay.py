import time

import pytest

from toolhelper.mcp.canonical import NAME_RE
from toolhelper.mcp.catalog import build_snapshot
from toolhelper.mcp.hub import Hub, ReplayMiss
from toolhelper.mcp.recorder import RecordingStore
from toolhelper.mcp.replay_server import build_replay_server

INVALID_PARAMS = -32602
from toolhelper.llm.base import estimate_tokens

pytestmark = pytest.mark.anyio


def replay_hub(server_configs, catalog, store, **kw) -> Hub:
    return Hub(server_configs, mode="replay", store=store, catalog=catalog, live_fallback=False, **kw)


def fake_live(catalog, store) -> dict:
    """In-process servers standing in for the live ones, serving recorded real responses."""
    return {name: build_replay_server(name, catalog.servers[name], store) for name in catalog.servers}


async def test_replay_lists_19_namespaced_tools(server_configs, catalog, store):
    async with replay_hub(server_configs, catalog, store) as hub:
        names = [t.name for t in hub.tools]
        assert len(names) == 19
        assert {s: sum(t.server == s for t in hub.tools) for s in server_configs} == \
            {"medical": 10, "biomcp": 7, "omop": 2}
        assert all(NAME_RE.match(n) for n in names)
        spec = next(t for t in hub.tool_specs() if t.name == "medical__search-drugs")
        assert spec.input_schema["required"] == ["query"]


async def test_replay_tool_list_hashes_identically_to_live(server_configs, catalog, store):
    async with replay_hub(server_configs, catalog, store) as hub:
        snap = build_snapshot(hub, estimate_tokens)
    assert snap.source == "replay"
    assert snap.catalog_hash == catalog.catalog_hash
    for name, server in catalog.servers.items():
        assert snap.servers[name].server_hash == server.server_hash
        assert [t.definition for t in snap.servers[name].tools] == [t.definition for t in server.tools]
        assert snap.servers[name].server_info == server.server_info


async def test_every_recording_replays_exactly(server_configs, catalog, store):
    recordings = list(store)
    assert len(recordings) >= 2 * 19 - 5  # a few real calls per tool
    assert {(r.server, r.tool) for r in recordings} == {
        (s, t.name) for s, srv in catalog.servers.items() for t in srv.tools}
    assert any(r.result and r.result.get("isError") for r in recordings)  # real error responses
    async with replay_hub(server_configs, catalog, store) as hub:
        for rec in recordings:
            result = await hub.call(f"{rec.server}__{rec.tool}", rec.arguments)
            assert result.source == "replay"
            assert result.raw == rec.result
            assert result.is_error == bool(rec.result.get("isError"))
            assert result.recorded_at == rec.recorded_at
            assert result.text


async def test_miss_raises_without_live_fallback(server_configs, catalog, store):
    async with replay_hub(server_configs, catalog, store) as hub:
        with pytest.raises(ReplayMiss):
            await hub.call("medical__search-drugs", {"query": "never recorded"})
        with pytest.raises(KeyError):
            await hub.call("medical__no-such-tool", {})


async def test_miss_falls_through_to_live_and_is_recorded(server_configs, catalog, store, tmp_path):
    rec = next(r for r in store if r.tool == "find_omop_concept")
    local = RecordingStore(tmp_path)
    hub = Hub(server_configs, mode="replay", store=local, catalog=catalog,
              live_targets=fake_live(catalog, store))
    async with hub:
        first = await hub.call("omop__find_omop_concept", rec.arguments)
        second = await hub.call("omop__find_omop_concept", rec.arguments)
    assert first.source == "live" and second.source == "replay"
    assert first.raw == second.raw == rec.result
    saved = local.get("omop", "find_omop_concept", rec.arguments)
    assert saved.server_version == server_configs["omop"].version
    assert saved.server_info == catalog.servers["omop"].server_info
    assert saved.schema_hash == next(t.schema_hash for t in catalog.servers["omop"].tools
                                     if t.name == "find_omop_concept")
    assert second.recorded_at == saved.recorded_at


async def test_protocol_errors_are_recorded_and_replayed(server_configs, catalog, store, tmp_path):
    # None of the three servers returns JSON-RPC errors (all report is_error results), so the
    # "live" side is a replay server over an empty store whose miss is sent as INVALID_PARAMS.
    local = RecordingStore(tmp_path)
    empty = RecordingStore(tmp_path / "empty")
    empty_live = {n: build_replay_server(n, catalog.servers[n], empty, miss_code=INVALID_PARAMS)
                  for n in catalog.servers}
    async with Hub(server_configs, mode="record", store=local, live_targets=empty_live) as hub:
        live = await hub.call("biomcp__search", {"entity": "gene", "query": "TP53"})
    assert live.is_error and live.protocol_error["code"] == INVALID_PARAMS and live.raw is None
    assert local.get("biomcp", "search", {"entity": "gene", "query": "TP53"}).protocol_error

    async with replay_hub(server_configs, catalog, local) as hub:
        replayed = await hub.call("biomcp__search", {"entity": "gene", "query": "TP53"})
    assert replayed.source == "replay"
    assert replayed.protocol_error == live.protocol_error


async def test_live_calls_are_throttled(server_configs, catalog, store, tmp_path):
    rec = next(r for r in store if r.tool == "find_omop_concept")
    configs = {"omop": server_configs["omop"].model_copy(update={"min_interval_s": 0.3})}
    async with Hub(configs, mode="record", store=RecordingStore(tmp_path),
                   live_targets=fake_live(catalog, store)) as hub:
        start = time.monotonic()
        await hub.call("omop__find_omop_concept", rec.arguments)
        await hub.call("omop__find_omop_concept", rec.arguments)
        assert time.monotonic() - start >= 0.25  # Windows clock granularity
