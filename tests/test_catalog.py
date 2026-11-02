import pytest

from toolhelper.mcp.canonical import canonical_json, namespaced, tool_hash
from toolhelper.mcp.catalog import diff_snapshots, latest_snapshot, llm_view, save_snapshot
from toolhelper.mcp.recorder import args_key


def test_live_snapshot_has_19_tools_with_tokens(catalog):
    assert catalog.tool_count == 19
    assert {n: len(s.tools) for n, s in catalog.servers.items()} == {"medical": 10, "biomcp": 7, "omop": 2}
    assert catalog.tokens_measured is False and catalog.tokenizer == "cl100k_base"
    for server in catalog.servers.values():
        assert all(t.tokens > 0 for t in server.tools)
        assert server.tokens == sum(t.tokens for t in server.tools)
        for t in server.tools:
            assert t.schema_hash == tool_hash(t.definition)


def test_llm_view_uses_namespaced_name(catalog):
    tool = catalog.servers["medical"].tools[0]
    assert f'"name":"{tool.namespaced}"' in llm_view(tool.namespaced, tool.definition)


def test_diff_identical_is_empty(catalog):
    assert diff_snapshots(catalog, catalog.model_copy(deep=True)).empty


def test_diff_detects_changed_added_removed(catalog):
    new = catalog.model_copy(deep=True)
    omop = new.servers["omop"]
    changed = omop.tools[1]
    changed.definition["description"] = "edited"
    changed.schema_hash = tool_hash(changed.definition)
    removed = omop.tools.pop(0)
    diff = diff_snapshots(catalog, new)
    assert diff.removed == [removed.namespaced]
    assert [(c.name, c.fields) for c in diff.changed] == [(changed.namespaced, ["description"])]
    assert diff_snapshots(new, catalog).added == [removed.namespaced]


def test_diff_detects_server_version_change(catalog):
    new = catalog.model_copy(deep=True)
    new.servers["biomcp"].version = "0.9.4"
    assert set(diff_snapshots(catalog, new).server_info) == {"biomcp"}


def test_save_and_load_latest(catalog, tmp_path):
    assert latest_snapshot(tmp_path) is None
    save_snapshot(catalog, tmp_path)
    assert latest_snapshot(tmp_path) == catalog


def test_canonical_forms():
    assert canonical_json({"b": 1, "a": [2, {"d": 1, "c": 0}]}) == '{"a":[2,{"c":0,"d":1}],"b":1}'
    assert args_key({"query": "x", "limit": 2}) == args_key({"limit": 2, "query": "x"})
    assert args_key({"limit": 2}) != args_key({"limit": "2"})


def test_namespaced_names():
    assert namespaced("medical", "search-drugs") == "medical__search-drugs"
    with pytest.raises(ValueError):
        namespaced("medical", "bad name")
    with pytest.raises(ValueError):
        namespaced("medical", "x" * 60)
