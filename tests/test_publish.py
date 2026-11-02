from types import SimpleNamespace

import pytest
import warmhub
from warmhub import WarmHubError

from toolhelper.context.ingest import import_catalog, tool_name
from toolhelper.context.local_store import LocalStore
from toolhelper.context.model import Add, EvidenceRef, Revise
from toolhelper.context.warmhub_store import WarmHubStore, publish

FIND = tool_name("omop", "find_omop_concept")


def recipe(depends_on, status) -> dict:
    return {"task_type": "omop_mapping", "steps": ["omop__find_omop_concept"], "arg_templates_json": "{}",
            "depends_on": depends_on, "evidence_runs": ["run-1"], "success_rate": 1.0, "mean_tokens": 900,
            "status": status}


@pytest.fixture
def local(catalog):
    store = LocalStore()
    import_catalog(store, catalog)
    store.apply("recipes", [
        Add(name="recipes/omop/accepted", shapes=["Recipe"], data=recipe([FIND], "accepted"),
            evidence=[EvidenceRef(kind="run", ref="run-1")]),
        Add(name="recipes/omop/proposed", shapes=["Recipe"], data=recipe([FIND], "proposed")),
    ])
    return store


def test_publish_pushes_curated_things_with_evidence(local):
    remote = LocalStore()
    plan = publish(local, remote)
    assert plan.skipped == {"recipes/omop/proposed": "not accepted"}
    assert len(remote.head(shape="ToolSchema")) == 19
    assert remote.get("recipes/omop/accepted").data["depends_on"] == [f"{FIND}@v1"]
    assert remote.get("evidence/recipes/omop/accepted/v1/0").data["kind"] == "run"
    assert remote.get(f"evidence/{FIND}/v1/0").data["kind"] == "schema_hash"
    assert remote.stale() == []
    assert publish(local, remote).written == []  # idempotent


def test_publish_remaps_pins_to_remote_versions(local):
    remote = LocalStore()
    publish(local, remote, shapes=["McpServer", "ToolSchema"])
    # Locally the schema moves twice; remotely it is published once more, as v2.
    for desc in ("v2", "v3"):
        find = local.get(FIND)
        local.apply("drift", [Revise(name=FIND, data={**find.data, "description": desc})])
    r = local.get("recipes/omop/accepted")
    assert publish(local, LocalStore(), shapes=["Recipe"]).skipped["recipes/omop/accepted"] == "stale"
    local.apply("re-pin", [Revise(name=r.name, data={**r.data, "depends_on": [FIND]})])
    assert local.get(r.name).data["depends_on"] == [f"{FIND}@v3"]

    plan = publish(local, remote)
    assert f"{FIND}@v2" in plan.written
    assert remote.get(FIND).data["description"] == "v3"
    assert remote.get("recipes/omop/accepted").data["depends_on"] == [f"{FIND}@v2"]
    assert remote.stale() == []


def test_dry_run_writes_nothing(local):
    remote = LocalStore()
    plan = publish(local, remote, dry_run=True)
    assert plan.written and remote.head() == []


class FakeRepo:
    """Stands in for the WarmHub SDK repository client."""

    def __init__(self, things):
        self.things = SimpleNamespace(get=self._get)
        self.store = things
        self.applied = []

    def _get(self, name, **kw):
        if name not in self.store:
            raise WarmHubError("NOT_FOUND", f"{name} not found")
        return self.store[name]

    def apply(self, message, ops, **kw):
        self.applied.append((message, ops, kw))


def test_warmhub_store_translates_writes_to_sdk_operations():
    server = SimpleNamespace(name="servers/omop", version=3, active=True, declared_shapes=["McpServer"],
                             data={"name": "omop"}, about_wref=None, affirmed_wrefs=[])
    repo = FakeRepo({"servers/omop": server})
    client = SimpleNamespace(repository=lambda locator: repo)
    store = WarmHubStore("org/toolhelper", client=client)
    data = {"server": "servers/omop", "tool": "t", "description": "d", "input_schema_json": "{}",
            "schema_hash": "h", "catalog_tokens": 1, "captured_at": "now"}
    result = store.apply("m", [Add(name="tools/omop/t", shapes=["ToolSchema"], data=data,
                                   evidence=[EvidenceRef(kind="schema_hash", ref="h")])])
    [(message, ops, kw)] = repo.applied
    assert [type(o) for o in ops] == [warmhub.Add, warmhub.Add]
    assert ops[0].data["server"] == "servers/omop@v3"  # pinned to the remote head
    assert ops[1].name == "evidence/tools/omop/t/v1/0"
    assert ops[1].affirmed_targets == ["tools/omop/t@v1"]
    assert "ToolSchema" in kw["shapes"]
    assert result.written == ["tools/omop/t@v1", "evidence/tools/omop/t/v1/0@v1"]

    store.apply("m", [Revise(name="servers/omop", data={"name": "omop"})])  # identical: no-op
    assert len(repo.applied) == 1
