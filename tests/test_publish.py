from types import SimpleNamespace

import pytest
import warmhub
from warmhub import WarmHubError

from toolhelper.context.ingest import import_catalog, tool_name
from toolhelper.context.local_store import LocalStore
from toolhelper.context.model import Add, ConflictError, EvidenceRef, Retract, Revise
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


def test_publish_takes_the_records_an_accepted_decision_cites(local):
    summary = {"experiment": "experiments/x", "arm": "A", "group": "x-A-r1", "repeat": 1, "runs": 1,
               "success_rate": 1.0, "tokens_per_success": 900, "trust_mean": 1.0, "run_ids": ["run-1"]}
    decision = {"title": "t", "objective": "o", "chosen_config_json": "{}", "constraints_json": "{}",
                "rejected_json": "[]"}
    local.apply("experiment", [
        Add(name="experiments/x", shapes=["Experiment"],
            data={"title": "x", "inputs_json": "{}", "plan_json": "{}", "folds_json": "{}", "status": "proposed"}),
        Add(name="runsummaries/x-A-r1", shapes=["RunSummary"], data=summary,
            evidence=[EvidenceRef(kind="run", ref="run-1")]),
        Add(name="runsummaries/x-B-r1", shapes=["RunSummary"], data={**summary, "arm": "B", "group": "x-B-r1"}),
        Add(name="runsummaries/y-A-r1", shapes=["RunSummary"], data={**summary, "group": "y-A-r1"}),
        Add(name="decisions/x", shapes=["Decision"],
            data={**decision, "evidence": ["runsummaries/x-A-r1", "runsummaries/x-B-r1"], "status": "accepted"}),
        Add(name="decisions/y", shapes=["Decision"],
            data={**decision, "evidence": ["runsummaries/y-A-r1"], "status": "proposed"}),
    ])
    remote = LocalStore()
    plan = publish(local, remote)
    assert plan.skipped == {"recipes/omop/proposed": "not accepted", "decisions/y": "not accepted"}
    assert remote.get("decisions/x").data["evidence"] == ["runsummaries/x-A-r1@v1", "runsummaries/x-B-r1@v1"]
    assert remote.get("runsummaries/x-A-r1").data["experiment"] == "experiments/x@v1"
    assert remote.get("experiments/x").data["status"] == "proposed"  # cited, so it goes out as it is
    assert remote.get("evidence/runsummaries/x-A-r1/v1/0").data["ref"] == "run-1"
    assert [t.name for t in remote.head(shape="RunSummary")] == ["runsummaries/x-A-r1", "runsummaries/x-B-r1"]
    assert remote.stale() == []
    assert publish(local, remote).written == []


def test_publish_adds_evidence_missing_from_an_unchanged_thing(local):
    remote = LocalStore()
    publish(local, remote)
    remote.apply("lost", [Retract(name=f"evidence/{FIND}/v1/0")])
    assert publish(local, remote).written == [f"evidence/{FIND}/v1/0@v1"]
    assert remote.get(f"evidence/{FIND}/v1/0").data["kind"] == "schema_hash"


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
        self.reject = set()

    def _get(self, name, **kw):
        if name not in self.store:
            raise WarmHubError("NOT_FOUND", f"{name} not found")
        return self.store[name]

    def apply(self, message, ops, **kw):
        self.applied.append((message, ops, kw))
        return SimpleNamespace(operations=[
            SimpleNamespace(name=o.name, submitted_name=o.name, status="error" if o.name in self.reject else "applied",
                            error=SimpleNamespace(message="rejected")) for o in ops])


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

    # A thing added earlier in the same commit can be pinned by a later operation.
    store.apply("m", [Add(name="servers/new", shapes=["McpServer"], data={"name": "new"}),
                      Add(name="tools/new/t", shapes=["ToolSchema"], data={**data, "server": "servers/new"})])
    assert repo.applied.pop()[1][1].data["server"] == "servers/new@v1"

    # WarmHub applies what it can; a rejected operation must not pass silently.
    repo.reject = {"servers/bad"}
    with pytest.raises(ConflictError, match="1 of 2 operations failed .servers/bad: rejected"):
        store.apply("m", [Add(name="servers/ok", shapes=["McpServer"], data={"name": "ok"}),
                          Add(name="servers/bad", shapes=["McpServer"], data={"name": "bad"})])
    repo.applied.pop()

    store.apply("m", [Revise(name="servers/omop", data={"name": "omop"})])  # identical: no-op
    assert len(repo.applied) == 1
