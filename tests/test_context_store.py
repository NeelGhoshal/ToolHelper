import json
import sqlite3

import pytest
from warmhub.rules.shape_definition import shape_definition_preflight_error

from toolhelper.context.ingest import import_catalog, tool_name
from toolhelper.context.local_store import LocalStore
from toolhelper.context.model import (
    Add,
    ConflictError,
    EvidenceRef,
    NameTakenError,
    NotFoundError,
    Reaffirm,
    Retract,
    Revise,
    ValidationError,
    parse_wref,
)
from toolhelper.context.shapes import SHAPES

FIND = tool_name("omop", "find_omop_concept")
BATCH = tool_name("omop", "batch_map_concepts_from_csv")


@pytest.fixture
def ctx(catalog):
    store = LocalStore()
    import_catalog(store, catalog)
    return store


def recipe(depends_on, status="accepted") -> dict:
    return {"task_type": "omop_mapping", "steps": ["omop__find_omop_concept"],
            "arg_templates_json": json.dumps({"omop_table": "condition_occurrence"}),
            "depends_on": depends_on, "evidence_runs": ["run-1"], "success_rate": 1.0,
            "mean_tokens": 900, "status": status}


def test_shapes_pass_warmhub_preflight():
    for name, fields in SHAPES.items():
        assert shape_definition_preflight_error(name, {"fields": fields}, "add") is None, name


def test_parse_wref():
    assert parse_wref("tools/omop/x") == ("tools/omop/x", None)
    assert parse_wref("tools/omop/x@HEAD") == ("tools/omop/x", None)
    assert parse_wref("tools/omop/x@v3") == ("tools/omop/x", 3)
    with pytest.raises(ValidationError):
        parse_wref("x@v0")


def test_catalog_import_creates_versions_with_evidence(ctx, catalog):
    assert len(ctx.head(shape="McpServer")) == 3
    schemas = ctx.head(shape="ToolSchema")
    assert len(schemas) == 19
    find = ctx.get(FIND)
    snap_tool = next(t for t in catalog.servers["omop"].tools if t.name == "find_omop_concept")
    assert find.version == 1
    assert find.data["schema_hash"] == snap_tool.schema_hash
    assert find.data["server"] == "servers/omop@v1"  # bare wref pinned at write time
    evidence = ctx.get(f"evidence/{FIND}/v1/0")
    assert evidence.kind == "assertion" and evidence.about == f"{FIND}@v1"
    assert evidence.affirmed == [f"{FIND}@v1"]
    assert evidence.data["kind"] == "schema_hash" and evidence.data["ref"] == snap_tool.schema_hash
    assert json.loads(evidence.data["detail_json"])["tokens_measured"] is False


def test_reimporting_the_same_catalog_is_a_no_op(ctx, catalog):
    result = import_catalog(ctx, catalog.model_copy(update={"captured_at": "2026-12-01T00:00:00+00:00"}))
    assert result.written == [] and len(result.unchanged) == 22
    assert ctx.get(FIND).version == 1


def test_revising_a_tool_schema_marks_dependent_recipes_stale(ctx):
    """Phase 3 done criterion."""
    ctx.apply("recipe", [Add(name="recipes/omop_mapping/condition", shapes=["Recipe"], data=recipe([FIND, BATCH]))])
    r = ctx.get("recipes/omop_mapping/condition")
    assert r.data["depends_on"] == [f"{FIND}@v1", f"{BATCH}@v1"]
    assert ctx.stale() == []

    find = ctx.get(FIND)
    result = ctx.apply("schema drift", [Revise(name=FIND, shapes=["ToolSchema"], expected_version=1,
                                               data={**find.data, "description": "changed upstream"})])
    assert result.written == [f"{FIND}@v2"]
    assert [s.thing.name for s in result.stale] == ["recipes/omop_mapping/condition"]
    [stale] = ctx.stale(shape="Recipe")
    assert [(l.field, l.target, l.target_version, l.target_head) for l in stale.links] == [
        ("depends_on", FIND, 1, 2)]
    assert ctx.stale(shape="ToolCard") == []  # evidence about v1 is not a dependency

    # Re-pinning the recipe to the new schema clears it.
    ctx.apply("re-pin", [Revise(name=r.name, data={**r.data, "depends_on": [FIND, BATCH]})])
    assert ctx.stale() == []


def test_catalog_drift_through_import_marks_recipes_stale(ctx, catalog):
    ctx.apply("recipe", [Add(name="recipes/omop_mapping/condition", shapes=["Recipe"], data=recipe([FIND]))])
    drifted = catalog.model_copy(deep=True)
    tool = next(t for t in drifted.servers["omop"].tools if t.name == "find_omop_concept")
    tool.schema_hash = "0" * 64
    result = import_catalog(ctx, drifted)
    assert result.written == [f"{FIND}@v2", f"evidence/{FIND}/v2/0@v1"]
    assert [s.thing.name for s in result.stale] == ["recipes/omop_mapping/condition"]


def test_retracting_a_dependency_marks_dependents_stale(ctx):
    ctx.apply("recipe", [Add(name="recipes/omop_mapping/batch", shapes=["Recipe"], data=recipe([BATCH]))])
    result = ctx.apply("remove tool", [Retract(name=BATCH, reason="tool removed upstream")])
    [stale] = result.stale
    assert stale.links[0].target_active is False and stale.links[0].stale
    retracted = ctx.get(BATCH, include_retracted=True)
    assert not retracted.active and retracted.reason == "tool removed upstream"
    assert BATCH not in [t.name for t in ctx.head(shape="ToolSchema")]
    assert ctx.get(f"{BATCH}@v1").active  # pinned versions stay readable
    with pytest.raises(NotFoundError):
        ctx.get(BATCH)


def test_versions_and_write_rules(ctx):
    name = "cards/omop/find_omop_concept"
    data = {"tool": FIND, "usage_notes": "pick one candidate", "example_tasks": ["map heart attack"],
            "status": "proposed"}
    ctx.apply("card", [Add(name=name, shapes=["ToolCard"], data=data)])
    with pytest.raises(NameTakenError):
        ctx.apply("dup", [Add(name=name, shapes=["ToolCard"], data=data)])
    assert ctx.apply("same", [Revise(name=name, data=data)]).unchanged == [f"{name}@v1"]
    ctx.apply("accept", [Revise(name=name, data={**data, "status": "accepted"}, expected_version=1)])
    with pytest.raises(ConflictError):
        ctx.apply("stale write", [Revise(name=name, data=data, expected_version=1)])
    assert [t.version for t in ctx.history(name)] == [1, 2]
    assert [t.data["status"] for t in ctx.history(name)] == ["proposed", "accepted"]

    # Retract releases the name; a new thing starts again at v1.
    ctx.apply("retract", [Retract(name=name)])
    ctx.apply("again", [Add(name=name, shapes=["ToolCard"], data=data)])
    assert ctx.get(name).version == 1


def test_commits_are_atomic(ctx):
    with pytest.raises(ValidationError):
        ctx.apply("half", [Add(name="recipes/a", shapes=["Recipe"], data=recipe([FIND])),
                           Add(name="recipes/b", shapes=["Recipe"], data={"task_type": "x"})])
    with pytest.raises(NotFoundError):
        ctx.get("recipes/a")


def test_validation_uses_warmhub_rules(ctx):
    with pytest.raises(ValidationError, match="Missing required field"):
        ctx.apply("bad", [Add(name="recipes/x", shapes=["Recipe"], data={"task_type": "x"})])
    with pytest.raises(ValidationError, match="unknown shape"):
        ctx.apply("bad", [Add(name="x/y", shapes=["Nope"], data={})])
    with pytest.raises(NotFoundError):
        ctx.apply("bad", [Add(name="recipes/x", shapes=["Recipe"], data=recipe(["tools/omop/missing"]))])
    with pytest.raises(ValidationError):
        ctx.apply("bad", [Add(name="hub/shapes/Recipe", shapes=["Recipe"], data=recipe([FIND]))])
    result = ctx.apply("extra", [Add(name="recipes/x", shapes=["Recipe"], data={**recipe([FIND]), "note": "x"})])
    assert result.warnings  # undeclared fields are warnings, as in WarmHub


def test_versions_are_append_only(ctx):
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ctx.db.execute("UPDATE versions SET data_json = '{}'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ctx.db.execute("DELETE FROM links")


def test_reaffirm_edits_affirmations_only(ctx):
    find = ctx.get(FIND)
    ctx.apply("drift", [Revise(name=FIND, data={**find.data, "description": "v2"})])
    ev = f"evidence/{FIND}/v1/0"
    before = ctx.get(ev)
    ctx.apply("reaffirm", [Reaffirm(name=ev, add=[f"{FIND}@v2"], remove=[f"{FIND}@v1"])])
    after = ctx.get(ev)
    assert after.version == 2 and after.data == before.data and after.affirmed == [f"{FIND}@v2"]
    with pytest.raises(ValidationError):
        ctx.apply("wrong target", [Reaffirm(name=ev, add=[f"{BATCH}@v1"])])
    with pytest.raises(ValidationError):
        ctx.apply("unpinned", [Reaffirm(name=ev, add=[FIND])])


def test_refs_head_match_and_search(ctx):
    inbound = ctx.refs("servers/omop")
    assert {l.source for l in inbound} == {FIND, BATCH}
    outbound = ctx.refs(FIND, direction="outbound")
    assert [(l.field, l.target) for l in outbound] == [("server", "servers/omop")]
    assert [t.name for t in ctx.head(match="tools/omop/*")] == [BATCH, FIND]
    assert len(ctx.head(match="tools/**", shape="ToolSchema")) == 19
    assert len(ctx.head(match="tools/{omop,biomcp}/*")) == 9
    assert [t.name for t in ctx.search("OMOP CDM table", shape="ToolSchema")] == [FIND]


def test_evidence_attached_to_writes(ctx):
    ctx.apply("recipe", [Add(name="recipes/omop_mapping/condition", shapes=["Recipe"], data=recipe([FIND]),
                             evidence=[EvidenceRef(kind="run", ref="run-1")])])
    ev = ctx.get("evidence/recipes/omop_mapping/condition/v1/0")
    assert ev.data["kind"] == "run" and ev.about == "recipes/omop_mapping/condition@v1"


def test_file_store_persists(tmp_path, catalog):
    path = tmp_path / "ctx.sqlite"
    store = LocalStore(path)
    import_catalog(store, catalog)
    store.close()
    assert len(LocalStore(path).head(shape="ToolSchema")) == 19
