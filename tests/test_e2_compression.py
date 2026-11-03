"""E2 schema compression over the 19 real tool schemas of the catalog snapshot."""

import pytest
from conftest import final, recorded, tool_call
from jsonschema import validators

from toolhelper.agent.hooks import Task
from toolhelper.agent.loop import Agent
from toolhelper.efficiency.compression import SchemaCompression, arg_error_rate, compress_schema
from toolhelper.llm.base import estimate_tokens
from toolhelper.llm.scripted import ScriptedBackend

pytestmark = pytest.mark.anyio


def schemas(catalog):
    return {f"{s}__{t.name}": t.definition["inputSchema"] for s, srv in catalog.servers.items() for t in srv.tools}


def keywords(schema, out=None):
    """Validation keywords anywhere in a schema, with their values."""
    out = out if out is not None else []
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k in ("type", "enum", "required", "minimum", "maximum", "maxItems", "minItems", "pattern",
                     "additionalProperties", "maxLength", "minLength"):
                out.append((k, repr(v)))
            keywords(v, out)
    elif isinstance(schema, list):
        for s in schema:
            keywords(s, out)
    return sorted(out)


@pytest.mark.parametrize("level", ["light", "aggressive"])
def test_validation_keywords_survive(catalog, level):
    for name, schema in schemas(catalog).items():
        assert keywords(compress_schema(schema, level)) == keywords(schema), name


def test_recorded_valid_calls_stay_valid(catalog, store):
    by_name = schemas(catalog)
    for rec in store:
        schema = by_name[f"{rec.server}__{rec.tool}"]
        if rec.result and not rec.result.get("isError") and not list(validators.validator_for(schema)(schema).iter_errors(rec.arguments)):
            compressed = compress_schema(schema, "aggressive")
            assert not list(validators.validator_for(compressed)(compressed).iter_errors(rec.arguments)), rec.tool


async def test_tool_definitions_shrink_with_evidence(replay_hub, store):
    task = Task(id="t", prompt="test")
    heart = recorded(store, "find_omop_concept", keyword="heart attack", omop_table="condition_occurrence")
    results = {}
    for level in ("light", "aggressive"):
        async with replay_hub:
            trace = await Agent(ScriptedBackend([tool_call("omop__find_omop_concept", heart), final("4329847")]),
                                replay_hub, [SchemaCompression(level)], counter=estimate_tokens).run(task)
        results[level] = trace
        assert trace.provenance_coverage == 1.0 and trace.problems() == []
        assert all(t.evidence[0].kind == "schema_hash" and t.after_tokens < t.before_tokens for t in trace.transformations)
        assert len(trace.transformations) >= 15  # most of the 19 tools have something to drop
    async with replay_hub:
        base = await Agent(ScriptedBackend([final("x")]), replay_hub, [], counter=estimate_tokens).run(task)
    tool_defs = {k: v.llm_calls[0].est_breakdown["tool_defs"] for k, v in results.items()}
    assert tool_defs["aggressive"] < tool_defs["light"] < base.llm_calls[0].est_breakdown["tool_defs"]
    assert arg_error_rate(results.values()) == {"calls": 2, "errors": 0, "rate": 0.0}
