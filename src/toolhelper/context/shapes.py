"""Context-store shapes, in WarmHub field syntax (README section 2, Shapes).

Field types the README leaves implicit are strings, except the clearly numeric
`success_rate`, `mean_tokens`, `expand_rate` and `ttl_seconds`. `Evidence` is an
addition: WarmHub has no evidence relation, so evidence is an assertion about the
exact version it supports. `Experiment` (frozen inputs, plan, folds) and `RunSummary` (one per run group) are Phase 10's.
"""

from __future__ import annotations

from typing import Any

STATUS = {"type": "string", "enum": ["proposed", "accepted"]}

SHAPES: dict[str, dict[str, Any]] = {
    "McpServer": {"name": "string", "package": "string", "version": "string", "launch": "string",
                  "env_vars?": ["string"]},
    "ToolSchema": {"server": "wref", "tool": "string", "description": "string",
                   "input_schema_json": "string", "schema_hash": "string", "catalog_tokens": "number",
                   "captured_at": "string"},
    "ToolCard": {"tool": "wref", "usage_notes": "string", "example_tasks": ["string"],
                 "pitfalls?": ["string"], "status": STATUS},
    "Dataset": {"name": "string", "source_url": "string", "license_note": "string", "version": "string",
                "sha256": "string", "rows": "number", "fetched_at": "string", "ttl_days": "number"},
    "EntityHint": {"surface": "string", "entity_type": "string", "canonical": "string",
                   "identifier": "string", "dataset": "wref", "row_key": "string", "confidence": "number"},
    "Recipe": {"task_type": "string", "steps": ["string"], "arg_templates_json": "string",
               "depends_on": ["wref"], "evidence_runs": ["string"], "success_rate": "number",
               "mean_tokens": "number", "status": STATUS},
    "Projection": {"tool": "wref", "keep_fields": ["string"], "drop_rule": "string",
                   "evidence_runs": ["string"], "expand_rate": "number"},
    "CachePolicy": {"tool": "wref", "ttl_seconds": "number", "source": "string"},
    "Decision": {"title": "string", "objective": "string", "chosen_config_json": "string",
                 "constraints_json": "string", "evidence": ["wref"], "rejected_json": "string",
                 "approved_by?": "string", "status": STATUS},
    "Experiment": {"title": "string", "inputs_json": "string", "plan_json": "string", "folds_json": "string",
                   "status": STATUS},
    "RunSummary": {"experiment": "wref", "arm": "string", "group": "string", "repeat": "number",
                   "runs": "number", "success_rate": "number", "tokens_per_success": "number",
                   "trust_mean": "number", "run_ids": ["string"]},
    "Evidence": {"about": "wref", "kind": {"type": "string", "enum": ["dataset_row", "run", "schema_hash"]},
                 "ref": "string", "detail_json?": "string"},
}


def field_name(spec: str) -> str:
    return spec[:-1] if spec.endswith("?") else spec


def wref_fields(shape: str) -> dict[str, bool]:
    """Fields holding wrefs, mapped to whether they are arrays."""
    out = {}
    for spec, kind in SHAPES.get(shape, {}).items():
        if kind in ("wref", "wref?"):
            out[field_name(spec)] = False
        elif isinstance(kind, list) and kind and kind[0] in ("wref", "wref?"):
            out[field_name(spec)] = True
    return out


def has_status(shape: str) -> bool:
    return "status" in SHAPES.get(shape, {})
