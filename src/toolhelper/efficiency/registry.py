"""Builds strategies from short specs (`E4:mode=repair`, `E1:top_k=8`) with their learned
inputs taken from a group of earlier runs (rules, usage policies, ToolCards, recipe paths,
evidence runs).

Strategies are always applied in ORDER, whatever order the specs come in: E3's rewrite
before E4's guard (so the guard checks the rewritten arguments), E7 after both (it caches
final arguments) and before E6 (it stores unshaped results), E8 and E9 last.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from toolhelper.agent.hooks import ToolDef

ORDER = ["E1", "E2", "E3", "E5", "E4", "E7", "E6", "E8", "E9"]


class Inputs(BaseModel):
    """What strategies learn from: earlier traces, the catalog tools, datasets and stores."""

    model_config = {"arbitrary_types_allowed": True}

    traces: list[Any] = []
    tools: list[ToolDef] = []
    recordings: list[Any] = []
    index: Any = None  # EntityIndex
    cdm_path: Path | None = None
    cache_path: Path | str = ":memory:"


def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """`E4:mode=repair,top_k=5` -> ("E4", {"mode": "repair", "top_k": 5})."""
    sid, _, rest = spec.partition(":")
    params: dict[str, Any] = {}
    for pair in filter(None, rest.split(",")):
        key, _, value = pair.partition("=")
        params[key.strip()] = int(value) if value.strip().isdigit() else value.strip()
    sid = sid.strip().upper()
    if sid not in ORDER:
        raise ValueError(f"unknown strategy {sid!r}; known: {ORDER}")
    return sid, params


def _e1(p, inp):
    from toolhelper.efficiency.arg_guard import learn_rules, observations_from_traces
    from toolhelper.efficiency.retrieval import ToolRetrieval, build_cards

    cards = build_cards(inp.tools, inp.traces, learn_rules(observations_from_traces(inp.traces)))
    return ToolRetrieval(cards, top_k=p.get("top_k", 5), mode=p.get("mode", "static"), route=p.get("route") == "on")


def _e2(p, inp):
    from toolhelper.efficiency.compression import SchemaCompression

    return SchemaCompression(p.get("level", "light"))


def _e3(p, inp):
    from toolhelper.efficiency.grounding import Grounding

    if inp.index is None:
        raise ValueError("E3 needs the dataset index (toolhelper datasets build)")
    modes = set(str(p.get("modes", "hints")).split("+"))
    datasets = str(p["datasets"]).split("+") if p.get("datasets") else None
    return Grounding(inp.index, modes, datasets=datasets)


def _e4(p, inp):
    from toolhelper.efficiency.arg_guard import (
        ArgumentGuard,
        CdmFields,
        learn_rules,
        observations_from_recordings,
        observations_from_traces,
    )

    obs = observations_from_traces(inp.traces) + observations_from_recordings(inp.recordings)
    cdm = CdmFields.load(inp.cdm_path) if inp.cdm_path and inp.cdm_path.exists() else None
    return ArgumentGuard(p.get("mode", "reject"), learn_rules(obs), cdm=cdm, index=inp.index)


def _e5(p, inp):
    from toolhelper.efficiency.recipes import Recipes, run_paths

    return Recipes(run_paths(inp.traces), p.get("mode", "plan"))


def _e6(p, inp):
    from toolhelper.efficiency.arg_guard import CdmFields
    from toolhelper.efficiency.shaping import ResultShaper, learn_policies

    cdm = CdmFields.load(inp.cdm_path) if inp.cdm_path and inp.cdm_path.exists() else None
    mode, budget = p.get("mode", "projection"), p.get("budget", 2000)
    policies = learn_policies(inp.traces, budgets=[budget] if mode == "budget" else [])
    return ResultShaper(mode, policies, top_n=p.get("top_n", 5), budget=budget, cdm=cdm)


def _e7(p, inp):
    from toolhelper.efficiency.cache import ResultCache, ResultCacheStrategy

    ttl = {k[4:]: float(v) for k, v in p.items() if k.startswith("ttl_")}
    return ResultCacheStrategy(ResultCache(inp.cache_path), ttl)


def _e8(p, inp):
    from toolhelper.efficiency.compaction import HistoryCompaction

    return HistoryCompaction(p.get("mode", "after_use"))


def _e9(p, inp):
    from toolhelper.efficiency.final_answer import FinalAnswerAtLimit

    runs = [t.run_id for t in inp.traces if t.stop_reason == "max_turns"]
    return FinalAnswerAtLimit(runs)


BUILDERS: dict[str, Callable[[dict[str, Any], Inputs], Any]] = {
    "E1": _e1, "E2": _e2, "E3": _e3, "E4": _e4, "E5": _e5, "E6": _e6, "E7": _e7, "E8": _e8, "E9": _e9}


def build_strategies(specs: list[str], inputs: Inputs) -> list[Any]:
    parsed = [parse_spec(s) for s in specs]
    ids = [sid for sid, _ in parsed]
    if len(set(ids)) != len(ids):
        raise ValueError(f"each strategy once: {ids}")
    return [BUILDERS[sid](params, inputs) for sid, params in sorted(parsed, key=lambda sp: ORDER.index(sp[0]))]
