"""What the experiment app's design form knows about strategies and designs, free of Streamlit
so it can be tested: the text shown for each strategy and parameter, slider ranges, building a
spec from widget values, and the factor rows the form edits.

The texts repeat the strategies' own docstrings and the measured Phase 10 results (README);
allowed values and defaults come from the registry, so a new variant shows up in the form
without a change here (it only lacks a description until one is added).
"""

from __future__ import annotations

import string
from typing import Any

from toolhelper.efficiency import custom
from toolhelper.efficiency.cache import DEFAULT_TTL
from toolhelper.efficiency.registry import BUILTIN, DEFAULTS, PARAMS, parse_spec

HOW_IT_WORKS = """
- **Strategy (E1 to E9):** one way of cutting tokens, with its settings (for example E6 with a budget of 2,000 tokens).
- **Base:** the strategies that are on in every run of this experiment, including its baseline.
- **Factor:** a switch with a letter (A, B, C or H, T, R). Switched on, it adds its strategies to the base.
  The experiment measures what switching each factor on changes.
- **Cell:** one combination of switches, named by the letters that are on (`AC`; `none` is the base alone).
  Each cell is one pass over every benchmark task.
- **Full factorial:** every combination, so k factors need 2^k cells. **Fraction:** half of them (or fewer);
  one factor is switched together with a product of others, which saves passes but mixes some interactions.
  **Confirm only:** no factorial; the named factors together are compared with the base on fresh paired repeats.
- **Choice rule (fixed before the run):** a factor is adopted when its tokens-per-success interval lies below 0,
  success does not drop by more than the gate allows and trust stays above the gate. The adopted combination
  is then confirmed with paired repeats against the base and checked live on one fold.
"""

# name, what it does, and what earlier experiments measured for it.
STRATEGIES: dict[str, dict[str, str]] = {
    "E1": {"name": "tool retrieval",
           "what": "Offers the model only the tools that match the task (lexical search over ToolCards) instead "
                   "of all 19, so fewer tool definitions are sent each turn.",
           "found": "p10, alone: success -9.9 points; retrieval dropped BioMCP tools that tasks needed. Failed the gate."},
    "E2": {"name": "schema compression",
           "what": "Shortens the tool definitions (44% of a baseline run's tokens). Any call valid under the "
                   "original schema stays valid.",
           "found": "p10b, minimal: tokens per success -18.1% (CI -26.3..-8.6). Light saved only 635 of 4,804 "
                    "definition tokens per turn. Lowers trust by about 0.04. Part of the accepted ABC config."},
    "E3": {"name": "dataset grounding",
           "what": "Resolves entities named in the task (genes, drugs, countries, OMOP terms) to canonical IDs "
                   "from the Phase 7 datasets before any call.",
           "found": "p10c, on top of ABC: hints cost +11.7% tokens per success; tool and rewrite had no clear "
                    "effect. p10d did not confirm tool + rewrite (-5.8%, CI includes 0)."},
    "E4": {"name": "argument guard",
           "what": "Checks a call's arguments before it reaches the server, with rules learned from real error "
                   "responses, the live schemas, OMOP's field map and the datasets.",
           "found": "p10b, repair: 0.45 fewer argument errors per run, but no token effect (+0.8%, CI includes 0)."},
    "E5": {"name": "recipes",
           "what": "Reuses the shortest frequent successful tool path of similar tasks (never the task's own runs).",
           "found": "p10, alone: +1.2 points, -12.7% tokens per success over 2 runs (within run-to-run noise). "
                    "Added to E2 light it cost 20.6% more tokens per success."},
    "E6": {"name": "result shaping",
           "what": "Keeps the part of a tool result that runs actually use and parks the rest behind "
                   "expand_result, so nothing is lost. Tool results carried across turns are 51% of a run's tokens.",
           "found": "p10b, budget 2000: tokens per success -16.8% (CI -27.1..-5.8), but +22.8% tokens per run on "
                    "OMOP tasks. Projection failed p10's gate (-4.3 points). Part of the accepted ABC config."},
    "E7": {"name": "result cache",
           "what": "Reuses the result of an identical call within its time to live instead of calling again.",
           "found": "p10, alone: no gain (tokens per success +3.4%, success -0.6 points)."},
    "E8": {"name": "history compaction",
           "what": "Replaces tool results in later turns with a short summary and a handle; recall_result "
                   "returns the original.",
           "found": "p10, after_use: success -11.7 points, tokens per success +36.6% (it hid facts still needed). "
                    "Failed the gate."},
    "E9": {"name": "final answer at the turn limit",
           "what": "When the turn limit runs out, asks once more, with no tools, for an answer from what was "
                   "gathered. Give the base or factor a turn limit to go with it.",
           "found": "p10b, with a limit of 6 turns: tokens per success -17.6%, success +6.0 points. Part of the "
                    "accepted ABC config."},
}

# Per (strategy, parameter): label, help, `range` (min, max, step) for numbers, `values` (what each
# choice means) and `when` (parameter, values): shown only while that earlier parameter has one of them.
PARAM_UI: dict[tuple[str, str], dict[str, Any]] = {
    ("E1", "top_k"): {"label": "tools offered (top_k)", "range": (1, 19, 1),
                      "help": "How many of the 19 tools the model sees. Share of needed tools still offered on "
                              "the benchmark: 0.79 at 3, 0.83 at 5, 0.96 at 8."},
    ("E1", "mode"): {"label": "how tools are offered", "values": {
        "static": "The top k tools, chosen once from the task.",
        "find_tools": "The top k tools plus a find_tools(query) tool that adds more during the run."}},
    ("E1", "route"): {"label": "server routing", "values": {
        "off": "Tools are ranked across all servers.",
        "on": "Keeps the best-scoring server's tools, plus any server scoring within half of it. "
              "It did not help on the benchmark."}},
    ("E2", "level"): {"label": "compression level", "values": {
        "light": "Drops titles, defaults and examples; descriptions keep their first paragraph or sentence.",
        "aggressive": "As light, but property descriptions go entirely and the tool description keeps one sentence.",
        "minimal": "As light, and also drops string lengths, minItems, uniqueItems and format "
                   "(the server and E4 still enforce them)."}},
    ("E3", "modes"): {"label": "how resolved entities reach the model (one or more)", "values": {
        "hints": 'One context line per entity, e.g. "HER2": gene ERBB2 [HGNC:3430].',
        "tool": "A local resolve_entity(text, type?) tool (adds one tool definition).",
        "rewrite": "Silently replaces a resolved name in arguments (gene symbols, ISO3 country codes, OMOP keywords)."}},
    ("E3", "datasets"): {"label": "datasets allowed to answer (empty = all)",
                         "help": "Restrict grounding to some datasets, e.g. only hgnc and ndc."},
    ("E4", "mode"): {"label": "what happens to a bad call", "values": {
        "reject": "The call is not sent; the model gets a one-line fix.",
        "repair": "The fix is applied when it is deterministic; what cannot be repaired is rejected."}},
    ("E5", "mode"): {"label": "how the recipe reaches the model", "values": {
        "plan": "The recipe is added to the context at the start.",
        "tool": "A local get_recipe tool the model may call."}},
    ("E6", "mode"): {"label": "what is cut", "values": {
        "projection": "Fields that no earlier run ever reused.",
        "top_n": "As projection, and only the first N items of a list.",
        "summary": "As top_n, with one line per kept item.",
        "budget": "As projection, and results over a token budget are cut to their leading whole items or sections."}},
    ("E6", "top_n"): {"label": "items kept per result (top_n)", "range": (1, 20, 1), "when": ("mode", ("top_n", "summary")),
                      "help": "A cut is applied only where earlier runs' used items all ranked within N."},
    ("E6", "budget"): {"label": "token budget per result", "range": (500, 8000, 250), "when": ("mode", ("budget",)),
                       "help": "Results longer than this are cut; the rest stays reachable through expand_result. "
                               "p10b used 2,000. Smaller saves more carried tokens but causes more expand calls."},
    ("E8", "mode"): {"label": "when a result is compacted", "values": {
        "after_turn": "Every result, once it is carried past the turn it arrived in.",
        "after_use": "Only once its identifiers appear in a later message or call."}},
}

RUN_MODES = {"replay": "Recordings first; a miss calls the real server and is recorded.",
             "live": "Always the real servers, no recordings.",
             "record": "Always the real servers, and the recordings are overwritten."}

KINDS = {"full": "Full factorial", "fraction": "Fraction", "confirm": "Confirm only"}
KIND_HELP = {"full": "Every combination of the factors: 2^k cells.",
             "fraction": "Fewer cells: some factors are switched together with a product of others.",
             "confirm": "No factorial: the named factors together vs the base, on fresh paired repeats."}

LEARNED = {"E1", "E4", "E5", "E6", "E9"}  # these learn from the runs of the `learn_from` groups


def info(sid: str, defs: dict[str, custom.Definition] | None = None) -> dict[str, str]:
    """Name, description and earlier result of a built-in or user-defined strategy."""
    if sid in STRATEGIES:
        return STRATEGIES[sid]
    d = (defs if defs is not None else custom.load()).get(sid)
    if d is None:
        return {"name": "unknown", "what": "Not defined in configs/strategies.yaml.", "found": "none"}
    return {"name": d.name, "what": f"{d.what.strip() + ' ' if d.what.strip() else ''}Your own strategy ({d.summary()}).",
            "found": "none yet; it has not been measured."}


def needs_runs(sid: str, defs: dict[str, custom.Definition]) -> bool:
    return sid in LEARNED or (sid in defs and custom.needs_runs(defs[sid]))


def spec_values(spec: str) -> tuple[str, dict[str, Any]]:
    """A spec as widget values: `E3:modes=tool+rewrite` -> ("E3", {"modes": ["tool", "rewrite"]})."""
    sid, params = parse_spec(spec)
    return sid, {k: str(v).split("+") if isinstance(PARAMS[sid].get(k), tuple) else v for k, v in params.items()}


def shown(sid: str, key: str, values: dict[str, Any]) -> bool:
    """Whether a parameter applies, given the parameters before it (E6's budget only in budget mode)."""
    when = PARAM_UI.get((sid, key), {}).get("when")
    return when is None or values.get(when[0]) in when[1]


def build_spec(sid: str, values: dict[str, Any]) -> str:
    """Widget values -> spec, in the registry's parameter order; empty values are left to the default."""
    order = [*PARAMS[sid], *(k for k in values if k not in PARAMS[sid])]
    parts = [f"{k}={'+'.join(values[k]) if isinstance(values[k], list) else values[k]}"
             for k in order if k in values and values[k] not in (None, "", [])]
    return sid + (":" + ",".join(parts) if parts else "")


def default_values(sid: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in PARAMS[sid]:
        if key in DEFAULTS[sid] and shown(sid, key, values):
            values[key] = DEFAULTS[sid][key]
    return values


def default_spec(sid: str) -> str:
    return build_spec(sid, default_values(sid))


def ttl_values(params: dict[str, Any]) -> dict[str, int]:
    """E7's time to live per server, in seconds: the cache's defaults overridden by `ttl_<server>`."""
    return {server: int(params.get(f"ttl_{server}", default)) for server, default in DEFAULT_TTL.items()}


def ttl_params(ttl: dict[str, int]) -> dict[str, int]:
    return {f"ttl_{server}": int(v) for server, v in ttl.items() if int(v) != DEFAULT_TTL.get(server)}


def next_letter(used: list[str]) -> str:
    return next((c for c in string.ascii_uppercase if c not in used), f"F{len(used) + 1}")


def rows_from(factors: dict[str, Any]) -> list[dict[str, Any]]:
    """The form's factor rows; `id` keys a row's widgets, so renaming a factor keeps its settings."""
    return [{"id": i, "name": name, "specs": list(f.get("specs", [])), "max_turns": f.get("max_turns"),
             "note": f.get("note", "")} for i, (name, f) in enumerate(factors.items())]


def factors_from(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for row in rows:
        f: dict[str, Any] = {"specs": row["specs"]}
        if row.get("max_turns"):
            f["max_turns"] = int(row["max_turns"])
        if row.get("note"):
            f["note"] = row["note"]
        out[row["name"]] = f
    return out


def row_problems(rows: list[dict[str, Any]]) -> list[str]:
    names = [r["name"] for r in rows]
    problems = [f"factor {n!r} is used {names.count(n)} times; every factor needs its own letter"
                for n in sorted(set(names)) if n and names.count(n) > 1]
    if "" in names:
        problems.append("a factor has no letter")
    return problems + [f"factor {r['name']}: nothing to switch on (pick a strategy or a turn limit)"
                       for r in rows if r["name"] and not r["specs"] and not r.get("max_turns")]


def kind_of(data: dict[str, Any]) -> str:
    return "confirm" if data.get("confirm_only") else "fraction" if data.get("generators") else "full"


def blank_like(data: dict[str, Any]) -> dict[str, Any]:
    """A new design with the run settings of `data` (LLM, mode, learning groups, gate, seeds) and
    no name, base or factors."""
    keep = ("llm", "mode", "max_turns", "parallel", "learn_from", "confirm_repeats", "live_fold", "order_seed",
            "gate", "bootstrap")
    return {"design": "factorial", "experiment": "", **{k: data[k] for k in keep if k in data}, "factors": {}}


def import_factors(data: dict[str, Any], source: dict[str, Any], names: list[str], into_base: bool) -> dict[str, Any]:
    """`data` with factors of another config added: as factors (a taken letter gets the next free
    one) or merged into the base, where they are on in every run. Raises ValueError when two
    specs set the same strategy differently."""
    from toolhelper.experiments.factorial import merge_specs

    out = {**data, "factors": dict(data.get("factors") or {})}
    picked = {n: source["factors"][n] for n in names}
    if into_base:
        base = dict(out.get("base") or {})
        base["specs"] = merge_specs([*base.get("specs", []), *(s for f in picked.values() for s in f.get("specs", []))])
        limits = [f["max_turns"] for f in picked.values() if f.get("max_turns")]
        if limits:
            base["max_turns"] = min([*limits, *([base["max_turns"]] if base.get("max_turns") else [])])
        out["base"] = base
        return out
    for name, factor in picked.items():
        letter = name if name not in out["factors"] else next_letter(list(out["factors"]))
        out["factors"][letter] = dict(factor)
    return out


assert set(STRATEGIES) == set(BUILTIN)
