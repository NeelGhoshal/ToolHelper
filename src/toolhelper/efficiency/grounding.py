"""E3: dataset grounding. Resolves entities in the task to canonical IDs before any call.

Entities are found in the prompt as word spans (longest first, up to 5 words) that the
entity index resolves exactly or after normalization; fuzzy matches are never used here.
Gene matches need an upper-case letter or digit in the span (HGNC aliases include plain
words), and spans made only of stopwords are skipped. An ambiguous span is reported with
its top two answers rather than a guess.

Variants (any combination):
- `hints`: one context item per entity, e.g. `"HER2": gene ERBB2 [HGNC:3430]`;
- `tool`: a local `resolve_entity(text, type?)` tool over the same index;
- `rewrite`: silently replace a resolved surface in arguments where its type fits (gene ids,
  WHO country codes, OMOP keywords).
crossed with `datasets`, the subset of datasets allowed to answer.

Evidence for every hint and rewrite is the dataset row (`datasets/<name>@vN#<row_key>`).
`uptake(trace)` reports how many hinted values the model then used in its arguments.
"""

from __future__ import annotations

import re
from typing import Any

from toolhelper.agent.hooks import BaseStrategy, CallDecision, ContextItem, ExtraTool, PendingCall, Task, ToolDef
from toolhelper.context.model import EvidenceRef
from toolhelper.datasets.base import normalize
from toolhelper.datasets.index import EntityIndex, Match
from toolhelper.mcp.canonical import canonical_json
from toolhelper.mcp.hub import ToolResult

RESOLVE = "toolhelper__resolve_entity"
STOPWORDS = set("""a an and are as at be by can do does find for from get give has have how i in into is it its list
map me of on or show tell than that the their them then there these this to use using via was what when where which
who why with within you your all any both each into most other some such only own same so very just also about
""".split())
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'.+-]*[A-Za-z0-9]|[A-Za-z0-9]")

# Where a resolved entity may replace an argument: (tool, argument, required other args, use identifier?)
REWRITE_TARGETS = {
    "gene": [("biomcp__search", "gene", {}, False), ("biomcp__search", "query", {"entity": "gene"}, False),
             ("biomcp__get", "id", {"entity": "gene"}, False)],
    "country": [("medical__get-health-statistics", "country", {}, True)],
    "omop_": [("omop__find_omop_concept", "keyword", {}, False)],
}


def _targets(entity_type: str):
    return REWRITE_TARGETS.get(entity_type) or (REWRITE_TARGETS["omop_"] if entity_type.startswith("omop_") else [])


def _evidence(m: Match, surface: str) -> EvidenceRef:
    return EvidenceRef(kind="dataset_row", ref=f"{m.dataset}#{m.row_key}", detail={
        "surface": surface, "type": m.type, "canonical": m.canonical, "identifier": m.identifier})


def describe(m: Match) -> str:
    return f"{m.type} {m.canonical} [{m.identifier}]"


class Grounding(BaseStrategy):
    name = "E3_grounding"

    def __init__(self, index: EntityIndex, modes: set[str] | frozenset[str] = frozenset({"hints"}), *,
                 datasets: list[str] | None = None, max_words: int = 5) -> None:
        unknown = set(modes) - {"hints", "tool", "rewrite"}
        if unknown:
            raise ValueError(f"unknown E3 modes {sorted(unknown)}")
        self.index = index
        self.modes = set(modes)
        self.datasets = datasets
        self.max_words = max_words
        self.found: list[tuple[str, list[Match]]] = []

    def config(self) -> dict[str, Any]:
        return {"modes": sorted(self.modes), "datasets": self.datasets,
                "versions": [d["wref"] for d in self.index.datasets()
                             if self.datasets is None or d["name"] in self.datasets]}

    def _allowed(self, m: Match) -> bool:
        return self.datasets is None or m.dataset.split("/", 1)[1].split("@")[0] in self.datasets

    def lookup(self, surface: str, type: str | None = None) -> tuple[str, list[Match]]:
        res = self.index.resolve(surface, type=type)
        if res.status not in ("exact", "normalized", "ambiguous"):
            return "none", []
        matches = [m for m in res.matches if self._allowed(m) and m.method != "fuzzy"]
        return (res.status if matches else "none"), matches

    def entities(self, text: str) -> list[tuple[str, list[Match]]]:
        words = [(m.group(), m.start(), m.end()) for m in WORD.finditer(text)]
        used = [False] * len(words)
        found = []
        for n in range(min(self.max_words, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                if any(used[i:i + n]):
                    continue
                span_words = [w for w, _, _ in words[i:i + n]]
                if all(w.casefold() in STOPWORDS for w in span_words):
                    continue
                surface = text[words[i][1]:words[i + n - 1][2]]
                if len(normalize(surface)) < 3:
                    continue
                status, matches = self.lookup(surface)
                cased = any(c.isupper() or c.isdigit() for c in surface)
                matches = [m for m in matches if m.type != "gene" or cased]
                if status == "none" or not matches:
                    continue
                found.append((surface, matches[:3]))
                used[i:i + n] = [True] * n
        order = {s: text.find(s) for s, _ in found}
        return sorted(found, key=lambda f: order[f[0]])

    def prepare(self, task: Task, ctx) -> list[ContextItem]:
        self.found = self.entities(task.prompt)
        if "hints" not in self.modes:
            return []
        items = []
        for surface, matches in self.found:
            same = [m for m in matches if m.type == matches[0].type]
            if len(same) > 1 and same[0].score - same[1].score <= 0.5:
                text = f'"{surface}" is ambiguous: ' + " or ".join(describe(m) for m in same[:2])
                shown = same[:2]
            else:
                shown = [matches[0]] + [m for m in matches[1:] if m.type != matches[0].type][:1]
                text = f'"{surface}": ' + "; ".join(describe(m) for m in shown)
            items.append(ContextItem(kind="entity", text=text, evidence=[_evidence(m, surface) for m in shown]))
        return items

    def before_call(self, call: PendingCall) -> CallDecision:
        if "rewrite" not in self.modes:
            return CallDecision()
        args, evidence, notes = dict(call.arguments), [], []
        for surface, matches in self.found:
            m = matches[0]
            for tool, arg, required, use_id in _targets(m.type):
                if tool != call.name or arg not in args or any(args.get(k) != v for k, v in required.items()):
                    continue
                new = m.identifier if use_id else m.canonical
                value = args[arg]
                if isinstance(value, str) and normalize(value) == normalize(surface) and value != new:
                    args[arg] = new
                    evidence.append(_evidence(m, surface))
                    notes.append(f"{arg}: {value!r} -> {new!r}")
        if not evidence:
            return CallDecision()
        return CallDecision(action="repaired", arguments=args, evidence=evidence, note="grounded " + "; ".join(notes))

    def extra_tools(self) -> list[ExtraTool]:
        if "tool" not in self.modes:
            return []

        async def resolve(call: PendingCall) -> ToolResult:
            text = str(call.arguments.get("text", ""))
            status, matches = self.lookup(text, call.arguments.get("type"))
            lines = [f"{status}: {text}"] + [f"- {describe(m)} via {m.source} '{m.surface}' ({m.dataset} row {m.row_key})"
                                             for m in matches]
            return ToolResult(name=RESOLVE, server="toolhelper", tool="resolve_entity", arguments=call.arguments,
                              text="\n".join(lines), is_error=False, source="local", latency_ms=0)

        return [ExtraTool(tool=ToolDef(name=RESOLVE, description=(
            "Resolve a gene alias, drug brand, country, variant or clinical term to its canonical name and ID "
            "(HGNC, FDA NDC, RxNorm, ISO3, ClinVar, OMOP) from local reference datasets."),
            input_schema={"type": "object", "properties": {
                "text": {"type": "string"},
                "type": {"type": "string", "description": "Optional: gene, drug_product, drug_ingredient, country, "
                                                          "variant, omop_condition, omop_measurement, ..."}},
                "required": ["text"]}), handler=resolve)]


def uptake(trace) -> dict[str, Any]:
    """Share of E3 hints whose canonical value or identifier later appears in tool arguments."""
    hints = [i for i in trace.context_items if i.strategy == Grounding.name]
    args = " ".join(canonical_json(c.final_arguments) for c in trace.tool_calls).casefold()
    taken = []
    for h in hints:
        values = {str(e.detail.get(k, "")).casefold() for e in h.evidence for k in ("canonical", "identifier")}
        if any(v and v in args for v in values):
            taken.append(h.text)
    return {"hints": len(hints), "taken": len(taken), "rate": len(taken) / len(hints) if hints else None,
            "taken_hints": taken}
