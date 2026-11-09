"""E6: result shaping. Keeps what traces show gets used and parks the rest behind
`expand_result(handle)`, so every drop is recorded and reversible.

Results are split into parts by format:

- OMOP `find_omop_concept` (JSON): candidates are items, their keys are fields. Candidates
  that are not Standard + Valid + in the requested table's domain are dropped first
  (evidence: omop_mcp's CDM field map; the domain is the table's, e.g. condition_occurrence
  -> Condition).
- Numbered lists (Medical MCP): `1. ...` blocks are items, `Label: value` lines are fields.
- Markdown (BioMCP): `## ` sections are fields of a single item.

Variants: `projection` drops fields no run ever used; `top_n` also keeps only the first N
items; `summary` keeps one line per kept item; `budget` cuts results longer than `budget`
tokens to their leading whole units (list items, `## ` sections, else lines; JSON keeps the
first items of its longest lists) after the projection. A field is "used" when its value (4+
characters, or a 3+ digit number) reappears in a later argument or the answer of the same
run; it is dropped only after `min_runs` runs saw it and none used it. Those runs are the
evidence; a top-N cut is evidenced by the runs whose used items all ranked within N, and a
budget cut by the runs whose own results of that tool, cut the same way, kept every
identifier (Phase 5's heuristic: a token with a digit) that the run later used.

Budget cuts target carried tokens: in Phase 10's baseline, results were 51% of all tokens,
mostly a few oversized ones re-sent on every later turn (an RxNorm list of 53,785 tokens,
BioMCP `get ... sections: all` at ~30,000).
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from toolhelper.agent.hooks import BaseStrategy, ExtraTool, PendingCall, Task, ToolDef
from toolhelper.context.model import EvidenceRef
from toolhelper.efficiency.arg_guard import CdmFields
from toolhelper.mcp.canonical import canonical_json
from toolhelper.mcp.hub import ToolResult
from toolhelper.monitoring.pathways import identifiers
from toolhelper.monitoring.tokens import TokenCounter

Mode = Literal["projection", "top_n", "summary", "budget"]
EXPAND = "toolhelper__expand_result"
LIST_ITEM = re.compile(r"^(\d+)\.\s", re.M)
LABEL = re.compile(r"^\s*([A-Z][\w /()-]{1,40}):\s+(.*\S)\s*$")
SECTION = re.compile(r"^## +(.+?)\s*$", re.M)


# --- parsing results into items and fields ------------------------------------------------


class Part(BaseModel):
    field: str
    text: str


class Item(BaseModel):
    rank: int
    head: str  # the item's first line (kept by every mode)
    parts: list[Part]
    raw: dict[str, Any] | None = None  # JSON item


class Parsed(BaseModel):
    format: Literal["omop", "list", "sections"]
    preamble: str
    items: list[Item]


def parse(tool: str, text: str) -> Parsed | None:
    if tool == "omop__find_omop_concept":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        cands = data.get("candidates") if isinstance(data, dict) else None
        if not isinstance(cands, list):
            return None
        items = [Item(rank=i + 1, head=f"{c.get('concept_id')} {c.get('name')}", raw=c,
                      parts=[Part(field=k, text=str(v)) for k, v in c.items()]) for i, c in enumerate(cands)]
        return Parsed(format="omop", preamble=json.dumps({k: v for k, v in data.items() if k != "candidates"}),
                      items=items)
    starts = [m.start() for m in LIST_ITEM.finditer(text)]
    if starts:
        items = []
        for i, start in enumerate(starts):
            block = text[start:starts[i + 1] if i + 1 < len(starts) else len(text)].rstrip("\n")
            head, *rest = block.split("\n")
            parts = [Part(field=m[1], text=line) if (m := LABEL.match(line)) else Part(field="(text)", text=line)
                     for line in rest if line.strip()]
            items.append(Item(rank=i + 1, head=head, parts=parts))
        return Parsed(format="list", preamble=text[:starts[0]].rstrip("\n"), items=items)
    sections = list(SECTION.finditer(text))
    if sections:
        parts = [Part(field=m[1], text=text[m.start():sections[i + 1].start() if i + 1 < len(sections) else len(text)]
                      .rstrip("\n")) for i, m in enumerate(sections)]
        return Parsed(format="sections", preamble=text[:sections[0].start()].rstrip("\n"),
                      items=[Item(rank=1, head="", parts=parts)])
    return None


def render(p: Parsed, items: list[Item], mode: Mode) -> str:
    if p.format == "omop":
        if mode == "summary":
            return "\n".join(f"{i.raw.get('concept_id')} {i.raw.get('name')} ({i.raw.get('vocab')} {i.raw.get('code')}, "
                             f"{i.raw.get('domain')})" for i in items)
        data = json.loads(p.preamble or "{}")
        data["candidates"] = [{part.field: i.raw[part.field] for part in i.parts} for i in items]
        return json.dumps(data, indent=2, ensure_ascii=False)
    if mode == "summary":
        return "\n".join(filter(None, [p.preamble] + [i.head.strip() for i in items]))
    blocks = [p.preamble] + ["\n".join(filter(None, [i.head] + [part.text for part in i.parts])) for i in items]
    return "\n\n".join(b for b in blocks if b)


# --- cutting to a token budget ------------------------------------------------------------

MORE = re.compile(r"^\[\.\.\. (\d+) more items\]$")
CHARS_PER_TOKEN_MAX = 12  # no tokenizer packs more characters than this into a token on these texts


class Cut(BaseModel):
    text: str
    note: str


def _fits(text: str, budget: int, counter: TokenCounter) -> bool:
    return len(text) <= budget * CHARS_PER_TOKEN_MAX and counter(text) <= budget


def _best_prefix(text: str, offsets: list[int], budget: int, counter: TokenCounter) -> int | None:
    """The largest offset (ascending list) whose prefix fits; prefixes only grow, so binary search."""
    best, lo, hi = None, 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fits(text[:offsets[mid]].rstrip(), budget, counter):
            best, lo = offsets[mid], mid + 1
        else:
            hi = mid - 1
    return best


def _cut_lines(text: str, budget: int, counter: TokenCounter) -> tuple[str, int, int]:
    """Leading whole lines that fit (part of the first if it alone is too long):
    (text, lines kept, lines)."""
    total = text.count("\n") + 1
    end = _best_prefix(text, [m.end() for m in re.finditer("\n", text)] + [len(text)], budget, counter)
    if end is None:
        chars = _best_prefix(text, list(range(1, min(len(text), budget * CHARS_PER_TOKEN_MAX) + 1)), budget, counter)
        return text[:chars or 1], 1, total
    kept = text[:end].rstrip()
    return kept, kept.count("\n") + 1, total


def _json_lists(value: Any) -> Iterable[list]:
    if isinstance(value, list):
        yield value
        for x in value:
            yield from _json_lists(x)
    elif isinstance(value, dict):
        for x in value.values():
            yield from _json_lists(x)


def _json_strings(value: Any) -> Iterable[tuple[Any, Any]]:
    """(container, key) for every string value."""
    pairs = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, x in pairs:
        if isinstance(x, str):
            yield value, key
        else:
            yield from _json_strings(x)


def _real(items: list) -> list:
    return [x for x in items if not (isinstance(x, str) and MORE.match(x))]


def _cut_json(data: Any, budget: int, counter: TokenCounter) -> tuple[str, str, bool]:
    """Halves the longest list (by serialized size) until the JSON fits, then the longest strings;
    a cut list ends with a "[... n more items]" marker, a cut string with "[... n more characters]".
    Returns the rendered JSON, a note, and whether it fits (many short keys may not)."""
    data, items, strings = copy.deepcopy(data), 0, 0
    while not _fits(text := json.dumps(data, indent=2, ensure_ascii=False), budget, counter):
        lists = [lst for lst in _json_lists(data) if len(_real(lst)) > 1]
        if lists:
            longest = max(lists, key=lambda lst: len(json.dumps(lst, ensure_ascii=False)))
            real = _real(longest)
            earlier = sum(int(m[1]) for x in longest if isinstance(x, str) and (m := MORE.match(x)))
            keep = max(1, len(real) // 2)
            longest[:] = real[:keep] + [f"[... {len(real) - keep + earlier} more items]"]
            items += len(real) - keep
            continue
        long = [(c, k) for c, k in _json_strings(data) if len(c[k]) > 120 and not MORE.match(c[k])]
        if not long:
            break
        container, key = max(long, key=lambda ck: len(ck[0][ck[1]]))
        value = container[key]
        container[key] = f"{value[:len(value) // 2]} [... {len(value) - len(value) // 2} more characters]"
        strings += 1
    note = f"kept the first items of long lists ({items} items dropped)"
    return text, note + (f", shortened {strings} long strings" if strings else ""), _fits(text, budget, counter)


def cut_to_budget(tool: str, text: str, budget: int, counter: TokenCounter) -> Cut | None:
    """`text` cut to its leading whole units within `budget` tokens, with a note saying what
    was kept; None when it already fits."""
    if _fits(text, budget, counter):
        return None
    if text.lstrip().startswith(("{", "[")):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            pruned, note, fits = _cut_json(data, budget, counter)
            if fits:
                return Cut(text=pruned, note=note)
            kept, n, total = _cut_lines(pruned, budget, counter)
            return Cut(text=kept, note=f"{note}, then the first {n} of {total} lines")
    items = [m.start() for m in LIST_ITEM.finditer(text)]
    if len(items) > 1:
        end = _best_prefix(text, items[1:], budget, counter)
        if end is not None:
            return Cut(text=text[:end].rstrip(), note=f"kept items 1-{items.index(end)} of {len(items)}")
    sections = list(SECTION.finditer(text))
    if sections:
        end = _best_prefix(text, [m.start() for m in sections[1:]], budget, counter)
        if end is not None:
            kept = [m[1] for m in sections if m.start() < end]
            dropped = [m[1] for m in sections if m.start() >= end]
            return Cut(text=text[:end].rstrip(), note=f"kept sections {kept}; dropped {dropped}")
    kept, n, total = _cut_lines(text, budget, counter)
    where = f" of section {sections[0][1]}" if len(sections) == 1 else ""
    return Cut(text=kept, note=f"kept the first {n} of {total} lines{where}")


# --- learning what gets used --------------------------------------------------------------


def _tokens(value: str) -> list[str]:
    return [t for t in re.findall(r"[\w.-]+", value.casefold())
            if (len(t) >= 4 and not t.isdigit()) or (t.isdigit() and len(t) >= 3)]


def _used(value: str, later: str) -> bool:
    toks = _tokens(value)
    return bool(toks) and any(t in later for t in toks)


class ToolPolicy(BaseModel):
    tool: str
    drop_fields: dict[str, list[str]] = Field(default_factory=dict)  # field -> evidence runs
    seen_runs: int = 0
    used_rank: dict[str, int] = Field(default_factory=dict)  # run -> highest rank of an item it used
    over_budget_runs: dict[int, list[str]] = Field(default_factory=dict)  # budget -> runs with a result over it
    budget_runs: dict[int, list[str]] = Field(default_factory=dict)  # budget -> those whose cut kept what they used

    def runs_within(self, n: int) -> list[str]:
        """Runs that used this tool's results and only items ranked <= n (evidence for a top-n cut)."""
        return sorted(r for r, top in self.used_rank.items() if 0 < top <= n)


def learn_policies(traces: Iterable[Any], *, min_runs: int = 3, budgets: Iterable[int] = (),
                   counter: TokenCounter | None = None) -> dict[str, ToolPolicy]:
    """Usage policies per tool; with `budgets`, also which runs support a cut at each budget
    (`counter` measures tokens, cl100k_base by default)."""
    budgets = list(budgets)
    if budgets and counter is None:
        from toolhelper.monitoring.tokens import tiktoken_counter

        counter = tiktoken_counter()
    seen: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    used: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    max_rank: dict[str, dict[str, int]] = defaultdict(dict)
    over: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    lost: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for t in traces:
        calls = t.tool_calls
        for i, c in enumerate(calls):
            if c.is_error or not c.raw_text or c.decision == "cached":
                continue
            later_raw = " ".join([canonical_json(x.final_arguments) for x in calls[i + 1:]] + [t.answer or ""])
            for b in budgets:
                if (cut := cut_to_budget(c.name, c.raw_text, b, counter)) is None:
                    continue
                over[c.name][b].add(t.run_id)
                if (identifiers(c.raw_text) & identifiers(later_raw)) - identifiers(cut.text):
                    lost[c.name][b].add(t.run_id)
            parsed = parse(c.name, c.raw_text)
            if parsed is None:
                continue
            later = later_raw.casefold()
            top = 0
            for item in parsed.items:
                for part in item.parts:
                    seen[c.name][part.field].add(t.run_id)
                    value = part.text.split(":", 1)[-1] if parsed.format == "list" else part.text
                    if _used(value, later):
                        used[c.name][part.field].add(t.run_id)
                # Rank goes by the item's identifying head (id and name), not shared values like "SNOMED".
                if _used(item.head, later):
                    top = max(top, item.rank)
            max_rank[c.name][t.run_id] = max(max_rank[c.name].get(t.run_id, 0), top)
    out = {}
    for tool in dict.fromkeys([*seen, *over]):
        fields = seen.get(tool, {})
        policy = ToolPolicy(tool=tool, seen_runs=len(set().union(*fields.values())) if fields else 0)
        for field, field_runs in fields.items():
            if len(field_runs) >= min_runs and not used[tool][field]:
                policy.drop_fields[field] = sorted(field_runs)
        policy.used_rank = dict(max_rank[tool])
        policy.over_budget_runs = {b: sorted(r) for b, r in over[tool].items()}
        policy.budget_runs = {b: sorted(r - lost[tool][b]) for b, r in over[tool].items()}
        out[tool] = policy
    return out


# --- the strategy -------------------------------------------------------------------------

TABLE_DOMAIN = {"condition_occurrence": "Condition", "drug_exposure": "Drug", "measurement": "Measurement",
                "procedure_occurrence": "Procedure", "observation": "Observation", "device_exposure": "Device",
                "visit_occurrence": "Visit", "specimen": "Specimen"}


class ResultShaper(BaseStrategy):
    name = "E6_shaping"

    def __init__(self, mode: Mode = "projection", policies: dict[str, ToolPolicy] | None = None, *, top_n: int = 5,
                 budget: int = 2000, cdm: CdmFields | None = None, min_reduction: float = 0.1) -> None:
        self.mode = mode
        self.policies = policies or {}
        self.top_n = top_n
        self.budget = budget
        self.cdm = cdm
        self.min_reduction = min_reduction
        self.parked: dict[str, ToolResult] = {}

    def config(self) -> dict[str, Any]:
        out = {"mode": self.mode, "top_n": self.top_n, "tools": sorted(self.policies),
               "cdm": self.cdm.ref if self.cdm else None}
        if self.mode == "budget":
            out |= {"budget": self.budget, "budget_tools": sorted(t for t, p in self.policies.items()
                                                                  if p.budget_runs.get(self.budget))}
        return out

    def prepare(self, task: Task, ctx) -> list:
        self.parked = {}
        return []

    def _omop_filter(self, call: PendingCall, items: list[Item]) -> tuple[list[Item], list[EvidenceRef], str]:
        table = call.arguments.get("omop_table")
        if not self.cdm or table not in self.cdm.tables:
            return items, [], ""
        domain = TABLE_DOMAIN.get(table)
        kept = [i for i in items if i.raw.get("concept") == "S" and i.raw.get("validity") == "Valid"
                and (domain is None or i.raw.get("domain") == domain)]
        if len(kept) == len(items):
            return items, [], ""
        return kept, [EvidenceRef(kind="dataset_row", ref=self.cdm.ref, detail={"table": table, "domain": domain})], \
            f"{len(items) - len(kept)} candidate(s) not Standard/Valid/{domain}"

    def after_result(self, call: PendingCall, result: ToolResult) -> ToolResult:
        if result.is_error or result.source == "local":
            return result
        parsed = parse(call.name, result.text)
        policy = self.policies.get(call.name)
        if parsed is None and self.mode != "budget":
            return result
        evidence: list[EvidenceRef] = []
        notes: list[str] = []
        text = result.text
        if parsed is not None:
            items = parsed.items
            if parsed.format == "omop":
                items, ev, note = self._omop_filter(call, items)
                evidence += ev
                notes += [note] if note else []
            if policy and policy.drop_fields:
                drop = {f for f in policy.drop_fields if any(p.field == f for i in items for p in i.parts)}
                if drop:
                    items = [i.model_copy(update={"parts": [p for p in i.parts if p.field not in drop]}) for i in items]
                    evidence += [EvidenceRef(kind="run", ref=r, detail={"unused_field": f})
                                 for f in sorted(drop) for r in policy.drop_fields[f][:5]]
                    notes.append(f"dropped fields {sorted(drop)}")
            if self.mode in ("top_n", "summary") and len(items) > self.top_n and policy:
                runs = policy.runs_within(self.top_n)
                if runs:
                    evidence += [EvidenceRef(kind="run", ref=r, detail={"used_within_top": self.top_n}) for r in runs[:5]]
                    notes.append(f"kept top {self.top_n} of {len(items)}")
                    items = items[:self.top_n]
            if evidence:
                text = render(parsed, items, self.mode)
                if self.mode == "summary":
                    notes.append("summary")
        if self.mode == "budget" and policy and (runs := policy.budget_runs.get(self.budget)):
            cut = cut_to_budget(call.name, text, self.budget, self.run.tokens)
            if cut is not None:
                support = f"{len(runs)}/{len(policy.over_budget_runs[self.budget])}"
                evidence += [EvidenceRef(kind="run", ref=r, detail={"kept_used_identifiers_at_budget": self.budget,
                                                                    "support": support}) for r in runs[:5]]
                notes.append(f"{cut.note} (budget {self.budget} tokens)")
                text = cut.text
        if not evidence:
            return result
        handle = f"r{len(self.parked) + 1}"
        text += f"\n\n[shaped: {'; '.join(notes)}. Full result: {EXPAND}(handle=\"{handle}\")]"
        # The handle note costs tokens too: small results whose cut does not pay for it stay whole.
        if self.run.tokens(text) > self.run.tokens(result.text) * (1 - self.min_reduction):
            return result
        self.parked[handle] = result
        self.run.transform(call.name, before=result.text, after=text, evidence=evidence,
                           note=f"handle {handle}: {'; '.join(notes)}")
        return result.model_copy(update={"text": text, "structured": None})

    def extra_tools(self) -> list[ExtraTool]:
        async def expand(call: PendingCall) -> ToolResult:
            handle = str(call.arguments.get("handle", ""))
            original = self.parked.get(handle)
            text = original.text if original else f"Error: no result with handle {handle!r}."
            return ToolResult(name=EXPAND, server="toolhelper", tool="expand_result", arguments=call.arguments,
                              text=text, is_error=original is None, source="local", latency_ms=0)

        return [ExtraTool(tool=ToolDef(name=EXPAND, description=(
            "Return the full, unshaped text of a tool result that was shortened. Pass the handle shown in it."),
            input_schema={"type": "object", "properties": {"handle": {"type": "string"}}, "required": ["handle"]}),
            handler=expand)]
