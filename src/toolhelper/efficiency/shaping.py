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
items; `summary` keeps one line per kept item. A field is "used" when its value (4+
characters, or a 3+ digit number) reappears in a later argument or the answer of the same
run; it is dropped only after `min_runs` runs saw it and none used it. Those runs are the
evidence; a top-N cut is evidenced by the runs whose used items all ranked within N.
"""

from __future__ import annotations

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

Mode = Literal["projection", "top_n", "summary"]
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

    def runs_within(self, n: int) -> list[str]:
        """Runs that used this tool's results and only items ranked <= n (evidence for a top-n cut)."""
        return sorted(r for r, top in self.used_rank.items() if 0 < top <= n)


def learn_policies(traces: Iterable[Any], *, min_runs: int = 3) -> dict[str, ToolPolicy]:
    seen: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    used: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    max_rank: dict[str, dict[str, int]] = defaultdict(dict)
    for t in traces:
        calls = t.tool_calls
        for i, c in enumerate(calls):
            if c.is_error or not c.raw_text or c.decision == "cached":
                continue
            parsed = parse(c.name, c.raw_text)
            if parsed is None:
                continue
            later = " ".join([canonical_json(x.final_arguments) for x in calls[i + 1:]] + [t.answer or ""]).casefold()
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
    for tool, fields in seen.items():
        runs = set().union(*fields.values())
        policy = ToolPolicy(tool=tool, seen_runs=len(runs))
        for field, field_runs in fields.items():
            if len(field_runs) >= min_runs and not used[tool][field]:
                policy.drop_fields[field] = sorted(field_runs)
        policy.used_rank = dict(max_rank[tool])
        out[tool] = policy
    return out


# --- the strategy -------------------------------------------------------------------------

TABLE_DOMAIN = {"condition_occurrence": "Condition", "drug_exposure": "Drug", "measurement": "Measurement",
                "procedure_occurrence": "Procedure", "observation": "Observation", "device_exposure": "Device",
                "visit_occurrence": "Visit", "specimen": "Specimen"}


class ResultShaper(BaseStrategy):
    name = "E6_shaping"

    def __init__(self, mode: Mode = "projection", policies: dict[str, ToolPolicy] | None = None, *, top_n: int = 5,
                 cdm: CdmFields | None = None, min_reduction: float = 0.1) -> None:
        self.mode = mode
        self.policies = policies or {}
        self.top_n = top_n
        self.cdm = cdm
        self.min_reduction = min_reduction
        self.parked: dict[str, ToolResult] = {}

    def config(self) -> dict[str, Any]:
        return {"mode": self.mode, "top_n": self.top_n, "tools": sorted(self.policies),
                "cdm": self.cdm.ref if self.cdm else None}

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
        if parsed is None:
            return result
        evidence: list[EvidenceRef] = []
        notes: list[str] = []
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
        text = render(parsed, items, self.mode if evidence else "projection")
        if self.mode == "summary" and evidence:
            notes.append("summary")
        if not evidence or len(text) > len(result.text) * (1 - self.min_reduction):
            return result
        handle = f"r{len(self.parked) + 1}"
        self.parked[handle] = result
        text += f"\n\n[shaped: {'; '.join(notes)}. Full result: {EXPAND}(handle=\"{handle}\")]"
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
