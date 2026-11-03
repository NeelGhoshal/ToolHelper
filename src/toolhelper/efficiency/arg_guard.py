"""E4: argument guard. Checks a call before it reaches the server; returns a one-line fix
(`reject` mode) or applies a deterministic repair (`repair` mode, rejecting what it can't repair).

Every rule comes from a real source, cited as evidence on the decision:

- schema: the tool's live input schema (enums, bounds, required, unknown keys). Repair clamps
  numbers into range. Evidence: the schema hash.
- learned: real error responses in traces and recordings. "unknown <entity> search field: X"
  rejects X (repaired by renaming it when a later successful call moved the same value to
  another field, e.g. article `query` -> `keyword`); "invalid <entity> section: S" drops S;
  BioMCP's read-only refusal of shell syntax (`| head`) cuts the command at the operator.
  Evidence: the failing call and, for a rename, the successful one.
- cdm: OMOP table/field pairs from omop_mcp's own field map. Repair maps a table prefix to
  its unique table (`condition` -> `condition_occurrence`) and a wrong field to the table's
  primary concept field. Evidence: the map file's sha256 and table.
- iso3: WHO statistics want an ISO3 country code; repair resolves a name through the
  countries dataset (`Japan` -> `JPN`). Evidence: the dataset row.
- ndc: product NDCs must look like 12345-678 and exist in the FDA NDC Directory (Medical MCP
  returns an unrelated product for an unknown one). A name in place of an NDC is resolved
  to the directory's products for the hint. Evidence: the dataset row, or the dataset version
  the NDC is absent from.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from jsonschema import validators
from pydantic import BaseModel, Field

from toolhelper.agent.hooks import BaseStrategy, CallDecision, PendingCall, Task, ToolDef
from toolhelper.context.model import EvidenceRef
from toolhelper.datasets.index import EntityIndex
from toolhelper.mcp.hub import content_text
from toolhelper.mcp.recorder import Recording

Mode = Literal["reject", "repair"]

UNKNOWN_FIELD = re.compile(r"^unknown ([\w-]+) search field: (\w+)")
INVALID_SECTION = re.compile(r"^invalid ([\w-]+) section: ([\w-]+)")
READ_ONLY = "BioMCP allows read-only commands only"
SHELL = re.compile(r"\s*(\|\|?|&&|;|[<>]|2>|\$\(|`)")
NDC_FORMAT = re.compile(r"^\d{4,5}-\d{3,4}$")
DEFAULT_CDM_FIELDS = Path(r"C:\Users\neelg\src\omop_mcp\src\omop_mcp\data\omop_concept_id_fields.json")


class Observation(BaseModel):
    """One real call and its response, from a trace or a recording."""

    tool: str  # namespaced
    arguments: dict[str, Any]
    is_error: bool
    text: str
    ref: str  # evidence ref: "<run_id>:<call_id>" or "recording:<server>/<tool>/<key>"
    sequence: str = ""  # calls of one run share it (rename learning follows the run's order)
    order: int = 0


def observations_from_traces(traces: Iterable[Any]) -> list[Observation]:
    return [Observation(tool=c.name, arguments=c.final_arguments, is_error=c.is_error, text=c.raw_text or "",
                        ref=f"{t.run_id}:{c.call_id}", sequence=t.run_id, order=i)
            for t in traces for i, c in enumerate(t.tool_calls) if c.raw_text is not None]


def observations_from_recordings(recordings: Iterable[Recording]) -> list[Observation]:
    out = []
    for r in recordings:
        if r.result is None:
            continue
        text = "\n".join(content_text(b) for b in r.result.get("content", []))
        out.append(Observation(tool=f"{r.server}__{r.tool}", arguments=r.arguments, is_error=bool(r.result.get("isError")),
                               text=text, ref=f"recording:{r.server}/{r.tool}/{r.args_key[:16]}"))
    return out


class Rule(BaseModel):
    kind: Literal["unknown_field", "invalid_section", "shell"]
    tool: str
    entity: str | None = None
    value: str | None = None  # the field or section
    evidence: list[str]
    rename_to: str | None = None
    rename_evidence: list[str] = Field(default_factory=list)


def _as_list(v: Any) -> list[str]:
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]


def learn_rules(observations: list[Observation]) -> list[Rule]:
    rules: dict[tuple, Rule] = {}
    for o in observations:
        if not o.is_error:
            continue
        text = o.text.strip()
        if m := UNKNOWN_FIELD.match(text):
            key = ("unknown_field", o.tool, m[1], m[2])
        elif m := INVALID_SECTION.match(text):
            key = ("invalid_section", o.tool, m[1], m[2])
        elif READ_ONLY in text and SHELL.search(str(o.arguments.get("command", ""))):
            key = ("shell", o.tool, None, None)
        else:
            continue
        rule = rules.setdefault(key, Rule(kind=key[0], tool=key[1], entity=key[2], value=key[3], evidence=[]))
        rule.evidence.append(o.ref)

    # A rename is learned when a successful call to the same tool and entity carries the
    # failed field's value under another field it did not have (later in the same run, or
    # anywhere among recordings).
    for rule in rules.values():
        if rule.kind != "unknown_field":
            continue
        failed = [o for o in observations if o.ref in rule.evidence]
        for f in failed:
            value = set(_as_list(f.arguments.get(rule.value)))
            for o in observations:
                if (o.is_error or o.tool != rule.tool or o.arguments.get("entity") != rule.entity
                        or (f.sequence and (o.sequence != f.sequence or o.order <= f.order))):
                    continue
                for field, v in o.arguments.items():
                    if field not in f.arguments and field != rule.value and value & set(_as_list(v)):
                        rule.rename_to, rule.rename_evidence = field, [f.ref, o.ref]
                        break
                if rule.rename_to:
                    break
            if rule.rename_to:
                break
    return list(rules.values())


class CdmFields(BaseModel):
    """omop_mcp's table -> concept-id fields map, pinned by sha256."""

    tables: dict[str, list[str]]
    ref: str

    @classmethod
    def load(cls, path: Path = DEFAULT_CDM_FIELDS) -> CdmFields:
        raw = path.read_bytes()
        return cls(tables=json.loads(raw), ref=f"omop_mcp:data/{path.name}@sha256:{hashlib.sha256(raw).hexdigest()[:16]}")


class Issue(BaseModel):
    rule: str
    hint: str
    evidence: list[EvidenceRef]
    repair: dict[str, Any] | None = None  # the full repaired arguments, when repairable


class ArgumentGuard(BaseStrategy):
    name = "E4_arg_guard"

    def __init__(self, mode: Mode = "reject", rules: list[Rule] | None = None, *, cdm: CdmFields | None = None,
                 index: EntityIndex | None = None) -> None:
        self.mode = mode
        self.rules = list(rules or [])
        self.cdm = cdm
        self.index = index
        self.schemas: dict[str, ToolDef] = {}

    def config(self) -> dict[str, Any]:
        return {"mode": self.mode, "learned_rules": len(self.rules), "cdm": self.cdm.ref if self.cdm else None,
                "datasets": [d["wref"] for d in self.index.datasets()] if self.index else []}

    def select_tools(self, task: Task, tools: list[ToolDef]) -> list[ToolDef]:
        # Original schemas, before any compression strategy rewrites them.
        self.schemas = {t.name: t for t in tools if t.server}
        return tools

    # --- checks -------------------------------------------------------------------------

    def _schema(self, tool: ToolDef, args: dict[str, Any]) -> list[Issue]:
        schema = tool.input_schema
        out = []
        for err in validators.validator_for(schema)(schema).iter_errors(args):
            path = ".".join(str(p) for p in err.absolute_path) or "(arguments)"
            ev = [EvidenceRef(kind="schema_hash", ref=tool.schema_hash or "", detail={
                "tool": tool.name, "validator": err.validator, "path": path})]
            repair = None
            if err.validator in ("maximum", "minimum") and len(err.absolute_path) == 1:
                repair = {**args, err.absolute_path[0]: err.validator_value}
            if err.validator == "enum":
                hint = f"`{path}` must be one of {', '.join(map(str, err.validator_value))}"
            elif err.validator in ("maximum", "minimum"):
                hint = f"`{path}` must be {'<=' if err.validator == 'maximum' else '>='} {err.validator_value}"
            elif err.validator == "additionalProperties":
                hint = f"unknown argument(s): {err.message.split('(')[-1].rstrip(')')}"
            else:
                hint = f"`{path}`: {err.message}"
            out.append(Issue(rule=f"schema:{err.validator}", hint=hint, evidence=ev, repair=repair))
        return out

    def _learned(self, name: str, args: dict[str, Any]) -> list[Issue]:
        out = []
        for r in self.rules:
            if r.tool != name or (r.entity and args.get("entity") != r.entity):
                continue
            ev = [EvidenceRef(kind="run", ref=e) for e in r.evidence]
            if r.kind == "unknown_field" and r.value in args:
                repair = hint_fix = None
                if r.rename_to and r.rename_to not in args:
                    value = args[r.value]
                    target = self.schemas.get(name)
                    wants_list = target and "array" in json.dumps(target.input_schema["properties"].get(r.rename_to, {}))
                    repair = {k: v for k, v in args.items() if k != r.value}
                    repair[r.rename_to] = [value] if wants_list and not isinstance(value, list) else value
                    ev += [EvidenceRef(kind="run", ref=e, detail={"rename": f"{r.value}->{r.rename_to}"})
                           for e in r.rename_evidence]
                    hint_fix = f"; use `{r.rename_to}` instead"
                out.append(Issue(rule="learned:unknown_field", evidence=ev, repair=repair,
                                 hint=f"`{r.value}` is not a {r.entity} search field{hint_fix or ''}"))
            elif r.kind == "invalid_section" and r.value in (args.get("sections") or []):
                kept = [s for s in args["sections"] if s != r.value]
                repair = {**args, "sections": kept} if kept else {k: v for k, v in args.items() if k != "sections"}
                out.append(Issue(rule="learned:invalid_section", evidence=ev, repair=repair,
                                 hint=f"section `{r.value}` is not valid for {r.entity}; drop it"))
            elif r.kind == "shell" and (m := SHELL.search(str(args.get("command", "")))):
                cut = str(args["command"])[:m.start()].strip()
                out.append(Issue(rule="learned:shell", evidence=ev, repair={**args, "command": cut} if cut else None,
                                 hint=f"no shell syntax in commands; send `{cut}` only"))
        return out

    def _cdm(self, args: dict[str, Any]) -> list[Issue]:
        if not self.cdm:
            return []
        table, field = args.get("omop_table"), args.get("omop_field")
        tables = self.cdm.tables
        if table not in tables:
            matches = [t for t in tables if t.startswith(f"{table}_")]
            ev = [EvidenceRef(kind="dataset_row", ref=self.cdm.ref, detail={"table": matches[0] if len(matches) == 1 else None})]
            repair = None
            if len(matches) == 1:
                table = matches[0]
                fields = tables[table]
                repair = {**args, "omop_table": table, "omop_field": field if field in fields else fields[0]}
            return [Issue(rule="cdm:table", evidence=ev, repair=repair,
                          hint=f"`{args.get('omop_table')}` is not a CDM table"
                               + (f"; use `{table}`" if repair else f"; tables: {', '.join(sorted(tables))}"))]
        if field not in tables[table]:
            ev = [EvidenceRef(kind="dataset_row", ref=self.cdm.ref, detail={"table": table})]
            return [Issue(rule="cdm:field", evidence=ev, repair={**args, "omop_field": tables[table][0]},
                          hint=f"`{field}` is not a concept field of {table}; use one of {', '.join(tables[table])}")]
        return []

    def _country(self, args: dict[str, Any]) -> list[Issue]:
        country = args.get("country")
        if not country or not self.index or re.fullmatch(r"[A-Z]{3}", str(country)):
            return []
        res = self.index.resolve(str(country), type="country")
        best = res.best if res.status in ("exact", "normalized") else None
        ev = ([EvidenceRef(kind="dataset_row", ref=f"{best.dataset}#{best.row_key}",
                           detail={"surface": best.surface, "iso3": best.identifier})] if best else [])
        if not best:
            return [Issue(rule="iso3", evidence=[EvidenceRef(kind="dataset_row", ref=d["wref"])
                                                 for d in self.index.datasets() if d["name"] == "countries"],
                          hint=f"`country` must be an ISO3 code; `{country}` is not a known country")]
        return [Issue(rule="iso3", evidence=ev, repair={**args, "country": best.identifier},
                      hint=f"`country` must be an ISO3 code; use `{best.identifier}` for {best.canonical}")]

    def _ndc(self, args: dict[str, Any]) -> list[Issue]:
        ndc = str(args.get("ndc", ""))
        if not self.index or not ndc:
            return []
        if NDC_FORMAT.match(ndc):
            if self.index.by_identifier(ndc, dataset="ndc"):
                return []
            wref = next((d["wref"] for d in self.index.datasets() if d["name"] == "ndc"), "datasets/ndc")
            return [Issue(rule="ndc:absent", evidence=[EvidenceRef(kind="dataset_row", ref=wref, detail={"absent": ndc})],
                          hint=f"NDC {ndc} is not in the FDA NDC Directory; find the product with search-drugs")]
        res = self.index.resolve(ndc, type="drug_product")
        if res.best and res.status != "fuzzy":
            b = res.best
            return [Issue(rule="ndc:name", hint=f"`ndc` takes a product code like {b.identifier} ({b.canonical}), not a name",
                          evidence=[EvidenceRef(kind="dataset_row", ref=f"{b.dataset}#{b.row_key}",
                                                detail={"surface": b.surface, "ndc": b.identifier})])]
        return [Issue(rule="ndc:format", hint="`ndc` must look like 12345-678 (labeler-product)",
                      evidence=[EvidenceRef(kind="dataset_row", ref=d["wref"]) for d in self.index.datasets()
                                if d["name"] == "ndc"])]

    def check(self, name: str, args: dict[str, Any]) -> list[Issue]:
        tool = self.schemas.get(name)
        issues = self._schema(tool, args) if tool else []
        issues += self._learned(name, args)
        if name == "omop__find_omop_concept":
            issues += self._cdm(args)
        elif name == "medical__get-health-statistics":
            issues += self._country(args)
        elif name == "medical__get-drug-details":
            issues += self._ndc(args)
        return issues

    def before_call(self, call: PendingCall) -> CallDecision:
        if call.name not in self.schemas:
            return CallDecision()
        args, applied, issues = call.arguments, [], self.check(call.name, call.arguments)
        if not issues:
            return CallDecision()
        if self.mode == "repair":
            # Apply repairs one at a time and re-check, so each repair sees the previous ones.
            for _ in range(8):
                fixable = next((i for i in issues if i.repair is not None), None)
                if fixable is None:
                    break
                applied.append(fixable)
                args = fixable.repair
                issues = self.check(call.name, args)
            if not issues:
                return CallDecision(action="repaired", arguments=args,
                                    evidence=[e for i in applied for e in i.evidence],
                                    note="; ".join(i.rule for i in applied))
        evidence = [e for i in applied + issues for e in i.evidence]
        hint = "Call not sent: " + "; ".join(i.hint for i in issues) + "."
        return CallDecision(action="rejected", hint=hint, evidence=evidence,
                            note="; ".join(i.rule for i in applied + issues))
