"""User-defined strategies (E10, E11, ...), declared in `configs/strategies.yaml` and usable in
specs like the built-in ones (`--strategy E10`, a factor's specs in an experiment config).

A definition picks one building block and fills it in; no code is written:

- `instruction`: a text added to the task's context before the first turn. Evidence: the
  learning runs that motivate it (those that ended at the turn limit, those with a failed tool
  call, or all of them), as E9 cites the runs that ended at `max_turns`.
- `tool_filter`: hides whole servers or single tools from the model. Evidence: the schema hash
  of every hidden tool.
- `closing_prompt`: E9 with its own wording (asked once, with no tools, when the turn limit
  runs out). Evidence: the learning runs that ended at the turn limit.

Custom strategies apply after the built-in ones, in the order of their numbers. The registry
loads the file at import (`registry.install_custom`), so every `toolhelper` process sees the
same definitions; `experiment freeze` stores the definitions a config uses with its inputs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from toolhelper.agent.hooks import BaseStrategy, ContextItem, Task, ToolDef
from toolhelper.config import CONFIG_DIR
from toolhelper.context.model import EvidenceRef
from toolhelper.efficiency.final_answer import FinalAnswerAtLimit

FILE = CONFIG_DIR / "strategies.yaml"
ID = re.compile(r"E[1-9]\d+")  # E10 and up; E1-E9 are built in
MAX_EVIDENCE_RUNS = 20

KINDS = {"instruction": "Adds a text to the task's context before the first turn.",
         "tool_filter": "Hides whole servers or single tools from the model.",
         "closing_prompt": "Your own wording for the last question at the turn limit (E9's place)."}
EVIDENCE_RUNS = {"max_turns": "runs that ended at the turn limit",
                 "tool_errors": "runs with a failed tool call",
                 "all": "all learning runs"}


class Definition(BaseModel):
    name: str
    kind: Literal["instruction", "tool_filter", "closing_prompt"]
    what: str = ""  # shown in the design form
    text: str = ""  # instruction, closing_prompt
    evidence_runs: Literal["max_turns", "tool_errors", "all"] = "all"  # instruction
    hide_servers: list[str] = Field(default_factory=list)  # tool_filter
    hide_tools: list[str] = Field(default_factory=list)  # tool_filter, namespaced names

    def problems(self) -> list[str]:
        out = [] if self.name.strip() else ["give the strategy a name"]
        if self.kind == "tool_filter":
            if not self.hide_servers and not self.hide_tools:
                out.append("pick at least one server or tool to hide")
        elif not self.text.strip():
            out.append("the text is empty")
        return out

    def summary(self) -> str:
        if self.kind == "tool_filter":
            return "hides " + ", ".join([*(f"server {s}" for s in self.hide_servers), *self.hide_tools])
        return f"{self.kind.replace('_', ' ')}: {self.text.strip()}"


def load(path: Path | None = None) -> dict[str, Definition]:
    path = path or FILE
    if not path.exists():
        return {}
    raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("strategies") or {}
    bad = [sid for sid in raw if not ID.fullmatch(str(sid))]
    if bad:
        raise ValueError(f"{path.name}: custom strategy ids must be E10 or higher, got {bad}")
    return {sid: Definition.model_validate(d) for sid, d in sorted(raw.items(), key=lambda kv: int(kv[0][1:]))}


def save(defs: dict[str, Definition], path: Path | None = None) -> None:
    body = {"strategies": {sid: d.model_dump(exclude_defaults=True) | {"name": d.name, "kind": d.kind}
                           for sid, d in sorted(defs.items(), key=lambda kv: int(kv[0][1:]))}}
    head = "# User-defined strategies (see src/toolhelper/efficiency/custom.py). Edited by the experiment app.\n"
    (path or FILE).write_text(head + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")


def next_id(defs: dict[str, Definition]) -> str:
    return f"E{max([9, *(int(sid[1:]) for sid in defs)]) + 1}"


def spec_ids(specs: list[str]) -> set[str]:
    return {s.split(":", 1)[0].strip().upper() for s in specs}


def config_specs(data: dict[str, Any]) -> list[str]:
    """Every spec an experiment config names (base, factors, arms)."""
    return [*(data.get("base") or {}).get("specs", []),
            *(s for f in (data.get("factors") or {}).values() for s in f.get("specs", [])),
            *(s for arm in (data.get("arms") or {}).values() for s in arm)]


def used_in(data: dict[str, Any], path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The definitions a config uses, for the frozen inputs."""
    ids = spec_ids(config_specs(data))
    return {sid: d.model_dump() for sid, d in load(path).items() if sid in ids}


def configs_using(sid: str, config_dir: Path | None = None) -> list[str]:
    """Experiment configs that name this strategy; changing it would change what they ran."""
    out = []
    for f in sorted((config_dir or CONFIG_DIR).glob("experiments*.yaml")):
        if sid in spec_ids(config_specs(yaml.safe_load(f.read_text(encoding="utf-8")) or {})):
            out.append(f.name)
    return out


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "custom"


def evidence_runs(traces: list[Any], which: str) -> list[str]:
    if which == "max_turns":
        picked = [t for t in traces if t.stop_reason == "max_turns"]
    elif which == "tool_errors":
        picked = [t for t in traces if any(c.is_error for c in t.tool_calls)]
    else:
        picked = traces
    return sorted(t.run_id for t in picked)[:MAX_EVIDENCE_RUNS]


class Instruction(BaseStrategy):
    def __init__(self, sid: str, d: Definition, runs: list[str]) -> None:
        if not runs:
            raise ValueError(f"{sid} needs evidence: the learning runs hold no {EVIDENCE_RUNS[d.evidence_runs]}")
        self.name, self.sid, self.d, self.runs = f"{sid}_{_slug(d.name)}", sid, d, runs

    def config(self) -> dict[str, Any]:
        return {"definition": self.d.model_dump(), "evidence_runs": len(self.runs)}

    def prepare(self, task: Task, ctx) -> list[ContextItem]:
        return [ContextItem(kind="instruction", text=self.d.text.strip(), evidence=[
            EvidenceRef(kind="run", ref=r, detail={"strategy": self.sid, "motivated_by": self.d.evidence_runs})
            for r in self.runs])]


class ToolFilter(BaseStrategy):
    def __init__(self, sid: str, d: Definition) -> None:
        self.name, self.sid, self.d = f"{sid}_{_slug(d.name)}", sid, d

    def config(self) -> dict[str, Any]:
        return {"definition": self.d.model_dump()}

    def select_tools(self, task: Task, tools: list[ToolDef]) -> list[ToolDef]:
        hidden = [t for t in tools if t.server and (t.server in self.d.hide_servers or t.name in self.d.hide_tools)]
        if not hidden:
            return tools
        kept = [t for t in tools if t not in hidden]
        if not any(t.server for t in kept):
            raise ValueError(f"{self.sid} hides every server tool; nothing is left to call")
        self.run.transform("tools", before=len(tools), after=len(kept), note=f"{len(hidden)} tools hidden by definition",
                           evidence=[EvidenceRef(kind="schema_hash", ref=t.schema_hash or t.name,
                                                 detail={"tool": t.name, "hidden_by": self.sid}) for t in hidden])
        return kept


class ClosingPrompt(FinalAnswerAtLimit):
    def __init__(self, sid: str, d: Definition, runs: list[str]) -> None:
        super().__init__(runs, d.text.strip())
        self.name = f"{sid}_{_slug(d.name)}"


def build(sid: str, d: Definition, inputs: Any) -> BaseStrategy:
    if problems := d.problems():
        raise ValueError(f"{sid}: {'; '.join(problems)}")
    if d.kind == "tool_filter":
        return ToolFilter(sid, d)
    if d.kind == "closing_prompt":
        return ClosingPrompt(sid, d, evidence_runs(inputs.traces, "max_turns"))
    return Instruction(sid, d, evidence_runs(inputs.traces, d.evidence_runs))


def needs_runs(d: Definition) -> bool:
    return d.kind != "tool_filter"
