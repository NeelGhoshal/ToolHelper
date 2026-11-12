"""E1: tool retrieval. Offers only the tools a task needs, found by lexical search over
ToolCards (Paper Librarian's method: FTS5 BM25 per field, fused by reciprocal rank; no
embeddings, see README section 5).

ToolCards are built from the catalog (description, required arguments), successful runs
(their prompts become example tasks, the runs their evidence) and E4's learned pitfalls,
and written to the context store as `toolcards/<server>/<tool>` with status `proposed`.
Example tasks are indexed per task and the running task's own examples are excluded, so a
benchmark task never retrieves tools through its own prompt (leave-one-out).

Variants: `top_k` (3/5/8); `static` (offer the top k) vs `find_tools` (offer the top k plus
a `find_tools(query)` meta-tool that adds more mid-run); `route` (server-first: keep the
best-scoring server's tools, plus any other server that scores within `route_margin`).
`servers` (`ServerMenu`) uses no search at all: the model sees one `load_tools(servers)` tool
that lists each server's tools by name and loads the servers it picks, so only their
definitions are re-sent on later turns (228 of 378 ABC runs called one server's tools only).
Evidence per offered tool: its schema hash with the retrieval score and rank;
`recall_at_k(trace, task)` compares what was offered with what the task needed.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from toolhelper.agent.hooks import BaseStrategy, ExtraTool, PendingCall, Task, ToolDef
from toolhelper.context.ingest import _upsert, tool_name
from toolhelper.context.model import ContextStore, EvidenceRef
from toolhelper.experiments.scoring import ANSWERED
from toolhelper.mcp.hub import ToolResult

FIND_TOOLS = "toolhelper__find_tools"
LOAD_TOOLS = "toolhelper__load_tools"
RRF_K = 60  # standard reciprocal-rank-fusion constant (as in Paper Librarian)
TOKEN = re.compile(r"[a-z0-9]+")
FIELDS = ("name", "description", "notes", "examples", "pitfalls")


class ToolCard(BaseModel):
    tool: str  # namespaced
    server: str
    description: str
    usage_notes: str
    example_tasks: list[tuple[str, str]] = Field(default_factory=list)  # (task_id, prompt)
    pitfalls: list[str] = Field(default_factory=list)
    evidence_runs: list[str] = Field(default_factory=list)


def build_cards(tools: Iterable[ToolDef], traces: Iterable[Any] = (), rules: Iterable[Any] = (),
                *, max_examples: int = 5) -> list[ToolCard]:
    """Cards for hub tools. `traces` should be scored (only successful runs give examples);
    `rules` are E4 learned rules."""
    cards = {}
    for t in tools:
        if not t.server:
            continue
        required = t.input_schema.get("required", [])
        notes = (t.description or "").split("\n\n", 1)[0].strip()
        notes += f" Required: {', '.join(required)}." if required else ""
        cards[t.name] = ToolCard(tool=t.name, server=t.server, description=t.description or "", usage_notes=notes)
    for tr in traces:
        if tr.stop_reason not in ANSWERED or not tr.success:
            continue
        for name in dict.fromkeys(c.name for c in tr.tool_calls if not c.is_error):
            card = cards.get(name)
            if card and len(card.example_tasks) < max_examples and tr.task_id not in {i for i, _ in card.example_tasks}:
                card.example_tasks.append((tr.task_id, tr.prompt))
                card.evidence_runs.append(tr.run_id)
    for r in rules:
        card = cards.get(r.tool)
        if card is None:
            continue
        if r.kind == "unknown_field":
            card.pitfalls.append(f"{r.entity} search does not take `{r.value}`"
                                 + (f"; use `{r.rename_to}`" if r.rename_to else ""))
        elif r.kind == "invalid_section":
            card.pitfalls.append(f"section `{r.value}` is not valid for {r.entity}")
        elif r.kind == "shell":
            card.pitfalls.append("no shell syntax (pipes, redirects) in commands")
    return list(cards.values())


def write_cards(store: ContextStore, cards: Iterable[ToolCard]):
    """Upserts `toolcards/<server>/<tool>` (status proposed), evidenced by the example runs."""
    ops = []
    for c in cards:
        server, tool = c.tool.split("__", 1)
        data = {"tool": tool_name(server, tool), "usage_notes": c.usage_notes,
                "example_tasks": [p for _, p in c.example_tasks], "pitfalls": c.pitfalls, "status": "proposed"}
        evidence = [EvidenceRef(kind="run", ref=r) for r in c.evidence_runs]
        ops.append(_upsert(store, f"toolcards/{server}/{tool}", "ToolCard", data, evidence=evidence))
    return store.apply("toolcards from catalog and runs", ops)


def fts_query(text: str) -> str | None:
    """Free text to a safe FTS5 query: each word quoted, OR-ed (as Paper Librarian's)."""
    terms = [t for t in TOKEN.findall(text.lower()) if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms)) or None


def rrf(rankings: dict[str, list[str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ids in rankings.values():
        for rank, tool in enumerate(ids, start=1):
            scores[tool] = scores.get(tool, 0.0) + 1.0 / (RRF_K + rank)
    return scores


class CardIndex:
    def __init__(self, cards: Iterable[ToolCard]) -> None:
        self.cards = {c.tool: c for c in cards}
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE docs USING fts5(text, tool UNINDEXED, field UNINDEXED, task UNINDEXED)")
        rows = []
        for c in self.cards.values():
            name = c.tool.replace("__", " ").replace("-", " ").replace("_", " ")
            rows += [(name, c.tool, "name", ""), (c.description, c.tool, "description", ""),
                     (c.usage_notes, c.tool, "notes", "")]
            rows += [(prompt, c.tool, "examples", task) for task, prompt in c.example_tasks]
            rows += [(p, c.tool, "pitfalls", "") for p in c.pitfalls]
        self.db.executemany("INSERT INTO docs (text, tool, field, task) VALUES (?, ?, ?, ?)", rows)

    def search(self, query: str, *, exclude_task: str | None = None) -> list[tuple[str, float]]:
        match = fts_query(query)
        if match is None:
            return []
        rankings = {}
        for field in FIELDS:
            cur = self.db.execute("SELECT tool, MIN(rank) FROM docs WHERE docs MATCH ? AND field = ? AND task != ? "
                                  "GROUP BY tool ORDER BY MIN(rank)", (match, field, exclude_task or "\0"))
            rankings[field] = [tool for tool, _ in cur]
        scores = rrf(rankings)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class ToolRetrieval(BaseStrategy):
    name = "E1_retrieval"

    def __init__(self, cards: list[ToolCard], *, top_k: int = 5, mode: Literal["static", "find_tools"] = "static",
                 route: bool = False, route_margin: float = 0.5) -> None:
        self.index = CardIndex(cards)
        self.top_k = top_k
        self.mode = mode
        self.route = route
        self.route_margin = route_margin
        self.catalog: dict[str, ToolDef] = {}

    def config(self) -> dict[str, Any]:
        return {"top_k": self.top_k, "mode": self.mode, "route": self.route, "cards": len(self.index.cards)}

    def rank(self, task: Task) -> list[tuple[str, float]]:
        ranked = self.index.search(task.prompt, exclude_task=task.id)
        if self.route and ranked:
            per_server: dict[str, float] = {}
            for tool, score in ranked:
                server = tool.split("__", 1)[0]
                per_server[server] = max(per_server.get(server, 0.0), score)
            best = max(per_server.values())
            keep = {s for s, v in per_server.items() if v >= best * (1 - self.route_margin)}
            ranked = [(t, s) for t, s in ranked if t.split("__", 1)[0] in keep]
        return ranked

    def select_tools(self, task: Task, tools: list[ToolDef]) -> list[ToolDef]:
        self.catalog = {t.name: t for t in tools if t.server}
        self.task = task
        ranked = [(t, s) for t, s in self.rank(task) if t in self.catalog][:self.top_k]
        chosen = {t for t, _ in ranked}
        kept = [t for t in tools if not t.server or t.name in chosen]
        evidence = [EvidenceRef(kind="schema_hash", ref=self.catalog[t].schema_hash or t,
                                detail={"tool": t, "score": round(s, 5), "rank": i + 1})
                    for i, (t, s) in enumerate(ranked)]
        if not evidence:  # nothing matched: offer everything rather than nothing
            return tools
        self.run.transform("tools", before=len(tools), after=len(kept), evidence=evidence,
                           note=f"top {self.top_k} of {len(self.catalog)} hub tools" + (" (routed)" if self.route else ""))
        return kept

    def extra_tools(self) -> list[ExtraTool]:
        if self.mode != "find_tools":
            return []

        async def find(call: PendingCall) -> ToolResult:
            query = str(call.arguments.get("query", ""))
            offered = set(self.run.trace.tools_offered)
            ranked = [(t, s) for t, s in self.index.search(query, exclude_task=self.task.id)
                      if t in self.catalog and t not in offered][:5]
            if ranked:
                self.run.offer(self.name, [self.catalog[t] for t, _ in ranked], note=f"find_tools({query!r})",
                               evidence=[EvidenceRef(kind="schema_hash", ref=self.catalog[t].schema_hash or t,
                                                     detail={"tool": t, "score": round(s, 5), "query": query})
                                         for t, s in ranked])
            lines = [f"- {t}: {self.catalog[t].description.splitlines()[0][:160]}" for t, _ in ranked]
            text = ("Now available:\n" + "\n".join(lines)) if lines else "No further tools match."
            return ToolResult(name=FIND_TOOLS, server="toolhelper", tool="find_tools", arguments=call.arguments,
                              text=text, is_error=False, source="local", latency_ms=0)

        return [ExtraTool(tool=ToolDef(name=FIND_TOOLS, description=(
            "Search for more tools by what you need to do; matching tools become callable on the next turn."),
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
            handler=find)]


class ServerMenu(BaseStrategy):
    """E1 `mode=servers`: no hub tool is offered at the start; `load_tools(servers)` adds the
    chosen servers' tools from the next turn. Evidence: the schema hash of every tool hidden
    at the start and of every tool loaded."""

    name = "E1_server_menu"

    def __init__(self, tools: Iterable[ToolDef]) -> None:
        self.by_server: dict[str, list[ToolDef]] = {}
        for t in tools:
            if t.server:
                self.by_server.setdefault(t.server, []).append(t)
        if not self.by_server:
            raise ValueError("E1 mode=servers needs the catalog's tools")

    def config(self) -> dict[str, Any]:
        return {"mode": "servers", "servers": {s: len(ts) for s, ts in self.by_server.items()}}

    def menu(self) -> str:
        lines = []
        for server, tools in self.by_server.items():
            # The servers' own tool names and the first line of their own descriptions.
            items = []
            for t in tools:
                first = next((line.strip() for line in t.description.splitlines() if line.strip()), "")
                # A required enum argument says what the tool covers (BioMCP `get`: entity gene, variant, trial, ...).
                props = t.input_schema.get("properties", {})
                enums = [f"{k}: {', '.join(map(str, props[k]['enum']))}" for k in t.input_schema.get("required", [])
                         if isinstance(props.get(k), dict) and props[k].get("enum")]
                items.append(t.tool + (f": {first[:70]}" if first else "") + (f" ({'; '.join(enums)})" if enums else ""))
            lines.append(f"- {server} ({len(tools)} tools): " + "; ".join(items))
        return "\n".join(lines)

    def select_tools(self, task: Task, tools: list[ToolDef]) -> list[ToolDef]:
        hidden = [t for t in tools if t.server]
        kept = [t for t in tools if not t.server]
        self.live = {t.name: t for t in hidden}  # the run's own definitions, hashes included
        self.run.transform("tools", before=len(tools), after=len(kept), note=f"server menu: {len(hidden)} tools load on request",
                           evidence=[EvidenceRef(kind="schema_hash", ref=t.schema_hash or t.name, detail={"tool": t.name})
                                     for t in hidden])
        return kept

    def extra_tools(self) -> list[ExtraTool]:
        async def load(call: PendingCall) -> ToolResult:
            asked = call.arguments.get("servers")
            asked = [asked] if isinstance(asked, str) else list(asked or [])
            unknown = [s for s in asked if s not in self.by_server]
            if unknown or not asked:
                text, tools = f"Error: unknown servers {unknown}; choose from {sorted(self.by_server)}.", []
            else:
                offered = set(self.run.trace.tools_offered)
                tools = [self.live.get(t.name, t) for s in asked for t in self.by_server[s] if t.name not in offered]
                if tools:
                    self.run.offer(self.name, tools, note=f"load_tools({asked})",
                                   evidence=[EvidenceRef(kind="schema_hash", ref=t.schema_hash or t.name,
                                                         detail={"tool": t.name, "server": t.server}) for t in tools])
                text = (f"Loaded {len(tools)} tools, callable from your next turn: " + ", ".join(t.name for t in tools)
                        if tools else f"The tools of {asked} are already loaded.")
            return ToolResult(name=LOAD_TOOLS, server="toolhelper", tool="load_tools", arguments=call.arguments,
                              text=text, is_error=not tools and bool(unknown or not asked), source="local", latency_ms=0)

        return [ExtraTool(tool=ToolDef(name=LOAD_TOOLS, description=(
            "Load the tools of one or more servers before using them; they become callable on your next turn. "
            "Load only the servers this task needs.\n" + self.menu()),
            input_schema={"type": "object", "required": ["servers"], "properties": {"servers": {
                "type": "array", "items": {"type": "string", "enum": sorted(self.by_server)}}}}),
            handler=load)]


def needed_tools(task: Task) -> set[str]:
    c = task.checks
    return set(c.get("must_call_all", [])) | ({c["must_call_any"][0]} if c.get("must_call_any") else set()) | \
        {t for path in task.expected_paths[:1] for t in path}


def recall_at_k(trace, task: Task) -> float | None:
    """Share of the tools the task needs (first expected path, required calls) that were offered."""
    needed = needed_tools(task)
    if not needed:
        return None
    offered = set(trace.llm_calls[0].tools_offered) if trace.llm_calls else set()
    any_of = set(task.checks.get("must_call_any", []))
    hit = {t for t in needed if t in offered or (t in any_of and any_of & offered)}
    return len(hit) / len(needed)
