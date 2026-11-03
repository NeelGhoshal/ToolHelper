"""E5: recipes. Reuses the shortest frequent successful tool path for similar tasks.

A recipe is learned per task category from successful runs: each run's path is its
sequence of successful hub calls (consecutive repeats collapsed); the most frequent path
wins, shorter first on ties. Argument values copied from the run's own prompt become
`<from task>`, the rest stay literal. A task never gets a recipe learned from its own runs
(leave-one-out). A task without a category takes the category of the most similar run
prompt (BM25).

Variants: `plan` (inject the recipe as context) vs `tool` (a local `get_recipe` tool).
Evidence: the runs the recipe came from, with its success rate; `write_recipes` stores
Recipe things whose `depends_on` pins the ToolSchema versions, so a schema change makes
the recipe stale (Phase 3).
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel

from toolhelper.agent.hooks import BaseStrategy, ContextItem, ExtraTool, PendingCall, Task, ToolDef
from toolhelper.context.ingest import _upsert, tool_name
from toolhelper.context.model import ContextStore, EvidenceRef, NotFoundError
from toolhelper.efficiency.retrieval import fts_query
from toolhelper.experiments.scoring import ANSWERED
from toolhelper.mcp.canonical import canonical_json
from toolhelper.mcp.hub import ToolResult

GET_RECIPE = "toolhelper__get_recipe"
FROM_TASK = "<from task>"


class RunPath(BaseModel):
    run_id: str
    task_id: str
    category: str | None
    prompt: str
    success: bool
    steps: list[str]
    templates: list[dict[str, Any]]
    prompt_tokens: int


class Recipe(BaseModel):
    task_type: str
    steps: list[str]
    arg_templates: list[dict[str, Any]]
    evidence_runs: list[str]
    success_rate: float
    mean_tokens: float


def _template(value: Any, prompt: str) -> Any:
    if isinstance(value, str):
        return FROM_TASK if len(value) >= 3 and value.casefold() in prompt.casefold() else value
    if isinstance(value, list):
        return [_template(v, prompt) for v in value]
    return value


def run_paths(traces: Iterable[Any]) -> list[RunPath]:
    out = []
    for t in traces:
        steps, templates = [], []
        for c in t.tool_calls:
            if c.is_error or not c.server:
                continue
            if steps and steps[-1] == c.name:
                continue
            steps.append(c.name)
            templates.append({k: _template(v, t.prompt) for k, v in c.final_arguments.items()})
        out.append(RunPath(run_id=t.run_id, task_id=t.task_id, category=(t.task or {}).get("category"), prompt=t.prompt,
                           success=bool(t.success) and t.stop_reason in ANSWERED, steps=steps, templates=templates,
                           prompt_tokens=sum(c.prompt_tokens for c in t.llm_calls)))
    return out


def learn_recipe(paths: list[RunPath], category: str, *, exclude_task: str | None = None) -> Recipe | None:
    pool = [p for p in paths if p.category == category and p.task_id != exclude_task]
    good = [p for p in pool if p.success and p.steps]
    if not good:
        return None
    freq = Counter(tuple(p.steps) for p in good)
    best = min(freq, key=lambda path: (-freq[path], len(path), path))
    runs = [p for p in good if tuple(p.steps) == best]
    return Recipe(task_type=category, steps=list(best), arg_templates=runs[0].templates,
                  evidence_runs=[p.run_id for p in runs], success_rate=len(good) / len(pool),
                  mean_tokens=sum(p.prompt_tokens for p in runs) / len(runs))


def render(recipe: Recipe) -> str:
    steps = "; ".join(f"{i}. {s} {canonical_json(a)}" for i, (s, a) in enumerate(zip(recipe.steps, recipe.arg_templates), 1))
    return (f"Plan that worked for similar '{recipe.task_type}' tasks ({len(recipe.evidence_runs)} runs, "
            f"{recipe.success_rate:.0%} of runs succeeded): {steps}. {FROM_TASK} means: take the value from this task.")


def write_recipes(store: ContextStore, recipes: Iterable[Recipe]):
    """Upserts `recipes/<task_type>` (status proposed) pinned to the current ToolSchemas."""
    ops = []
    for r in recipes:
        depends = []
        for step in dict.fromkeys(r.steps):
            name = tool_name(*step.split("__", 1))
            try:
                store.get(name)
            except NotFoundError:
                continue
            depends.append(name)
        data = {"task_type": r.task_type, "steps": r.steps, "arg_templates_json": canonical_json(r.arg_templates),
                "depends_on": depends, "evidence_runs": r.evidence_runs, "success_rate": r.success_rate,
                "mean_tokens": r.mean_tokens, "status": "proposed"}
        ops.append(_upsert(store, f"recipes/{r.task_type}", "Recipe", data,
                           evidence=[EvidenceRef(kind="run", ref=x) for x in r.evidence_runs]))
    return store.apply("recipes from successful runs", ops)


class Recipes(BaseStrategy):
    name = "E5_recipes"

    def __init__(self, paths: list[RunPath], mode: Literal["plan", "tool"] = "plan") -> None:
        self.paths = paths
        self.mode = mode
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE prompts USING fts5(prompt, category UNINDEXED, task UNINDEXED)")
        self.db.executemany("INSERT INTO prompts VALUES (?, ?, ?)",
                            [(p.prompt, p.category, p.task_id) for p in paths if p.category])

    def config(self) -> dict[str, Any]:
        return {"mode": self.mode, "runs": len(self.paths)}

    def category(self, task: Task) -> str | None:
        if task.category:
            return task.category
        match = fts_query(task.prompt)
        row = self.db.execute("SELECT category FROM prompts WHERE prompts MATCH ? AND task != ? ORDER BY rank LIMIT 1",
                              (match, task.id)).fetchone() if match else None
        return row[0] if row else None

    def recipe_for(self, task: Task, category: str | None = None) -> Recipe | None:
        category = category or self.category(task)
        return learn_recipe(self.paths, category, exclude_task=task.id) if category else None

    def prepare(self, task: Task, ctx) -> list[ContextItem]:
        self.task = task
        if self.mode != "plan":
            return []
        recipe = self.recipe_for(task)
        if recipe is None:
            return []
        return [ContextItem(kind="recipe", text=render(recipe), evidence=[
            EvidenceRef(kind="run", ref=r, detail={"task_type": recipe.task_type, "success_rate": recipe.success_rate})
            for r in recipe.evidence_runs])]

    def extra_tools(self) -> list[ExtraTool]:
        if self.mode != "tool":
            return []

        async def get(call: PendingCall) -> ToolResult:
            recipe = self.recipe_for(self.task, call.arguments.get("task_type"))
            text = render(recipe) if recipe else "No recipe for tasks like this one."
            return ToolResult(name=GET_RECIPE, server="toolhelper", tool="get_recipe", arguments=call.arguments,
                              text=text, is_error=False, source="local", latency_ms=0)

        return [ExtraTool(tool=ToolDef(name=GET_RECIPE, description=(
            "Get the tool plan that worked for similar tasks before (steps with argument templates)."),
            input_schema={"type": "object", "properties": {"task_type": {"type": "string"}}}), handler=get)]


def categories(paths: list[RunPath]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for p in paths:
        if p.category and p.success:
            out[p.category] += 1
    return dict(out)
