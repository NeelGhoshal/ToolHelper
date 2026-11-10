"""Phase 10 experiment setup: frozen inputs, folds and the Experiment thing.

Inputs are frozen as hashes and versions, so any later run can be checked against them:
catalog hash, benchmark file sha256, dataset versions, LLM profile and options, the replay
set (count and a hash of its keys), the learning group's run ids and the code commit.

Folds: two-fold cross-fitting. Tasks are split into halves balanced by server kind and
trap status; a strategy evaluated on one fold learns only from the other fold's runs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from toolhelper.agent.hooks import Task
from toolhelper.config import ROOT
from toolhelper.context.ingest import _upsert
from toolhelper.context.model import ContextStore
from toolhelper.mcp.canonical import canonical_json, sha256_hex

EXPERIMENTS_CONFIG = ROOT / "configs" / "experiments.yaml"


class ExperimentConfig(BaseModel):
    experiment: str
    llm: str
    mode: str = "replay"
    max_turns: int = 8
    parallel: int = 1
    learn_from: str
    baseline_repeats: int = 3
    close_call_repeats: int = 2
    finalist_repeats: int = 3
    arms: dict[str, list[str]]
    e3_grid: dict[str, list[str]]
    gate: dict[str, float]
    greedy: dict[str, float]
    bootstrap: dict[str, int]


def load_experiment_config(path: Path | str = EXPERIMENTS_CONFIG) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def kind(task: Task) -> str:
    return "cross" if len(task.servers) > 1 else (task.servers[0] if task.servers else "none")


def folds(tasks: list[Task]) -> dict[str, list[str]]:
    """Two halves balanced by server kind and traps: within each (kind, trapped) group,
    tasks sorted by id alternate between A and B."""
    groups: dict[tuple[str, bool], list[str]] = {}
    for t in sorted(tasks, key=lambda t: t.id):
        groups.setdefault((kind(t), bool(t.traps)), []).append(t.id)
    out: dict[str, list[str]] = {"A": [], "B": []}
    flip = 0
    for key in sorted(groups):
        for i, task_id in enumerate(groups[key]):
            out["AB"[(i + flip) % 2]].append(task_id)
        flip += len(groups[key])  # odd-sized groups alternate which fold gets the extra task
    return {k: sorted(v) for k, v in out.items()}


def other_fold(fold: str) -> str:
    return "B" if fold == "A" else "A"


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def replay_set(store_dir: Path) -> dict[str, Any]:
    keys = sorted(str(p.relative_to(store_dir)).replace("\\", "/") for p in store_dir.rglob("*.json"))
    return {"recordings": len(keys), "keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest()}


def freeze_inputs(*, catalog_hash: str, tasks_file: Path, datasets: list[dict[str, Any]], llm_profile: dict[str, Any],
                  store_dir: Path, learn_from_runs: list[str],
                  custom_strategies: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        # user-defined strategies the config uses (absent when none, so earlier input hashes still match)
        **({"custom_strategies": custom_strategies} if custom_strategies else {}),
        "catalog_hash": catalog_hash,
        "benchmark": {"file": tasks_file.name, "sha256": hashlib.sha256(tasks_file.read_bytes()).hexdigest()},
        "datasets": {d["name"]: {"wref": d["wref"], "sha256": d["sha256"]} for d in datasets},
        "llm": llm_profile,
        "replay_set_at_start": replay_set(store_dir),
        "learn_from_runs": sorted(learn_from_runs),
        "code_commit": _git_commit(),
    }


def write_experiment(store: ContextStore, cfg: Any, inputs: dict[str, Any], fold_map: dict[str, list[str]]):
    name = f"experiments/{cfg.experiment}"
    design = "fractional factorial" if getattr(cfg, "design", None) == "factorial" else "trimmed plan"
    data = {"title": f"Phase 10 {cfg.experiment}: {design}", "inputs_json": canonical_json(inputs),
            "plan_json": canonical_json(cfg.model_dump()), "folds_json": canonical_json(fold_map), "status": "proposed"}
    store.apply(f"experiment {cfg.experiment}", [_upsert(store, name, "Experiment", data)])
    return store.get(name)


def inputs_hash(inputs: dict[str, Any]) -> str:
    """Hash of what must not change mid-experiment (the replay set may grow: misses are recorded)."""
    return sha256_hex({k: v for k, v in inputs.items() if k not in ("replay_set_at_start", "code_commit")})
