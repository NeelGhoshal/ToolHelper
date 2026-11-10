"""Streamlit app for Phase 10 experiments: edit a design, launch / pause / resume it, watch its
progress and read its results.

    uv run --extra dashboard toolhelper experiment app

Designs are the factorial configs (`design: factorial`, configs/experiments-p10b..d.yaml); any
config, including the original p10 plan, can also be edited as raw YAML. Runs are the same
`toolhelper experiment freeze|auto` the CLI uses, launched detached, so closing the app does not
stop them. Everything shown is read from the monitoring DB, the context store and the
orchestrator's state.json.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from toolhelper.config import CONTEXT_DB, MONITOR_DB, ROOT, TASKS_FILE, load_llm_config
from toolhelper.efficiency.registry import ORDER, PARAMS, check_spec
from toolhelper.experiments import control
from toolhelper.experiments.experiment import folds
from toolhelper.experiments.factorial import FactorialConfig, cells, interaction_groups, merge_specs
from toolhelper.experiments.runner import load_tasks

st.set_page_config(page_title="ToolHelper experiments", layout="wide")
# Keys and the dataset dir from the Windows user env, for apps started before they were set.
os.environ.update({k: v for k, v in control.run_env().items() if k in control.USER_ENV_KEYS and not os.environ.get(k)})

STRATEGY_NAMES = {"E1": "tool retrieval", "E2": "schema compression", "E3": "dataset grounding",
                  "E4": "argument guard", "E5": "recipes", "E6": "result shaping", "E7": "result cache",
                  "E8": "history compaction", "E9": "final answer at the turn limit"}


# --- data -----------------------------------------------------------------------------------


@st.cache_data
def tasks_info() -> tuple[int, dict[str, int]]:
    tasks = load_tasks(TASKS_FILE)
    return len(tasks), {f: len(ids) for f, ids in folds(tasks).items()}


@st.cache_data(ttl=30)
def monitor_rows(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    from toolhelper.monitoring.db import MonitorDB

    return MonitorDB(MONITOR_DB).rows(sql, params)


def groups() -> list[str]:
    return [r["group_id"] for r in monitor_rows("SELECT group_id FROM runs GROUP BY group_id ORDER BY MIN(started_at)")]


def read_state(experiment: str) -> dict[str, Any]:
    path = control.workdir(experiment) / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def store_thing(name: str):
    from toolhelper.context.local_store import LocalStore

    try:
        return LocalStore(CONTEXT_DB).get(name)
    except Exception:  # not written yet
        return None


def status_of(experiment: str) -> tuple[str, str]:
    info, state = control.process_info(experiment), read_state(experiment)
    log = state.get("log", [])
    if control.is_running(info):
        return "running", f"pid {info['pid']}"
    if log and log[-1].endswith("experiment finished"):
        return "finished", log[-1][:19]
    if log:
        return "paused", f"last entry {log[-1][:19]}"
    return "not started", ""


def pct(x: float | None, scale: float = 100, fmt: str = "+.1f") -> str:
    return "n/a" if x is None else format(x * scale, fmt)


def ci(c: list[float] | None, scale: float = 100) -> str:
    return "n/a" if not c else f"{c[0] * scale:+.1f} .. {c[1] * scale:+.1f}"


# --- design ---------------------------------------------------------------------------------


def spec_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def factors_frame(factors: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"factor": k, "specs": " ; ".join(v.get("specs", [])), "max_turns": v.get("max_turns"),
                          "note": v.get("note", "")} for k, v in factors.items()],
                        columns=["factor", "specs", "max_turns", "note"])


def _cell(v: Any) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()


def frame_factors(df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for row in df.to_dict("records"):
        name = _cell(row.get("factor"))
        if not name:
            continue
        f: dict[str, Any] = {"specs": [s.strip() for s in _cell(row.get("specs")).split(";") if s.strip()]}
        if _cell(row.get("max_turns")):
            f["max_turns"] = int(float(row["max_turns"]))
        if _cell(row.get("note")):
            f["note"] = _cell(row["note"])
        out[name] = f
    return out


def validate(data: dict[str, Any]) -> tuple[FactorialConfig | None, list[str]]:
    problems = []
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(data.get("experiment", ""))):
        return None, ["the experiment name may only use letters, digits, '-' and '_'"]
    try:
        cfg = FactorialConfig.model_validate(data)
    except Exception as e:
        return None, [str(e)]
    for where, specs in [("base", cfg.base.specs), *((f"factor {k}", f.specs) for k, f in cfg.factors.items())]:
        problems += [f"{where}: {p}" for s in specs for p in check_spec(s)]
    for f, word in cfg.generators.items():
        if f not in cfg.factors or any(g not in cfg.factors or g in cfg.generators for g in word):
            problems.append(f"generator {f} = {word}: use factor letters that are not generated themselves")
    if set(cfg.confirm_only) - set(cfg.factors):
        problems.append(f"confirm_only names unknown factors {sorted(set(cfg.confirm_only) - set(cfg.factors))}")
    if not cfg.factors:
        problems.append("at least one factor is needed")
    if not problems:
        try:
            cells(cfg)
        except ValueError as e:
            problems.append(str(e))
    return (cfg if not problems else None), problems


def spec_builder() -> None:
    with st.expander("Spec builder: compose a strategy spec"):
        sid = st.selectbox("strategy", ORDER, format_func=lambda s: f"{s} {STRATEGY_NAMES[s]}", key="sb_sid")
        parts = []
        for key, allowed in PARAMS[sid].items():
            if allowed is int:
                v = st.number_input(key, min_value=0, value=0, step=1, key=f"sb_{sid}_{key}",
                                    help="0 leaves the builder's default")
                if v:
                    parts.append(f"{key}={int(v)}")
            elif isinstance(allowed, tuple):
                options = allowed[1] or [d["name"] for d in datasets()]
                v = st.multiselect(key, options, key=f"sb_{sid}_{key}")
                if v:
                    parts.append(f"{key}={'+'.join(v)}")
            else:
                v = st.selectbox(key, ["(default)", *allowed], key=f"sb_{sid}_{key}")
                if v != "(default)":
                    parts.append(f"{key}={v}")
        if sid == "E7":
            ttl = st.text_input("ttl per server (server=seconds, comma-separated)", key="sb_e7_ttl")
            parts += [f"ttl_{p.strip()}" for p in ttl.split(",") if "=" in p]
        st.code(sid + (":" + ",".join(parts) if parts else ""), language=None)
        st.caption("Copy it into a factor's specs (factors take several specs separated by ` ; `) or the base.")


@st.cache_data
def datasets() -> list[dict[str, Any]]:
    try:
        from toolhelper.datasets.base import INDEX_PATH
        from toolhelper.datasets.index import EntityIndex

        return EntityIndex(INDEX_PATH).datasets()
    except Exception:
        return []


def load_draft(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    st.session_state.update(draft_src=str(path), draft_text=text, draft=yaml.safe_load(text) or {},
                            draft_header=control.leading_comment(text), draft_rev=st.session_state.get("draft_rev", 0) + 1)


def design_tab(path: Path) -> None:
    d: dict[str, Any] = st.session_state.draft
    rev = st.session_state.draft_rev
    if d.get("design") != "factorial":
        st.info("This config is the original p10 plan (singles, E3 grid, greedy). Edit it as raw YAML below; "
                "the form is for factorial designs. Start a factorial design from experiments-p10b/c/d.yaml.")
        raw_yaml_editor(path)
        return
    n_tasks, fold_sizes = tasks_info()
    profiles = load_llm_config().profiles
    known_groups = groups()

    st.subheader("Experiment")
    c1, c2, c3, c4 = st.columns(4)
    name = c1.text_input("name", d["experiment"], key=f"name{rev}",
                         help="Groups are named <name>-f-<cell>-r<repeat>. Reusing a name resumes that experiment.")
    llm = c2.selectbox("LLM profile", list(profiles), index=list(profiles).index(d["llm"]) if d["llm"] in profiles else 0,
                       key=f"llm{rev}")
    mode = c3.selectbox("hub mode", ["replay", "live", "record"], index=["replay", "live", "record"].index(d.get("mode", "replay")),
                        key=f"mode{rev}", help="replay serves recordings and records misses live; live never replays")
    parallel = c4.number_input("parallel passes", 1, 8, int(d.get("parallel", 3)), key=f"par{rev}",
                               help="~0.6-1 GB RAM per pass with its MCP servers")
    learn = st.multiselect("learn from (groups the learned strategies E1 E4 E5 E6 and E9 learn from, cross-fitted by fold)",
                           sorted(set(known_groups) | set(d.get("learn_from", []))), d.get("learn_from", []), key=f"learn{rev}")

    st.subheader("Base (on in every cell, the baseline and the live baseline)")
    base = d.get("base") or {}
    b1, b2 = st.columns([3, 1])
    base_specs = b1.text_area("base specs, one per line", "\n".join(base.get("specs", [])), key=f"bspecs{rev}", height=100)
    base_turns = b2.number_input("base max_turns (0 = experiment default)", 0, 20, int(base.get("max_turns") or 0),
                                 key=f"bturns{rev}")
    max_turns = b2.number_input("experiment max_turns", 1, 20, int(d.get("max_turns", 8)), key=f"mt{rev}")
    base_note = st.text_input("base note", base.get("note", ""), key=f"bnote{rev}")

    st.subheader("Factors (two levels each: specs off / on)")
    spec_builder()
    df = st.data_editor(factors_frame(d.get("factors", {})), num_rows="dynamic", width="stretch",
                        key=f"factors{rev}", column_config={
                            "factor": st.column_config.TextColumn(help="a letter; cells are named by the letters on"),
                            "specs": st.column_config.TextColumn(width="large", help="specs separated by ' ; '"),
                            "max_turns": st.column_config.NumberColumn(min_value=1, max_value=20, step=1),
                        })
    factors = frame_factors(df)

    st.subheader("Design and decision rule")
    g1, g2, g3 = st.columns(3)
    gen_text = g1.text_input("generators (fraction), e.g. D=ABC", ", ".join(f"{k}={v}" for k, v in d.get("generators", {}).items()),
                             key=f"gen{rev}", help="empty = full factorial (2^k cells)")
    confirm_only = g2.multiselect("confirm only (skip the factorial; confirm these factors on fresh runs)", list(factors),
                                  [f for f in d.get("confirm_only", []) if f in factors], key=f"co{rev}")
    subsets = g3.multiselect("report subsets", ["grounded"], d.get("report_subsets", []), key=f"sub{rev}",
                             help="effects also on the tasks whose prompt names a dataset entity")
    h1, h2, h3, h4, h5 = st.columns(5)
    gate = d.get("gate", {"max_success_drop": 0.02, "min_trust": 0.9})
    drop = h1.number_input("max success drop (points)", 0.0, 50.0, gate["max_success_drop"] * 100, 0.5, key=f"drop{rev}")
    trust = h2.number_input("min trust", 0.0, 1.0, float(gate["min_trust"]), 0.01, key=f"trust{rev}")
    repeats = h3.number_input("confirmation repeats", 1, 10, int(d.get("confirm_repeats", 3)), key=f"rep{rev}")
    live_fold = h4.selectbox("live fold", ["A", "B"], index="AB".index(d.get("live_fold", "A")), key=f"lf{rev}")
    seed = h5.number_input("pass order seed", 0, 10_000, int(d.get("order_seed", 7)), key=f"seed{rev}")
    require_saving = st.checkbox("require saving (adopt only if the confirmation's tokens-per-success CI is below 0)",
                                 bool(d.get("require_saving")), key=f"rs{rev}")
    boot = d.get("bootstrap", {"samples": 2000, "seed": 7})
    k1, k2 = st.columns(2)
    samples = k1.number_input("bootstrap samples", 100, 20_000, int(boot["samples"]), 100, key=f"bs{rev}")
    bseed = k2.number_input("bootstrap seed", 0, 10_000, int(boot["seed"]), key=f"bseed{rev}")
    header = st.text_area("pre-registration (written as the config's leading comment: hypothesis, design, choice rule)",
                          st.session_state.draft_header, key=f"hdr{rev}", height=140)

    data: dict[str, Any] = {"design": "factorial", "experiment": name.strip(), "llm": llm, "mode": mode,
                            "max_turns": int(max_turns), "parallel": int(parallel), "learn_from": learn}
    base_out: dict[str, Any] = {"specs": spec_lines(base_specs)}
    if base_turns:
        base_out["max_turns"] = int(base_turns)
    if base_note.strip():
        base_out["note"] = base_note.strip()
    if base_out["specs"] or base_turns:
        data["base"] = base_out
    data["factors"] = factors
    gens = dict(p.replace(" ", "").split("=", 1) for p in gen_text.split(",") if "=" in p)
    if gens:
        data["generators"] = gens
    if confirm_only:
        data["confirm_only"] = confirm_only
    if require_saving:
        data["require_saving"] = True
    if subsets:
        data["report_subsets"] = subsets
    data.update(confirm_repeats=int(repeats), live_fold=live_fold, order_seed=int(seed),
                gate={"max_success_drop": round(drop / 100, 4), "min_trust": trust},
                bootstrap={"samples": int(samples), "seed": int(bseed)})

    st.subheader("Check")
    cfg, problems = validate(data)
    for p in problems:
        st.error(p)
    if cfg:
        preview(cfg, n_tasks, fold_sizes)
    save_box(cfg, data, header, path)

    with st.expander("Raw YAML"):
        raw_yaml_editor(path)


def preview(cfg: FactorialConfig, n_tasks: int, fold_sizes: dict[str, int]) -> None:
    cs = cells(cfg)
    if not cfg.confirm_only:
        st.write(f"**{len(cs)} cells** ({'fraction ' + str(cfg.generators) if cfg.generators else 'full factorial'}); "
                 f"aliased interactions: {[g for g in interaction_groups(cfg) if len(g) > 1] or 'none'}")
        st.dataframe(pd.DataFrame([{"cell": c.name, "specs": " ; ".join(c.specs) or "(none)", "max_turns": c.max_turns or cfg.max_turns}
                                   for c in cs]), width="stretch", hide_index=True)
    else:
        on = merge_specs([*cfg.base.specs, *(s for f in cfg.confirm_only for s in cfg.factors[f].specs)])
        st.write(f"**Confirmation only:** {' ; '.join(on)} vs base {' ; '.join(cfg.base.specs) or '(no strategies)'}, "
                 f"{cfg.confirm_repeats} paired repeats.")
    plan = control.plan_runs(cfg, n_tasks, fold_sizes[cfg.live_fold])
    rate = st.number_input("expected task runs per hour (all passes together; ~60-175 overnight, ~12 on US mornings)",
                           1, 500, 60, key="rate")
    st.write(f"Task runs, at most: factorial {plan['factorial']}, confirmation {plan['confirm']}, live {plan['live']} "
             f"= **{plan['total']}**, about **{plan['total'] / rate:.1f} h** at {rate}/h (replay; misses run live).")


def save_box(cfg: FactorialConfig | None, data: dict[str, Any], header: str, path: Path) -> None:
    target = ROOT / "configs" / f"experiments-{data['experiment']}.yaml"
    started = bool(read_state(data["experiment"]).get("log"))
    st.write(f"Saves to `{target.relative_to(ROOT)}`")
    if started:
        st.warning(f"{data['experiment']} has already run (data/experiments/{data['experiment']}/state.json). Changing "
                   "its design would mix runs of two designs in one experiment; give it a new name instead.")
    if st.button("Save config", disabled=cfg is None or started, type="primary"):
        target.write_text(control.dump_config(data, header), encoding="utf-8")
        load_draft(target)
        st.session_state.pending_select = target.name
        st.rerun()


def raw_yaml_editor(path: Path) -> None:
    text = st.text_area("YAML", st.session_state.draft_text, height=420, key=f"raw{st.session_state.draft_rev}")
    try:
        data = yaml.safe_load(text) or {}
        if data.get("design") == "factorial":
            _, problems = validate(data)
        else:
            from toolhelper.experiments.experiment import ExperimentConfig

            ExperimentConfig.model_validate(data)
            problems = [p for specs in data.get("arms", {}).values() for s in specs for p in check_spec(s)]
    except Exception as e:
        data, problems = {}, [str(e)]
    for p in problems:
        st.error(p)
    name = data.get("experiment", "")
    started = bool(name and read_state(name).get("log"))
    target = ROOT / "configs" / ("experiments.yaml" if name == "p10" else f"experiments-{name}.yaml")
    if started:
        st.warning(f"{name} has already run; save under a new `experiment:` name.")
    if st.button(f"Save YAML to {target.name}", disabled=bool(problems) or not name or started, key="save_raw"):
        target.write_text(text, encoding="utf-8")
        load_draft(target)
        st.session_state.pending_select = target.name
        st.rerun()


# --- run ------------------------------------------------------------------------------------


def run_tab(path: Path, data: dict[str, Any]) -> None:
    name = data.get("experiment", "")
    state, info = status_of(name)
    st.write(f"**{name}**: {state} {info}")
    env = control.run_env()
    profile = load_llm_config().profiles.get(data.get("llm", ""))
    problems = control.missing_requirements(env, profile.model_dump() if profile else None)
    free = control.free_memory_gb()
    c1, c2, c3 = st.columns(3)
    c1.metric("free RAM", "n/a" if free is None else f"{free:.1f} GB")
    c2.metric("OneDrive", "running" if control.onedrive_running() else "off")
    c3.metric("parallel passes", data.get("parallel", 1))
    for p in problems:
        st.error(p)
    if free is not None and free < data.get("parallel", 1) * 0.8:
        st.warning("Little free memory for this many passes; Windows may kill runs. Lower `parallel` or close apps.")
    if control.onedrive_running():
        st.warning("OneDrive syncs every trace and recording the run writes and grew to several GB of RAM in earlier "
                   "runs. Consider stopping it (it restarts itself after updates).")

    frozen = store_thing(f"experiments/{name}")
    st.subheader("1. Freeze")
    st.write(f"Frozen as `experiments/{name}@v{frozen.version}`" if frozen else "Not frozen yet. Freezing records the "
             "inputs (catalog, benchmark, datasets, LLM profile, recordings, learning runs, code commit) and the folds.")
    if st.button("Freeze inputs" if not frozen else "Freeze again (new version)", disabled=bool(problems)):
        with st.spinner("freezing..."):
            r = control.run_cli(["experiment", "freeze", "--config", str(path)], env)
        (st.success if r.returncode == 0 else st.error)(r.stdout[-3000:] + r.stderr[-3000:])

    st.subheader("2. Run")
    st.caption("Launch starts `toolhelper experiment auto` detached; it keeps running if the app closes. Pause kills it "
               "and its passes; launching again resumes from the saved runs (tasks in flight are redone).")
    b1, b2 = st.columns(2)
    if b1.button("Launch / resume", type="primary", disabled=state == "running" or bool(problems) or not frozen):
        info_ = control.launch(path, name, env)
        st.success(f"started pid {info_['pid']}; logs in {Path(info_['out']).name}")
        st.rerun()
    if b2.button("Pause", disabled=state != "running"):
        st.info(control.stop(name))
        st.rerun()

    st.subheader("3. Decision")
    wref = read_state(name).get("decision")
    if not wref:
        st.write("No Decision yet (written after the confirmation).")
    else:
        thing = store_thing(wref.split("@")[0])
        status = thing.data.get("status") if thing else "?"
        st.write(f"`{thing.name}@v{thing.version}`: **{status}**" if thing else wref)
        by = st.text_input("approved by", key="approver")
        if st.button("Approve decision", disabled=status == "accepted" or not by.strip()):
            r = control.run_cli(["curate", wref.split("@")[0], "--by", by.strip()], env)
            (st.success if r.returncode == 0 else st.error)(r.stdout + r.stderr)


# --- progress -------------------------------------------------------------------------------


@st.fragment(run_every=30)
def progress_tab(data: dict[str, Any]) -> None:
    name = data.get("experiment", "")
    state, info = status_of(name)
    n_tasks, fold_sizes = tasks_info()
    st.write(f"**{name}**: {state} {info} (refreshes every 30 s)")
    monitor_rows.clear()
    rows = monitor_rows("SELECT group_id, COUNT(DISTINCT task_id) AS tasks, COUNT(DISTINCT CASE WHEN success THEN task_id END) "
                        "AS passed, SUM(error LIKE 'APITimeout%' OR error LIKE 'APIConnection%') AS transient, "
                        "MIN(started_at) AS first, MAX(started_at) AS last FROM runs WHERE group_id LIKE ? "
                        "GROUP BY group_id ORDER BY MIN(started_at)", (f"{name}-%",))
    if rows:
        df = pd.DataFrame(rows)
        expected = [fold_sizes[data.get("live_fold", "A")] if "-live-" in g else n_tasks for g in df["group_id"]]
        df.insert(2, "of", expected)
        df["done"] = df["tasks"] / df["of"]
        st.dataframe(df, width="stretch", hide_index=True, column_config={
            "done": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
        last = monitor_rows("SELECT COUNT(*) AS n FROM runs WHERE group_id LIKE ? AND started_at >= datetime('now', '-1 hour')",
                            (f"{name}-%",))
        st.write(f"Task runs started in the last hour: {last[0]['n']}")
    else:
        st.write("No runs saved yet.")
    log = read_state(name).get("log", [])
    st.text_area("orchestrator log (latest last)", "\n".join(log[-60:]), height=300)
    proc = control.process_info(name)
    if proc and Path(proc["err"]).exists() and (err := Path(proc["err"]).read_text(encoding="utf-8", errors="replace")[-4000:]).strip():
        st.text_area("stderr of the latest launch", err, height=160)


# --- results --------------------------------------------------------------------------------


def effects_frame(effects: dict[str, Any], labels: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for f, e in effects.items():
        if not e.get("tasks"):
            continue
        rows.append({"effect": f, "what": (labels or {}).get(f, ""), "tasks": e["tasks"],
                     "tokens/success %": pct(e.get("tps_rel")), "CI": ci(e.get("tps_rel_ci")),
                     "tokens/run %": pct(e.get("tokens_rel")), "CI ": ci(e.get("tokens_rel_ci")),
                     "success pts": pct(e.get("success_pts")), "CI  ": ci(e.get("success_pts_ci")),
                     "trust on": "" if e.get("trust_on") is None else f"{e['trust_on']:.3f}",
                     **{f"Δ {m}": v for m, v in (e.get("mediators") or {}).items()}})
    return pd.DataFrame(rows)


def summary_frame(arms: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"arm": k, "runs": s.get("runs"), "success %": pct(s.get("success_rate"), 100, ".1f"),
                          "tokens/success": f"{s['tokens_per_success']:,.0f}" if s.get("tokens_per_success") else "n/a",
                          "median tokens": s.get("median_tokens"), "p90 tokens": s.get("p90_tokens"),
                          "trust": f"{s['trust_mean']:.3f}" if s.get("trust_mean") is not None else ""}
                         for k, s in arms.items()])


def results_tab(data: dict[str, Any]) -> None:
    name = data.get("experiment", "")
    s = read_state(name)
    if not s:
        st.write("No results yet.")
        return
    labels = {f: " ; ".join(v.get("specs", [])) for f, v in (data.get("factors") or {}).items()}
    st.caption("Effects are paired by task; CIs are 95% paired-bootstrap intervals. Tokens are measured. "
               "A factor is chosen when its tokens/success CI is below 0, success holds within the gate and trust stays above it.")
    if chosen := s.get("chosen"):
        st.subheader("Choice")
        st.write(f"Adopted: **{chosen['cell']}** ({' ; '.join(chosen['specs']) or 'no strategies'}, "
                 f"max_turns {chosen.get('max_turns')})" if chosen["factors"] else
                 f"Nothing adopted; the baseline stays ({' ; '.join(chosen['specs']) or 'no strategies'}).")
        st.table(pd.DataFrame([{"factor": f, "reason": r} for f, r in chosen.get("reasons", {}).items()]))
    if "effects" in s:
        st.subheader("Main effects (all tasks)")
        st.dataframe(effects_frame(s["effects"], labels), width="stretch", hide_index=True)
        st.subheader("Interactions")
        st.dataframe(effects_frame(s.get("interactions", {})), width="stretch", hide_index=True)
        st.caption("Interactions were not part of the choice rule; one with a CI below 0 is a lead to confirm, not a result.")
        for sub, effs in s.get("effects_by_subset", {}).items():
            st.subheader(f"Main effects, {sub} tasks only")
            st.dataframe(effects_frame(effs, labels), width="stretch", hide_index=True)
        with st.expander("Effects by server"):
            for k, effs in s.get("effects_by_kind", {}).items():
                st.write(f"**{k}**")
                st.dataframe(effects_frame(effs, labels), width="stretch", hide_index=True)
        with st.expander("Cells (one pass each)"):
            st.dataframe(summary_frame(s.get("cells", {})), width="stretch", hide_index=True)
    if conf := s.get("confirm"):
        st.subheader(f"Confirmation: {conf['cell']} vs baseline, {conf['summary'].get('repeats')} repeats")
        vs = conf["vs_baseline"]
        g = conf["gate"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("success change (pts)", pct(vs.get("delta_success")), ci(vs.get("delta_success_ci")))
        c2.metric("tokens/success change (%)", pct(vs.get("delta_tokens_per_success")), ci(vs.get("delta_tokens_per_success_ci")),
                  delta_color="inverse")
        c3.metric("McNemar lost / gained", f"{vs['mcnemar']['lost']} / {vs['mcnemar']['gained']}", f"p {vs['mcnemar']['p']:.3g}",
                  delta_color="off")
        c4.metric("gate", "pass" if g["passed"] else "fail", "; ".join(g.get("reasons", [])) or None, delta_color="off")
        st.dataframe(summary_frame({"chosen": conf["summary"], **{f"chosen, {k}": v for k, v in conf.get("by_kind", {}).items()}}),
                     width="stretch", hide_index=True)
    if live := s.get("live"):
        st.subheader(f"Live check, fold {data.get('live_fold', 'A')}")
        st.dataframe(summary_frame({**{f"live: {k}": v for k, v in live["live"].items()},
                                    **{f"replay, same tasks: {k}": v for k, v in live["replay_same_tasks"].items()}}),
                     width="stretch", hide_index=True)
    if wref := s.get("decision"):
        thing = store_thing(wref.split("@")[0])
        st.write(f"Decision `{wref}`" + (f", now `{thing.name}@v{thing.version}` ({thing.data.get('status')})" if thing else ""))


# --- page -----------------------------------------------------------------------------------


def main() -> None:
    st.title("ToolHelper experiments")
    files = control.config_files()
    names = [f.name for f in files]
    if st.session_state.get("pending_select") in names:
        st.session_state.selected = st.session_state.pop("pending_select")
    if st.session_state.get("selected") not in names:
        st.session_state.selected = names[-1]
    with st.sidebar:
        pick = st.selectbox("config", names, key="selected")
        path = ROOT / "configs" / pick
        if st.session_state.get("draft_src") != str(path):
            load_draft(path)
        st.caption("Edit a config and save it under a new experiment name to run a variant.")
        st.divider()
        for f in files:
            exp = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("experiment", "?")
            state, info = status_of(exp)
            st.write(f"`{exp}`: {state}")
    saved = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    design, run, progress, results = st.tabs(["Design", "Run", "Progress", "Results"])
    with design:
        design_tab(path)
    with run:
        run_tab(path, saved)
    with progress:
        progress_tab(saved)
    with results:
        if saved.get("design") == "factorial":
            results_tab(saved)
        else:
            st.write("Results of the original p10 plan are in its Decision (decisions/p10) and "
                     "data/experiments/p10/state.json.")


main()
