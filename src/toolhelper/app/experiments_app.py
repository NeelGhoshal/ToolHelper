"""Streamlit app for Phase 10 experiments: build a design, launch / pause / resume it, watch its
progress and read its results.

    uv run --extra dashboard toolhelper experiment app

Designs are the factorial configs (`design: factorial`, configs/experiments-p10b..d.yaml), built
in a form: strategies are picked and set with sliders and choices (texts and ranges in
experiments/design_form.py), factors are cards and can be taken over from other configs. The
Strategies tab defines new strategies (E10, E11, ...) from building blocks (efficiency/custom.py,
saved to configs/strategies.yaml). Any config, including the original p10 plan, can also be
edited as raw YAML. Runs are the same
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

from toolhelper.config import CATALOG_DIR, CONTEXT_DB, MONITOR_DB, ROOT, TASKS_FILE, load_llm_config, load_servers_config
from toolhelper.efficiency import custom
from toolhelper.efficiency.registry import BUILTIN, DEFAULTS, ORDER, PARAMS, check_spec, install_custom
from toolhelper.experiments import control
from toolhelper.experiments import design_form as form
from toolhelper.experiments.experiment import folds
from toolhelper.experiments.factorial import FactorialConfig, cells, interaction_groups, merge_specs
from toolhelper.experiments.runner import load_tasks

st.set_page_config(page_title="ToolHelper experiments", layout="wide")
# Keys and the dataset dir from the Windows user env, for apps started before they were set.
os.environ.update({k: v for k, v in control.run_env().items() if k in control.USER_ENV_KEYS and not os.environ.get(k)})
# User-defined strategies (configs/strategies.yaml), re-read on every rerun, as the runs will see them.
DEFS: dict[str, custom.Definition] = install_custom()

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


def validate(data: dict[str, Any]) -> tuple[FactorialConfig | None, list[str]]:
    problems = []
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(data.get("experiment", ""))):
        return None, ["give the experiment a name (letters, digits, '-' and '_' only)"]
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


@st.cache_data
def datasets() -> list[dict[str, Any]]:
    try:
        from toolhelper.datasets.base import INDEX_PATH
        from toolhelper.datasets.index import EntityIndex

        return EntityIndex(INDEX_PATH).datasets()
    except Exception:
        return []


def set_draft(data: dict[str, Any], header: str, text: str, src: str) -> None:
    """A new revision of the form: widget keys carry the revision, so every widget starts from `data`."""
    st.session_state.update(draft_src=src, draft_text=text, draft=data, draft_header=header,
                            draft_rev=st.session_state.get("draft_rev", 0) + 1,
                            rows=form.rows_from(data.get("factors") or {}))
    st.session_state.row_seq = len(st.session_state.rows)


def load_draft(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    set_draft(yaml.safe_load(text) or {}, control.leading_comment(text), text, str(path))


def strategy_label(sid: str) -> str:
    return f"{sid} {form.info(sid, DEFS)['name']}"


def param_widgets(kp: str, sid: str, loaded: dict[str, Any]) -> dict[str, Any]:
    """One widget per parameter of a strategy: a slider for numbers, a choice with its meaning for modes."""
    values: dict[str, Any] = {}
    for key, allowed in PARAMS[sid].items():
        if not form.shown(sid, key, values):
            continue
        ui = form.PARAM_UI.get((sid, key), {})
        label, k = ui.get("label", key), f"{kp}_{sid}_{key}"
        current = loaded.get(key, DEFAULTS[sid].get(key))
        if allowed is int:
            lo, hi, step = ui.get("range", (0, 100, 1))
            values[key] = st.slider(label, min(lo, int(current)), max(hi, int(current)), int(current), step, key=k,
                                    help=ui.get("help"))
        elif isinstance(allowed, tuple):
            picked = current if isinstance(current, list) else str(current).split("+") if current else []
            options = list(allowed[1] or [d["name"] for d in datasets()])
            options += [p for p in picked if p not in options]
            values[key] = st.multiselect(label, options, picked, key=k, help=ui.get("help"))
            for v in values[key]:
                if v in ui.get("values", {}):
                    st.caption(f"**{v}:** {ui['values'][v]}")
            if key in DEFAULTS[sid] and not values[key]:
                st.warning(f"Nothing picked: the strategy falls back to `{DEFAULTS[sid][key]}`.")
        else:
            options = [*allowed, *([current] if current not in allowed else [])]
            values[key] = st.radio(label, options, options.index(current), horizontal=True, key=k, help=ui.get("help"))
            if values[key] in ui.get("values", {}):
                st.caption(ui["values"][values[key]])
    if sid == "E7":
        ttl = {}
        for server, seconds in form.ttl_values(loaded).items():
            ttl[server] = st.number_input(f"time to live for {server} results (seconds)", 0, None, seconds, 3600,
                                          key=f"{kp}_E7_ttl_{server}",
                                          help="How long a cached result may be reused. 86,400 = one day; 0 = never cache.")
        values.update(form.ttl_params(ttl))
    return values


def strategy_editor(kp: str, specs: list[str], label: str) -> list[str]:
    """Pick strategies and set their parameters; returns the specs."""
    loaded: dict[str, dict[str, Any]] = {}
    for spec in specs:
        try:
            sid, params = form.spec_values(spec)
            loaded[sid] = params
        except ValueError as e:
            st.error(f"{spec}: {e}")
    picked = st.multiselect(label, ORDER, [s for s in ORDER if s in loaded], format_func=strategy_label, key=f"{kp}_sids",
                            help="Each strategy once. They always run in the order E1 E2 E3 E5 E4 E7 E6 E8 E9, then "
                                 "your own (E10 and up, made in the Strategies tab).")
    out = []
    for sid in picked:
        with st.expander(strategy_label(sid), expanded=True):
            info = form.info(sid, DEFS)
            st.info(f"{info['what']}\n\n**Earlier result:** {info['found']}")
            spec = form.build_spec(sid, param_widgets(kp, sid, loaded.get(sid, {})))
            st.caption(f"Written to the config as `{spec}`")
        out.append(spec)
    return out


def turn_limit(kp: str, current: int | None, label: str, help: str) -> int | None:
    on = st.checkbox(label, bool(current), key=f"{kp}_limit_on", help=help)
    if not on:
        return None
    return st.slider("turn limit", 1, 20, int(current or 6), key=f"{kp}_limit",
                     help="Tool-calling turns before the run stops. The baseline uses 8; the accepted ABC config uses 6 with E9.")


def factor_card(row: dict[str, Any], rev: int) -> dict[str, Any]:
    kp = f"r{rev}f{row['id']}"
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 6, 1], vertical_alignment="bottom")
        name = c1.text_input("letter", row["name"], key=f"{kp}_name",
                             help="Cells are named by the letters that are on. A fraction needs single letters.")
        note = c2.text_input("what this factor tests", row["note"], key=f"{kp}_note",
                             help="A note for the pre-registration, e.g. the cost it aims at.")
        if c3.button("Remove", key=f"{kp}_rm"):
            st.session_state.rows = [r for r in st.session_state.rows if r["id"] != row["id"]]
            st.rerun()
        specs = strategy_editor(kp, row["specs"], "strategies this factor switches on")
        turns = turn_limit(kp, row["max_turns"], "this factor also sets a turn limit",
                           "For E9's limit variant: with the factor on, runs stop after this many turns.")
    return {"id": row["id"], "name": name.strip(), "specs": specs, "max_turns": turns, "note": note.strip()}


def add_factor_row(rev: int) -> None:
    a1, a2 = st.columns([4, 1], vertical_alignment="bottom")
    sid = a1.selectbox("add a factor for", ["(empty factor)", *ORDER], key=f"add_sid{rev}",
                       format_func=lambda s: strategy_label(s) if s in ORDER else s,
                       help="The new factor gets the next free letter and this strategy at its default settings; "
                            "change both in its card.")
    if a2.button("Add factor", key="add_factor", type="primary"):
        rows = st.session_state.rows
        rows.append({"id": st.session_state.row_seq, "name": form.next_letter([r["name"] for r in rows]),
                     "specs": [form.default_spec(sid)] if sid in ORDER else [], "max_turns": None, "note": ""})
        st.session_state.row_seq += 1
        st.rerun()


def import_factors_box(rev: int) -> None:
    with st.expander("Add factors from another config (for example H, T, R from p10c, or A, B, C as the base)"):
        files = {f.name: f for f in control.config_files()}
        pick = st.selectbox("take factors from", list(files), key=f"imp_file{rev}")
        source = yaml.safe_load(files[pick].read_text(encoding="utf-8")) or {}
        factors = source.get("factors") or {}
        if not factors:
            st.write("This config has no factors.")
            return

        def label(n: str) -> str:
            limit = f", turn limit {factors[n]['max_turns']}" if factors[n].get("max_turns") else ""
            return f"{n}: {' ; '.join(factors[n].get('specs', [])) or 'no strategies'}{limit}"

        names = st.multiselect("factors", list(factors), key=f"imp_names{rev}_{pick}", format_func=label)
        where = st.radio("add them", ["as factors", "to the base"], horizontal=True, key=f"imp_where{rev}",
                         captions=["Switches this experiment tests. A letter already in use gets the next free one.",
                                   "Always on, in every run and the baseline (e.g. an accepted config to build on)."])
        if st.button("Add to this design", key="imp_add", disabled=not names):
            st.session_state.pending_import = (source, names, where == "to the base")
            st.rerun()


def factor_multiselect(label: str, key: str, ids: list[int], initial: list[int], names: dict[int, str], help: str) -> list[int]:
    """A multiselect over factor rows (by id, shown by letter) that survives renaming and removing factors."""
    st.session_state[key] = [i for i in st.session_state.get(key, initial) if i in ids]
    return st.multiselect(label, ids, key=key, format_func=lambda i: names.get(i) or "?", help=help)


def design_kind(d: dict[str, Any], rows: list[dict[str, Any]], rev: int) -> tuple[dict[str, str], list[str]]:
    """The design type and what it needs: generators for a fraction, the factors of a confirmation."""
    names = {r["id"]: r["name"] for r in rows}
    by_name = {r["name"]: r["id"] for r in rows}
    kinds = list(form.KINDS)
    kind = st.radio("design", kinds, kinds.index(form.kind_of(d)), horizontal=True, key=f"kind{rev}",
                    format_func=form.KINDS.get, captions=[form.KIND_HELP[k] for k in kinds])
    generators: dict[str, str] = {}
    confirm_only: list[str] = []
    if kind == "fraction":
        st.info("For each factor you want to save passes on, pick the factors whose product it is switched with "
                "(at least two). Example: with A, B, C free and D = ABC, 4 factors need 8 cells instead of 16; "
                "D's effect is then mixed with the ABC interaction, and two-factor interactions with each other. "
                "The preview below lists what is mixed (aliased).")
        for r in rows:
            word = d.get("generators", {}).get(r["name"], "")
            picked = factor_multiselect(f"{r['name'] or '?'} is switched as the product of", f"gen{rev}_{r['id']}",
                                        [i for i in names if i != r["id"]], [by_name[g] for g in word if g in by_name],
                                        names, "Leave empty for a free factor.")
            if picked:
                generators[r["name"]] = "".join(names[i] for i in picked)
    elif kind == "confirm":
        picked = factor_multiselect("factors to confirm together", f"co{rev}", list(names),
                                    [by_name[f] for f in d.get("confirm_only", []) if f in by_name], names,
                                    "These factors, all on, are compared with the base. Use it for a lead from an "
                                    "earlier experiment; fresh runs only.")
        confirm_only = [names[i] for i in picked]
        if not picked:
            st.warning("Pick the factors to confirm, or the design runs as a full factorial.")
    return generators, confirm_only


def design_tab(path: Path) -> None:
    d: dict[str, Any] = st.session_state.draft
    rev = st.session_state.draft_rev
    if d.get("design") != "factorial":
        st.info("This config is the original p10 plan (singles, E3 grid, greedy). Edit it as raw YAML below; "
                "the form is for factorial designs. Pick experiments-p10b/c/d.yaml or press New design.")
        raw_yaml_editor(path)
        return
    if pending := st.session_state.pop("pending_import", None):
        # Applied to the form as it stood on the last run, then loaded as a new revision of the form.
        try:
            set_draft(form.import_factors(st.session_state.form_data, *pending), st.session_state.form_header,
                      st.session_state.draft_text, st.session_state.draft_src)
            d, rev = st.session_state.draft, st.session_state.draft_rev
        except ValueError as e:
            st.error(f"Not added: {e}")
    n_tasks, fold_sizes = tasks_info()
    profiles = load_llm_config().profiles
    known_groups = groups()
    with st.expander("How a design works", expanded=not d.get("factors")):
        st.markdown(form.HOW_IT_WORKS)

    st.subheader("1. Experiment")
    c1, c2 = st.columns(2)
    name = c1.text_input("name", d.get("experiment", ""), key=f"name{rev}",
                         help="Runs are grouped as <name>-f-<cell>-r<repeat>. An experiment that has already run "
                              "cannot be changed; save a variant under a new name.")
    llm = c2.selectbox("LLM profile", list(profiles), index=list(profiles).index(d["llm"]) if d.get("llm") in profiles else 0,
                       key=f"llm{rev}", help="From configs/llm.yaml. Results of different models are not comparable.")
    modes = list(form.RUN_MODES)
    mode = st.radio("tool calls", modes, modes.index(d.get("mode", "replay")), horizontal=True, key=f"mode{rev}",
                    captions=[form.RUN_MODES[m] for m in modes])
    s1, s2 = st.columns(2)
    parallel = s1.slider("passes running at once", 1, 8, int(d.get("parallel", 3)), key=f"par{rev}",
                         help="Each pass with its MCP servers takes about 0.6 to 1 GB of RAM. More passes finish sooner, "
                              "but each LLM call gets slower on the free tier.")
    max_turns = s2.slider("turn limit when nothing else sets one", 1, 20, int(d.get("max_turns", 8)), key=f"mt{rev}",
                          help="Tool-calling turns per task. A limit on the base or on a factor that is on overrides it.")
    learn = st.multiselect("runs the learned strategies learn from", sorted(set(known_groups) | set(d.get("learn_from", []))),
                           d.get("learn_from", []), key=f"learn{rev}",
                           help="E1, E4, E5, E6 and E9 learn rules from earlier runs. Tasks are split in two folds and "
                                "each fold learns only from the other fold's runs of these groups.")

    st.subheader("2. Base")
    st.caption("On in every run of the experiment, including its baseline (`none`) and the live baseline. "
               "Leave it empty to compare against no strategies at all.")
    base = d.get("base") or {}
    with st.container(border=True):
        base_specs = strategy_editor(f"r{rev}base", base.get("specs", []), "strategies always on")
        base_turns = turn_limit(f"r{rev}base", base.get("max_turns"), "the base sets a turn limit",
                                "Every run stops after this many turns, whatever the factors.")
        base_note = st.text_input("note", base.get("note", ""), key=f"bnote{rev}",
                                  help="Where the base comes from, e.g. the Decision that accepted it.")

    st.subheader("3. Factors")
    st.caption("Each factor is one switch. Give it a letter and pick what it switches on.")
    rows = [factor_card(row, rev) for row in list(st.session_state.rows)]
    add_factor_row(rev)
    import_factors_box(rev)

    st.subheader("4. Design")
    generators, confirm_only = design_kind(d, rows, rev)
    subsets = st.multiselect("also report effects on", ["grounded"], d.get("report_subsets", []), key=f"sub{rev}",
                             help="grounded = the tasks whose prompt names a dataset entity. The choice still uses all tasks.")

    st.subheader("5. Decision rule")
    st.caption("Fixed before the run. A factor is adopted only if it saves tokens per success without breaking these limits.")
    gate = d.get("gate", {"max_success_drop": 0.02, "min_trust": 0.9})
    h1, h2, h3 = st.columns(3)
    drop_now = float(gate["max_success_drop"]) * 100
    drop = h1.slider("largest success drop allowed (points)", 0.0, max(10.0, drop_now), drop_now, 0.5, key=f"drop{rev}",
                     help="With 2, a config that solves 80% where the base solves 82% still passes.")
    trust = h2.slider("lowest mean trust allowed", 0.5, 1.0, max(0.5, float(gate["min_trust"])), 0.01, key=f"trust{rev}",
                      help="Trust (0 to 1) scores how well answers and transformations agree with their evidence. "
                           "The baseline runs at about 0.97; ABC at about 0.92.")
    repeats = h3.slider("confirmation repeats", 1, max(6, int(d.get("confirm_repeats", 3))), int(d.get("confirm_repeats", 3)),
                        key=f"rep{rev}", help="Paired repeats of the chosen config and of the base. One run per arm "
                                              "detects only changes of about 14%; three, about 8.5%.")
    g1, g2 = st.columns(2)
    live_fold = g1.radio("fold for the live check", ["A", "B"], "AB".index(d.get("live_fold", "A")), horizontal=True,
                         key=f"lf{rev}", format_func=lambda f: f"{f} ({fold_sizes.get(f, '?')} tasks)",
                         help="After the confirmation, the chosen config and the base run once against the real servers on this fold.")
    require_saving = g2.toggle("adopt only if the confirmation itself shows a saving", bool(d.get("require_saving")),
                               key=f"rs{rev}", help="Also requires the confirmation's tokens-per-success interval to lie "
                                                    "below 0. Use it when confirming a lead found after the fact.")
    boot = d.get("bootstrap", {"samples": 2000, "seed": 7})
    with st.expander("Seeds and bootstrap"):
        k1, k2, k3 = st.columns(3)
        seed = k1.number_input("pass order seed", 0, 10_000, int(d.get("order_seed", 7)), key=f"seed{rev}",
                               help="Cells run in a shuffled order so time-of-day effects do not line up with a factor.")
        samples = k2.number_input("bootstrap samples", 100, 20_000, int(boot["samples"]), 100, key=f"bs{rev}",
                                  help="Resamples of tasks behind each 95% interval.")
        bseed = k3.number_input("bootstrap seed", 0, 10_000, int(boot["seed"]), key=f"bseed{rev}")

    st.subheader("6. Pre-registration")
    header = st.text_area("why this experiment, what it tests and how the choice is made (saved as the config's leading comment)",
                          st.session_state.draft_header, key=f"hdr{rev}", height=140)

    data: dict[str, Any] = {"design": "factorial", "experiment": name.strip(), "llm": llm, "mode": mode,
                            "max_turns": int(max_turns), "parallel": int(parallel), "learn_from": learn}
    base_out: dict[str, Any] = {"specs": base_specs}
    if base_turns:
        base_out["max_turns"] = base_turns
    if base_note.strip():
        base_out["note"] = base_note.strip()
    if base_specs or base_turns:
        data["base"] = base_out
    data["factors"] = form.factors_from(rows)
    if generators:
        data["generators"] = generators
    if confirm_only:
        data["confirm_only"] = confirm_only
    if require_saving:
        data["require_saving"] = True
    if subsets:
        data["report_subsets"] = subsets
    data.update(confirm_repeats=int(repeats), live_fold=live_fold, order_seed=int(seed),
                gate={"max_success_drop": round(drop / 100, 4), "min_trust": round(trust, 2)},
                bootstrap={"samples": int(samples), "seed": int(bseed)})

    st.subheader("7. Check and save")
    row_problems = form.row_problems(rows)
    cfg, problems = (None, row_problems) if row_problems else validate(data)
    for p in problems:
        st.error(p)
    used = {form.spec_values(s)[0] for s in [*base_specs, *(s for r in rows for s in r["specs"])]}
    learned = sorted(s for s in used if form.needs_runs(s, DEFS))
    if not learn and learned:
        st.warning(f"{', '.join(learned)} learn from earlier runs or cite them as evidence, but no runs are picked in section 1.")
    st.session_state.form_data, st.session_state.form_header = data, header
    if cfg:
        preview(cfg, n_tasks, fold_sizes)
    save_box(cfg, data, header, path)

    with st.expander("The YAML this form saves"):
        st.code(control.dump_config(data, header), language="yaml")
    with st.expander("Edit the saved file as raw YAML"):
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


# --- strategies -----------------------------------------------------------------------------


@st.cache_data(ttl=60)
def catalog_tools() -> tuple[list[str], list[str]]:
    """Server names and namespaced tool names, from the latest catalog snapshot."""
    from toolhelper.mcp.catalog import latest_snapshot

    try:
        snap = latest_snapshot(CATALOG_DIR)
        if snap:
            return list(snap.servers), [t.namespaced for s in snap.servers.values() for t in s.tools]
        return list(load_servers_config()), []
    except Exception:
        return [], []


def definition_form(sid: str, d: custom.Definition | None, k: str) -> None:
    """Fields of one user-defined strategy; saves to configs/strategies.yaml."""
    defs = DEFS
    used = custom.configs_using(sid) if d else []
    locked = bool(used)
    if used:
        st.warning(f"{sid} is used by {', '.join(used)}, so it can no longer be changed or deleted "
                   "(their runs must stay reproducible). Make a new strategy for a variant.")
    c1, c2 = st.columns([1, 2])
    name = c1.text_input("name", d.name if d else "", key=f"{k}_name", disabled=locked, placeholder="answer early")
    what = c2.text_input("what it is for", d.what if d else "", key=f"{k}_what", disabled=locked,
                         help="Shown in the design form's info box, like the descriptions of E1 to E9.")
    kinds = list(custom.KINDS)
    kind = st.radio("building block", kinds, kinds.index(d.kind) if d else 0, horizontal=True, key=f"{k}_kind", disabled=locked,
                    format_func=lambda x: x.replace("_", " "), captions=[custom.KINDS[x] for x in kinds])
    fields: dict[str, Any] = {}
    if kind == "tool_filter":
        servers, tools = catalog_tools()
        hide_servers, hide_tools = (d.hide_servers, d.hide_tools) if d else ([], [])
        fields["hide_servers"] = st.multiselect("servers to hide", sorted({*servers, *hide_servers}), hide_servers,
                                                key=f"{k}_servers", disabled=locked,
                                                help="None of the server's tools are offered to the model.")
        fields["hide_tools"] = st.multiselect("single tools to hide", sorted({*tools, *hide_tools}), hide_tools,
                                              key=f"{k}_tools", disabled=locked)
        st.caption("Evidence recorded with each run: the schema hash of every hidden tool.")
    else:
        label = ("text added to the task's context" if kind == "instruction"
                 else "what the model is asked at the turn limit, with no tools")
        fields["text"] = st.text_area(label, d.text if d else "", key=f"{k}_text", disabled=locked, height=110)
        if kind == "instruction":
            options = list(custom.EVIDENCE_RUNS)
            fields["evidence_runs"] = st.radio(
                "earlier runs that motivate it (cited as evidence)", options, options.index(d.evidence_runs) if d else 0,
                key=f"{k}_ev", disabled=locked, horizontal=True, format_func=custom.EVIDENCE_RUNS.get,
                help="Every change a strategy makes must cite evidence. An instruction cites runs from the "
                     "experiment's learning groups; without any such run it refuses to start.")
        else:
            st.caption("Evidence: the learning runs that ended at the turn limit. Use it in place of E9, and give "
                       "the base or factor a turn limit.")
    new = custom.Definition(name=name.strip(), what=what.strip(), kind=kind, **fields)
    b1, b2 = st.columns([1, 5])
    if b1.button("Save changes" if d else f"Create {sid}", key=f"{k}_save", disabled=locked, type="primary"):
        if problems := new.problems():
            for p in problems:
                st.error(p)
        else:
            custom.save({**defs, sid: new})
            st.rerun()
    if d and b2.button("Delete", key=f"{k}_del", disabled=locked):
        custom.save({s: x for s, x in defs.items() if s != sid})
        st.rerun()


def strategies_tab() -> None:
    defs = DEFS
    st.caption("Everything a base or a factor can switch on. E1 to E9 are built in; E10 and up are yours.")
    st.dataframe(pd.DataFrame([{"id": sid, "name": form.info(sid, defs)["name"],
                                "kind": "built in" if sid in BUILTIN else defs[sid].kind.replace("_", " "),
                                "what it does": form.info(sid, defs)["what"]} for sid in ORDER]),
                 width="stretch", hide_index=True)
    new_id = custom.next_id(defs)
    st.subheader(f"Make your own strategy ({new_id})")
    st.info("A new strategy is put together from one building block; no code is needed. It gets the next free "
            "number and then appears in the Design tab next to E1 to E9, for the base or any factor. "
            "Your strategies run after the built-in ones. Variants of a built-in strategy (say E6 with another "
            "budget) need no new strategy: add a factor for it in the Design tab.")
    with st.container(border=True):
        definition_form(new_id, None, f"defnew{len(defs)}_{new_id}")
    if defs:
        st.subheader("Your strategies")
    for sid, d in defs.items():
        with st.expander(f"{sid} {d.name} ({d.kind.replace('_', ' ')})"):
            definition_form(sid, d, f"def_{sid}")


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
        b1, b2 = st.columns(2)
        if b1.button("New design", help="An empty factorial design that keeps this config's run settings "
                                        "(LLM, learning runs, gate, seeds)."):
            set_draft(form.blank_like(st.session_state.draft), "", path.read_text(encoding="utf-8"), str(path))
            st.rerun()
        if b2.button("Reload", help="Discard the form's unsaved changes and load the file again."):
            load_draft(path)
            st.rerun()
        st.divider()
        for f in files:
            exp = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("experiment", "?")
            state, info = status_of(exp)
            st.write(f"`{exp}`: {state}")
    saved = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    design, strategies, run, progress, results = st.tabs(["Design", "Strategies", "Run", "Progress", "Results"])
    with design:
        design_tab(path)
    with strategies:
        strategies_tab()
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
