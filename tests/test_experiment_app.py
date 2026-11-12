"""The experiment app's helpers (spec checks, run environment, configs, run counts) and a smoke
run of the Streamlit page itself on the repo's configs, with empty monitoring and context DBs."""

from pathlib import Path

import pytest
import yaml

from toolhelper.config import ROOT
from toolhelper.efficiency.registry import DEFAULTS, ORDER, PARAMS, check_spec
from toolhelper.experiments import control
from toolhelper.experiments import design_form as form
from toolhelper.experiments.factorial import load_config

APP = ROOT / "src" / "toolhelper" / "app" / "experiments_app.py"


def test_every_spec_in_the_repo_configs_passes_the_check():
    from toolhelper.efficiency import registry

    registry.install_custom(ROOT / "configs" / "strategies.yaml")  # configs may name the repo's own E10+
    for path in control.config_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        specs = [*data.get("base", {}).get("specs", []), *(s for f in data.get("factors", {}).values() for s in f.get("specs", [])),
                 *(s for arm in data.get("arms", {}).values() for s in arm)]
        assert specs, path
        assert [p for s in specs for p in check_spec(s)] == [], path


@pytest.mark.parametrize("spec, problem", [
    ("E10", "unknown strategy"),
    ("E2:level=tiny", "level='tiny' not in"),
    ("E6:mode=budget,budget=lots", "budget must be a whole number"),
    ("E3:modes=hints+guess", "['guess'] not in"),
    ("E4:mode=repair,top_k=3", "unknown parameter 'top_k'"),
    ("E9:x=1", "unknown parameter 'x'"),
])
def test_bad_specs_are_named(spec, problem):
    assert any(problem in p for p in check_spec(spec)), check_spec(spec)


def test_good_specs_pass():
    for spec in ["E1:top_k=8,mode=find_tools,route=on", "E3:modes=tool+rewrite,datasets=hgnc+ndc", "E7:ttl_biomcp=3600",
                 "E6:mode=budget,budget=2000", "E9"]:
        assert check_spec(spec) == [], spec


def test_the_form_rebuilds_every_factorial_spec_in_the_repo_configs_unchanged():
    from toolhelper.efficiency import registry

    registry.install_custom(ROOT / "configs" / "strategies.yaml")
    for path in control.config_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data.get("design") != "factorial":
            continue
        for spec in [*data.get("base", {}).get("specs", []), *(s for f in data["factors"].values() for s in f.get("specs", []))]:
            sid, values = form.spec_values(spec)
            if "mode" in PARAMS[sid] and "mode" not in values:
                continue  # a partial spec that only changes one setting of the base's (p10e: E6:budget=1000)
            assert form.build_spec(sid, {**form.default_values(sid), **values}) == spec, path


def test_form_specs_split_lists_and_hide_parameters_that_do_not_apply():
    assert form.spec_values("E3:modes=tool+rewrite,datasets=hgnc") == ("E3", {"modes": ["tool", "rewrite"], "datasets": ["hgnc"]})
    assert form.build_spec("E3", {"datasets": [], "modes": ["hints", "tool"]}) == "E3:modes=hints+tool"
    assert form.default_spec("E6") == "E6:mode=projection" and form.default_spec("E9") == "E9"
    assert form.shown("E6", "budget", {"mode": "budget"}) and not form.shown("E6", "budget", {"mode": "top_n"})
    assert form.shown("E6", "top_n", {"mode": "summary"}) and not form.shown("E6", "top_n", {"mode": "budget"})
    assert form.ttl_params(form.ttl_values({"ttl_biomcp": 3600})) == {"ttl_biomcp": 3600}
    assert form.build_spec("E7", form.ttl_params(form.ttl_values({}))) == "E7"


def test_every_strategy_parameter_and_choice_has_a_description_and_a_valid_default():
    for sid in ORDER:
        assert form.STRATEGIES[sid]["what"] and form.STRATEGIES[sid]["found"]
        assert check_spec(form.default_spec(sid)) == [] and set(DEFAULTS[sid]) <= set(PARAMS[sid])
        for key, allowed in PARAMS[sid].items():
            ui = form.PARAM_UI[sid, key]
            assert ui["label"], (sid, key)
            if allowed is int:
                lo, hi, _ = ui["range"]
                assert lo <= DEFAULTS[sid][key] <= hi
            elif isinstance(allowed, list) or allowed[1]:
                choices = allowed if isinstance(allowed, list) else allowed[1]
                assert set(ui["values"]) == set(choices), (sid, key)


def test_factor_rows_round_trip_and_name_their_problems():
    data = yaml.safe_load((ROOT / "configs" / "experiments-p10b.yaml").read_text(encoding="utf-8"))
    rows = form.rows_from(data["factors"])
    assert form.factors_from(rows) == data["factors"] and form.row_problems(rows) == []
    assert form.next_letter([r["name"] for r in rows]) == "E" and form.next_letter([]) == "A"
    rows += [{"id": 9, "name": "A", "specs": [], "max_turns": None, "note": ""},
             {"id": 10, "name": "", "specs": ["E9"], "max_turns": None, "note": ""},
             {"id": 11, "name": "Z", "specs": [], "max_turns": None, "note": ""}]
    problems = form.row_problems(rows)
    assert any("'A' is used 2 times" in p for p in problems) and "a factor has no letter" in problems
    assert any(p.startswith("factor Z: nothing to switch on") for p in problems)


def test_design_kind_and_a_blank_design_that_keeps_the_run_settings():
    load = lambda n: yaml.safe_load((ROOT / "configs" / f"experiments-{n}.yaml").read_text(encoding="utf-8"))  # noqa: E731
    assert [form.kind_of(load(n)) for n in ("p10b", "p10c", "p10d")] == ["fraction", "full", "confirm"]
    blank = form.blank_like(load("p10d"))
    assert blank["experiment"] == "" and blank["factors"] == {} and "base" not in blank and "confirm_only" not in blank
    assert blank["llm"] == "nim-mistral" and blank["gate"] == {"max_success_drop": 0.02, "min_trust": 0.9}


def test_run_env_fills_keys_from_the_user_env_and_puts_tool_dirs_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(control, "_windows_user_env", lambda k: {"NVIDIA_API_KEY": "from-user-env"}.get(k))
    monkeypatch.setattr(control, "tool_dirs", lambda: [tmp_path])
    env = control.run_env({"PATH": "C:\\bin", "OMOPHUB_API_KEY": "already-set"})
    assert env["NVIDIA_API_KEY"] == "from-user-env" and env["OMOPHUB_API_KEY"] == "already-set"
    assert env["PATH"].split(control.os.pathsep)[:2] == [str(tmp_path), "C:\\bin"]
    assert control.missing_requirements({"PATH": ""}, {"api_key_env": "NVIDIA_API_KEY"})[:3] == [
        "`uv` not found on PATH", "`biomcp` not found on PATH", "NVIDIA_API_KEY is not set"]


def test_a_dumped_config_round_trips_with_its_preregistration_header():
    text = (ROOT / "configs" / "experiments-p10d.yaml").read_text(encoding="utf-8")
    data, header = yaml.safe_load(text), control.leading_comment(text)
    out = control.dump_config(data, header)
    assert yaml.safe_load(out) == data and control.leading_comment(out) == header
    assert header.startswith("Phase 10, p10d")


def test_run_counts_follow_the_design():
    p10b = control.plan_runs(load_config("configs/experiments-p10b.yaml"), 54, 27)
    assert p10b == {"cells": 8, "factorial": 432, "confirm": 216, "live": 54, "total": 702}
    p10d = control.plan_runs(load_config("configs/experiments-p10d.yaml"), 54, 27)
    assert p10d == {"cells": 0, "factorial": 0, "confirm": 324, "live": 54, "total": 378}


def test_no_process_file_means_not_running(monkeypatch, tmp_path):
    monkeypatch.setattr(control, "workdir", lambda name: tmp_path / name)
    assert control.process_info("nothing") is None and not control.is_running(None)
    assert control.stop("nothing") == "nothing is not running"


def test_the_page_renders_every_config(monkeypatch, tmp_path):
    testing = pytest.importorskip("streamlit.testing.v1")
    import toolhelper.config

    monkeypatch.setattr(toolhelper.config, "MONITOR_DB", tmp_path / "monitoring.sqlite")
    monkeypatch.setattr(toolhelper.config, "CONTEXT_DB", tmp_path / "context.sqlite")
    at = testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    for name in [p.name for p in control.config_files()]:
        at.sidebar.selectbox[0].select(name).run()
        assert not at.exception, (name, at.exception)
    at.sidebar.selectbox[0].select("experiments-p10b.yaml").run()
    assert any("8 cells" in m.value for m in at.markdown), [m.value for m in at.markdown][:20]
    assert not at.error, [e.value for e in at.error]


def _app(monkeypatch, tmp_path, config):
    testing = pytest.importorskip("streamlit.testing.v1")
    import toolhelper.config

    monkeypatch.setattr(toolhelper.config, "MONITOR_DB", tmp_path / "monitoring.sqlite")
    monkeypatch.setattr(toolhelper.config, "CONTEXT_DB", tmp_path / "context.sqlite")
    at = testing.AppTest.from_file(str(APP), default_timeout=120).run()
    return at.sidebar.selectbox[0].select(config).run()


def _saved_yaml(at):
    return yaml.safe_load(next(c.value for c in at.code if "design: factorial" in c.value))


@pytest.mark.parametrize("name", ["p10b", "p10c", "p10d"])
def test_the_form_reproduces_a_loaded_config(monkeypatch, tmp_path, name):
    at = _app(monkeypatch, tmp_path, f"experiments-{name}.yaml")
    assert not at.exception, at.exception
    assert _saved_yaml(at) == yaml.safe_load((ROOT / "configs" / f"experiments-{name}.yaml").read_text(encoding="utf-8"))


def test_adding_a_factor_and_moving_a_slider_change_the_saved_yaml(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, "experiments-p10c.yaml")
    at.selectbox(key=next(w.key for w in at.selectbox if str(w.key).startswith("add_sid"))).select("E6").run()
    at.button(key="add_factor").click().run()
    assert not at.exception, at.exception
    data = _saved_yaml(at)
    assert list(data["factors"]) == ["H", "T", "R", "A"] and data["factors"]["A"] == {"specs": ["E6:mode=projection"]}
    budget = next(w for w in at.slider if str(w.key).endswith("base_E6_budget"))
    budget.set_value(1500).run()
    assert "E6:mode=budget,budget=1500" in _saved_yaml(at)["base"]["specs"]
    # A second E6 with another mode cannot merge with the base's: the check names the conflict.
    assert any("conflicting mode" in e.value for e in at.error), [e.value for e in at.error]


def test_new_design_starts_empty_with_the_run_settings(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, "experiments-p10b.yaml")
    next(b for b in at.sidebar.button if b.label == "New design").click().run()
    assert not at.exception, at.exception
    data = _saved_yaml(at)
    assert data["experiment"] == "" and data["factors"] == {} and data["llm"] == "nim-mistral"
    assert any("give the experiment a name" in e.value for e in at.error)


def test_a_strategy_made_in_the_app_is_saved_and_offered_in_the_design(monkeypatch, tmp_path):
    from toolhelper.efficiency import custom

    at = _app(monkeypatch, tmp_path, "experiments-p10c.yaml")
    at.text_input(key="defnew0_E10_name").set_value("Answer early")
    at.text_area(key="defnew0_E10_text").set_value("Answer as soon as one result holds the ID.")
    at.button(key="defnew0_E10_save").click().run()
    assert not at.exception, at.exception
    assert custom.load()["E10"].kind == "instruction" and custom.load()["E10"].name == "Answer early"
    picker = next(w for w in at.multiselect if str(w.key).endswith("base_sids"))
    assert "E10" in picker.options or any("E10" in str(o) for o in picker.options)
    picker.select("E10").run()
    assert not at.exception, at.exception
    assert _saved_yaml(at)["base"]["specs"] == ["E2:level=minimal", "E6:mode=budget,budget=2000", "E9", "E10"]
    assert not at.error, [e.value for e in at.error]


def test_factors_are_taken_over_from_another_config(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, "experiments-p10c.yaml")
    key = lambda kind, start: next(w.key for w in getattr(at, kind) if str(w.key).startswith(start))  # noqa: E731
    at.selectbox(key=key("selectbox", "imp_file")).select("experiments-p10b.yaml").run()
    at.multiselect(key=key("multiselect", "imp_names")).select("D").run()
    at.button(key="imp_add").click().run()
    assert not at.exception, at.exception
    data = _saved_yaml(at)
    assert list(data["factors"]) == ["H", "T", "R", "D"] and data["factors"]["D"]["specs"] == ["E4:mode=repair"]
    assert data["base"]["specs"] == ["E2:level=minimal", "E6:mode=budget,budget=2000", "E9"] and data["experiment"] == "p10c"
