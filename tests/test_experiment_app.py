"""The experiment app's helpers (spec checks, run environment, configs, run counts) and a smoke
run of the Streamlit page itself on the repo's configs, with empty monitoring and context DBs."""

from pathlib import Path

import pytest
import yaml

from toolhelper.config import ROOT
from toolhelper.efficiency.registry import check_spec
from toolhelper.experiments import control
from toolhelper.experiments.factorial import load_config

APP = ROOT / "src" / "toolhelper" / "app" / "experiments_app.py"


def test_every_spec_in_the_repo_configs_passes_the_check():
    for path in control.config_files():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        specs = [*data.get("base", {}).get("specs", []), *(s for f in data.get("factors", {}).values() for s in f["specs"]),
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
