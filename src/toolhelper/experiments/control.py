"""Launching, pausing and inspecting experiment runs from outside the CLI (the Streamlit app).

`toolhelper experiment auto` runs as a detached process, so it outlives the app (and Claude
Code's shells), with the API keys and the dataset dir read from the Windows user environment
(shells and apps started before they were set do not see them) and the uv and biomcp
directories on PATH (the orchestrator launches its passes with `uv run`). Its pid and creation
time go to `<workdir>/process.json`; the creation time guards against a reused pid.

Pausing kills the process tree, as `taskkill /T /F` did by hand; the orchestrator resumes from
saved runs on the next launch (tasks in flight are redone).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from toolhelper.config import ROOT

USER_ENV_KEYS = ("NVIDIA_API_KEY", "OMOPHUB_API_KEY", "TOOLHELPER_DATASETS_DIR", "ANTHROPIC_API_KEY", "WH_TOKEN")
PROCESS_FILE = "process.json"


def workdir(experiment: str) -> Path:
    return ROOT / "data" / "experiments" / experiment


def _windows_user_env(key: str) -> str | None:
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, key)[0])
    except OSError:
        return None


def tool_dirs() -> list[Path]:
    """Directories holding uv and biomcp when they are not on PATH (README section 5, Environment notes)."""
    dirs = [Path.home() / ".local" / "bin"]
    if appdata := os.environ.get("APPDATA"):
        dirs += sorted((Path(appdata) / "Python").glob("Python3*/Scripts"))
    return [d for d in dirs if d.is_dir()]


def run_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for an experiment process; keys missing here are filled from the user env."""
    env = dict(os.environ if base is None else base)
    for key in USER_ENV_KEYS:
        if not env.get(key) and (value := _windows_user_env(key)):
            env[key] = value
    missing = [str(d) for d in tool_dirs() if str(d) not in env.get("PATH", "").split(os.pathsep)]
    env["PATH"] = os.pathsep.join([*missing, env.get("PATH", "")])
    env.setdefault("PYTHONUTF8", "1")
    return env


def missing_requirements(env: dict[str, str], llm_profile: dict[str, Any] | None = None) -> list[str]:
    """What an experiment run would fail on: no uv, no biomcp, no keys."""
    path = env.get("PATH")
    problems = [f"`{exe}` not found on PATH" for exe in ("uv", "biomcp") if not shutil.which(exe, path=path)]
    key = (llm_profile or {}).get("api_key_env")
    for k in filter(None, [key, "OMOPHUB_API_KEY", "TOOLHELPER_DATASETS_DIR"]):
        if not env.get(k):
            problems.append(f"{k} is not set")
    return problems


def free_memory_gb() -> float | None:
    """Free physical memory (long runs need ~1 GB per parallel pass)."""
    if sys.platform != "win32":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    *((f, ctypes.c_ulonglong) for f in ("total", "avail", "total_page", "avail_page", "total_virtual",
                                                        "avail_virtual", "avail_extended"))]

    status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
    return status.avail / 2**30 if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None


def onedrive_running() -> bool:
    """OneDrive syncing the run's files grew to several GB of RAM during p10/p10b (README section 5)."""
    if sys.platform != "win32":
        return False
    r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq OneDrive.exe", "/NH"], capture_output=True, text=True)
    return "onedrive.exe" in r.stdout.lower()


# --- process state --------------------------------------------------------------------------


def _creation_time(pid: int) -> float | None:
    """Process creation time (seconds since 1601 on Windows), or None when it is not running."""
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except OSError:
            return None
        return 0.0
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        return ((times[0].dwHighDateTime << 32) + times[0].dwLowDateTime) / 1e7
    finally:
        k32.CloseHandle(handle)


def process_info(experiment: str) -> dict[str, Any] | None:
    path = workdir(experiment) / PROCESS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def is_running(info: dict[str, Any] | None) -> bool:
    if not info:
        return False
    created = _creation_time(info["pid"])
    return created is not None and (info.get("created") is None or abs(created - info["created"]) < 1)


def launch(config_path: Path, experiment: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Starts (or resumes) `toolhelper experiment auto` for a config as a detached process."""
    info = process_info(experiment)
    if is_running(info):
        raise RuntimeError(f"{experiment} is already running (pid {info['pid']})")
    wd = workdir(experiment)
    wd.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out, err = wd / f"auto-{stamp}.out.log", wd / f"auto-{stamp}.err.log"
    cmd = ["uv", "run", "toolhelper", "experiment", "auto", "--config", str(config_path)]
    env = env or run_env()
    exe = shutil.which("uv", path=env.get("PATH")) or "uv"
    flags = 0x00000008 | 0x00000200 | 0x08000000 if sys.platform == "win32" else 0  # detached, own group, no window
    with open(out, "ab") as o, open(err, "ab") as e:
        proc = subprocess.Popen([exe, *cmd[1:]], cwd=ROOT, env=env, stdout=o, stderr=e, stdin=subprocess.DEVNULL,
                                creationflags=flags, start_new_session=sys.platform != "win32")
    info = {"pid": proc.pid, "created": _creation_time(proc.pid), "config": str(config_path), "command": cmd,
            "out": str(out), "err": str(err), "started_at": datetime.now(UTC).isoformat()}
    (wd / PROCESS_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def stop(experiment: str) -> str:
    """Pauses a run: kills the orchestrator with its bench passes and MCP servers."""
    info = process_info(experiment)
    if not is_running(info):
        return f"{experiment} is not running"
    if sys.platform == "win32":
        r = subprocess.run(["taskkill", "/T", "/F", "/PID", str(info["pid"])], capture_output=True, text=True)
        return (r.stdout or r.stderr).strip()
    os.killpg(os.getpgid(info["pid"]), 15)
    return f"stopped {info['pid']}"


def run_cli(args: list[str], env: dict[str, str] | None = None, timeout: float = 900) -> subprocess.CompletedProcess:
    """A short `toolhelper` command (freeze, curate) run to completion."""
    env = env or run_env()
    exe = shutil.which("uv", path=env.get("PATH")) or "uv"
    return subprocess.run([exe, "run", "toolhelper", *args], cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


# --- configs --------------------------------------------------------------------------------


def config_files() -> list[Path]:
    return sorted((ROOT / "configs").glob("experiments*.yaml"))


def dump_config(data: dict[str, Any], header: str = "") -> str:
    """YAML for a config dict; a header becomes leading `#` comment lines."""
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110)
    head = "".join(f"# {line}\n".replace("# \n", "#\n") for line in header.strip().splitlines()) if header.strip() else ""
    return head + body


def leading_comment(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        lines.append(line[1:].removeprefix(" "))
    return "\n".join(lines)


def plan_runs(cfg, n_tasks: int, live_tasks: int) -> dict[str, int]:
    """Task runs a factorial config needs, at most (the first confirmation attempt; fallbacks add more)."""
    from toolhelper.experiments.factorial import cells

    cs = cells(cfg)
    r = cfg.confirm_repeats
    if cfg.confirm_only:
        factorial, confirm = 0, 2 * r * n_tasks
    else:
        factorial, confirm = len(cs) * n_tasks, 2 * max(r - 1, 0) * n_tasks
    return {"cells": 0 if cfg.confirm_only else len(cs), "factorial": factorial, "confirm": confirm,
            "live": 2 * live_tasks, "total": factorial + confirm + 2 * live_tasks}
