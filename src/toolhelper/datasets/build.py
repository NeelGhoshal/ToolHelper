"""Fetching dataset files and building them into the index and the context store.

`fetch_dataset` streams each configured file to <datasets dir>/raw/<name>/ and writes
`_fetch.json` (URL, fetch time, Last-Modified, bytes, sha256). Hand-downloaded files
(no URL) are only checksummed. `build_dataset` loads the files through the dataset's
loader into the index and writes `datasets/<name>` as a Dataset thing: a new version
only when the files' sha256 changes, so evidence stays pinned to the data it came from.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from toolhelper.context.model import Add, ContextStore, NotFoundError, Revise
from toolhelper.datasets.base import DatasetConfig
from toolhelper.datasets.index import EntityIndex
from toolhelper.datasets.loaders import get_loader

FETCH_META = "_fetch.json"
USER_AGENT = "toolhelper-datasets/0.1 (+https://github.com/NeelGhoshal)"

Opener = Callable[[str, Path], dict[str, str]]


def http_opener(url: str, path: Path) -> dict[str, str]:
    """Streams `url` to `path` (never the whole file in memory). Returns response headers."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
        return dict(resp.headers)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_thing_name(name: str) -> str:
    return f"datasets/{name}"


def fetch_dataset(cfg: DatasetConfig, dest: Path, *, opener: Opener = http_opener) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    meta: dict[str, Any] = {"dataset": cfg.name, "files": {}}
    for file, url in cfg.files.items():
        path = dest / file
        if url is None:
            if not path.exists():
                raise FileNotFoundError(f"{cfg.name}: {path} is downloaded by hand. {cfg.manual or ''}")
            fetched_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
            entry = {"url": None, "fetched_at": fetched_at, "last_modified": None}
        else:
            tmp = path.with_name(path.name + ".part")
            headers = opener(url, tmp)
            tmp.replace(path)
            entry = {"url": url, "fetched_at": now,
                     "last_modified": next((v for k, v in headers.items() if k.lower() == "last-modified"), None)}
        entry.update(bytes=path.stat().st_size, sha256=sha256_file(path))
        meta["files"][file] = entry
    (dest / FETCH_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class BuildResult(BaseModel):
    name: str
    wref: str
    version: int
    sha256: str
    rows: int
    reindexed: bool


def _file_meta(cfg: DatasetConfig, raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-file sha256 and fetch details; files without `_fetch.json` are checksummed now."""
    saved = {}
    if (raw_dir / FETCH_META).exists():
        saved = json.loads((raw_dir / FETCH_META).read_text(encoding="utf-8"))["files"]
    out = {}
    for file in cfg.files:
        path = raw_dir / file
        if not path.exists():
            hint = cfg.manual if cfg.files[file] is None else f"run `toolhelper datasets fetch --only {cfg.name}`"
            raise FileNotFoundError(f"{cfg.name}: missing {path}. {hint}")
        sha = sha256_file(path)
        entry = saved.get(file, {})
        if entry.get("sha256") != sha:  # replaced by hand since the fetch
            entry = {"url": cfg.files[file], "last_modified": None,
                     "fetched_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")}
        out[file] = {**entry, "sha256": sha, "bytes": path.stat().st_size}
    return out


def _version_label(meta: dict[str, dict[str, Any]], source_version: str | None) -> str:
    if source_version:
        return source_version
    modified = [m["last_modified"] for m in meta.values() if m.get("last_modified")]
    if modified:
        return max(parsedate_to_datetime(m) for m in modified).date().isoformat()
    return max(m["fetched_at"] for m in meta.values())[:10]


def build_dataset(cfg: DatasetConfig, raw_dir: Path, index: EntityIndex, store: ContextStore) -> BuildResult:
    loader = get_loader(cfg)
    meta = _file_meta(cfg, raw_dir)
    combined = (next(iter(meta.values()))["sha256"] if len(meta) == 1 else
                hashlib.sha256("".join(f"{f}:{m['sha256']}\n" for f, m in sorted(meta.items())).encode()).hexdigest())
    name = dataset_thing_name(cfg.name)
    try:
        current = store.get(name)
    except NotFoundError:
        current = None
    indexed = index.dataset(cfg.name)

    if current and current.data["sha256"] == combined:
        wref = f"{name}@v{current.version}"
        if indexed and indexed["sha256"] == combined and indexed["wref"] == wref:
            return BuildResult(name=cfg.name, wref=wref, version=current.version, sha256=combined,
                               rows=indexed["rows"], reindexed=False)
        rows = index.replace(cfg.name, loader.rows(raw_dir), wref=wref, sha256=combined,
                             priority=cfg.priority, fuzzy=cfg.fuzzy)
        return BuildResult(name=cfg.name, wref=wref, version=current.version, sha256=combined, rows=rows,
                           reindexed=True)

    # New data: index under the version this commit will create, then record that version.
    version = current.version + 1 if current else 1
    wref = f"{name}@v{version}"
    rows = index.replace(cfg.name, loader.rows(raw_dir), wref=wref, sha256=combined, priority=cfg.priority,
                         fuzzy=cfg.fuzzy)
    data = {"name": cfg.name, "source_url": cfg.source_url, "license_note": cfg.license_note,
            "version": _version_label(meta, loader.source_version(raw_dir)), "sha256": combined, "rows": rows,
            "fetched_at": max(m["fetched_at"] for m in meta.values()), "ttl_days": cfg.ttl_days}
    # The Dataset is itself the evidence source: its sha256 identifies the exact files
    # (per-file details are in raw/<name>/_fetch.json).
    op = (Revise(name=name, shapes=["Dataset"], data=data, expected_version=current.version)
          if current else Add(name=name, shapes=["Dataset"], data=data))
    store.apply(f"dataset {cfg.name} {data['version']}", [op])
    committed = store.get(name).version
    if committed != version:  # someone else revised it meanwhile; re-pin the index to what was written
        wref = f"{name}@v{committed}"
        index.set_wref(cfg.name, wref)
    return BuildResult(name=cfg.name, wref=wref, version=committed, sha256=combined, rows=rows, reindexed=True)
