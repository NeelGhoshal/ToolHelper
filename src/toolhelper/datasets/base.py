"""Shared types for dataset loaders: config, EntityRecord, rows and normalization.

A loader turns one downloaded dataset into rows. Each row is one source record (its
`row_key` and the columns kept as evidence) plus the EntityRecords it yields: the
surface forms a task might use and what they resolve to.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from toolhelper.config import ROOT

DATASETS_CONFIG = ROOT / "configs" / "datasets.yaml"
DATASETS_ENV = "TOOLHELPER_DATASETS_DIR"


def datasets_dir(env: Mapping[str, str] = os.environ) -> Path:
    """Where raw downloads and the index live: `TOOLHELPER_DATASETS_DIR` if set, else
    data/datasets/. The index is GBs and rewritten on every build, so on this machine it is
    kept outside the OneDrive-synced repo."""
    return Path(env[DATASETS_ENV]) if env.get(DATASETS_ENV) else ROOT / "data" / "datasets"


DATASETS_DIR = datasets_dir()
RAW_DIR = DATASETS_DIR / "raw"
INDEX_PATH = DATASETS_DIR / "index.sqlite"


class DatasetConfig(BaseModel):
    name: str
    loader: str
    source_url: str
    # Local file name -> download URL; None means the file is downloaded by hand (see `manual`).
    files: dict[str, str | None]
    license_note: str
    ttl_days: float
    manual: str | None = None
    fuzzy: bool = True
    vocabularies: list[str] = Field(default_factory=list)
    # Read members from this zip (one of `files`) under `member_prefix`, instead of loose files.
    archive: str | None = None
    member_prefix: str = ""
    delimiter: str = "\t"
    # Position in datasets.yaml; breaks ties between equally good matches of different types.
    priority: int = 0


def load_config(path: Path | str = DATASETS_CONFIG) -> dict[str, DatasetConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["datasets"]
    return {name: DatasetConfig(name=name, priority=i, **cfg) for i, (name, cfg) in enumerate(raw.items())}


class EntityRecord(BaseModel):
    surface: str
    type: str
    canonical: str
    identifier: str
    source: str  # "<dataset>:<field>" the surface came from, e.g. "hgnc:alias_symbol"
    row_key: str


class Row(BaseModel):
    row_key: str
    data: dict[str, Any]
    records: list[EntityRecord]


class ColumnError(ValueError):
    """A dataset file lacks a column its loader needs (the source format changed)."""


def normalize(text: str) -> str:
    """Case-, accent- and punctuation-insensitive key: "HER-2" and "her2" both give "her2"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return re.sub(r"[^0-9a-z]", "", "".join(c for c in decomposed if not unicodedata.combining(c)).casefold())


def require_columns(file: str, header: Iterable[str], required: Iterable[str]) -> None:
    missing = [c for c in required if c not in set(header)]
    if missing:
        raise ColumnError(f"{file}: missing columns {missing}; header is {list(header)}")


def read_tsv(lines: Iterable[str], file: str, required: Iterable[str], *, quoting: int = csv.QUOTE_MINIMAL,
             show_header: bool = True) -> Iterator[dict[str, str]]:
    """Tab-separated rows as dicts. Prints the real header first (README section 5: never trust
    remembered column names) and checks the loader's required columns against it."""
    csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
    reader = csv.DictReader(lines, delimiter="\t", quoting=quoting)
    header = reader.fieldnames or []
    if show_header:
        print(f"[{file}] {len(header)} columns: {header}", file=sys.stderr)
    require_columns(file, header, required)
    yield from reader


def split_multi(value: str) -> list[str]:
    """HGNC-style multi-value cells: 'A|B|C' (already unquoted by the csv reader)."""
    return [v.strip() for v in value.split("|") if v.strip()]
