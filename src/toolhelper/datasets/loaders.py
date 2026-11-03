"""Loader registry. Each loader reads one dataset's downloaded files and yields Rows."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from toolhelper.datasets.base import ColumnError, DatasetConfig, Row

__all__ = ["ColumnError", "Loader", "get_loader"]


class Loader:
    name = "base"

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg

    def rows(self, raw_dir: Path) -> Iterator[Row]:
        raise NotImplementedError

    def source_version(self, raw_dir: Path) -> str | None:
        """The publisher's version, when the files carry one (else the fetch date is used)."""
        return None


def get_loader(cfg: DatasetConfig) -> Loader:
    from toolhelper.datasets import clinvar, countries, hgnc, ndc, omop_vocab, rxnorm, who_gho

    known = {m.Loader.name: m.Loader for m in (ndc, rxnorm, hgnc, countries, who_gho, clinvar, omop_vocab)}
    if cfg.loader not in known:
        raise KeyError(f"unknown loader {cfg.loader!r}; known: {sorted(known)}")
    return known[cfg.loader](cfg)
