"""OMOP vocabularies from Athena: concept name or synonym -> standard concept_id.

Keeps standard (standard_concept = 'S'), valid concepts from the configured vocabularies.
The type is the concept's domain (`omop_condition`, `omop_measurement`, ...). Synonyms are
staged in an on-disk SQLite table first, so memory stays flat on multi-million-row files.

Two sources use this loader: Athena's full download (tab-separated, lower-case headers;
not yet seen, downloaded by hand) and OHDSI's Eunomia GiBleed sample CDM (comma-separated,
upper-case headers, inside a zip; verified 2026-11-02). Columns are matched
case-insensitively and checked against the real header on load.
"""

from __future__ import annotations

import csv
import io
import sqlite3
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from toolhelper.datasets import loaders
from toolhelper.datasets.base import EntityRecord, Row, require_columns

CONCEPT = ["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_class_id", "standard_concept",
           "concept_code", "invalid_reason"]
SYNONYM = ["concept_id", "concept_synonym_name"]
VOCABULARY = ["vocabulary_id", "vocabulary_version"]


class Loader(loaders.Loader):
    name = "omop_vocab"

    @contextmanager
    def _read(self, raw_dir: Path, file: str, required: list[str]) -> Iterator[Iterator[dict[str, str]]]:
        """Rows with lower-cased column names: Athena ships tab-separated, unquoted, lower-case
        files; Eunomia's sample CDM ships comma-separated, upper-case ones inside a zip."""
        cfg = self.cfg
        if cfg.archive:
            z = zipfile.ZipFile(raw_dir / cfg.archive)
            f = io.TextIOWrapper(z.open(cfg.member_prefix + file), encoding="utf-8", newline="")
        else:
            z, f = None, open(raw_dir / file, encoding="utf-8", newline="")
        try:
            quoting = csv.QUOTE_NONE if cfg.delimiter == "\t" else csv.QUOTE_MINIMAL
            reader = csv.reader(f, delimiter=cfg.delimiter, quoting=quoting)
            header = next(reader)
            print(f"[{file}] {len(header)} columns: {header}", file=sys.stderr)
            header = [h.strip().lower() for h in header]
            require_columns(file, header, required)
            yield (dict(zip(header, r)) for r in reader)
        finally:
            f.close()
            if z:
                z.close()

    def source_version(self, raw_dir: Path) -> str | None:
        wanted = set(self.cfg.vocabularies)
        with self._read(raw_dir, "VOCABULARY.csv", VOCABULARY) as rows:
            versions = {r["vocabulary_id"]: r["vocabulary_version"] for r in rows if r["vocabulary_id"] in wanted}
        return "; ".join(f"{v}={versions[v]}" for v in sorted(versions)) or None

    def rows(self, raw_dir: Path) -> Iterator[Row]:
        wanted = set(self.cfg.vocabularies)
        with tempfile.TemporaryDirectory() as tmp:
            db = sqlite3.connect(Path(tmp) / "synonyms.sqlite")
            db.execute("CREATE TABLE syn (concept_id TEXT, name TEXT)")
            with self._read(raw_dir, "CONCEPT_SYNONYM.csv", SYNONYM) as syn:
                batch = []
                for r in syn:
                    batch.append((r["concept_id"], r["concept_synonym_name"]))
                    if len(batch) >= 50_000:
                        db.executemany("INSERT INTO syn VALUES (?, ?)", batch)
                        batch.clear()
                db.executemany("INSERT INTO syn VALUES (?, ?)", batch)
            db.execute("CREATE INDEX syn_concept ON syn (concept_id)")
            try:
                with self._read(raw_dir, "CONCEPT.csv", CONCEPT) as concepts:
                    for r in concepts:
                        if r["vocabulary_id"] not in wanted or r["standard_concept"] != "S" or r["invalid_reason"]:
                            continue
                        key, name = r["concept_id"], r["concept_name"]
                        kind = f"omop_{r['domain_id'].lower().replace(' ', '_')}"
                        surfaces = [("concept_name", name)] + [
                            ("synonym", s) for (s,) in db.execute("SELECT name FROM syn WHERE concept_id = ?", (key,))
                            if s and s != name]
                        yield Row(row_key=key, data={k: r[k] for k in CONCEPT}, records=[
                            EntityRecord(surface=s, type=kind, canonical=name, identifier=key,
                                         source=f"{self.cfg.name}:{r['vocabulary_id']}:{field}", row_key=key)
                            for field, s in surfaces])
            finally:
                db.close()
