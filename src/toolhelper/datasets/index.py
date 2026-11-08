"""Entity index over all loaded datasets (SQLite at <datasets dir>/index.sqlite).

`resolve(surface, type=None)` tries, in order:

- exact: the surface equals a known surface, ignoring case;
- normalized: equal after `normalize` (case, accents and punctuation ignored);
- fuzzy: rapidfuzz `ratio` >= 85 (whole-string edit similarity, so typos match but a shared
  word does not: WRatio scored "heart attack" ~ "Heart (Bovine), Lung (Bovine)" at 85)
  over candidates from an FTS5 trigram index (only for datasets with `fuzzy: true`).

Matches are grouped by (type, canonical): Zoloft's four NDC products are one answer. If
the best type has two different answers that score alike, the status is `ambiguous` and
both are returned with their evidence rather than guessing. A surface that is the
canonical name itself (an approved symbol) beats one that is only a synonym.
Every match carries its dataset pinned to the version it came from and the source row.

Storage: one `rows` entry per source row (its type, canonical, identifier and the evidence
columns, zlib-compressed JSON) and one small `names` entry per surface pointing at it.
ClinVar alone has ~10M surfaces, so nothing per-row is repeated per surface.
"""

from __future__ import annotations

import json
import sqlite3
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from rapidfuzz import fuzz

from toolhelper.datasets.base import Row, normalize

FUZZY_MIN = 85
SCORES = {"exact": 100.0, "normalized": 95.0}
SYNONYM_PENALTY = 2.0  # surface is a synonym, not the canonical name itself
FUZZY_SCALE = 0.9  # fuzzy scores stay below every exact or normalized match
AMBIGUOUS_MARGIN = {"exact": 0.5, "normalized": 0.5, "fuzzy": 3.0}

SCHEMA = """
CREATE TABLE datasets (id INTEGER PRIMARY KEY, name TEXT UNIQUE, wref TEXT, sha256 TEXT, rows INTEGER,
                       priority INTEGER, fuzzy INTEGER);
CREATE TABLE rows (id INTEGER PRIMARY KEY, dataset_id INTEGER, row_key TEXT, type TEXT, canonical TEXT,
                   identifier TEXT, data BLOB);
CREATE INDEX rows_dataset ON rows (dataset_id);
CREATE INDEX rows_identifier ON rows (identifier);
CREATE TABLE names (id INTEGER PRIMARY KEY, row_id INTEGER, surface TEXT, norm TEXT, source TEXT);
CREATE INDEX names_norm ON names (norm);
CREATE INDEX names_row ON names (row_id);
CREATE VIRTUAL TABLE names_fts USING fts5(text, content='', contentless_delete=1, tokenize='trigram');
"""

Method = Literal["exact", "normalized", "fuzzy"]


class Match(BaseModel):
    type: str
    canonical: str
    identifier: str
    surface: str  # the known surface that matched
    source: str
    row_key: str
    dataset: str  # pinned wref, e.g. datasets/hgnc@v1
    row: dict[str, Any]
    rows: int  # source rows giving this same answer
    method: Method
    score: float


class Resolution(BaseModel):
    query: str
    status: Literal["exact", "normalized", "fuzzy", "ambiguous", "none"]
    matches: list[Match]

    @property
    def best(self) -> Match | None:
        return self.matches[0] if self.matches else None


class EntityIndex:
    def __init__(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        # Only a new file needs the schema (and a write lock): lookups must work while a
        # long build (ClinVar) holds the writer. WAL lets readers run alongside it.
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name = 'names'").fetchone():
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
        elif not self.db.execute("SELECT 1 FROM sqlite_master WHERE name = 'rows_identifier'").fetchone():
            try:  # added after the first real build; skipped if a build holds the writer
                with self.db:
                    self.db.execute("CREATE INDEX rows_identifier ON rows (identifier)")
            except sqlite3.OperationalError:
                pass

    def close(self) -> None:
        self.db.close()

    def dataset(self, name: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT wref, sha256, rows FROM datasets WHERE name = ?", (name,)).fetchone()
        return dict(zip(("wref", "sha256", "rows"), row)) if row else None

    def datasets(self) -> list[dict[str, Any]]:
        cur = self.db.execute("SELECT name, wref, sha256, rows FROM datasets ORDER BY priority")
        return [dict(zip(("name", "wref", "sha256", "rows"), r)) for r in cur]

    def set_wref(self, name: str, wref: str) -> None:
        with self.db:
            self.db.execute("UPDATE datasets SET wref = ? WHERE name = ?", (wref, name))

    def replace(self, name: str, rows: Iterable[Row], *, wref: str, sha256: str, priority: int,
                fuzzy: bool, batch: int = 20_000) -> int:
        """Swaps a dataset's rows for new ones in one transaction. Returns the row count."""
        db, count = self.db, 0
        with db:
            db.execute("INSERT INTO datasets (name, wref, sha256, rows, priority, fuzzy) VALUES (?, ?, ?, 0, ?, ?) "
                       "ON CONFLICT (name) DO UPDATE SET wref = excluded.wref, sha256 = excluded.sha256, "
                       "priority = excluded.priority, fuzzy = excluded.fuzzy",
                       (name, wref, sha256, priority, int(fuzzy)))
            (dataset_id,) = db.execute("SELECT id FROM datasets WHERE name = ?", (name,)).fetchone()
            old = "SELECT n.id FROM names n JOIN rows r ON r.id = n.row_id WHERE r.dataset_id = ?"
            ids = [i for (i,) in db.execute(old, (dataset_id,))]
            for start in range(0, len(ids), 50_000):
                chunk = ids[start:start + 50_000]
                db.execute(f"DELETE FROM names_fts WHERE rowid IN ({','.join('?' * len(chunk))})", chunk)
            db.execute("DELETE FROM names WHERE row_id IN (SELECT id FROM rows WHERE dataset_id = ?)", (dataset_id,))
            db.execute("DELETE FROM rows WHERE dataset_id = ?", (dataset_id,))
            next_row = db.execute("SELECT COALESCE(MAX(id), 0) FROM rows").fetchone()[0] + 1
            next_name = db.execute("SELECT COALESCE(MAX(id), 0) FROM names").fetchone()[0] + 1
            row_buf, name_buf = [], []

            def flush() -> None:
                db.executemany("INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?)", row_buf)
                db.executemany("INSERT INTO names VALUES (?, ?, ?, ?, ?)", name_buf)
                if fuzzy:
                    db.executemany("INSERT INTO names_fts (rowid, text) VALUES (?, ?)",
                                   [(n[0], n[2].casefold()) for n in name_buf])
                row_buf.clear()
                name_buf.clear()

            for row in rows:
                if not row.records:
                    continue
                answers = {(r.type, r.canonical, r.identifier) for r in row.records}
                if len(answers) != 1:
                    raise ValueError(f"{name} row {row.row_key}: records disagree on type/canonical/identifier "
                                     f"{sorted(answers)}; split it into one row per answer")
                (kind, canonical, identifier), count = answers.pop(), count + 1
                data = zlib.compress(json.dumps(row.data, ensure_ascii=False).encode("utf-8"))
                row_buf.append((next_row, dataset_id, row.row_key, kind, canonical, identifier, data))
                seen = set()
                for r in row.records:
                    if (r.surface, r.source) not in seen:
                        seen.add((r.surface, r.source))
                        name_buf.append((next_name, next_row, r.surface, normalize(r.surface), r.source))
                        next_name += 1
                next_row += 1
                if len(name_buf) >= batch:
                    flush()
            flush()
            db.execute("UPDATE datasets SET rows = ? WHERE id = ?", (count, dataset_id))
        return count

    # --- lookup -------------------------------------------------------------------------

    # Column positions used below: 0 dataset name, 1 type, 2 surface, 3 canonical, 4 identifier,
    # 5 source, 6 row_key, 7 wref, 8 priority, 9 row id, 10 data.
    COLS = ("d.name, r.type, n.surface, r.canonical, r.identifier, n.source, r.row_key, d.wref, d.priority, "
            "r.id, r.data")
    JOIN = "names n JOIN rows r ON r.id = n.row_id JOIN datasets d ON d.id = r.dataset_id"

    def _by_norm(self, key: str, type: str | None) -> list[tuple]:
        sql, args = f"SELECT {self.COLS} FROM {self.JOIN} WHERE n.norm = ?", [key]
        if type:
            sql += " AND r.type = ?"
            args.append(type)
        return self.db.execute(sql, args).fetchall()

    def _fuzzy(self, query: str, type: str | None, limit: int = 300) -> list[tuple[tuple, float]]:
        text = query.casefold()
        grams = {text[i:i + 3] for i in range(len(text) - 2)}
        if not grams:
            return []
        match = " OR ".join('"' + g.replace('"', '""') + '"' for g in grams)
        sql = (f"SELECT {self.COLS} FROM names_fts f JOIN names n ON n.id = f.rowid "
               "JOIN rows r ON r.id = n.row_id JOIN datasets d ON d.id = r.dataset_id "
               "WHERE names_fts MATCH ? AND d.fuzzy = 1")
        args: list[Any] = [match]
        if type:
            sql += " AND r.type = ?"
            args.append(type)
        sql += " ORDER BY rank LIMIT ?"
        args.append(limit)
        out = []
        for r in self.db.execute(sql, args):
            score = fuzz.ratio(text, r[2].casefold())
            if score >= FUZZY_MIN:
                out.append((r, score * FUZZY_SCALE))
        return out

    def by_identifier(self, identifier: str, *, dataset: str | None = None,
                      type: str | None = None) -> list[Match]:
        """Rows whose identifier is exactly `identifier` (an NDC, RxCUI, ISO3, concept_id ...)."""
        sql = (f"SELECT {self.COLS} FROM rows r JOIN datasets d ON d.id = r.dataset_id "
               "JOIN names n ON n.id = (SELECT MIN(id) FROM names WHERE row_id = r.id) WHERE r.identifier = ?")
        args: list[Any] = [identifier]
        if dataset:
            sql += " AND d.name = ?"
            args.append(dataset)
        if type:
            sql += " AND r.type = ?"
            args.append(type)
        return [Match(type=r[1], canonical=r[3], identifier=r[4], surface=r[2], source=r[5], row_key=r[6],
                      dataset=r[7], row=json.loads(zlib.decompress(r[10])), rows=1, method="exact", score=100.0)
                for r in self.db.execute(sql + " ORDER BY d.priority, r.id", args)]

    def resolve(self, surface: str, type: str | None = None, *, fuzzy: bool = True) -> Resolution:
        """`fuzzy=False` stops after exact and normalized matching (the fuzzy stage is the slow one)."""
        query = surface.strip()
        key = normalize(query)
        scored: list[tuple[tuple, Method, float]] = []
        for r in self._by_norm(key, type) if key else []:
            method: Method = "exact" if r[2].casefold() == query.casefold() else "normalized"
            scored.append((r, method, SCORES[method]))
        if not scored and fuzzy:
            scored = [(r, "fuzzy", s) for r, s in self._fuzzy(query, type)]
        if not scored:
            return Resolution(query=query, status="none", matches=[])

        # `raw` ranks types (match quality, then dataset priority); `score` (raw minus the synonym
        # penalty) ranks answers within a type, so an approved symbol beats another gene's alias.
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for r, method, raw in scored:
            score = raw - (SYNONYM_PENALTY if normalize(r[2]) != normalize(r[3]) else 0)
            g = groups.setdefault((r[1], r[3]), {"best": None, "score": -1.0, "raw": -1.0, "method": method,
                                                 "rows": set()})
            g["rows"].add(r[9])
            g["raw"] = max(g["raw"], raw)
            if score > g["score"]:
                g.update(best=r, score=score, method=method)
        ranked = sorted(groups.values(), key=lambda g: (-g["raw"], g["best"][8], -g["score"], g["best"][3]))
        top_type = ranked[0]["best"][1]
        same = sorted((g for g in ranked if g["best"][1] == top_type),
                      key=lambda g: (-g["score"], g["best"][3]))[:2]
        others = [g for g in ranked if g["best"][1] != top_type]
        others = [g for i, g in enumerate(others) if g["best"][1] not in {o["best"][1] for o in others[:i]}]
        top = same[0]
        ambiguous = len(same) > 1 and same[1]["score"] >= top["score"] - AMBIGUOUS_MARGIN[top["method"]]
        chosen = (same if ambiguous else same[:1]) + others

        def as_match(g: dict[str, Any]) -> Match:
            r = g["best"]
            return Match(type=r[1], canonical=r[3], identifier=r[4], surface=r[2], source=r[5], row_key=r[6],
                         dataset=r[7], row=json.loads(zlib.decompress(r[10])), rows=len(g["rows"]),
                         method=g["method"], score=round(g["score"], 2))

        return Resolution(query=query, status="ambiguous" if ambiguous else top["method"],
                          matches=[as_match(g) for g in chosen])
