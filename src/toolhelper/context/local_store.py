"""Local SQLite context store (the default backend).

Tables: `things` (identity and head pointer), `versions` (append-only, enforced by
triggers), `links` (one row per wref in a version, with the pinned target version)
and `commits`. Evidence is stored as `Evidence` things. Data is validated with the
WarmHub SDK's own shape validator, so local writes fail the way WarmHub's would.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from warmhub.rules import validate_against_shape

from toolhelper.context.model import (
    Add,
    CommitResult,
    ConflictError,
    Link,
    NameTakenError,
    NotFoundError,
    Operation,
    Reaffirm,
    Retract,
    Revise,
    StaleThing,
    Thing,
    ValidationError,
    evidence_add,
    glob_regex,
    group_stale,
    parse_wref,
    pin,
    pin_data,
    validate_name,
    wref_values,
)
from toolhelper.context.shapes import SHAPES

SCHEMA = """
CREATE TABLE IF NOT EXISTS commits (
    id INTEGER PRIMARY KEY, message TEXT NOT NULL, committed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS things (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, head INTEGER NOT NULL, active INTEGER NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS things_active_name ON things(name) WHERE active = 1;
CREATE TABLE IF NOT EXISTS versions (
    thing_id INTEGER NOT NULL REFERENCES things(id), version INTEGER NOT NULL,
    op TEXT NOT NULL, shapes_json TEXT NOT NULL, data_json TEXT NOT NULL,
    affirmed_json TEXT NOT NULL, active INTEGER NOT NULL, reason TEXT,
    commit_id INTEGER NOT NULL REFERENCES commits(id),
    PRIMARY KEY (thing_id, version));
CREATE TABLE IF NOT EXISTS links (
    thing_id INTEGER NOT NULL, version INTEGER NOT NULL, field TEXT NOT NULL,
    target_id INTEGER NOT NULL REFERENCES things(id), target_version INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS links_target ON links(target_id);
CREATE INDEX IF NOT EXISTS links_source ON links(thing_id, version);
"""
APPEND_ONLY = ["versions", "links", "commits"]


class LocalStore:
    def __init__(self, path: Path | str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        for table in APPEND_ONLY:
            for action in ("UPDATE", "DELETE"):
                self.db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_{action.lower()} BEFORE {action} ON {table} "
                    f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END")

    def close(self) -> None:
        self.db.close()

    # ---------- writes ----------

    def apply(self, message: str, operations: list[Operation]) -> CommitResult:
        result = CommitResult(message=message)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            commit_id = self.db.execute(
                "INSERT INTO commits (message, committed_at) VALUES (?, ?)",
                (message, datetime.now(UTC).isoformat())).lastrowid
            for op in operations:
                written = self._apply_one(op, commit_id, result)
                if written and isinstance(op, (Add, Revise)):
                    for i, ev in enumerate(op.evidence):
                        self._apply_one(evidence_add(op.name, written, i, ev), commit_id, result)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        targets = {parse_wref(w)[0] for w in result.written}
        result.stale = [s for s in self.stale() if any(l.stale and l.target in targets for l in s.links)]
        return result

    def _apply_one(self, op: Operation, commit_id: int, result: CommitResult) -> int | None:
        """Applies one operation; returns the new version, or None for a no-op."""
        if isinstance(op, Add):
            validate_name(op.name)
            if self._active_id(op.name) is not None:
                raise NameTakenError(f"{op.name!r} already exists")
            data = self._check(op.name, op.shapes, op.data, op.affirmed_targets, result)
            thing_id = self.db.execute("INSERT INTO things (name, head, active) VALUES (?, 1, 1)",
                                       (op.name,)).lastrowid
            self._insert_version(thing_id, 1, "add", op.shapes, data, op.affirmed_targets, True, None, commit_id)
            result.written.append(pin(op.name, 1))
            return 1

        thing_id, current = self._current(op.name)
        if op.expected_version is not None and op.expected_version != current.version:
            raise ConflictError(f"{op.name}: expected v{op.expected_version}, head is v{current.version}")
        version = current.version + 1

        if isinstance(op, Revise):
            shapes = op.shapes if op.shapes is not None else current.shapes
            data = self._check(op.name, shapes, op.data, op.affirmed_targets, result)
            if (shapes, data, op.affirmed_targets) == (current.shapes, current.data, current.affirmed):
                result.unchanged.append(current.wref)
                return None
            self._insert_version(thing_id, version, "revise", shapes, data, op.affirmed_targets, True, None,
                                 commit_id)
        elif isinstance(op, Retract):
            self._insert_version(thing_id, version, "retract", current.shapes, current.data, current.affirmed,
                                 False, op.reason, commit_id)
        elif isinstance(op, Reaffirm):
            affirmed = [t for t in current.affirmed if t not in op.remove]
            affirmed += [t for t in op.add if t not in affirmed]
            self._check_affirmed(op.name, current.data, affirmed)
            self._insert_version(thing_id, version, "reaffirm", current.shapes, current.data, affirmed, True,
                                 None, commit_id)
        self.db.execute("UPDATE things SET head = ?, active = ? WHERE id = ?",
                        (version, 0 if isinstance(op, Retract) else 1, thing_id))
        result.written.append(pin(op.name, version))
        return version

    def _check(self, name: str, shapes: list[str], data: dict[str, Any], affirmed: list[str],
               result: CommitResult) -> dict[str, Any]:
        if not shapes:
            raise ValidationError(f"{name}: at least one shape is required")
        data = pin_data(shapes, data, self._head_version)
        for shape in shapes:
            if shape not in SHAPES:
                raise ValidationError(f"{name}: unknown shape {shape!r}")
            verdict = validate_against_shape(data, SHAPES[shape])
            if not verdict.valid:
                raise ValidationError(f"{name} ({shape}): {'; '.join(verdict.errors)}")
            if verdict.warnings is not None:
                fields = ", ".join(verdict.warnings.undeclared_fields)
                result.warnings.append(f"{name} ({shape}): undeclared fields {fields}")
        self._check_affirmed(name, data, affirmed)
        return data

    def _check_affirmed(self, name: str, data: dict[str, Any], affirmed: list[str]) -> None:
        if not affirmed:
            return
        about = data.get("about")
        if not about:
            raise ValidationError(f"{name}: affirmed targets need an `about` reference")
        about_name = parse_wref(about)[0]
        for target in affirmed:
            target_name, version = parse_wref(target)
            if version is None or target_name != about_name:
                raise ValidationError(f"{name}: affirmed target {target!r} must be a pinned version of {about_name}")
            head = self._head_version(target_name)
            if head is None or version > head:
                raise NotFoundError(f"{name}: affirmed target {target!r} does not exist")

    def _insert_version(self, thing_id: int, version: int, op: str, shapes: list[str], data: dict[str, Any],
                        affirmed: list[str], active: bool, reason: str | None, commit_id: int) -> None:
        self.db.execute(
            "INSERT INTO versions (thing_id, version, op, shapes_json, data_json, affirmed_json, active, reason, "
            "commit_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (thing_id, version, op, json.dumps(shapes), json.dumps(data, sort_keys=True), json.dumps(affirmed),
             int(active), reason, commit_id))
        refs = wref_values(shapes, data) + [("affirmed", t) for t in affirmed]
        for field, wref in refs:
            target_name, target_version = parse_wref(wref)
            self.db.execute(
                "INSERT INTO links (thing_id, version, field, target_id, target_version) VALUES (?, ?, ?, ?, ?)",
                # A retract or reaffirm copies links whose targets may since have been retracted.
                (thing_id, version, field, self._thing_id(target_name, include_retracted=True), target_version))

    # ---------- lookups ----------

    def _active_id(self, name: str) -> int | None:
        row = self.db.execute("SELECT id FROM things WHERE name = ? AND active = 1", (name,)).fetchone()
        return row["id"] if row else None

    def _thing_id(self, name: str, include_retracted: bool) -> int | None:
        thing_id = self._active_id(name)
        if thing_id is None and include_retracted:
            row = self.db.execute("SELECT id FROM things WHERE name = ? ORDER BY id DESC LIMIT 1",
                                  (name,)).fetchone()
            thing_id = row["id"] if row else None
        return thing_id

    def _head_version(self, name: str) -> int | None:
        row = self.db.execute("SELECT head FROM things WHERE name = ? AND active = 1", (name,)).fetchone()
        return row["head"] if row else None

    def _current(self, name: str) -> tuple[int, Thing]:
        thing_id = self._active_id(name)
        if thing_id is None:
            raise NotFoundError(f"no active thing named {name!r}")
        return thing_id, self._version(thing_id, None)

    def _version(self, thing_id: int, version: int | None) -> Thing:
        row = self.db.execute(
            "SELECT t.name, t.head, v.*, c.message, c.committed_at FROM things t "
            "JOIN versions v ON v.thing_id = t.id JOIN commits c ON c.id = v.commit_id "
            "WHERE t.id = ? AND v.version = COALESCE(?, t.head)", (thing_id, version)).fetchone()
        if row is None:
            raise NotFoundError(f"thing {thing_id} has no version {version}")
        return self._row_thing(row)

    @staticmethod
    def _row_thing(row: sqlite3.Row) -> Thing:
        data = json.loads(row["data_json"])
        return Thing(name=row["name"], version=row["version"], active=bool(row["active"]),
                     shapes=json.loads(row["shapes_json"]), data=data, about=data.get("about"),
                     affirmed=json.loads(row["affirmed_json"]), reason=row["reason"],
                     message=row["message"], committed_at=row["committed_at"])

    # ---------- reads ----------

    def get(self, wref: str, version: int | None = None, *, include_retracted: bool = False) -> Thing:
        name, pinned = parse_wref(wref)
        version = version or pinned
        # A pinned version stays readable after retraction.
        thing_id = self._thing_id(name, include_retracted or version is not None)
        if thing_id is None:
            raise NotFoundError(f"{wref!r} not found")
        return self._version(thing_id, version)

    def head(self, *, shape: str | None = None, match: str | None = None, include_retracted: bool = False,
             limit: int | None = None) -> list[Thing]:
        where = "" if include_retracted else "WHERE t.active = 1"
        rows = self.db.execute(
            "SELECT t.name, t.head, v.*, c.message, c.committed_at FROM things t "
            "JOIN versions v ON v.thing_id = t.id AND v.version = t.head "
            f"JOIN commits c ON c.id = v.commit_id {where} ORDER BY t.name, t.id").fetchall()
        pattern = glob_regex(match) if match else None
        things = [self._row_thing(r) for r in rows]
        things = [t for t in things if (shape is None or shape in t.shapes)
                  and (pattern is None or pattern.match(t.name))]
        return things[:limit] if limit else things

    def history(self, name: str) -> list[Thing]:
        thing_id = self._thing_id(name, include_retracted=True)
        if thing_id is None:
            raise NotFoundError(f"{name!r} not found")
        head = self.db.execute("SELECT head FROM things WHERE id = ?", (thing_id,)).fetchone()["head"]
        return [self._version(thing_id, v) for v in range(1, head + 1)]

    def search(self, text: str, *, shape: str | None = None, limit: int | None = None) -> list[Thing]:
        needle = text.lower()
        hits = [t for t in self.head(shape=shape)
                if needle in t.name.lower() or needle in json.dumps(t.data).lower()]
        return hits[:limit] if limit else hits

    def _links(self, where: str, params: tuple) -> list[Link]:
        rows = self.db.execute(
            "SELECT s.name AS source, l.version AS source_version, l.field, tt.name AS target, "
            "l.target_version, tt.head AS target_head, tt.active AS target_active FROM links l "
            "JOIN things s ON s.id = l.thing_id AND s.head = l.version AND s.active = 1 "
            f"JOIN things tt ON tt.id = l.target_id WHERE {where} ORDER BY s.name, l.field", params).fetchall()
        return [Link(**{k: r[k] for k in r.keys()}) for r in rows]

    def refs(self, wref: str, *, direction: Literal["inbound", "outbound"] = "inbound") -> list[Link]:
        name, _ = parse_wref(wref)
        thing_id = self._thing_id(name, include_retracted=True)
        if thing_id is None:
            raise NotFoundError(f"{wref!r} not found")
        column = "l.target_id" if direction == "inbound" else "l.thing_id"
        return self._links(f"{column} = ?", (thing_id,))

    def stale(self, *, shape: str | None = None) -> list[StaleThing]:
        return group_stale(self, self._links("1 = 1", ()), shape)
