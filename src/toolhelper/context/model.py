"""Context-store model mirroring WarmHub: named things with immutable versions.

- A wref is `name`, `name@HEAD` or `name@vN`. Names are unique among active things.
- Writes: `Add`, `Revise` (full data replacement; identical data is a no-op),
  `Retract` (new inactive version, optional reason) and `Reaffirm` (edits an
  assertion's pinned `affirmed_targets` without touching data).
- Bare wrefs in wref-typed fields are pinned to the target's current version at
  write time, so every dependency names the exact version it was built on. A link
  is stale once its target has a newer version or is retracted.
- Evidence attached to a write becomes an `Evidence` assertion about the version
  the write produced.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from toolhelper.context.shapes import wref_fields

WREF_RE = re.compile(r"^(?P<name>[^@]+?)(?:@(?P<mod>HEAD|ALL|v[1-9][0-9]*))?$")


class ContextError(Exception):
    pass


class NotFoundError(ContextError):
    pass


class NameTakenError(ContextError):
    pass


class ConflictError(ContextError):
    """`expected_version` did not match the current version."""


class ValidationError(ContextError):
    pass


def parse_wref(wref: str) -> tuple[str, int | None]:
    """Returns (name, version); version is None for a floating reference."""
    m = WREF_RE.match(wref)
    if not m:
        raise ValidationError(f"invalid wref {wref!r}")
    mod = m.group("mod")
    return m.group("name"), int(mod[1:]) if mod and mod.startswith("v") else None


def pin(name: str, version: int) -> str:
    return f"{name}@v{version}"


def validate_name(name: str) -> None:
    if not name or name.startswith("/") or name.endswith("/") or "//" in name or "@" in name:
        raise ValidationError(f"invalid name {name!r}")
    if name.startswith("hub/"):
        raise ValidationError(f"the hub/ prefix is reserved for shapes: {name!r}")


class EvidenceRef(BaseModel):
    kind: Literal["dataset_row", "run", "schema_hash"]
    ref: str
    detail: dict[str, Any] | None = None


class Add(BaseModel):
    op: Literal["add"] = "add"
    name: str
    shapes: list[str]
    data: dict[str, Any] = Field(default_factory=dict)
    affirmed_targets: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class Revise(BaseModel):
    op: Literal["revise"] = "revise"
    name: str
    # None keeps the current shapes (WarmHub's keep_shapes).
    shapes: list[str] | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    # Complete set, as in WarmHub: omitting it clears the snapshot.
    affirmed_targets: list[str] = Field(default_factory=list)
    expected_version: int | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class Retract(BaseModel):
    op: Literal["retract"] = "retract"
    name: str
    reason: str | None = None
    expected_version: int | None = None


class Reaffirm(BaseModel):
    op: Literal["reaffirm"] = "reaffirm"
    name: str
    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)
    expected_version: int | None = None


Operation = Add | Revise | Retract | Reaffirm


class Thing(BaseModel):
    name: str
    version: int
    active: bool
    shapes: list[str]
    data: dict[str, Any]
    about: str | None = None
    affirmed: list[str] = Field(default_factory=list)
    reason: str | None = None
    message: str | None = None
    committed_at: str | None = None

    @property
    def kind(self) -> str:
        return "assertion" if self.about else "thing"

    @property
    def wref(self) -> str:
        return pin(self.name, self.version)


class Link(BaseModel):
    source: str
    source_version: int
    field: str
    target: str
    target_version: int
    target_head: int
    target_active: bool

    @property
    def stale(self) -> bool:
        return not self.target_active or self.target_head > self.target_version


class StaleThing(BaseModel):
    thing: Thing
    links: list[Link]


class CommitResult(BaseModel):
    message: str
    written: list[str] = Field(default_factory=list)  # pinned wrefs of new versions
    unchanged: list[str] = Field(default_factory=list)  # no-op revises
    warnings: list[str] = Field(default_factory=list)
    # Things whose links this commit made stale.
    stale: list[StaleThing] = Field(default_factory=list)


# Shapes whose links record what they are about, not what they depend on.
NON_DEPENDENT_SHAPES = {"Evidence"}


class ContextStore(Protocol):
    def apply(self, message: str, operations: list[Operation]) -> CommitResult: ...
    def head(self, *, shape: str | None = None, match: str | None = None,
             include_retracted: bool = False, limit: int | None = None) -> list[Thing]: ...
    def get(self, wref: str, version: int | None = None, *, include_retracted: bool = False) -> Thing: ...
    def history(self, name: str) -> list[Thing]: ...
    def refs(self, wref: str, *, direction: Literal["inbound", "outbound"] = "inbound") -> list[Link]: ...
    def search(self, text: str, *, shape: str | None = None, limit: int | None = None) -> list[Thing]: ...
    def stale(self, *, shape: str | None = None) -> list[StaleThing]: ...


def glob_regex(pattern: str) -> re.Pattern[str]:
    """WarmHub name globs: `*` matches one segment, `**` any number, `{a,b}` alternatives."""
    out, i = "", 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif c == "*":
            out += "[^/]*"
            i += 1
        elif c == "{":
            end = pattern.index("}", i)
            out += "(?:" + "|".join(re.escape(p) for p in pattern[i + 1:end].split(",")) + ")"
            i = end + 1
        else:
            out += re.escape(c)
            i += 1
    return re.compile(f"^{out}$")


def wref_values(shapes: list[str], data: dict[str, Any]) -> list[tuple[str, str]]:
    """(field, wref) pairs for every wref-typed field across a thing's shapes."""
    pairs = []
    fields: dict[str, bool] = {}
    for shape in shapes:
        fields.update(wref_fields(shape))
    for field, is_array in fields.items():
        value = data.get(field)
        if value is None:
            continue
        for item in (value if is_array else [value]):
            pairs.append((field, item))
    return pairs


def pin_data(shapes: list[str], data: dict[str, Any], head_version) -> dict[str, Any]:
    """Pins bare wrefs in wref-typed fields to `head_version(name)`; raises NotFoundError
    if a target does not exist (WarmHub enforces referential integrity on wref fields)."""
    fields: dict[str, bool] = {}
    for shape in shapes:
        fields.update(wref_fields(shape))
    out = dict(data)

    def resolve(wref: str) -> str:
        name, version = parse_wref(wref)
        head = head_version(name)
        if head is None:
            raise NotFoundError(f"wref target {name!r} not found")
        if version is not None and version > head:
            raise NotFoundError(f"{wref!r}: {name} has no version {version}")
        return pin(name, version or head)

    for field, is_array in fields.items():
        value = out.get(field)
        if value is None:
            continue
        out[field] = [resolve(v) for v in value] if is_array else resolve(value)
    return out


def evidence_name(target: str, version: int, index: int) -> str:
    return f"evidence/{target}/v{version}/{index}"


def evidence_add(target: str, version: int, index: int, ev: EvidenceRef) -> Add:
    data: dict[str, Any] = {"about": pin(target, version), "kind": ev.kind, "ref": ev.ref}
    if ev.detail is not None:
        data["detail_json"] = json.dumps(ev.detail, sort_keys=True)
    return Add(name=evidence_name(target, version, index), shapes=["Evidence"], data=data,
               affirmed_targets=[pin(target, version)])


def group_stale(store: ContextStore, links: list[Link], shape: str | None = None) -> list[StaleThing]:
    """Groups stale links by source thing, skipping non-dependent shapes."""
    links = [link for link in links if link.stale]
    out = []
    for source in dict.fromkeys(link.source for link in links):
        thing = store.get(source)
        if NON_DEPENDENT_SHAPES & set(thing.shapes) or (shape and shape not in thing.shapes):
            continue
        out.append(StaleThing(thing=thing, links=[link for link in links if link.source == source]))
    return out
