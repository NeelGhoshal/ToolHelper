"""WarmHub adapter: the ContextStore interface over the WarmHub SDK, plus `publish`.

Writes translate one-to-one to the SDK's Add/Revise/Retract/Reaffirm. Inbound refs
and staleness are computed from pinned wrefs in head data, the same way for any
store, instead of relying on the wire shape of `thing.refs`. `publish` ran against a
live repository on 2026-11-11; the tests use a local store and a fake SDK repo as remote.
"""

from __future__ import annotations

import re
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from toolhelper.context.model import (
    ConflictError,
    Add,
    CommitResult,
    ContextStore,
    Link,
    NotFoundError,
    Operation,
    Retract,
    Revise,
    StaleThing,
    Thing,
    evidence_add,
    evidence_name,
    group_stale,
    parse_wref,
    pin,
    pin_data,
    wref_values,
)
from toolhelper.context.shapes import SHAPES, has_status, wref_fields

def scan_links(store: ContextStore) -> list[Link]:
    everything = {t.name: t for t in store.head(include_retracted=True)}
    links = []
    for t in everything.values():
        if not t.active:
            continue
        for field, wref in wref_values(t.shapes, t.data) + [("affirmed", a) for a in t.affirmed]:
            name, version = parse_wref(wref)
            target = everything.get(name)
            if target is None or version is None:
                continue
            links.append(Link(source=t.name, source_version=t.version, field=field, target=name,
                              target_version=version, target_head=target.version, target_active=target.active))
    return links


DURABLE_ID = re.compile(r"[0-9A-Z]{40,}")

# WarmHub takes `affirmed_targets` only on things whose shape composes its built-in Assertion shape.
SHAPE_COMPOSES = {"Evidence": ["Assertion"]}


class WarmHubStore:
    def __init__(self, locator: str, client: Any = None) -> None:
        if client is None:
            from warmhub import WarmHubClient

            client = WarmHubClient.from_env()  # reads WH_TOKEN
        self.wh = client
        self.org, self.repo_name = locator.split("/", 1)
        self.repo = client.repository(locator)
        self._names: dict[str, str] = {}  # durable id -> name

    # ---------- writes ----------

    def apply(self, message: str, operations: list[Operation]) -> CommitResult:
        import warmhub

        result = CommitResult(message=message)
        sdk_ops: list[Any] = []
        written: dict[str, int] = {}  # versions this commit creates: later operations may pin them

        def head_version(name: str) -> int | None:
            return written.get(name) or self._head_version(name)

        for op in operations:
            current = self._maybe_get(op.name)
            if isinstance(op, Add):
                data = pin_data(op.shapes, op.data, head_version)
                sdk_ops.append(warmhub.Add(op.name, shapes=op.shapes, data=data,
                                           **_opt(affirmed_targets=op.affirmed_targets)))
                version = 1
            elif isinstance(op, Revise):
                if current is None:
                    raise NotFoundError(f"no active thing named {op.name!r}")
                shapes = op.shapes if op.shapes is not None else current.shapes
                data = pin_data(shapes, op.data, head_version)
                if (shapes, data, op.affirmed_targets) == (current.shapes, current.data, current.affirmed):
                    result.unchanged.append(current.wref)
                    continue
                sdk_ops.append(warmhub.Revise(op.name, shapes=shapes, data=data,
                                              **_opt(affirmed_targets=op.affirmed_targets,
                                                     expected_version=op.expected_version)))
                version = current.version + 1
            elif isinstance(op, Retract):
                sdk_ops.append(warmhub.Retract(op.name, **_opt(reason=op.reason,
                                                               expected_version=op.expected_version)))
                version = (current.version if current else 0) + 1
            else:
                sdk_ops.append(warmhub.Reaffirm(op.name, **_opt(add=op.add, remove=op.remove,
                                                                expected_version=op.expected_version)))
                version = (current.version if current else 0) + 1
            result.written.append(pin(op.name, version))
            written[op.name] = version
            if isinstance(op, (Add, Revise)):
                for i, ev in enumerate(op.evidence):
                    e = evidence_add(op.name, version, i, ev)
                    sdk_ops.append(warmhub.Add(e.name, shapes=e.shapes, data=e.data,
                                               affirmed_targets=e.affirmed_targets))
                    result.written.append(pin(e.name, 1))
        if sdk_ops:
            submitted = self.repo.apply(message, sdk_ops, shapes=SHAPES)
            # WarmHub applies what it can and reports the rest per operation; an unread report hides them.
            failed = [o for o in submitted.operations if o.status == "error"]
            if failed:
                first = "; ".join(f"{o.submitted_name or o.name}: {o.error.message if o.error else 'error'}"
                                  for o in failed[:3])
                raise ConflictError(f"{len(failed)} of {len(sdk_ops)} operations failed ({first})")
        return result

    def ensure_shapes(self, names: list[str]) -> list[str]:
        """Creates missing shapes in the repository; returns the names created."""
        from warmhub import WarmHubError

        created = []
        for name in names:
            composes = SHAPE_COMPOSES.get(name, [])
            try:
                shape = self.repo.things.get(f"hub/shapes/{name}")
            except WarmHubError as e:
                if getattr(e, "kind", None) != "NOT_FOUND":
                    raise
                self.wh.shape.create(self.org, self.repo_name, name, SHAPES[name], event_request_id=str(uuid4()),
                                     **_opt(composes=composes))
                created.append(name)
                continue
            if set(composes) - set((shape.data or {}).get("composes") or []):
                self.wh.shape.revise(self.org, self.repo_name, name, composes=composes, event_request_id=str(uuid4()))
        return created

    # ---------- reads ----------

    def _maybe_get(self, name: str) -> Thing | None:
        try:
            return self.get(name)
        except NotFoundError:
            return None

    def _head_version(self, name: str) -> int | None:
        thing = self._maybe_get(name)
        return thing.version if thing else None

    def get(self, wref: str, version: int | None = None, *, include_retracted: bool = False) -> Thing:
        from warmhub import WarmHubError

        name, pinned = parse_wref(wref)
        try:
            return self._thing(self.repo.things.get(name, **_opt(version=version or pinned,
                                                               include_retracted=include_retracted or None)))
        except WarmHubError as e:
            if getattr(e, "kind", None) == "NOT_FOUND":
                raise NotFoundError(f"{wref!r} not found") from e
            raise

    def head(self, *, shape: str | None = None, match: str | None = None, include_retracted: bool = False,
             limit: int | None = None) -> list[Thing]:
        res = self.repo.things.head(**_opt(shape=shape, match=match, include_retracted=include_retracted or None,
                                           limit=limit, exclude_infra_shapes=True))
        return [self._thing(i) for i in res.items]

    def _thing(self, item: Any) -> Thing:
        """An SDK item as a Thing in the local store's terms: shapes without their version, and
        references as `name@vN` (WarmHub returns them as `durableId@vN`)."""
        shapes = [parse_wref(s)[0] for s in getattr(item, "declared_shapes", None) or []]
        data = dict(getattr(item, "data", None) or {})
        for shape in shapes:
            for field, is_array in wref_fields(shape).items():
                if data.get(field) is not None:
                    data[field] = [self._named(w) for w in data[field]] if is_array else self._named(data[field])
        about = getattr(item, "about_wref", None)
        return Thing(name=item.name, version=item.version, active=bool(getattr(item, "active", True)),
                     shapes=shapes, data=data, about=self._named(about) if about else None,
                     affirmed=[self._named(w) for w in getattr(item, "affirmed_wrefs", None) or []])

    def _named(self, wref: str) -> str:
        target, version = parse_wref(wref)
        if DURABLE_ID.fullmatch(target):
            if target not in self._names:
                self._names[target] = self.repo.things.get(target, include_retracted=True).name
            target = self._names[target]
        return pin(target, version) if version is not None else target

    def history(self, name: str) -> list[Thing]:
        res = self.wh.thing.history(self.org, self.repo_name, wref=name, include_retracted=True)
        return sorted((self._thing(i) for i in res.items), key=lambda t: t.version)

    def search(self, text: str, *, shape: str | None = None, limit: int | None = None) -> list[Thing]:
        return [self._thing(i) for i in self.repo.things.search(text, **_opt(shape=shape, limit=limit)).items]

    def refs(self, wref: str, *, direction: Literal["inbound", "outbound"] = "inbound") -> list[Link]:
        name, _ = parse_wref(wref)
        key = "target" if direction == "inbound" else "source"
        return [l for l in scan_links(self) if getattr(l, key) == name]

    def stale(self, *, shape: str | None = None) -> list[StaleThing]:
        return group_stale(self, scan_links(self), shape)


def _opt(**kwargs: Any) -> dict[str, Any]:
    """Drops unset (None or empty) keyword arguments so the SDK sees UNSET."""
    return {k: v for k, v in kwargs.items() if v not in (None, [], ())}


# ---------- publish ----------

DEFAULT_PUBLISH_SHAPES = ["McpServer", "ToolSchema", "ToolCard", "Dataset", "EntityHint", "Recipe",
                          "Projection", "CachePolicy", "Decision"]
# Records a published thing cites (a Decision's evidence is RunSummaries, which pin their Experiment).
# They go out only with something that pins them, and without the status check: an Experiment is a
# frozen plan that nothing ever accepts, and its own status travels with it.
SUPPORT_SHAPES = ["Experiment", "RunSummary"]


class PublishPlan(CommitResult):
    skipped: dict[str, str] = Field(default_factory=dict)


def publish(local: ContextStore, remote: ContextStore, *, shapes: list[str] | None = None,
            dry_run: bool = False, message: str = "toolhelper publish") -> PublishPlan:
    """Pushes curated local things (accepted, not stale) and their evidence to `remote`,
    targets before dependents, with pinned versions remapped to the remote's numbering.
    Experiments and run summaries are pushed when a published thing pins them."""
    shapes = shapes or DEFAULT_PUBLISH_SHAPES
    plan = PublishPlan(message=message)
    stale = {s.thing.name for s in local.stale()}
    candidates: dict[str, Thing] = {}
    for t in local.head():
        if not set(t.shapes) & set(shapes):
            continue
        if any(has_status(s) for s in t.shapes) and t.data.get("status") != "accepted":
            plan.skipped[t.name] = "not accepted"
        elif t.name in stale:
            plan.skipped[t.name] = "stale"
        else:
            candidates[t.name] = t
    support = {t.name: t for t in local.head()
               if set(t.shapes) & set(SUPPORT_SHAPES) and t.name not in candidates and t.name not in stale}

    remote_version: dict[str, int] = {}
    ops: list[Operation] = []

    def remap(wref: str) -> str:
        name, _ = parse_wref(wref)
        if name in remote_version:
            return pin(name, remote_version[name])
        current = _maybe(remote, name)
        if current is None:
            raise NotFoundError(name)
        return pin(name, current.version)

    def visit(name: str, path: tuple[str, ...] = ()) -> None:
        if name in remote_version or name in plan.skipped or name in path:
            return
        thing = candidates.get(name) or support[name]
        deps = [parse_wref(w)[0] for _, w in wref_values(thing.shapes, thing.data)]
        for dep in deps:
            if dep in candidates or dep in support:
                visit(dep, path + (name,))
        try:
            data = _remap_data(thing, remap)
        except NotFoundError as e:
            plan.skipped[name] = f"depends on {e.args[0]}, which is neither published nor publishable"
            return
        current = _maybe(remote, name)
        if current is None:
            ops.append(Add(name=name, shapes=thing.shapes, data=data))
            remote_version[name] = 1
        elif (current.shapes, current.data) == (thing.shapes, data):
            plan.unchanged.append(current.wref)
            remote_version[name] = current.version
            _publish_evidence(local, thing, current.version, ops, plan, published=remote)
            return
        else:
            ops.append(Revise(name=name, shapes=thing.shapes, data=data, expected_version=current.version))
            remote_version[name] = current.version + 1
        plan.written.append(pin(name, remote_version[name]))
        _publish_evidence(local, thing, remote_version[name], ops, plan)

    for name in candidates:
        visit(name)
    if ops and not dry_run:
        remote.apply(message, ops)
    return plan


def _publish_evidence(local: ContextStore, thing: Thing, version: int, ops: list[Operation],
                      plan: PublishPlan, published: ContextStore | None = None) -> None:
    """Adds a thing's evidence; with `published` (the thing is already there), only what is missing."""
    for ev in local.head(shape="Evidence", match=f"evidence/{thing.name}/v{thing.version}/*"):
        index = int(ev.name.rsplit("/", 1)[1])
        target = pin(thing.name, version)
        if published is not None and _maybe(published, evidence_name(thing.name, version, index)) is not None:
            continue
        ops.append(Add(name=evidence_name(thing.name, version, index), shapes=["Evidence"],
                       data={**ev.data, "about": target}, affirmed_targets=[target]))
        plan.written.append(pin(evidence_name(thing.name, version, index), 1))


def _remap_data(thing: Thing, remap) -> dict[str, Any]:
    data = dict(thing.data)
    for shape in thing.shapes:
        for field, is_array in wref_fields(shape).items():
            if data.get(field) is not None:
                data[field] = [remap(w) for w in data[field]] if is_array else remap(data[field])
    return data


def _maybe(store: ContextStore, name: str) -> Thing | None:
    try:
        return store.get(name)
    except NotFoundError:
        return None
