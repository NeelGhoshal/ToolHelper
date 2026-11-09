"""Moves stale recordings out of the replay store, so the next replay miss re-records them from the
real server. Nothing is deleted: each file moves to data/replay-archive/<stamp>/<server>/<tool>/ and
gets a line in data/replay-archive/manifest.jsonl with the reason and its evidence.

Selections (each can be run alone):
  who-outage   get-health-statistics answered "No health indicators found" while the WHO GHO API
               was down (HTTP 500, 2026-11-01..05; the first real data was recorded 2026-11-05T03),
               plus any later such answers from a failed live call (--include-since).
  from-list    files named in a text file (one path per line), e.g. empty PubMed answers that a
               logged HTTP 429 or a live PubMed hit count shows were transient failures.

    uv run python scripts/archive_recordings.py who-outage --before 2026-11-05T03 --reason "..."
    uv run python scripts/archive_recordings.py from-list paths.txt --reason "..." [--evidence ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "replay"
ARCHIVE = ROOT / "data" / "replay-archive"


def text_of(rec: dict) -> str:
    return "".join(b.get("text", "") for b in (rec.get("result") or {}).get("content", []))


def who_outage(before: str, include_since: str | None) -> list[Path]:
    out = []
    for f in sorted((STORE / "medical" / "get-health-statistics").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if text_of(rec).startswith("No health indicators found") and (
                rec["recorded_at"] < before or (include_since and rec["recorded_at"] >= include_since)):
            out.append(f)
    return out


def archive(files: list[Path], reason: str, evidence: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    with (ARCHIVE / "manifest.jsonl").open("a", encoding="utf-8") as manifest:
        for f in files:
            rel = f.resolve().relative_to(STORE)
            rec = json.loads(f.read_text(encoding="utf-8"))
            target = ARCHIVE / stamp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), target)
            manifest.write(json.dumps({"moved_at": stamp, "from": str(rel).replace("\\", "/"),
                                       "to": str(target.relative_to(ARCHIVE)).replace("\\", "/"),
                                       "recorded_at": rec.get("recorded_at"), "arguments": rec.get("arguments"),
                                       "text": text_of(rec)[:200], "reason": reason, "evidence": evidence}) + "\n")
    print(f"archived {len(files)} recordings to {ARCHIVE / stamp}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("who-outage")
    w.add_argument("--before", required=True, help="ISO time the API came back (recorded_at before it is outage)")
    w.add_argument("--include-since", help="also take 'No health indicators found' recorded at or after this time")
    lst = sub.add_parser("from-list")
    lst.add_argument("paths", type=Path)
    for p in (w, lst):
        p.add_argument("--reason", required=True)
        p.add_argument("--evidence", default="")
        p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "who-outage":
        files = who_outage(args.before, args.include_since)
    else:
        files = [ROOT / line.strip() if not Path(line.strip()).is_absolute() else Path(line.strip())
                 for line in args.paths.read_text(encoding="utf-8").splitlines() if line.strip()]
        files = [f for f in files if f.exists()]
    print(f"{len(files)} recordings selected")
    if not args.dry_run:
        archive(files, args.reason, args.evidence)


if __name__ == "__main__":
    main()
