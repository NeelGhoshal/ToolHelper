"""ClinVar variant_summary: gene + protein or cDNA change -> HGVS name and rsID.

Only GRCh38 rows whose name carries a protein change are kept (each variant appears once
per assembly). Surfaces: "BRAF p.Val600Glu", "BRAF V600E" (missense only) and "BRAF
c.1799T>A". The dataset is configured exact/normalized only: a near-miss spelling of a
variant is a different variant.
"""

from __future__ import annotations

import csv
import gzip
import re
from collections.abc import Iterator
from pathlib import Path

from toolhelper.datasets import loaders
from toolhelper.datasets.base import EntityRecord, Row, read_tsv

FILE = "variant_summary.txt.gz"
KEEP = ["#AlleleID", "Type", "Name", "GeneSymbol", "HGNC_ID", "ClinicalSignificance", "RS# (dbSNP)",
        "Assembly", "ReviewStatus", "VariationID", "OtherIDs"]
NAME = re.compile(r"^(?P<tx>[^(\s]+)\((?P<gene>[^)]+)\):(?P<c>c\.\S+) \((?P<p>p\.[^)]+)\)$")
MISSENSE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")
AA = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
      "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
      "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}


def one_letter(protein: str) -> str | None:
    """p.Val600Glu -> V600E; None for anything but a plain missense change."""
    m = MISSENSE.match(protein)
    if not m or m[1] not in AA or m[3] not in AA:
        return None
    return f"{AA[m[1]]}{m[2]}{AA[m[3]]}"


class Loader(loaders.Loader):
    name = "clinvar"

    def rows(self, raw_dir: Path) -> Iterator[Row]:
        with gzip.open(raw_dir / FILE, "rt", encoding="utf-8", newline="") as f:
            for r in read_tsv(f, FILE, KEEP, quoting=csv.QUOTE_NONE):
                if r["Assembly"] != "GRCh38" or not (m := NAME.match(r["Name"])):
                    continue
                gene, rs, key = m["gene"], r["RS# (dbSNP)"], r["VariationID"]
                identifier = f"rs{rs}" if rs not in ("", "-1") else f"clinvar:{key}"
                surfaces = [("protein", f"{gene} {m['p']}"), ("cdna", f"{gene} {m['c']}")]
                if short := one_letter(m["p"]):
                    surfaces.insert(0, ("protein_short", f"{gene} {short}"))
                yield Row(row_key=key, data={k: r[k] for k in KEEP}, records=[
                    EntityRecord(surface=s, type="variant", canonical=r["Name"], identifier=identifier,
                                 source=f"clinvar:{field}", row_key=key) for field, s in surfaces])
