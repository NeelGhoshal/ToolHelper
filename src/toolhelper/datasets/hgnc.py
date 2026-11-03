"""HGNC complete set: approved symbol, aliases, previous symbols and name -> approved symbol, HGNC ID."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from toolhelper.datasets import loaders
from toolhelper.datasets.base import EntityRecord, Row, read_tsv, split_multi

KEEP = ["hgnc_id", "symbol", "name", "status", "locus_group", "alias_symbol", "prev_symbol", "entrez_id",
        "ensembl_gene_id"]
SINGLE, MULTI = ("symbol", "name"), ("alias_symbol", "prev_symbol")


class Loader(loaders.Loader):
    name = "hgnc"

    def rows(self, raw_dir: Path) -> Iterator[Row]:
        with open(raw_dir / "hgnc_complete_set.txt", encoding="utf-8", newline="") as f:
            for r in read_tsv(f, "hgnc_complete_set.txt", KEEP):
                if r["status"] != "Approved":
                    continue
                key, symbol = r["hgnc_id"], r["symbol"]
                surfaces = [(field, r[field]) for field in SINGLE if r[field]]
                surfaces += [(field, s) for field in MULTI for s in split_multi(r[field])]
                records = [EntityRecord(surface=s, type="gene", canonical=symbol, identifier=key,
                                        source=f"hgnc:{field}", row_key=key) for field, s in surfaces]
                yield Row(row_key=key, data={k: r[k] for k in KEEP}, records=records)
