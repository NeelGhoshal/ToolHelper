"""FDA NDC Directory (product.txt inside ndctext.zip): brand or generic name -> generic, product NDC."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

from toolhelper.datasets import loaders
from toolhelper.datasets.base import EntityRecord, Row, read_tsv

KEEP = ["PRODUCTID", "PRODUCTNDC", "PRODUCTTYPENAME", "PROPRIETARYNAME", "NONPROPRIETARYNAME",
        "DOSAGEFORMNAME", "ROUTENAME", "LABELERNAME", "SUBSTANCENAME"]


class Loader(loaders.Loader):
    name = "ndc"

    def rows(self, raw_dir: Path) -> Iterator[Row]:
        with zipfile.ZipFile(raw_dir / "ndctext.zip") as z:
            # Latin-1 and unquoted; a couple of stray quote characters must not merge rows.
            text = io.TextIOWrapper(z.open("product.txt"), encoding="latin-1", newline="")
            for r in read_tsv(text, "product.txt", KEEP, quoting=csv.QUOTE_NONE):
                generic = r["NONPROPRIETARYNAME"].strip()
                if not generic:
                    continue
                canonical, key, ndc = generic.lower(), r["PRODUCTID"], r["PRODUCTNDC"]
                records = [EntityRecord(surface=generic, type="drug_product", canonical=canonical,
                                        identifier=ndc, source="ndc:NONPROPRIETARYNAME", row_key=key)]
                if brand := r["PROPRIETARYNAME"].strip():
                    records.insert(0, EntityRecord(surface=brand, type="drug_product", canonical=canonical,
                                                   identifier=ndc, source="ndc:PROPRIETARYNAME", row_key=key))
                yield Row(row_key=key, data={k: r[k] for k in KEEP}, records=records)
