"""RxNorm ingredients (IN) and brand names (BN) from RxNav's allconcepts: name -> RxCUI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from toolhelper.datasets import loaders
from toolhelper.datasets.base import EntityRecord, Row, require_columns

TYPES = {"IN": "drug_ingredient", "BN": "drug_brand"}


class Loader(loaders.Loader):
    name = "rxnorm"

    def rows(self, raw_dir: Path) -> Iterator[Row]:
        for tty, entity_type in TYPES.items():
            file = f"allconcepts_{tty}.json"
            concepts = json.loads((raw_dir / file).read_text(encoding="utf-8"))["minConceptGroup"]["minConcept"]
            if concepts:
                require_columns(file, concepts[0], ["rxcui", "name", "tty"])
            for c in concepts:
                key = f"{c['tty']}:{c['rxcui']}"
                yield Row(row_key=key, data=c, records=[EntityRecord(
                    surface=c["name"], type=entity_type, canonical=c["name"], identifier=c["rxcui"],
                    source=f"rxnorm:{c['tty']}", row_key=key)])
