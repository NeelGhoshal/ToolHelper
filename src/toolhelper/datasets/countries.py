"""Country name, ISO3 or ISO2 -> ISO 3166 alpha-3, from the World Bank country list.

README 1.3 names WHO GHO for this; its API was down (HTTP 500) on 2026-11-02, and the
World Bank list uses the same alpha-3 codes. Regional and income aggregates are skipped.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from toolhelper.datasets import loaders
from toolhelper.datasets.base import EntityRecord, Row, require_columns

FILE = "worldbank_countries.json"


class Loader(loaders.Loader):
    name = "countries"

    def rows(self, raw_dir: Path) -> Iterator[Row]:
        _, countries = json.loads((raw_dir / FILE).read_text(encoding="utf-8"))
        if countries:
            require_columns(FILE, countries[0], ["id", "iso2Code", "name", "region"])
        for c in countries:
            region = c["region"]["value"].strip()
            if region == "Aggregates":
                continue
            data = {"id": c["id"], "iso2Code": c["iso2Code"], "name": c["name"], "region": region,
                    "capitalCity": c.get("capitalCity", "")}
            yield Row(row_key=c["id"], data=data, records=[
                EntityRecord(surface=s, type="country", canonical=c["name"], identifier=c["id"],
                             source=f"worldbank:{field}", row_key=c["id"])
                for field, s in (("name", c["name"]), ("id", c["id"]), ("iso2Code", c["iso2Code"])) if s])
