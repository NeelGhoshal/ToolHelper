"""WHO GHO indicators: indicator phrase -> IndicatorCode, for `get-health-statistics`.

Unverified: the GHO API returned HTTP 500 on 2026-11-01 and 2026-11-02, so the column
names below come from the API's documentation, not a real file. `require_columns` fails
loudly if the real response differs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from toolhelper.datasets import loaders
from toolhelper.datasets.base import EntityRecord, Row, require_columns

FILE = "Indicator.json"


class Loader(loaders.Loader):
    name = "who_gho"

    def rows(self, raw_dir: Path) -> Iterator[Row]:
        indicators = json.loads((raw_dir / FILE).read_text(encoding="utf-8"))["value"]
        if indicators:
            require_columns(FILE, indicators[0], ["IndicatorCode", "IndicatorName"])
        for i in indicators:
            key = i["IndicatorCode"]
            yield Row(row_key=key, data=i, records=[EntityRecord(
                surface=i["IndicatorName"], type="who_indicator", canonical=i["IndicatorName"],
                identifier=key, source="who_gho:IndicatorName", row_key=key)])
