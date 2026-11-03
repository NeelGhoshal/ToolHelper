"""Cuts small real excerpts of the downloaded datasets into tests/fixtures/datasets/.

Every fixture row is copied verbatim from the real file (headers included); nothing is
hand-written. Run after `toolhelper datasets fetch`:

    uv run python scripts/make_dataset_fixtures.py
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path

from toolhelper.datasets.base import RAW_DIR as RAW

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "datasets"

NDC_BRANDS = {"zoloft", "zepbound", "tylenol"}
NDC_GENERICS = {"sertraline hydrochloride"}
NDC_PRODUCTS = {"47781-951"}  # sertraline (Alvogen); Medical MCP returns another product for it
HGNC_SYMBOLS = {"ERBB2", "BRAF", "TP53", "ABCA1", "HEATR6"}  # ABC1 is an alias of both ABCA1 and HEATR6
COUNTRIES = {"JPN", "BRA", "IND", "WLD"}  # WLD (World) is an aggregate the loader must skip
RXNORM = {"IN": {"metformin", "sertraline", "acetaminophen"}, "BN": {"Zoloft", "Tylenol"}}
CLINVAR = {("BRAF", "p.Val600Glu"), ("BRAF", "p.Val600Lys"), ("TP53", "p.Arg175His")}


def excerpt_tsv(lines, keep) -> str:
    header = next(lines)
    rows = [line for line in lines if keep(dict(zip(header.rstrip("\n").split("\t"), line.rstrip("\n").split("\t"))))]
    return header + "".join(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(RAW / "ndc" / "ndctext.zip") as z:
        text = io.TextIOWrapper(z.open("product.txt"), encoding="latin-1", newline="")
        product = excerpt_tsv(iter(text), lambda r: r["PROPRIETARYNAME"].lower() in NDC_BRANDS
                              or r["PRODUCTNDC"] in NDC_PRODUCTS
                              or (r["NONPROPRIETARYNAME"].lower() in NDC_GENERICS and r["PROPRIETARYNAME"].lower() == "sertraline"))
    (OUT / "ndc").mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT / "ndc" / "ndctext.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("product.txt", product.encode("latin-1"))

    with open(RAW / "hgnc" / "hgnc_complete_set.txt", encoding="utf-8", newline="") as f:
        hgnc = excerpt_tsv(iter(f), lambda r: r["symbol"] in HGNC_SYMBOLS)
    (OUT / "hgnc").mkdir(exist_ok=True)
    (OUT / "hgnc" / "hgnc_complete_set.txt").write_text(hgnc, encoding="utf-8", newline="")

    meta, countries = json.loads((RAW / "countries" / "worldbank_countries.json").read_text(encoding="utf-8"))
    (OUT / "countries").mkdir(exist_ok=True)
    (OUT / "countries" / "worldbank_countries.json").write_text(
        json.dumps([meta, [c for c in countries if c["id"] in COUNTRIES]], indent=1), encoding="utf-8")

    (OUT / "rxnorm").mkdir(exist_ok=True)
    for tty, names in RXNORM.items():
        data = json.loads((RAW / "rxnorm" / f"allconcepts_{tty}.json").read_text(encoding="utf-8"))
        kept = [c for c in data["minConceptGroup"]["minConcept"] if c["name"] in names]
        (OUT / "rxnorm" / f"allconcepts_{tty}.json").write_text(
            json.dumps({"minConceptGroup": {"minConcept": kept}}, indent=1), encoding="utf-8")

    with gzip.open(RAW / "clinvar" / "variant_summary.txt.gz", "rt", encoding="utf-8", newline="") as f:
        clinvar = excerpt_tsv(iter(f), lambda r: any(r["GeneSymbol"] == g and f"({p})" in r["Name"] for g, p in CLINVAR))
    (OUT / "clinvar").mkdir(exist_ok=True)
    with gzip.open(OUT / "clinvar" / "variant_summary.txt.gz", "wt", encoding="utf-8", newline="") as f:
        f.write(clinvar)

    # Eunomia: the heart-attack concepts, one LOINC concept, one non-standard (ICD10CM) row the
    # loader must skip, their synonyms and the vocabulary versions, in the original zip layout.
    keep = {"4329847", "4163874", "3016723", "35208414"}
    with zipfile.ZipFile(RAW / "omop_eunomia" / "GiBleed_5.3.zip") as z:
        def cut(member, test):
            lines = io.TextIOWrapper(z.open("GiBleed_5.3/" + member), encoding="utf-8", newline="").read()
            lines = lines.splitlines(keepends=True)
            return lines[0] + "".join(line for line in lines[1:] if test(line))

        members = {
            "CONCEPT.csv": cut("CONCEPT.csv", lambda line: line.split(",", 1)[0] in keep),
            "CONCEPT_SYNONYM.csv": cut("CONCEPT_SYNONYM.csv", lambda line: line.split(",", 1)[0] in keep),
            "VOCABULARY.csv": cut("VOCABULARY.csv", lambda line: line.split(",", 1)[0] in {"SNOMED", "LOINC", "RxNorm"}),
        }
    (OUT / "omop_eunomia").mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT / "omop_eunomia" / "GiBleed_5.3.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for member, text in members.items():
            z.writestr("GiBleed_5.3/" + member, text)

    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(ROOT)}  {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
