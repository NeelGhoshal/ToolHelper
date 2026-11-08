"""Phase 7: dataset loaders, the entity index and Dataset versions.

Fixtures in tests/fixtures/datasets/ are real excerpts cut from the downloaded files by
scripts/make_dataset_fixtures.py (headers included).
"""

import gzip
import json
import shutil
from pathlib import Path

import pytest

from toolhelper.context.local_store import LocalStore
from toolhelper.config import ROOT
from toolhelper.datasets.base import DatasetConfig, datasets_dir, load_config, normalize
from toolhelper.datasets.build import build_dataset, fetch_dataset
from toolhelper.datasets.index import EntityIndex
from toolhelper.datasets.loaders import ColumnError, get_loader

FIXTURES = Path(__file__).parent / "fixtures" / "datasets"
CONFIG = load_config()


@pytest.fixture
def raw(tmp_path):
    shutil.copytree(FIXTURES, tmp_path / "raw")
    return tmp_path / "raw"


@pytest.fixture
def built(tmp_path, raw):
    index, store = EntityIndex(tmp_path / "index.sqlite"), LocalStore()
    for name in ("ndc", "rxnorm", "hgnc", "countries", "clinvar", "omop_eunomia"):
        build_dataset(CONFIG[name], raw / name, index, store)
    return index, store


def test_normalize_ignores_case_punctuation_and_accents():
    assert normalize("HER-2") == normalize("her2") == "her2"
    assert normalize("  Heart  Attack ") == "heartattack"
    assert normalize("Côte d’Ivoire") == "cotedivoire"


def test_her2_resolves_to_erbb2_with_hgnc_row(built):
    index, _ = built
    res = index.resolve("HER2")
    assert res.status == "exact"
    best = res.best
    assert (best.type, best.canonical, best.identifier) == ("gene", "ERBB2", "HGNC:3430")
    assert best.source == "hgnc:alias_symbol"
    assert best.dataset == "datasets/hgnc@v1"
    assert best.row["hgnc_id"] == "HGNC:3430" and "HER2" in best.row["alias_symbol"].split("|")


def test_zoloft_resolves_to_sertraline_with_ndc_row(built):
    index, _ = built
    res = index.resolve("Zoloft")
    best = res.best
    assert res.status == "exact" and best.type == "drug_product"
    assert best.canonical == "sertraline hydrochloride"
    assert best.identifier in {"58151-574", "58151-575", "58151-576", "58151-601"}
    assert best.row["PROPRIETARYNAME"] == "Zoloft" and best.rows == 4
    # RxNorm's brand entry is a separate type and stays available.
    assert {m.type for m in res.matches} >= {"drug_product", "drug_brand"}


def test_japan_resolves_to_iso3_and_aggregates_are_skipped(built):
    index, _ = built
    best = index.resolve("Japan").best
    assert (best.type, best.canonical, best.identifier) == ("country", "Japan", "JPN")
    assert best.row["id"] == "JPN" and best.dataset == "datasets/countries@v1"
    assert index.resolve("JPN").best.canonical == "Japan"
    assert index.resolve("World").status == "none"


def test_ambiguous_alias_returns_top_two_with_evidence(built):
    index, _ = built
    res = index.resolve("ABC1", type="gene")
    assert res.status == "ambiguous"
    assert {m.canonical for m in res.matches} == {"ABCA1", "HEATR6"}
    assert all(m.row and m.row_key.startswith("HGNC:") for m in res.matches)


def test_fuzzy_match_is_labelled(built):
    index, _ = built
    res = index.resolve("Zolof")
    assert res.status == "fuzzy" and res.best.canonical == "sertraline hydrochloride"
    assert 0 < res.best.score < 100
    assert index.resolve("Zolof", fuzzy=False).status == "none"  # callers can skip the slow fuzzy stage
    # A shared word is not a typo: no partial-string matches.
    assert index.resolve("Zoloft tablets").status == "none"


def test_rxnorm_ingredient_rxcui(built):
    index, _ = built
    best = index.resolve("metformin", type="drug_ingredient").best
    assert (best.canonical, best.identifier, best.source) == ("metformin", "6809", "rxnorm:IN")


def test_clinvar_variant_resolves_on_grch38_only_and_never_fuzzy(built):
    index, _ = built
    for surface in ("BRAF V600E", "BRAF p.Val600Glu", "braf c.1799T>A"):
        best = index.resolve(surface, type="variant").best
        assert best.canonical == "NM_004333.6(BRAF):c.1799T>A (p.Val600Glu)", surface
        assert best.identifier == "rs113488022" and best.row["Assembly"] == "GRCh38"
    # V600K exists and is close in spelling; ClinVar is exact/normalized only.
    assert index.resolve("BRAF V600", type="variant").status == "none"


def test_dataset_versions_follow_file_content(tmp_path, raw):
    index, store = EntityIndex(tmp_path / "index.sqlite"), LocalStore()
    first = build_dataset(CONFIG["hgnc"], raw / "hgnc", index, store)
    thing = store.get("datasets/hgnc")
    assert thing.version == 1 and thing.data["rows"] == 5 and thing.data["sha256"] == first.sha256
    assert thing.data["ttl_days"] == CONFIG["hgnc"].ttl_days
    build_dataset(CONFIG["hgnc"], raw / "hgnc", index, store)
    assert store.get("datasets/hgnc").version == 1  # same bytes: no new version

    path = raw / "hgnc" / "hgnc_complete_set.txt"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(l for l in lines if "	HEATR6	" not in l), encoding="utf-8")  # one real row fewer
    build_dataset(CONFIG["hgnc"], raw / "hgnc", index, store)
    assert store.get("datasets/hgnc").version == 2 and store.get("datasets/hgnc").data["rows"] == 4
    # The index serves only the current version's rows.
    assert index.resolve("HER2").best.dataset == "datasets/hgnc@v2"


def test_missing_column_names_the_column(tmp_path, raw):
    path = raw / "hgnc" / "hgnc_complete_set.txt"
    lines = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
    drop = lines[0].index("alias_symbol")
    path.write_text("\n".join("\t".join(c for i, c in enumerate(row) if i != drop) for row in lines),
                    encoding="utf-8")
    with pytest.raises(ColumnError, match="alias_symbol"):
        list(get_loader(CONFIG["hgnc"]).rows(raw / "hgnc"))


def test_fetch_writes_files_and_provenance(tmp_path):
    body = (FIXTURES / "hgnc" / "hgnc_complete_set.txt").read_bytes()
    seen = []

    def opener(url, path):
        seen.append(url)
        path.write_bytes(body)
        return {"Last-Modified": "Fri, 30 Oct 2026 14:04:37 GMT"}

    info = fetch_dataset(CONFIG["hgnc"], tmp_path / "hgnc", opener=opener)
    assert seen == [CONFIG["hgnc"].source_url]
    assert (tmp_path / "hgnc" / "hgnc_complete_set.txt").read_bytes() == body
    meta = json.loads((tmp_path / "hgnc" / "_fetch.json").read_text(encoding="utf-8"))
    assert meta["files"]["hgnc_complete_set.txt"]["last_modified"] == "Fri, 30 Oct 2026 14:04:37 GMT"
    assert meta["files"]["hgnc_complete_set.txt"]["sha256"] == info["files"]["hgnc_complete_set.txt"]["sha256"]


def test_manual_dataset_explains_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="athena.ohdsi.org"):
        fetch_dataset(CONFIG["omop_vocab"], tmp_path / "omop_vocab", opener=lambda url, path: {})


def test_config_lists_every_loader():
    assert set(CONFIG) == {"ndc", "rxnorm", "hgnc", "countries", "who_gho", "clinvar", "omop_eunomia",
                           "omop_vocab"}
    assert all(isinstance(c, DatasetConfig) for c in CONFIG.values())
    for cfg in CONFIG.values():
        get_loader(cfg)


def test_clinvar_fixture_is_real_excerpt():
    with gzip.open(FIXTURES / "clinvar" / "variant_summary.txt.gz", "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    assert header[:3] == ["#AlleleID", "Type", "Name"] and len(header) == 43


def test_datasets_dir_can_live_outside_the_repo(tmp_path):
    assert datasets_dir({}) == ROOT / "data" / "datasets"
    assert datasets_dir({"TOOLHELPER_DATASETS_DIR": ""}) == ROOT / "data" / "datasets"
    assert datasets_dir({"TOOLHELPER_DATASETS_DIR": str(tmp_path)}) == tmp_path


def test_heart_attack_resolves_to_omop_condition_with_concept_row(built):
    index, store = built
    res = index.resolve("heart attack")
    best = res.best
    assert res.status == "exact"
    assert (best.type, best.canonical, best.identifier) == ("omop_condition", "Myocardial infarction", "4329847")
    assert best.source == "omop_eunomia:SNOMED:synonym" and best.surface == "Heart attack"
    assert best.row["concept_id"] == "4329847" and best.row["standard_concept"] == "S"
    assert best.dataset == "datasets/omop_eunomia@v1"
    # The concept's own name is a surface too, and the version label comes from VOCABULARY.csv.
    assert index.resolve("Myocardial infarction").best.identifier == "4329847"
    assert store.get("datasets/omop_eunomia").data["version"] == (
        "LOINC=2.65; RxNorm=RxNorm 20181203; SNOMED=Snomed Release 20180731")


def test_omop_loader_keeps_only_standard_concepts(built):
    index, _ = built
    assert index.resolve("Gastrointestinal hemorrhage, unspecified").status == "none"  # ICD10CM, non-standard
    assert index.resolve("Creatinine serum/plasma").best.type == "omop_measurement"
