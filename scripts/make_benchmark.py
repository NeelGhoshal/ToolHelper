"""Writes data/tasks/benchmark.jsonl (the Phase 6 benchmark) and its CSV input.

Prompts and checks are task specifications, not response data. Expected answer strings
are only facts confirmed by recorded real responses or well established; run
`toolhelper bench check` after a record run to find expectations no tool output contains.

    uv run python scripts/make_benchmark.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tasks"

M, B, O = "medical__", "biomcp__", "omop__"
FIND, BATCH = O + "find_omop_concept", O + "batch_map_concepts_from_csv"


def task(id, servers, category, prompt, paths, *, any=None, all=None, called_with=None, answer_any=None,
         answer_all=None, traps=None):
    checks = {}
    if any:
        checks["must_call_any"] = any
    if all:
        checks["must_call_all"] = all
    if called_with:
        checks["called_with"] = [{"tool": t, "args": a} for t, a in called_with]
    if answer_any:
        checks["answer_includes_any"] = answer_any
    if answer_all:
        checks["answer_includes_all"] = answer_all
    return {"id": id, "servers": servers, "category": category, "prompt": prompt, "expected_paths": paths,
            "checks": checks, "traps": traps or []}


def omop(id, prompt, table, field, answer=None, traps=None, category="concept_mapping"):
    return task(id, ["omop"], category, prompt, [[FIND]], any=[FIND],
                called_with=[(FIND, {"omop_table": table, "omop_field": field})],
                answer_any=answer, traps=traps)


TASKS = [
    # ---------------- medical (16) ----------------
    task("med-001", ["medical"], "drug_search", "What is the generic name of Zoloft?",
         [[M + "search-drugs"], [M + "search-drug-nomenclature"]],
         any=[M + "search-drugs", M + "search-drug-nomenclature"], answer_any=["sertraline"], traps=["brand_name"]),
    task("med-002", ["medical"], "drug_label", "Pull the FDA label details for Zoloft.",
         [[M + "get-drug-details"], [M + "search-drugs", M + "get-drug-details"]],
         any=[M + "get-drug-details"], answer_any=["sertraline"], traps=["brand_name", "ndc_needed"]),
    task("med-003", ["medical"], "drug_search", "Find FDA drug products containing ibuprofen and name two manufacturers.",
         [[M + "search-drugs"]], any=[M + "search-drugs"], answer_any=["ibuprofen"]),
    task("med-004", ["medical"], "drug_label", "Get the FDA details for the drug product with NDC 47781-951.",
         [[M + "get-drug-details"]], any=[M + "get-drug-details"],
         called_with=[(M + "get-drug-details", {"ndc": "47781-951"})], answer_any=["sertraline"]),
    task("med-005", ["medical"], "health_statistics", "What is the WHO life expectancy at birth in Japan?",
         [[M + "get-health-statistics"]], any=[M + "get-health-statistics"],
         called_with=[(M + "get-health-statistics", {"country": "JPN"})], traps=["country_name"]),
    task("med-006", ["medical"], "health_statistics",
         "Get WHO infant mortality statistics for the country with ISO3 code BRA.",
         [[M + "get-health-statistics"]], any=[M + "get-health-statistics"],
         called_with=[(M + "get-health-statistics", {"country": "BRA"})]),
    task("med-007", ["medical"], "literature",
         "Find recent PubMed articles on metformin and cardiovascular outcomes and give their PMIDs.",
         [[M + "search-medical-literature"]], any=[M + "search-medical-literature"]),
    task("med-008", ["medical"], "article", "Summarize PubMed article 42957066.",
         [[M + "get-article-details"]], any=[M + "get-article-details"], answer_any=["GLP-1", "aneurysm"]),
    task("med-009", ["medical"], "rxnorm", "What is the RxNorm RxCUI for metformin?",
         [[M + "search-drug-nomenclature"]], any=[M + "search-drug-nomenclature"], answer_any=["6809"]),
    task("med-010", ["medical"], "rxnorm", "Find the RxNorm concept for Tylenol and name its active ingredient.",
         [[M + "search-drug-nomenclature"]], any=[M + "search-drug-nomenclature", M + "search-drugs"],
         answer_any=["acetaminophen"], traps=["brand_name"]),
    task("med-011", ["medical"], "guidelines", "Find clinical practice guidelines for hypertension from the AHA.",
         [[M + "search-clinical-guidelines"]], any=[M + "search-clinical-guidelines"]),
    task("med-012", ["medical"], "scholar", "Search Google Scholar for papers on BRAF V600E melanoma.",
         [[M + "search-google-scholar"]], any=[M + "search-google-scholar"]),
    task("med-013", ["medical"], "databases", "Search the medical databases for treatments of type 2 diabetes.",
         [[M + "search-medical-databases"]], any=[M + "search-medical-databases"]),
    task("med-014", ["medical"], "journals", "Find recent medical journal articles about CRISPR gene therapy.",
         [[M + "search-medical-journals"], [M + "search-medical-literature"]],
         any=[M + "search-medical-journals", M + "search-medical-literature"]),
    task("med-015", ["medical"], "literature", "Find peer-reviewed literature on statins and dementia risk.",
         [[M + "search-medical-literature"]], any=[M + "search-medical-literature"], traps=["overlapping_tools"]),
    task("med-016", ["medical"], "drug_search", "What FDA drug products contain naproxen? Give two brand names.",
         [[M + "search-drugs"]], any=[M + "search-drugs"], answer_any=["naproxen"]),

    # ---------------- biomcp (15) ----------------
    task("bio-001", ["biomcp"], "gene_pathways", "What pathways is HER2 involved in?",
         [[B + "get"], [B + "biomcp"]], any=[B + "get", B + "biomcp"], answer_any=["ERBB2"], traps=["gene_alias"]),
    task("bio-002", ["biomcp"], "variant_significance", "What is the ClinVar clinical significance of BRAF V600E?",
         [[B + "get"], [B + "search", B + "get"], [B + "search"]], any=[B + "get", B + "search", B + "biomcp"],
         answer_any=["pathogenic"]),
    task("bio-003", ["biomcp"], "trials", "Find recruiting clinical trials for BRAF V600E melanoma.",
         [[B + "search"]], any=[B + "search", B + "biomcp"], answer_any=["NCT"]),
    task("bio-004", ["biomcp"], "gene_summary", "Summarize what is known about the BRAF gene.",
         [[B + "get"]], any=[B + "get", B + "biomcp"]),
    task("bio-005", ["biomcp"], "articles", "Find research articles about EGFR mutations in lung cancer.",
         [[B + "search"]], any=[B + "search", B + "biomcp"]),
    task("bio-006", ["biomcp"], "variant_id", "What is the dbSNP rsID of BRAF V600E?",
         [[B + "get"], [B + "search"]], any=[B + "get", B + "search", B + "biomcp"], answer_any=["rs113488022"]),
    task("bio-007", ["biomcp"], "variant_normalize",
         "Normalize the HGVS expression NM_004333.6:c.1799T>A with the ClinGen Allele Registry.",
         [[B + "variant_normalize_car"]], any=[B + "variant_normalize_car"], answer_any=["CA123643"]),
    task("bio-008", ["biomcp"], "drug_targets", "Search BioMCP for the drug imatinib and list its targets.",
         [[B + "biomcp"], [B + "get"]], any=[B + "biomcp", B + "get"], answer_any=["ABL1", "BCR-ABL", "KIT"],
         traps=["search_enum"]),
    task("bio-009", ["biomcp"], "gene_diseases", "Which diseases are associated with the TP53 gene?",
         [[B + "get"]], any=[B + "get", B + "biomcp"], answer_any=["Li-Fraumeni"]),
    task("bio-010", ["biomcp"], "pgx", "What is the pharmacogenomic guidance for CYP2C19 and clopidogrel?",
         [[B + "search"]], any=[B + "search", B + "get", B + "biomcp"]),
    task("bio-011", ["biomcp"], "erepo", "List ClinGen expert curated assertions for BRAF variants.",
         [[B + "variant_erepo"]], any=[B + "variant_erepo"]),
    task("bio-012", ["biomcp"], "cspec", "Get the ClinGen CSpec variant curation specification for MYH7.",
         [[B + "gene_cspec"]], any=[B + "gene_cspec"], answer_any=["GN002"]),
    task("bio-013", ["biomcp"], "variant_articles", "Find a short literature list for the BRAF p.V600E variant.",
         [[B + "variant_articles"], [B + "search"]], any=[B + "variant_articles", B + "search", B + "biomcp"]),
    task("bio-014", ["biomcp"], "disease_genes", "Search BioMCP for melanoma as a disease and list associated genes.",
         [[B + "get"], [B + "biomcp"]], any=[B + "get", B + "biomcp"], answer_any=["BRAF"], traps=["search_enum"]),
    task("bio-015", ["biomcp"], "drug_label", "What are the FDA-approved indications of pembrolizumab?",
         [[B + "get"], [B + "biomcp"]], any=[B + "get", B + "biomcp"], answer_any=["melanoma", "lung"]),

    # ---------------- omop (13) ----------------
    omop("omop-001", "Map 'heart attack' to the OMOP field condition_occurrence.condition_concept_id.",
         "condition_occurrence", "condition_concept_id", ["4329847"]),
    omop("omop-002", "Map HbA1c to the OMOP field measurement.measurement_concept_id.",
         "measurement", "measurement_concept_id", ["3004410"]),
    omop("omop-003", "Map 'heart attack' to an OMOP concept in the condition table.",
         "condition_occurrence", "condition_concept_id", ["4329847"], traps=["wrong_table_name"]),
    omop("omop-004", "Map type 2 diabetes mellitus to condition_occurrence.condition_concept_id.",
         "condition_occurrence", "condition_concept_id", ["201826"]),
    omop("omop-005", "Map metformin to drug_exposure.drug_concept_id.", "drug_exposure", "drug_concept_id",
         ["1503297"]),
    omop("omop-006", "Map hypertension to condition_occurrence.condition_concept_id.",
         "condition_occurrence", "condition_concept_id", ["320128", "316866"]),
    omop("omop-007", "Map systolic blood pressure to measurement.measurement_concept_id.",
         "measurement", "measurement_concept_id", ["3004249"]),
    omop("omop-008", "Map appendectomy to procedure_occurrence.procedure_concept_id.",
         "procedure_occurrence", "procedure_concept_id"),
    omop("omop-009", "Map body weight to measurement.measurement_concept_id.", "measurement",
         "measurement_concept_id", ["3025315"]),
    omop("omop-010", "Map aspirin to drug_exposure.drug_concept_id.", "drug_exposure", "drug_concept_id",
         ["1112807"]),
    task("omop-011", ["omop"], "batch_mapping",
         "Map every concept listed in the CSV file at {repo}\\data\\tasks\\inputs\\conditions.csv to OMOP.",
         [[BATCH]], any=[BATCH], answer_any=["4329847"]),
    omop("omop-012", "Map asthma to condition_occurrence.condition_concept_id.", "condition_occurrence",
         "condition_concept_id", ["317009"]),
    omop("omop-013", "Map serum creatinine to an OMOP concept in the measurements table.",
         "measurement", "measurement_concept_id", ["3016723"], traps=["wrong_table_name"]),

    # ---------------- cross-server (10) ----------------
    task("x-001", ["medical", "omop"], "cross", "What is metformin's RxNorm RxCUI, and its OMOP drug_exposure concept?",
         [[M + "search-drug-nomenclature", FIND]], all=[M + "search-drug-nomenclature", FIND],
         answer_all=["6809", "1503297"]),
    task("x-002", ["medical", "omop"], "cross",
         "Find FDA drug products for sertraline, then map sertraline to OMOP drug_exposure.drug_concept_id.",
         [[M + "search-drugs", FIND]], all=[M + "search-drugs", FIND], answer_any=["739138"]),
    task("x-003", ["biomcp", "medical"], "cross",
         "Which gene is HER2, and what does recent PubMed literature say about HER2-targeted therapy?",
         [[B + "get", M + "search-medical-literature"]], all=[M + "search-medical-literature"],
         any=[B + "get", B + "search", B + "biomcp"], answer_all=["ERBB2"], traps=["gene_alias"]),
    task("x-004", ["biomcp", "medical"], "cross",
         "What is the clinical significance of BRAF V600E, and what does recent PubMed literature say about it?",
         [[B + "get", M + "search-medical-literature"]],
         all=[M + "search-medical-literature"], any=[B + "get", B + "search", B + "biomcp"],
         answer_any=["pathogenic"]),
    task("x-005", ["omop", "biomcp"], "cross",
         "Map melanoma to OMOP condition_occurrence.condition_concept_id and find recruiting melanoma trials.",
         [[FIND, B + "search"]], all=[FIND, B + "search"], answer_any=["NCT"]),
    task("x-006", ["medical", "omop"], "cross",
         "What is the RxNorm concept for acetaminophen, and its OMOP drug_exposure.drug_concept_id?",
         [[M + "search-drug-nomenclature", FIND]], all=[M + "search-drug-nomenclature", FIND],
         answer_any=["1125315"]),
    task("x-007", ["medical", "biomcp"], "cross",
         "Get FDA product information for imatinib and its gene targets from BioMCP.",
         [[M + "search-drugs", B + "biomcp"], [M + "search-drugs", B + "get"]],
         all=[M + "search-drugs"], any=[B + "biomcp", B + "get"], answer_any=["ABL1", "BCR-ABL", "KIT"]),
    task("x-008", ["medical", "omop"], "cross",
         "Find WHO statistics on diabetes prevalence in India, and map type 2 diabetes to OMOP condition_occurrence.",
         [[M + "get-health-statistics", FIND]], all=[M + "get-health-statistics", FIND],
         called_with=[(M + "get-health-statistics", {"country": "IND"})], answer_any=["201826"],
         traps=["country_name"]),
    task("x-009", ["biomcp", "medical"], "cross",
         "What pathways involve EGFR, and what FDA drug products exist for erlotinib?",
         [[B + "get", M + "search-drugs"]], all=[M + "search-drugs"], any=[B + "get", B + "biomcp"],
         answer_any=["erlotinib"]),
    task("x-010", ["omop", "medical"], "cross",
         "Map atrial fibrillation to OMOP condition_occurrence.condition_concept_id and find clinical guidelines for it.",
         [[FIND, M + "search-clinical-guidelines"]], all=[FIND, M + "search-clinical-guidelines"],
         answer_any=["313217"]),
]


def main() -> None:
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT / "inputs" / "conditions.csv").write_text(
        "keyword,omop_table,omop_field\nheart attack,condition_occurrence,condition_concept_id\n"
        "asthma,condition_occurrence,condition_concept_id\n", encoding="utf-8")
    with (OUT / "benchmark.jsonl").open("w", encoding="utf-8") as f:
        for t in TASKS:
            f.write(json.dumps(t) + "\n")
    print(f"wrote {len(TASKS)} tasks")


if __name__ == "__main__":
    main()
