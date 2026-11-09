# ToolHelper

ToolHelper monitors how LLM agents call the tools of three biomedical MCP servers (Medical MCP, BioMCP and OMOP MCP) and makes those calls cheaper, while keeping every change traceable. It follows WarmHub's principles: context that is trusted, curated, versioned, permissioned, timely and usable at scale.

It has two parts:

- **Monitoring:** exact token usage per tool call, how the model reaches each tool (its pathway), and statistics a human can act on.
- **Efficiency:** strategies, several of them backed by public datasets, that reduce the tokens needed to call these tools. Every strategy is tested in experiments, the optimum is chosen with an explicit decision log, and each change carries a trust score and evidence.

Ollama is the default LLM, and any other backend can be swapped in to rerun the same experiments.

> **Status:** build plan. This README is the specification for building the prototype phase by phase in Claude Code.
>
> **Verified on 2026-11-01:** items marked *verified* were checked against the servers' source code and the SDK versions installed from PyPI and npm on that date. Versions drift, so Phase 2 re-checks them against the live servers.
>
> **Medical data notice:** the underlying servers return information for research and education. ToolHelper measures and optimizes tool calls; it does not provide medical advice.

## Contents

1. [Verified metadata](#1-verified-metadata)
2. [Architecture](#2-architecture)
3. [Build steps](#3-build-steps)
4. [Gotchas](#4-gotchas)
5. [Working in Claude Code](#5-working-in-claude-code)
6. [References](#6-references)

---

## 1. Verified metadata

### 1.1 The three MCP servers

| | Medical MCP | BioMCP | OMOP MCP |
|---|---|---|---|
| Source | `JamesANZ/medical-mcp` (TypeScript, MIT) | `genomoncology/biomcp` (Rust, MIT) | `OHNLP/omop_mcp` (Python, Apache-2.0) |
| Version checked | npm `medical-mcp` 1.0.8 | 0.9.4 (commit of Oct 30, 2026) | 0.2.0 (commit of May 10, 2026) |
| Install | none (runs via npx) | `uv tool install biomcp-cli` | `git clone https://github.com/OHNLP/omop_mcp.git` |
| Launch (stdio) | `npx -y medical-mcp` | `biomcp serve` | `uv --directory <repo> run omop_mcp` |
| Keys | none | none for core sources | `OMOPHUB_API_KEY` |
| Tools | 10 | 7 | 2, plus 3 resources and 1 prompt |

- For BioMCP, install `biomcp-cli`; the PyPI package named `biomcp` is a different, unrelated project. The same binary also has an HTTP mode and works as a standalone CLI.
- For OMOP MCP, the server does its vocabulary search and mapping through the OMOPHub API (`https://api.omophub.com/v1`), and you get the key from omophub.com.
- Medical MCP's README describes a cache with per-source lifetimes: FDA data 24 hours, PubMed 1 hour, WHO 7 days, RxNorm 30 days. Check whether your installed version has it before crediting ToolHelper's cache for latency wins. Those lifetimes are good defaults for your own freshness policy.

#### Medical MCP tools (verified from `build/index.js`)

| Tool | Arguments |
|---|---|
| `search-drugs` | `query` (brand or generic), `limit` 1–50, default 10 |
| `get-drug-details` | `ndc` (matched against openFDA `product_ndc`) |
| `get-health-statistics` | `indicator` (substring of a WHO indicator name), `country` (ISO3 such as `USA`, optional), `limit` 1–20 |
| `search-medical-literature` | `query`, `max_results` 1–20 |
| `get-article-details` | `pmid` |
| `search-drug-nomenclature` | `query` (RxNorm) |
| `search-google-scholar` | `query` |
| `search-clinical-guidelines` | `query`, `organization` (optional) |
| `search-medical-databases` | `query` |
| `search-medical-journals` | `query` |

Medical MCP behaviors that create efficiency targets:

- **Dead end:** `search-drugs` returns markdown with brand, generic, manufacturer, route and form, but no NDC. `get-drug-details` needs an NDC, so search followed by details stalls and the model starts guessing. An NDC dataset fixes this directly.
- **Country codes:** `get-health-statistics` wants a 3-letter code, but models send "Japan" or "United States".
- **Overlapping tools:** four literature tools overlap heavily, which causes wrong-tool-first calls.
- **Stray stdout line:** the server prints a startup line to stdout, so your client must tolerate one non-JSON line.
- **Tool list drift:** the GitHub README lists more tools than npm 1.0.8 has. Always snapshot the live tool list.

#### BioMCP tools (verified from `src/mcp/catalog.rs` and `src/mcp/shell.rs`)

| Tool | Arguments |
|---|---|
| `biomcp` | `command` (one read-only CLI command, e.g. `get gene BRAF pathways`), `json` |
| `search` | `entity` ∈ author, gene, pgx, gwas, article, trial, variant, protein; `limit` 1–25 (default 10); `offset` 0–1000; `json`; plus per-entity fields |
| `get` | `entity` ∈ gene, variant, article, author, trial, diagnostic, drug, disease, pgx, pathway, cell-line, protein, adverse-event; `id`; `sections[]`; `json`; also variant `assembly`, trial `source` (ctgov or nci), article `pdf` |
| `variant_normalize_car` | `inputs`: 1–50 RefSeq HGVS strings |
| `variant_erepo` | `caid`, `caids` (1–50) or `gene`; `limit` 1–100; `offset`; `detail`; `assertion_id`; `version` |
| `gene_cspec` | `gene`; optional `version_iri`, `capture_id`, `files` |
| `variant_articles` | `items` (1–10 variants), `strategy`, `limit` 1–50, `offset`, `verify_identity`, `confirmed_only`, `debug_plan` |

Search fields per entity:

- **trial:** condition[], intervention[], mutation[], criteria[], biomarker[], phase, status, source
- **variant:** query, gene, hgvsp, significance, max_frequency, consequence, review_status, revel_min
- **article:** keyword[], gene, disease[], drug[], author[], journal[], date_from, date_to, article_type, source, open_access, no_preprints, sort
- **gene:** query, gene_type, chromosome, region
- **pgx:** gene, drug, cpic_level
- **gwas:** gene, trait, p_value
- **protein:** query, all_species, reviewed, disease, existence
- **author:** query, source

`get` sections:

- **gene:** pathways, ontology, diseases, diagnostics, protein, go, interactions, civic, expression, hpa, druggability, clingen, gencc, constraint, disgenet, funding, all
- **variant:** predict, predictions, clinvar, population, population-details, conservation, cosmic, cgi, civic, cbioportal, gwas, all
- **trial:** eligibility, contacts, locations, outcomes, arms, references, all
- **drug:** label, regulatory, safety, shortage, targets, indications, interactions, civic, approvals, approval-history, cell_lines, all
- **disease:** genes, pathways, phenotypes, diagnostics, variants, models, prevalence, survival, funding, civic, disgenet, clinical_features, all

BioMCP behaviors that create efficiency targets:

- **Already optimized:** the team cut 36 tools (about 16,600 tokens) down to 7. Their CI caps the current catalog at 5,800 tokens, they report about 75% smaller outputs, and the server has a 24-hour disk cache. Expect smaller gains here; treat it as your "already efficient" control.
- **Sections are the main lever:** a gene summary card is about 200 tokens, while `all` is about 2,000.
- **A guaranteed failure:** `drug` and `disease` are valid for `get` but missing from the typed `search` enum. `search(entity="drug")` fails, and drug or disease search has to go through the raw `biomcp` command. This is an ideal argument-guard and recipe target.
- **Built-in measurement:** `biomcp mcp tools` dumps the tool list, and their `scripts/measure-mcp-tools.py` counts it with `cl100k_base`. Use both to sanity-check your own counts.
- **Suggested next commands:** every response ends with suggestions. Track whether the agent follows them.

#### OMOP MCP tools (verified from `src/omop_mcp/server.py`)

| Tool | Arguments | Returns |
|---|---|---|
| `find_omop_concept` | `keyword`, `omop_table`, `omop_field`, `max_results` (default 20) | up to 20 candidates (concept_id, code, name, class, concept/standard flag, validity, domain, vocab, url) plus selection guidance |
| `batch_map_concepts_from_csv` | `csv_path` (CSV with keyword, omop_table, omop_field) | CSV text with the top candidate appended |

Resources and prompt:

- `omop://tables`: 23 CDM tables mapped to their concept_id fields, from `data/omop_concept_id_fields.json`.
- `omop://preferred_vocabularies`: the preferred vocabulary order per table.
- `omop://documentation`: fetches the full OHDSI vocabulary page live, so it's large.
- Prompt: `map_clinical_concept`.

Preferred vocabularies, verbatim:

| Table | Vocabularies, in order |
|---|---|
| measurement | LOINC, SNOMED |
| condition_occurrence | SNOMED, ICD10CM, ICD9CM |
| drug_exposure | RxNorm, RxNorm Extension, SNOMED |
| procedure_occurrence | SNOMED, CPT4, ICD10PCS |
| observation | SNOMED |
| device_exposure | SNOMED |

OMOP MCP behaviors that create efficiency targets:

- **Oversized results:** 20 candidates with 9 fields each is a lot for a pick-one task. This is the clearest result-shaping win.
- **Exact argument pairs:** `omop_table` and `omop_field` must be an exact pair from the CDM map. Models write `condition` instead of `condition_occurrence`.
- **Separate environment:** its `uv.lock` pins `mcp` 1.7.1, and it imports the 1.x `FastMCP` API. Run it from its own repo environment, never inside ToolHelper's, which uses mcp 2.x.
- **Permissions:** `batch_map_concepts_from_csv` reads any local path, which makes it a natural place for a permission rule.

### 1.2 SDK APIs (verified against installed versions)

**MCP Python SDK 2.3.0**

```python
from mcp import Client, StdioServerParameters

params = StdioServerParameters(
    command="uv",
    args=["--directory", OMOP_REPO, "run", "omop_mcp"],
    env={"OMOPHUB_API_KEY": os.environ["OMOPHUB_API_KEY"]},
)
async with Client(params) as c:
    tools = (await c.list_tools()).tools            # .name .description .input_schema
    r = await c.call_tool("find_omop_concept", {...})
    texts = [b.text for b in r.content if hasattr(b, "text")]   # also r.is_error, r.structured_content
```

- **Environment variables:** stdio servers inherit only `HOME, LOGNAME, PATH, SHELL, TERM, USER`. API keys must go in `env` explicitly.
- **Replay server:** `Client(MCPServer(...))` connects to an in-process server. ToolHelper uses this to serve *recorded real responses* captured from the live servers, under the exact real tool names and schemas. No response data is hand-written. Note that a plain exception inside a tool comes back as `is_error=True` with a generic message.

**Ollama Python 0.6.3**

```python
from ollama import Client

llm = Client(host="http://localhost:11434")
resp = llm.chat(
    model="qwen3:8b", messages=msgs, tools=tool_defs, think=False,
    options={"num_ctx": 32768, "temperature": 0, "seed": 7},
)
resp.message.tool_calls          # each: .function.name, .function.arguments (dict)
resp.prompt_eval_count, resp.eval_count, resp.prompt_eval_duration, resp.eval_duration, resp.done_reason
msgs.append({"role": "tool", "tool_name": name, "content": result_text})
```

Tool format: `{"type": "function", "function": {"name", "description", "parameters": input_schema}}`.

**WarmHub Python SDK 0.29.0** (`pip install warmhub`)

```python
from uuid import uuid4
from warmhub import WarmHubClient, Add, Revise, Retract

wh = WarmHubClient.from_env()                         # reads WH_TOKEN
repo = wh.repository("your-org/toolhelper")
repo.apply("seed catalog", [Add("<thing name>", shapes=["ToolSchema"], data={...})])
b = repo.batch(message="recipes from experiment 7"); b.add(...); b.commit()
repo.things.head(shape="Recipe", limit=50)            # also .get(wref, version=), .query(...), .search(text, shape=...)
wh.shape.create("your-org", "toolhelper", "Recipe", fields, event_request_id=str(uuid4()))
```

- **Write options:** `Revise` takes `expected_version` (optimistic concurrency), `Retract` takes `reason`, and `Add` takes `affirmed_targets`.
- **Shape field types:** `string`, `number`, `boolean`, `wref`, arrays like `["string"]`, a `?` suffix for optional fields, and `{"type": "string", "enum": [...]}`.
- **Tokens:** `npm install -g @warmhub/cli`, then `wh auth login`, then `wh token create --name toolhelper`.
- **Read first:** have Claude Code read the Things, Naming as Navigation, Wrefs and Assertions docs pages before writing the adapter.

**Other libraries**

- Core: pydantic, typer, rich, pyyaml, jsonschema, rapidfuzz, tiktoken, pytest, anyio.
- Optional: transformers (exact model tokenizer), streamlit (dashboard), openai, anthropic.

### 1.3 Datasets for the efficiency layer

| Dataset | Where | Access | Helps | Use |
|---|---|---|---|---|
| FDA NDC Directory (`product.txt`) | `https://www.accessdata.fda.gov/cder/ndctext.zip` (updated daily) | public | Medical drug tools | brand/generic → product NDC; brand → generic |
| RxNorm | RxNav REST API `https://rxnav.nlm.nih.gov/REST` | API free; full files need a free UMLS license | Medical, OMOP drug_exposure, BioMCP drug | name → RxCUI/ingredient |
| HGNC complete set | `https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt` | public | BioMCP gene/variant | alias/previous symbol → approved symbol (HER2 → ERBB2) |
| WHO GHO indicators | `https://ghoapi.azureedge.net/api/Indicator` | public | `get-health-statistics` | phrase → indicator name; country → ISO3 |
| OMOP vocabularies | athena.ohdsi.org (CONCEPT, CONCEPT_SYNONYM, CONCEPT_RELATIONSHIP) | free account; some vocabularies need extra licences | OMOP | synonym → standard concept candidates, locally |
| CDM field map + preferred vocabularies | ships inside omop_mcp | Apache-2.0 | OMOP | argument validation, result filtering |
| ClinVar `variant_summary.txt.gz` | `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/` | public | BioMCP variant | gene + protein change → HGVS/rsID |
| Your own traces | monitoring DB | yours | all | recipes, projections, failure patterns |

Have Claude Code print each file's header on first load instead of trusting remembered column names.

---

## 2. Architecture

**Five layers:**

1. **LLM backends:** Ollama by default, behind one interface.
2. **MCP hub:** connects all three servers and namespaces tools as `medical__search-drugs`. Use only letters, digits, `_` and `-`, at most 64 characters, so names work with Ollama, OpenAI and Anthropic.
3. **Agent loop:** has hooks where efficiency strategies plug in.
4. **Monitoring:** tracer, then SQLite, then stats, then reports.
5. **Context store:** WarmHub-style versioned knowledge with evidence. It runs on local SQLite by default, with an optional WarmHub adapter.

### WarmHub principles, mapped

| Principle | ToolHelper implementation |
|---|---|
| Trusted | every hint, recipe, projection and decision links to evidence (dataset row, run IDs, schema hash) |
| Curated | strategies *propose*; `toolhelper curate` approves; only accepted items are used |
| Versioned | tool schemas hashed and versioned; recipes pin versions; a schema change marks dependents stale |
| Permissioned | per-profile tool allowlist and path rules (the OMOP CSV tool) |
| Timely | per-source TTLs; context injected at the moment of the call |
| At scale | machine-readable, queryable objects, publishable to WarmHub |

### Shapes

All of these were validated with the WarmHub SDK's own shape-definition preflight check.

- `McpServer {name, package, version, launch, env_vars?: [string]}`
- `ToolSchema {server: wref, tool, description, input_schema_json, schema_hash, catalog_tokens: number, captured_at}`
- `ToolCard {tool: wref, usage_notes, example_tasks: [string], pitfalls?: [string], status: enum(proposed, accepted)}`
- `Dataset {name, source_url, license_note, version, sha256, rows: number, fetched_at, ttl_days: number}`
- `EntityHint {surface, entity_type, canonical, identifier, dataset: wref, row_key, confidence: number}`
- `Recipe {task_type, steps: [string], arg_templates_json, depends_on: [wref], evidence_runs: [string], success_rate, mean_tokens, status}`
- `Projection {tool: wref, keep_fields: [string], drop_rule, evidence_runs: [string], expand_rate}`
- `CachePolicy {tool: wref, ttl_seconds, source}`
- `RunSummary`, `Experiment`, `Decision {title, objective, chosen_config_json, constraints_json, evidence: [wref], rejected_json, approved_by?, status}`

Raw traces stay in local SQLite. Only curated summaries go to WarmHub: "the right facts, not all the facts."

### Repo layout

```
toolhelper/
├─ CLAUDE.md                README.md
├─ configs/                 servers.yaml llm.yaml experiments.yaml datasets.yaml policy.yaml strategies/
├─ data/                    tasks/benchmark.jsonl  datasets/ (gitignored)  replay/
├─ src/toolhelper/
│  ├─ cli.py config.py
│  ├─ llm/          base.py ollama.py openai_compat.py anthropic.py scripted.py
│  ├─ mcp/          hub.py catalog.py recorder.py replay.py replay_server.py
│  ├─ context/      model.py local_store.py warmhub_store.py shapes.py
│  ├─ agent/        loop.py hooks.py
│  ├─ monitoring/   tracer.py tokens.py stats.py pathways.py report.py
│  ├─ datasets/     ndc.py rxnorm.py hgnc.py who_gho.py omop_vocab.py clinvar.py index.py
│  ├─ efficiency/   retrieval.py compression.py grounding.py arg_guard.py recipes.py shaping.py cache.py compaction.py trust.py
│  └─ experiments/  runner.py scoring.py optimize.py decisions.py
├─ dashboard/app.py
└─ tests/
```

---

## 3. Build steps

### Phase 0: Environment

1. **Install the runtimes:** Python 3.12+, uv, Node 22 and Ollama. Then `ollama pull qwen3:8b`, or any model tagged "tools" in the Ollama library.
2. **Install the servers and smoke-test each one** with the MCP Inspector, e.g. `npx @modelcontextprotocol/inspector npx -y medical-mcp`. Also run `biomcp health --apis-only`.
3. **Create the project:** `uv init toolhelper`, then add the dependencies above.
4. **Write `CLAUDE.md`** with these rules:
   - tests never call live APIs; they use recorded real responses served by the replay server (`Client(MCPServer)`);
   - never hand-write response data; record it from the real servers;
   - every efficiency change emits a transformation record with evidence;
   - never install omop_mcp into this environment.
5. **Optional:** register the servers in Claude Code so it can explore real outputs, e.g. `claude mcp add medical --scope user -- npx -y medical-mcp`.

**Done when:** all three servers list their tools in the Inspector and Ollama answers a tool-calling test.

### Phase 1: LLM backends

- **Interface:** `chat(messages, tools, options) -> ChatResult(message, tool_calls, usage{prompt, completion, measured: bool}, latency_ms, raw)`.
- **Backends:** Ollama, OpenAI-compatible (covers vLLM, LM Studio, OpenAI and Ollama's own `/v1` endpoint), Anthropic, and Scripted (deterministic, for tests).
- **Profiles:** live in `configs/llm.yaml`; every command takes `--llm <profile>`.

> Prompt: "Read README.md Phase 1. Implement src/toolhelper/llm with the LLMBackend protocol and the four backends. Normalize usage with a measured flag. Tests use only the Scripted backend."

### Phase 2: MCP hub and catalog

1. **Configure:** `servers.yaml` holds the launch commands and env vars from 1.1.
2. **Build the hub:** connect, build the namespaced registry, call with timeout, and normalize results into text plus structured content.
3. **Build `toolhelper catalog`:**
   - snapshot each schema as canonical JSON and hash it with sha256;
   - store it as a ToolSchema version;
   - diff against the last snapshot;
   - measure token cost per tool and per server.
4. **Build the recorder and replay server:**
   - the recorder captures real responses from the live servers, keyed by server, tool and canonical args, with the time recorded and the server version;
   - the replay server serves those recordings in-process under the exact real tool names and schemas, taken from the catalog snapshot;
   - for tests, record a small fixture set (a few real calls per tool, including real error responses).

**Done when:** the catalog reports 19 tools (10, 7 and 2) and tokens per tool, and the replay server's tool list hashes identically to the live servers'.

### Phase 3: Context store

- **Storage:** local SQLite tables for things, append-only versions, links (with pinned versions) and evidence.
- **Operations:** add, revise, retract and reaffirm, mirroring WarmHub; reads via head, get, history, refs and search.
- **WarmHub adapter:** the same interface over the SDK, plus `toolhelper publish`.

**Done when:** revising a ToolSchema marks the Recipes that depend on it as stale.

### Phase 4: Agent loop and tracer

**The loop:**

1. `prepare` adds hints and recipes.
2. `select_tools`, then `transform_schemas`.
3. Call the LLM.
4. For each tool call: `before_call` (guard or cache), then the hub call, then `after_result` (shaping).
5. `compact_history`, then repeat until a final answer or `max_turns` (about 8).

**Strategy hook interface:**

```python
class Strategy(Protocol):
    def prepare(self, task, ctx) -> list[ContextItem]: ...      # each item carries evidence
    def select_tools(self, task, tools) -> list[ToolDef]: ...
    def transform_schemas(self, tools) -> list[ToolDef]: ...
    def before_call(self, call) -> CallDecision: ...            # proceed | repaired | rejected(hint) | cached
    def after_result(self, call, result) -> ToolResult: ...
    def compact_history(self, messages) -> list[Message]: ...
    def extra_tools(self) -> list[ToolDef]: ...                 # find_tools, resolve_entity, expand_result
```

**Done when:** one task produces a complete trace.

### Phase 5: Monitoring

**SQLite tables:**

- `runs(run_id, group_id, task_id, backend, model, params_json, config_hash, catalog_version, success, trust_json, answer)`
- `llm_calls(run_id, turn, prompt_tokens, completion_tokens, measured, est_breakdown_json, latency_ms, done_reason, tools_offered)`
- `tool_calls(run_id, turn, server, tool, args_json, args_tokens, raw_result_tokens, shaped_result_tokens, carried_tokens, latency_ms, is_error, error_class, cache_hit, replayed, used)`
- `transformations(run_id, turn, strategy, hook, target, before_tokens, after_tokens, evidence_json, note)`
- `context_items(run_id, kind, wref, version, tokens, used)`

**Token accounting:**

- **Measured vs estimated:** measured comes from the backend. Estimated is a per-component breakdown (system prompt, tool definitions, injected context, history, tool results), calibrated so the parts sum to the measured total. Always label which is which.
- **Exact tool-definition cost:** send the same messages with and without tools, using `num_predict: 1`. The difference in `prompt_eval_count` is the overhead as that model actually sees it. Repeat per tool for marginal cost. Put a random nonce at the start of the system prompt so prompt caching can't distort the count.
- **Carried tokens:** a tool result's tokens times the number of later LLM calls that include it. This is often the biggest hidden cost.

**Metrics:**

- **Per tool:** calls, error %, args tokens, raw vs shaped result tokens, carried tokens, p50/p95 latency, share of total.
- **Per run:** tokens split by component, turns, calls, success, trust.

**Pathways ("how the LLM gets to the tool"):**

- the tools offered on each turn;
- the first tool chosen and whether it was correct;
- turns until the first useful call;
- retries, redundant identical calls, dead ends (results never used) and detours;
- a tool-to-tool transition matrix;
- the most common sequences in successful vs failed runs;
- BioMCP suggestion-following rate.

**Warnings:** flag prompts near `num_ctx` or `done_reason == "length"`.

**Output:** `toolhelper report` writes HTML and JSON. The optional Streamlit dashboard has pages for Overview, Tools, Pathways (Sankey), Runs (trace cards), Experiments and Decisions.

### Phase 6: Benchmark and record/replay

**Size:** 40–60 tasks. That's about 15 per server and 10 cross-server, with roughly 20% "traps": brand names, gene aliases, wrong table names, country names.

**Format:**

```json
{"id": "med-004", "servers": ["medical"], "category": "drug_label",
 "prompt": "Pull the FDA label details for Zoloft.",
 "expected_paths": [["medical__get-drug-details"], ["medical__search-drugs", "medical__get-drug-details"]],
 "checks": {"must_call_any": ["medical__get-drug-details"], "answer_includes_any": ["sertraline"]},
 "traps": ["brand_name", "ndc_needed"]}
```

**Example tasks:**

- Pathways for HER2 (alias of ERBB2).
- ClinVar significance of BRAF V600E.
- Trials for BRAF V600E melanoma.
- Map "heart attack" to `condition_occurrence.condition_concept_id`.
- Map HbA1c to `measurement`.
- WHO life expectancy for Japan (needs `JPN`).
- Metformin's RxCUI and its OMOP drug_exposure concept (cross-server).

**Record/replay:** every response the agent sees is a real server response. The hub has three modes:

- **record:** call the live servers and save every response;
- **replay:** serve saved real responses, so every strategy sees identical tool outputs; misses fall through to live and get recorded;
- **live:** call the live servers with no replay, for final confirmation.

The first benchmark run uses record mode, experiments use replay, and the final step uses live (Phase 10). This layer sits *below* ToolHelper's own cache, so the cache experiment stays honest.

**Scoring:**

- Hard checks: required tool called, and required strings present in the answer.
- Grounding: IDs in the answer must appear in tool outputs.
- Optional: an LLM judge using a different model.

**Done when:** a record run produces the baseline, and a strict replay run (`--strict-replay`, no live fallback) serves every matching call from recordings with zero live calls. Hosted LLMs may not be reproducible, so the strict run's divergence rate (tasks ending in `ReplayMiss`) is measured and recorded rather than required to be zero; experiments then use replay with live fallback and repeat each configuration.

### Phase 7: Datasets

1. **One loader per dataset.** Each emits `EntityRecord(surface, type, canonical, identifier, source, row_key)` and a versioned Dataset thing (URL, fetch date, sha256, row count, license note, TTL).
2. **Index:** exact, normalized and fuzzy matching (rapidfuzz). If a match is ambiguous, return the top two with evidence instead of guessing.

**Done when:** "HER2", "Zoloft", "heart attack" and "Japan" each resolve with a dataset row as evidence.

### Phase 8: Efficiency strategies

| ID | Strategy | Mechanism | Variants to test | Records |
|---|---|---|---|---|
| E1 | Tool retrieval | offer only relevant tools via hybrid search over ToolCards (reuse the Paper Librarian search) | top-k 3/5/8; static vs `find_tools` meta-tool; server-first routing | scores, recall@k vs tools actually needed |
| E2 | Schema compression | shorter descriptions, drop titles/defaults, keep types, enums, required | light vs aggressive vs minimal (also drops bounds that only reject values: string lengths, `minItems`, `uniqueItems`, `format`) | original vs compressed hash, later arg-error rate |
| E3 | Dataset grounding | resolve task entities to canonical IDs before calls | inject hints vs `resolve_entity` tool vs silent arg rewrite, crossed with dataset subsets | dataset version and row per hint, *uptake* (hinted value used in args) |
| E4 | Argument guard | validate against schema and domain rules (CDM pairs, BioMCP enums and sections, ISO3, NDC format) before calling; return a one-line fix | reject-with-hint vs deterministic auto-repair | rule fired, original vs repaired args |
| E5 | Recipes | reuse the shortest frequent successful path for similar tasks | plan injection vs recipe tool | evidence runs, success rate, pinned schema versions |
| E6 | Result shaping | keep fields traces show get used; park the rest behind `expand_result(handle)`; OMOP: Standard + Valid + preferred vocabulary, top 3–5; BioMCP: minimal `sections` | projection vs top-N vs summary vs token budget (results over N tokens cut to leading whole items or sections) | dropped fields and counts, expand rate |
| E7 | Cache | reuse identical canonical calls within TTL | TTL per source | hit rate, age of served result |
| E8 | History compaction | replace consumed results in later turns with a summary plus handle | after one turn vs after use is detected | carried tokens saved, re-expansions |
| E9 | Final answer at turn limit | when `max_turns` is reached, make one last LLM call with no tools asking for an answer from what was gathered | on vs off; limit 6/8/12 | turns used, answers produced at the limit |

### Phase 9: Trust

**Components** (each 0–1, always shown separately as well as combined):

- **Grounding:** IDs in the answer that appear in this run's tool outputs or in evidenced hints.
- **Provenance coverage:** injected items and transformations that carry evidence. This should be 1.0, and anything less is a bug.
- **Fidelity:** penalizes expand calls, low-confidence rewrites, and argument errors caused by compressed schemas.
- **Freshness:** cached results and datasets that are within their TTL.

**Default weighting** (configurable): trust = 0.35·grounding + 0.25·provenance + 0.25·fidelity + 0.15·freshness.

**Trace card:** each run gets a card with the timeline, tokens per step, every transformation with before/after and evidence links, and the trust components with reasons. WarmHub's Veritas docs (opinions, reputation, oracles) are worth reading for this layer.

### Phase 10: Experiments and finding the optimum

1. **Freeze the inputs:** catalog version, benchmark version, dataset versions, model and params (temperature 0, seed, `num_ctx`, `think=False`) and replay set. Store them as an Experiment.
2. **Baseline:** 3 repeats over all tasks. Temperature 0 isn't fully deterministic in Ollama, so repeats matter.
3. **Single-strategy ablations:** run each strategy alone, with per-server breakdowns.
4. **Dataset experiments for E3:** cross datasets (NDC, RxNorm, HGNC, GHO, OMOP) with modes (inject, tool, rewrite).
   - **Measure per tool:** uptake, change in argument errors, change in calls, tokens added by hints vs tokens saved, and wrong-hint rate.
   - **Output:** a per-tool dataset routing table, i.e. which dataset helps which tool, in which mode.
5. **Gate:** drop strategies that cut success by more than 2 points or push trust below 0.9.
6. **Greedy forward selection:** start from the best single strategy and add whichever cuts tokens most while meeting the constraints. Stop when the gain is under 3%. Check pairwise interactions among the finalists.
7. **Objective:** minimize **tokens per successful task**, subject to success ≥ baseline − 2 points and trust ≥ 0.9. Allow per-server optima.
8. **Statistics:**
   - comparisons are paired by task;
   - bootstrap 95% CIs on the changes;
   - McNemar's test for success flips;
   - report median and p90, not just means.
9. **Decision log:** write a Decision thing containing:
   - the chosen config;
   - each key decision with its numbers ("E2 rejected: −8% tokens but −10 points success on BioMCP, argument confusion");
   - the rejected alternatives;
   - links to the evidence runs.

   It stays `proposed` until you approve it with `toolhelper curate`.
10. **Live confirmation:** rerun the baseline and the chosen config once in live mode on the test split. Report any gap between replay and live results; a large gap means the recordings are stale and should be refreshed.

**Second iteration, when the first finds no significant effect** (p10 did not, 2026-11-05; `configs/experiments-p10b.yaml`):

1. **Diagnose from the runs.** Split each run's measured tokens by component. Compute each candidate variant's saving at fixed behaviour on the baseline traces (turn-1 prompts are deterministic, so definition savings are exact). Compare it with the smallest change the planned repeats can detect, which follows from the per-task run-to-run spread in log tokens. A variant whose mechanism cannot reach that size is not worth an arm.
2. **Pick variants that can reach it,** aimed at the measured cost centres: tool definitions re-sent every turn, large results carried across turns, turn loops and error turns.
3. **Run a two-level fractional factorial** instead of steps 3 and 6. Every cell runs every task once, in a seeded random order. Each factor's effect compares half of the runs with the other half, paired by task, with bootstrap CIs; two-factor interactions come from the same runs; per-server effects are reported too.
4. **Fix the choice rule before running.** A factor is on when the CI of its change in tokens per success lies below zero, success holds within the gate and trust stays ≥ 0.9. Confirm the chosen config against the baseline with 3 paired repeats each, then follow steps 9 and 10.
5. **Keep the replay set honest.** Live fallbacks from parallel passes share a cross-process throttle. Recordings of transient failures (an API outage, rate-limit errors reported as empty results) move to `data/replay-archive/` with a manifest, so the real server re-records them (`scripts/archive_recordings.py`).

### Phase 11: Swap the LLM and rerun

Run `toolhelper optimize --llm <profile> --replay` with everything else frozen, then compare which decisions hold across models. Store an optimum per model.

### Phase 12 (optional): Gateway

`toolhelper serve` exposes the optimized tool surface as an MCP server (`MCPServer`), so Claude Desktop or Claude Code get the efficiency layer directly. `toolhelper publish` pushes recipes and decisions to WarmHub.

**Full CLI:** `catalog`, `datasets fetch|index|status`, `run`, `bench`, `report`, `mine`, `optimize`, `curate`, `publish`, `serve`.

---

## 4. Gotchas

- **Ollama context:** set `num_ctx` explicitly and log it. The default is small and has changed between versions, and all 19 tool definitions plus history can overflow it.
- **Ollama prompt caching:** reused prompt caches can make later turns' `prompt_eval_count` look smaller. Keep measured and estimated "logical" prompt tokens side by side.
- **Qwen3 thinking:** use `think=False` for fair comparisons, or count thinking tokens separately.
- **PubMed rate limits:** E-utilities allow 3 requests per second without an API key, so throttle benchmark runs. Replay mode mostly solves this.
- **Medical MCP first run:** it depends on Puppeteer for Google Scholar, so the first `npx` run may be slow.
- **Shaping can hide information:** every drop must be recorded and expandable. This is the line between efficient and black-box.

## 5. Working in Claude Code

- Keep this README in the repo root and point `CLAUDE.md` at it.
- Work one phase per session: plan mode first, tests first, run pytest, commit.
- Once the hub and catalog exist, have Claude Code run `toolhelper catalog` against the live servers and diff the result against the tables in section 1.
- Rough timeline: Phases 0–5 in week 1, 6–7 in week 2, 8–9 in week 3, 10–11 in week 4.

## 6. References

- Medical MCP: [JamesANZ/medical-mcp](https://github.com/JamesANZ/medical-mcp) · [Glama listing](https://glama.ai/mcp/servers/JamesANZ/medical-mcp/tree)
- BioMCP: [genomoncology/biomcp](https://github.com/genomoncology/biomcp) · ["We Deleted 35 Tools and Our Agent Got Better"](https://github.com/genomoncology/biomcp/blob/main/docs/blog/we-deleted-35-tools.md) · [mcpservers.org listing](https://mcpservers.org/vi/servers/genomoncology/biomcp) · [BioMCP Server: Complete Guide (2026)](https://mcp.directory/blog/biomcp-complete-guide-2026)
- OMOP MCP: [OHNLP/omop_mcp](https://github.com/OHNLP/omop_mcp) · [mdskills listing](https://www.mdskills.ai/de/mcp-servers/omop-mcp)
- WarmHub: [Docs: Get started](https://docs.warmhub.ai/get-started/)
- SDKs: [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) · [Ollama Python](https://github.com/ollama/ollama-python)
- Datasets: [HGNC downloads](https://genenames.org/download/) · [openFDA NDC](https://open.fda.gov/data/ndc/) · [FDA NDC product file definitions](https://www.fda.gov/drugs/drug-approvals-and-databases/ndc-product-file-definitions)
