# CLAUDE.md

ToolHelper monitors and reduces the token cost of LLM tool calls against three biomedical MCP servers (Medical MCP, BioMCP, OMOP MCP), keeping every change traceable.

**`README.md` is the spec.** Read the relevant phase in section 3 before starting work, and check section 1 (verified metadata) and section 4 (gotchas) before touching servers, SDKs or datasets. When the README and the code disagree, raise it rather than silently picking one.

## How we work

- One phase per session, in order. Plan first, then write tests, then implement, then run `uv run pytest`. Never run `git commit`; the user commits themselves.
- Stop at each phase's **Done when** criterion and confirm it's met before moving on.
- Update the progress table below at the end of each phase.
- This repo root *is* the `toolhelper/` project from the README's repo layout. Run `uv init` here; don't create a nested `toolhelper/` folder.

## Hard rules

- **No live APIs in tests.** Tests use the Scripted LLM backend and recorded real responses served in-process by the replay server (`Client(MCPServer(...))`), under the exact real tool names and schemas from the catalog snapshot.
- **Never hand-write response data.** Every response the agent or a test sees is a real server response, captured by the recorder (keyed by server, tool and canonical args, with record time and server version). Test fixtures are a small recorded set: a few real calls per tool, including real error responses.
- **Hub modes:** `record` (live, saves everything), `replay` (serves recordings; misses fall through to live and get recorded), `live` (no replay, for final confirmation). Record/replay sits *below* ToolHelper's own cache.
- **Every efficiency change emits a transformation record with evidence** (dataset row, run IDs or schema hash). Provenance coverage below 1.0 is a bug.
- **Never install omop_mcp into this environment.** It pins `mcp` 1.7.1; ToolHelper uses `mcp` 2.x. Launch it from its own repo with `uv --directory <repo> run omop_mcp`.
- **Pass API keys explicitly** in `StdioServerParameters(env=...)`. Stdio servers only inherit `HOME, LOGNAME, PATH, SHELL, TERM, USER`.
- **Shaping must be reversible:** every dropped field is recorded and reachable via `expand_result(handle)`.
- **Label tokens as measured or estimated**, always.
- **Don't trust remembered column names or tool lists.** Print dataset headers on first load; snapshot live tool lists rather than relying on the tables in the README.
- Datasets go in `data/datasets/` (gitignored), or in `$TOOLHELPER_DATASETS_DIR` when set (on this machine `C:\Users\neelg\toolhelper-data\datasets`, outside OneDrive; the index is GBs). Secrets (`OMOPHUB_API_KEY`, `WH_TOKEN`) come from the environment, never from committed files.

## Conventions

- Python 3.12+, managed with uv. Source in `src/toolhelper/`, tests in `tests/`, configs in `configs/`.
- Namespaced tool names: `<server>__<tool>`, only `[A-Za-z0-9_-]`, at most 64 characters.
- Ollama defaults for experiments: `temperature 0`, fixed `seed`, explicit `num_ctx`, `think=False`.

## Environment notes

- Platform is Windows; the Bash tool is Git Bash and PowerShell is also available. In Git Bash, set `MSYS_NO_PATHCONV=1` before passing `/c`-style flags to Windows programs.
- `uv` lives in `C:\Users\neelg\AppData\Roaming\Python\Python313\Scripts` and `biomcp` in `C:\Users\neelg\.local\bin`; neither is on PATH by default, so prepend them in shell commands.
- Python scripts run with `-I` ignore `PYTHON*` env vars; use `-X utf8` so emoji in server output don't crash on the cp1252 console.

### Verified launch commands (Phase 0, 2026-11-01)

These differ from README section 1.1 because of Windows. Use them in `configs/servers.yaml`.

| Server | Version | Command |
|---|---|---|
| medical | npm 1.0.8 | `node C:\Users\neelg\src\medical-mcp\node_modules\medical-mcp\build\index.js` |
| biomcp | biomcp-cli 0.9.3 (README says 0.9.4) | `biomcp serve` |
| omop | commit 7628155 (May 10, 2026), mcp 1.7.1 | `uv --directory C:\Users\neelg\src\omop_mcp run omop_mcp` |

- **`npx -y medical-mcp` does not work on Windows.** The package's `build/index.js` has no shebang, so npm's `.cmd` shim runs it through Windows Script Host, which exits silently. Use `node <path>` against the pinned install in `C:\Users\neelg\src\medical-mcp`. Its postinstall (writes a Claude Desktop config) is denied; Puppeteer's (downloads Chrome for Google Scholar) is approved.
- **Medical MCP prints one non-JSON line to stdout** (`✅ Medical MCP Server running on stdio`). The MCP SDK logs a parse error and carries on; this is expected.
- **Always connect with `Client(params, mode="legacy")`.** MCP SDK 2.3's default `mode="auto"` sends a `server/discover` probe first; BioMCP exits on it ("expects an MCP client on stdin (initialize handshake)"). All three servers speak the handshake protocol.
- **BioMCP health:** core sources OK. GWAS Catalog (HTTP 410), FDA Orphan Drug Designations and ComplexPortal (timeout) errored; NCI, OncoKB, DisGeNET, AlphaGenome and UMLS are excluded for lack of keys. Expect failures from tasks touching those sections.
- **MCP Inspector `--cli` hangs on OMOP MCP** on this machine, though the server answers raw stdio fine. Don't rely on the Inspector for smoke tests; use the SDK.
- `OMOPHUB_API_KEY` is set as a Windows user environment variable (verified with a live `find_omop_concept` call). Shells started before it was set won't see it; read it with `[Environment]::GetEnvironmentVariable("OMOPHUB_API_KEY", "User")` if `$env:OMOPHUB_API_KEY` is empty.
- **OMOPHub free tier: 3,000 calls/month, 2 requests/sec** (failed requests not counted). Each `find_omop_concept` is one OMOPHub call; `batch_map_concepts_from_csv` is one per CSV row, so keep benchmark CSVs small. The recorder must throttle OMOP calls; experiments should run in replay mode.

### LLM backends (Phase 1, 2026-11-01)

- Backends report options they can't honour in `ChatResult.ignored_options` instead of dropping them silently. Check it when comparing runs across backends.
- **Ollama `/v1` ignores `think`**; qwen3 thinks by default there (186 vs 29 completion tokens). Send `reasoning_effort: none` via `options.extra` (the `ollama-qwen3-v1` profile does); verified to match native token counts.
- **Current Claude models take no `temperature`/`seed`** (the installed SDK's `messages.create` has no such parameters), and thinking can't be disabled on `claude-opus-5-5`; use `effort`. Anthropic runs are therefore not seed-reproducible.
- LLM test fixtures in `tests/fixtures/llm/` are real responses recorded by `scripts/record_llm_fixtures.py`. The Anthropic parsing test still uses a hand-built SDK object (no key yet); replace it with a recording when a key is available.

### MCP hub and catalog (Phase 2, 2026-11-01)

- **Connect every client with `mode="legacy"`, including in-process replay servers.** On the 2026-07-28 protocol the SDK drops Medical MCP's `execution` tool field (breaking hash identity) and stamps `_meta.io.modelcontextprotocol/serverInfo` into every result.
- **Reported serverInfo is not the package version:** Medical MCP reports `1.0.0` (npm 1.0.8); OMOP reports `1.7.1` (FastMCP's default, i.e. the mcp SDK). Recordings and snapshots store both the configured version (`configs/servers.yaml`) and the reported serverInfo.
- **No server returns JSON-RPC errors**, not even for schema violations; all errors are `is_error` results (Medical MCP: `MCP error -32602: Input validation error...`). Medical MCP also reports "not found" as normal text, never `is_error`.
- **Silent wrong answers worth targeting later:** `get-drug-details` with `0000-0000` returns an unrelated sunscreen; `find_omop_concept` with `omop_table: condition` returns candidates instead of an error.
- **WHO GHO API was down (HTTP 500) on 2026-11-01.** Medical MCP turns that into "No health indicators found", so both `get-health-statistics` fixtures are outage responses. Re-record them (`toolhelper record ... --only-missing` after deleting the two files) once `https://ghoapi.azureedge.net/api/Indicator` answers.
- Catalog snapshots are versioned JSON in `data/catalog/` until Phase 3 can store them as ToolSchema versions. Tokens are counted on the LLM view (function format, namespaced name) with `cl100k_base`, labelled estimated. BioMCP's own `biomcp mcp tools` dump counts 2,545 vs our 2,329 (it includes titles/annotations).
- Live schema drift from README §1.1: `gene_cspec` also takes `limit` and `offset`.

### Context store (Phase 3, 2026-11-01)

- Local store is `data/context.sqlite` (gitignored). `toolhelper catalog` now also writes McpServer (`servers/<server>`) and ToolSchema (`tools/<server>/<tool>`) versions; `toolhelper context head|show|history|stale|import-catalog` read it.
- Writes follow WarmHub's documented rules: names unique among active things, retract releases the name, identical revise is a no-op, `expected_version` conflicts, data validated with the SDK's own `warmhub.rules.validate_against_shape` (undeclared fields are warnings, not errors).
- **Bare wrefs in wref fields are pinned to `name@vN` at write time.** A link is stale when its target has a newer version or is retracted; `CommitResult.stale` lists dependents a commit just made stale. Staleness is derived, not written back into Recipe `status`.
- **Evidence is stored as `Evidence` assertion things** (`evidence/<thing>/v<N>/<i>`, `about` + `affirmed_targets` pinned to the version it supports), not a separate table: WarmHub has no evidence relation and its docs say relationships belong in assertions. Evidence links never count as dependencies.
- Shape types the README leaves implicit: `success_rate`, `mean_tokens`, `expand_rate`, `ttl_seconds` are numbers; Recipe and Decision `status` use ToolCard's `proposed|accepted` enum. `RunSummary` and `Experiment` have no fields yet.
- `toolhelper publish --repo org/name` pushes only accepted, non-stale things with their evidence, targets first, remapping pins to remote version numbers. **Untested live: no `WH_TOKEN` yet**; tested against a second local store and a fake SDK repo. `warmhub.rules.shape_definition` (used to preflight shapes in a test) is not a root export of the SDK.

### Agent loop and tracer (Phase 4, 2026-11-01)

- `toolhelper run --llm <profile> --prompt ... [--mode replay|live|record]` runs one task and saves a trace to `data/traces/<run_id>.json` (gitignored); `toolhelper trace show <run_id>` prints it. Replay mode falls through to live on a miss and records into `data/replay/`.
- **Strategies report changes with `self.run.transform(target, before=, after=, evidence=)`.** The loop diffs every hook's input and output; an unreported change is recorded with `reported=False` and no evidence, so provenance coverage drops below 1.0 and `Trace.problems()` names the hook. `before_call` decisions carry their own evidence.
- Cached results go through `after_result` like fresh ones. Calls to tools that exist but weren't offered get `error_class="not_offered"`; the loop's own error texts (not offered, unknown tool, timeout) are not server responses.
- Ollama and the scripted backend number tool calls per turn (`call_0`); the loop renames repeats to `t<turn>_call_<i>` so call ids are unique per run.
- Each LLM call record keeps measured tokens next to an estimated per-component breakdown (cl100k_base) and the tool results present in that prompt (for carried tokens in Phase 5). First live run: qwen3 measured 2,791 prompt tokens where cl100k estimated ~3,724 (3,628 of it tool definitions), so the estimates need Phase 5's calibration before they're compared to measured counts.

### Monitoring (Phase 5, 2026-11-01)

- `toolhelper run` ingests each trace into `data/monitoring.sqlite` (gitignored); `toolhelper monitor ingest` reloads `data/traces/`; `toolhelper report` writes `report.json` + `report.html` to `data/reports/<ts>/` (gitignored).
- **`toolhelper tokens tools --llm <profile>`** measures the exact tool-definition cost (2 + 2×N calls, `max_tokens=1`, a 12-digit nonce per call; hex nonces tokenize to varying lengths and add noise). Saved to `data/catalog/tool-costs-<model>-<ts>.json`. qwen3:8b, 2026-11-01: all 19 tools = **2,674 prompt tokens** (cl100k estimated 3,628); fixed tools header 77; `biomcp__search` alone 696 marginal. Marginals + header summed exactly to the total, so caching didn't distort it.
- Token splits per LLM call: estimated parts (cl100k) scaled to the measured total; when a tool-cost measurement matches the run's model and catalog version, `tool_defs` is anchored to the measured value and only the rest is scaled (basis `measured, tool_defs anchored`). Re-run `tokens tools` after any catalog change or model swap.
- "Used" results and context items are a heuristic: identifiers (tokens containing a digit) introduced by a result that reappear in later arguments or the answer. Results with no new identifiers are `NULL`, not dead ends. Gene symbols without digits (BRAF) aren't tracked.
- BioMCP suggestion-following compares the next call (typed calls mapped to their CLI command) with backticked commands in the previous BioMCP result, words up to the first flag.
- The Streamlit dashboard (optional per README) is not built.

### Benchmark (Phase 6, 2026-11-01)

- `data/tasks/benchmark.jsonl` is generated by `scripts/make_benchmark.py` (edit the script, not the JSONL): 54 tasks, 16 medical / 15 biomcp / 13 omop / 10 cross, 12 traps (22%). `{repo}` in prompts expands to the repo root. Extra checks beyond the README: `must_call_all`, `called_with` (trap handling, e.g. `country: JPN`), `answer_grounded`.
- `toolhelper bench validate|run|check|rescore`. `bench run --mode record` overwrites recordings; `--strict-replay` turns live fallback off. Change a task, then `bench rescore --group <g>` instead of rerunning the LLM. Summaries count the latest run per task.
- **Every expected answer string was confirmed in a real response** on 2026-11-01, but those recordings were deleted with the Ollama results. After the new record run, re-confirm what `bench check` lists with direct `toolhelper call --mode record` calls along the task's expected path. `bench check` only searches the group's own traces, so it still lists expectations the model never reached.
- **Experiments moved off Ollama (2026-11-02).** Local qwen3:8b ran out of memory (CUDA OOM), so the default LLM is now `nim-mistral` (Mistral Large 4, open-weight, on NVIDIA's free hosted API; `NVIDIA_API_KEY` is a Windows user env var, ~40 RPM; the profile throttles to 30 RPM and retries 429s). Kimi K2.6 and Nemotron 3 Super return 404 for this account; GLM/DeepSeek/Gemma/gpt-oss timed out. Mistral always reasons (it can't be disabled; reasoning tokens are reported separately) and may ignore `seed`. All Ollama run results were deleted and the baseline is re-recorded from scratch.
- **Baseline `bench-baseline-record` (nim-mistral, all 19 tools, 2026-11-02): 43/54 = 80%** (medical 10/16, biomcp 12/15, omop 12/13, cross 9/10). bio-005 was re-run after an NVIDIA `APITimeoutError`. Failures: WHO API down (HTTP 500, med-005/006 loop to max turns), NDC bug (med-004), BioMCP typed-search misuse (bio-003/014, plus `entity: drug` and shell pipes in med-002/009), ClinGen CSpec `capture_corrupt` (bio-012), never answering before the 8-turn limit (med-002/007/009, x-008), omop-009 choosing 3013762 over 3025315. `bench check`: only med-004's "sertraline" is not in any tool output (Medical MCP never shows NDCs).
- **Strict replay `bench-baseline-replay-strict`: 0 live calls, 67 served from recordings, but 38/54 tasks diverged (ReplayMiss), 15 on the first call; median 1 call before divergence.** Clean by server: omop 10/13, medical 5/16, biomcp 1/15, cross 0/10. All 16 clean tasks had the same outcome as the record run. The hosted model is not reproducible (seed apparently ignored), so experiments need normal replay (live fallback) plus repeated runs per configuration.
- **`bench run --retry-timeouts N` (default 1)** re-runs, after the pass, tasks that ended in `APITimeoutError`/`APIConnectionError`. The timed-out trace is kept; the retry's `retry_of` names it, and scores use the latest run per task. Model failures and `ReplayMiss` are never retried. SDK-level retries inside one request are still not logged in the trace.
- **NVIDIA free tier can stall a request indefinitely** (no reply for 18+ min); profiles set `timeout_s: 300`, `max_retries: 3`. Turn latency median 25 s, tail up to 850 s; a full 54-task pass takes ~3–4 h. Long runs are launched as detached processes (`Start-Process`), because Claude Code kills its own background shells when system memory is low.
- **Measured tool cost, Mistral Large 4 (2026-11-02): all 19 tools = 4,804 prompt tokens**, `biomcp__search` 1,489 marginal; marginals + 3-token header sum exactly.
- **Medical MCP's `get-drug-details` sends `openfda.product_ndc:<ndc>` unquoted** and returns another product from the same labeler (47781-951 = sertraline in openFDA; the server returns estradiol). med-004 cannot pass without grounding.
- Medical MCP's RxNorm search lists products only, never ingredient RxCUIs (metformin's 6809 is not in its output; OMOP's `find_omop_concept` shows it as the RxNorm code).
- OMOPHub's top 5 for "HbA1c" or "serum creatinine" miss the standard LOINC concepts (3004410, 3016723); LOINC-style keywords find them. A grounding target.
- **Memory:** qwen3:8b at `num_ctx` 32768 holds ~10.3 GB (6.3 GB VRAM) of 15.3 GB. A long run was killed for low memory, and Ollama once failed to load the model (out of memory at startup). Check free memory before long runs.

### Datasets (Phase 7, 2026-11-02)

- `toolhelper datasets fetch|build|list|resolve`; sources in `configs/datasets.yaml`. Fetch streams files to `<datasets dir>/raw/<name>/` with `_fetch.json` (URL, Last-Modified, sha256); build loads them into `<datasets dir>/index.sqlite` and writes `datasets/<name>` Dataset things (a new version only when the files' sha256 changes). Every match carries its dataset pinned (`datasets/hgnc@v1`) and the source row.
- **`TOOLHELPER_DATASETS_DIR` is a Windows user env var** pointing outside OneDrive; shells started earlier must read it like the API keys. The index is 2.4 GB (ClinVar: 3.7M rows, ~10M surfaces). Evidence rows are zlib-compressed one at a time, which barely shrinks ~400-byte rows (a shared zlib dictionary would). A full ClinVar build takes ~11 min.
- Matching: exact (case-insensitive) > normalized (`normalize`: case, accents, punctuation) > fuzzy (rapidfuzz `ratio` >= 85 over FTS5 trigram candidates; `WRatio` matched "heart attack" to "Heart (Bovine), Lung (Bovine)"). Grouped by (type, canonical); a near-tie within the best type is `ambiguous` and returns the top two. Types rank by match quality, then config order, so "Zoloft" gives NDC's sertraline before RxNorm's brand entry. ClinVar is exact/normalized only.
- **Country -> ISO3 uses the World Bank list** (README says WHO GHO, down with HTTP 500 on 2026-11-01/02; same alpha-3 codes, aggregates skipped). The `who_gho` indicator loader is written against documented columns but **unverified**; build it when GHO answers.
- **OMOP: `omop_eunomia` is a stand-in**, OHDSI's public Eunomia GiBleed sample CDM (Apache-2.0, GitHub): real vocabulary rows, but only 418 standard concepts from 2018 releases. It gives "heart attack" -> 4329847 (Done-when) but lacks e.g. HbA1c 3004410 and body weight 3025315, so grounding experiments need the full **`omop_vocab`, a manual Athena download** (account + licence acceptance by the user; SNOMED, LOINC, RxNorm: CONCEPT, CONCEPT_SYNONYM and VOCABULARY into `raw/omop_vocab/`). Same loader (`delimiter`, `archive`, case-insensitive headers); the Athena format itself is still unverified. 'Maps to' relationships are not loaded.
- RxNorm comes from RxNav `allconcepts` (IN and BN only); brand -> ingredient is not mapped (NDC covers brand -> generic). NDC `product.txt` is Latin-1 and read unquoted (it has two stray quotes); HGNC quotes multi-value cells.
- Test fixtures in `tests/fixtures/datasets/` are real excerpts cut by `scripts/make_dataset_fixtures.py`.

### Efficiency strategies (Phase 8 plan, 2026-11-02)

- **Done when (agreed; README has none):** each of E1–E9 is a configurable strategy in `src/toolhelper/efficiency/` with its README variants, unit tests on the scripted backend plus recorded replays, and a replay smoke run on a few benchmark tasks with provenance coverage 1.0. Measuring which variants win is Phase 10.
- **E1 uses lexical search only:** FTS5 BM25 plus reciprocal-rank fusion over ToolCard fields (Paper Librarian's method, `C:\Users\neelg\OneDrive\Desktop\MCP\paper-librarian\src\librarian\search.py`), no embeddings (PyTorch is too heavy for this machine). No ToolCards existed yet; E1 generates `proposed` ones from the catalog and baseline traces.
- Build order: E9 (final answer at the turn limit; needs a loop hook), E4 (argument guard; the main baseline failure), E7, E6, E3, E2, E8, E1, E5.
- **Built (2026-11-02):** `src/toolhelper/efficiency/` (retrieval, compression, grounding, recipes, arg_guard, cache, shaping, compaction, final_answer, registry). `toolhelper bench run --strategy E4:mode=repair --strategy E1:top_k=8 ... --learn-from <group>` builds them with inputs learned from that group's traces (default `bench-baseline-record`), the recordings, the dataset index and omop_mcp's CDM map. The registry always applies them in the order E1 E2 E3 E5 E4 E7 E6 E8 E9.
- Loop changes: `at_turn_limit` hook (E9; one extra LLM call with no tools; stop reason `final_at_limit`, which scoring counts as answered) and `RunContext.offer(strategy, tools, evidence=)` (tools added mid-run, from the next turn; E1 `find_tools`).
- **E4 rules are learned from real errors**, not hand-written: "unknown <entity> search field: X" (with a rename when a later successful call moved the value, e.g. article `query` -> `keyword`), "invalid <entity> section: S", BioMCP's read-only refusal of shell syntax. Plus the live JSON Schema, omop_mcp's CDM field map, ISO3 via the countries dataset, and NDC existence in the FDA directory (`EntityIndex.by_identifier`; 47781-951 is sertraline there). Test fixtures gained 7 real BioMCP recordings of these errors and their fixes.
- E6 drops a field only after `min_runs` runs saw it and none reused its value; top-N cuts need runs whose used items (judged by the item's id/name line) ranked within N. E8 uses Phase 5's identifier heuristic (tokens with a digit). Local tools: `toolhelper__expand_result` (E6), `toolhelper__recall_result` (E8), `toolhelper__resolve_entity` (E3), `toolhelper__find_tools` (E1), `toolhelper__get_recipe` (E5).
- **Leave-one-out:** E1 example tasks and E5 recipes never use the running task's own runs. E1 recall of needed tools on the benchmark (cards with baseline runs): k=3 0.79, k=5 0.83, k=8 0.96; server routing did not help.
- **Smoke run `phase8-smoke` (all nine strategies, nim-mistral, replay, 2026-11-02): done criterion met.** Every strategy acted at least once; provenance 1.00 on all 5 tasks. 4/5 passed vs 2/5 in the baseline (med-005 and bio-003 rescued by E9 at the turn limit; med-004 still hits the NDC server bug). Measured prompt tokens fell on 4 of 5 tasks (e.g. bio-001 24,198 -> 9,116); bio-003 rose (50,741 -> 67,268: 9 turns of BioMCP results, E9's closing turn re-sends the history). Tool definitions ~3,600 -> 900-2,000 estimated tokens per turn (E1+E2). One run per task: not evidence of improvement (Phase 10).
- Found while smoke-testing: `Trace.stop_reason` lacked `final_at_limit` (traces saved but would not reload; fixed, round-trip tested). E3 resolves names, not codes: an identifier hint (`47781-951` = sertraline in the FDA directory) could catch Medical MCP's wrong-product answer for med-004.
- Not done: E6's "minimal BioMCP `sections`" (an argument change, not shaping). E7's cache is in-memory per `bench run` unless `Inputs.cache_path` is set.

## Progress

| Phase | Topic | Status |
|---|---|---|
| 0 | Environment | done |
| 1 | LLM backends | done |
| 2 | MCP hub and catalog | done |
| 3 | Context store | done |
| 4 | Agent loop and tracer | done |
| 5 | Monitoring | done |
| 6 | Benchmark and record/replay | done |
| 7 | Datasets | done (WHO indicators and full Athena vocabularies pending) |
| 8 | Efficiency strategies | done |
| 9 | Trust | not started |
| 10 | Experiments and optimum | not started |
| 11 | Swap the LLM and rerun | not started |
| 12 | Gateway (optional) | not started |
