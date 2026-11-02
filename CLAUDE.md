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
- Datasets go in `data/datasets/` (gitignored). Secrets (`OMOPHUB_API_KEY`, `WH_TOKEN`) come from the environment, never from committed files.

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

## Progress

| Phase | Topic | Status |
|---|---|---|
| 0 | Environment | done |
| 1 | LLM backends | done |
| 2 | MCP hub and catalog | done |
| 3 | Context store | done |
| 4 | Agent loop and tracer | not started |
| 5 | Monitoring | not started |
| 6 | Benchmark and record/replay | not started |
| 7 | Datasets | not started |
| 8 | Efficiency strategies | not started |
| 9 | Trust | not started |
| 10 | Experiments and optimum | not started |
| 11 | Swap the LLM and rerun | not started |
| 12 | Gateway (optional) | not started |
