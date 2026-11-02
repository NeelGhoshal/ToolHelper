# CLAUDE.md

ToolHelper monitors and reduces the token cost of LLM tool calls against three biomedical MCP servers (Medical MCP, BioMCP, OMOP MCP), keeping every change traceable.

**`README.md` is the spec.** Read the relevant phase in section 3 before starting work, and check section 1 (verified metadata) and section 4 (gotchas) before touching servers, SDKs or datasets. When the README and the code disagree, raise it rather than silently picking one.

## How we work

- One phase per session, in order. Plan first, then write tests, then implement, then run `uv run pytest`, then commit.
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

## Progress

| Phase | Topic | Status |
|---|---|---|
| 0 | Environment | done |
| 1 | LLM backends | not started |
| 2 | MCP hub and catalog | not started |
| 3 | Context store | not started |
| 4 | Agent loop and tracer | not started |
| 5 | Monitoring | not started |
| 6 | Benchmark and record/replay | not started |
| 7 | Datasets | not started |
| 8 | Efficiency strategies | not started |
| 9 | Trust | not started |
| 10 | Experiments and optimum | not started |
| 11 | Swap the LLM and rerun | not started |
| 12 | Gateway (optional) | not started |
