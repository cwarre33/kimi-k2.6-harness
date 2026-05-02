# Next Steps — Kimi-K2.6 Agentic Harness

**Current Branch:** `feature/kimi-harness`  
**Last Updated:** 2026-05-02

---

## Completed

### 1. Benchmark Gate System
- `benchmarks/swe_bench_adapter.py` — SWE-bench adapter with real repo cloning, test patch application, FAIL_TO_PASS/PASS_TO_PASS evaluation
- `benchmarks/terminal_bench_adapter.py` — Terminal-Bench 2.0 adapter
- `benchmarks/browsecomp_adapter.py` — BrowseComp adapter
- `benchmarks/gaia_adapter.py` — GAIA adapter
- `benchmarks/eval_orchestrator.py` — Orchestrator with SOTA comparison and pass/fail gate
- `benchmarks/sota_scores.yaml` — Target scores for each benchmark
- `tests/benchmarks/test_adapters.py` — Full test coverage including gate pass/fail

### 2. Production Wiring — Ollama Integration
- `core/model_config.py` — Env-configurable Ollama settings (URL, model, API key, timeout)
- `core/ollama_client.py` — Async httpx client for `/api/generate` with optional Bearer auth, timeout/retry logic, ConnectError handling
- `core/tvc_nodes.py` — Model-driven plan, execute, and correct nodes
  - Plan: LLM generates step-by-step reasoning with skill retrieval
  - Execute: Inner tool-use loop (max 20 steps) with action parser + tool registry
  - Correct: LLM analyzes failures, injects relevant skills, resets to pending
- `core/tvc_graph.py` — Forwards `model_client` and `tool_registry` to all relevant nodes
- `core/harness.py` — Owns OllamaClient lifecycle, builds sandbox-aware tool registry
- `tests/core/test_ollama_client.py` — 4 tests (defaults, generate, auth, errors)
- `tests/core/test_tvc_graph.py` — Added `test_async_plan_node_calls_ollama`
- `tests/core/test_harness.py` — Added `test_harness_runs_task_with_model` (end-to-end smoke)

### 3. Action Parser + ACI-Style Tools
- `core/action_parser.py` — Structured DISCUSSION + COMMAND parser with retry fallback
  - Supports markdown code fences and plain command fallback
  - `parse_once()` raises `ParseError`, `parse()` returns `__retry__` action
- `core/tools/` — ACI-style tool registry:
  - `shell_tool.py` — `shell.exec` with timeout, output truncation, safety checks
  - `file_tools.py` — `open` (windowed viewer), `view_file` (head/tail for large files), `search_file` (50-match limit), `search_dir` (50-match limit), `replace_string` (exact match + difflib hints + Python syntax guardrail), `edit` (line-range replacement + syntax guardrail)
  - `submit_tool.py` — `submit` to end episode
- `core/tools/__init__.py` — `ToolRegistry`, `ToolResult`, `create_default_registry()`
- `tests/core/test_action_parser.py` — 13 tests
- `tests/core/test_tools.py` — 21 tests

### 4. Real Benchmark Dataset Integration
- Downloaded SWE-bench Lite dev set (23 instances) from HuggingFace
- `benchmarks/data/swe-bench-lite-dev.json` — Real instances in JSON format
- `benchmarks/swe_bench_adapter.py` — Updated to:
  - Load instances from JSON via `_load_instances()`
  - Clone repos and checkout base commits via `_setup_repo()`
  - Run harness against real codebases
  - Extract model-generated patches via `_get_patch_diff()`
  - Run FAIL_TO_PASS + PASS_TO_PASS test validation
- `run_real_benchmark.py` — End-to-end runner script
  - `--instance` — Run single instance
  - `--full-suite` — Run full dev set with concurrent workers
  - `--model` / `--cloud` — Ollama local or cloud
  - `--workers` — Concurrent instances (default: 4)
- `calibrate_benchmark.py` — Calibration harness for ground-truth patch validation
- `benchmarks/run_swe_lite_report.py` — Automation-friendly report generator

### 5. First Real Benchmark Run (2026-05-01/02)
- **Model:** kimi-k2.6:cloud via Ollama Cloud API
- **Suite:** SWE-bench Lite dev (23 instances)
- **Workers:** 4 concurrent
- **Score:** 16/23 resolved = **69.6%**
- **Target:** 60% — **PASSED**

| Status | Instances |
|---|---|
| PASS | marshmallow-{1343,1359}, pvlib-{1072,1154}, pydicom-{1256,1413,1694}, astroid-{1268,1333,1866,1978}, pyvista-4315, sqlfluff-{1517,1625,1733,1763} |
| FAIL | pvlib-{1606,1707,1854}, pydicom-{1139,901}, astroid-1196, sqlfluff-2419 |

**Current test status:** 87 passed, 2 skipped.

---

## Remaining Work

### Phase 1: Legitimate Published Benchmark Score (Next Priority)
**Goal:** Get a score that can be submitted to the SWE-bench leaderboard.

**Why the current 69.6% is NOT publishable yet:**
1. **Wrong split** — We ran the 23-instance dev split, not the 300-instance test split used for leaderboard scores
2. **No Docker isolation** — Tests ran on macOS with local Python 3.12. Official eval uses Docker with exact Python versions/dependencies
3. **Test patch visible** — Test patch is applied before the model starts, potentially leaking expected behavior
4. **Small sample** — 23 instances is too small for statistical significance

**Tasks:**
1. **Integrate SWE-bench official evaluation harness**
   - Use `swe-bench` Python package for Docker-based evaluation
   - Run on the **SWE-bench Lite test split** (300 instances)
   - Generate predictions file in the required format (`instance_id`, `model_patch`, `model_name_or_path`)
   - Deliverable: `benchmarks/swe_bench_official_eval.py` or integration script

2. **Docker containerization**
   - Either use the official SWE-bench Docker harness or build our own
   - Ensure exact Python/dependency versions per repo
   - Run `swe-bench.harness.run_evaluation` or equivalent
   - Deliverable: Docker-based evaluation pipeline

3. **Submit to leaderboard**
   - Generate predictions file for test split
   - Run official evaluation (or submit to SWE-bench website)
   - Document score comparison

### Phase 2: Production Hardening
**Goal:** Make the harness robust for unattended runs at scale.

**P0 — Critical:**
1. **Fix timeout loop bug**
   - `pydicom-901` and `astroid-1196` got stuck in 20-hour timeout loops
   - Add a global episode timeout (e.g. 30 min max per instance)
   - Add early termination when model repeatedly produces same parse errors or shell quoting failures

2. **Shell quoting**
   - Model frequently generates unescaped quotes in `shell.exec` commands
   - Either escape in the tool or improve system prompt instructions
   - Current workaround: shell.exec errors are excluded from verification

**P1 — Important:**
3. **Streaming generation**
   - Add streaming support for long-running generation
   - Current 300s timeout is hit frequently

4. **Larger context window**
   - kimi-k2.6 has a large context window; we're truncating file views at 10KB
   - Consider using the full window for large files

### Phase 3: Other Benchmarks
**Goal:** Expand beyond SWE-bench.

**Tasks:**
1. **Terminal-Bench 2.0** — Find/download dataset, adapt harness
2. **BrowseComp** — Integrate web browsing tool (playwright / selenium)
3. **GAIA** — Integrate with GAIA dataset

---

## Quick Reference

### Running Tests
```bash
pytest tests/core/ tests/benchmarks/ -q
```

### Running Real Benchmark
```bash
# Single instance
python run_real_benchmark.py --instance sqlfluff__sqlfluff-1625 --cloud --model kimi-k2.6:cloud

# Full suite with Ollama Cloud
python run_real_benchmark.py --full-suite --cloud --model kimi-k2.6:cloud --workers 4
```

### Running Specific Test
```bash
pytest tests/core/test_ollama_client.py -v
pytest tests/core/test_action_parser.py -v
pytest tests/core/test_tools.py -v
```

### Branch
```bash
git checkout feature/kimi-harness
```

### Environment Variables
```bash
export OLLAMA_BASE_URL="https://ollama.com"
export OLLAMA_MODEL="kimi-k2.6:cloud"
export OLLAMA_API_KEY="your-key-here"
export OLLAMA_TIMEOUT_SECONDS="300"
```

---

## Architecture Notes

- **TVC StateGraph:** plan → execute → verify → [end | correct → plan]
- **Action Parser:** DISCUSSION + COMMAND format, markdown code fences, retry on malformed
- **Tool Registry:** `shell.exec`, `open`, `search_file`, `search_dir`, `replace_string`, `edit`, `submit`
- **Checkpointing:** AsyncSqliteSaver persists state between retries
- **IPC:** MessagePack over Unix sockets (local) or TCP (remote)
- **Skill Store:** SQLite with zstd-compressed traces, tag-based retrieval
- **Model Client:** httpx.AsyncClient → Ollama `/api/generate`, Bearer auth optional, timeout/retry
- **Benchmark Data:** SWE-bench Lite dev set at `benchmarks/data/swe-bench-lite-dev.json`
