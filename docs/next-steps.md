# Next Steps — Kimi-K2.6 Agentic Harness

**Current Branch:** `feature/kimi-harness`  
**Last Updated:** 2026-04-25

---

## Completed

### 1. Benchmark Gate System
- `benchmarks/swe_bench_adapter.py` — SWE-bench adapter with mock mode
- `benchmarks/terminal_bench_adapter.py` — Terminal-Bench 2.0 adapter
- `benchmarks/browsecomp_adapter.py` — BrowseComp adapter
- `benchmarks/gaia_adapter.py` — GAIA adapter
- `benchmarks/eval_orchestrator.py` — Orchestrator with SOTA comparison and pass/fail gate
- `benchmarks/sota_scores.yaml` — Target scores for each benchmark
- `tests/benchmarks/test_adapters.py` — Full test coverage including gate pass/fail

### 2. Production Wiring — Ollama Integration
- `core/model_config.py` — Env-configurable Ollama settings (URL, model, API key, timeout)
- `core/ollama_client.py` — Async httpx client for `/api/generate` with optional Bearer auth
- `core/tvc_nodes.py` — Model-driven plan, execute, and correct nodes
  - Plan: LLM generates step-by-step reasoning
  - Execute: LLM suggests shell command, node runs it via subprocess
  - Correct: LLM analyzes failures and suggests fixes
- `core/tvc_graph.py` — Forwards `model_client` to all relevant nodes
- `core/harness.py` — Owns OllamaClient lifecycle
- `tests/core/test_ollama_client.py` — 4 tests (defaults, generate, auth, errors)
- `tests/core/test_tvc_graph.py` — Added `test_async_plan_node_calls_ollama`
- `tests/core/test_harness.py` — Added `test_harness_runs_task_with_model` (end-to-end smoke)

### 3. Research — Existing Harnesses
- `docs/research/harness_comparison.md` — Comparison of SWE-agent, AutoCodeRover, and Devin-style agents
  - Loop structure, tool set, memory representation, failure recovery
  - Design implications and prioritized recommendations for our harness

### 4. Action Parser + ACI-Style Tools (This Session)
- `core/action_parser.py` — Structured DISCUSSION + COMMAND parser with retry fallback
  - Supports markdown code fences and plain command fallback
  - `parse_once()` raises `ParseError`, `parse()` returns `__retry__` action
- `core/tools/` — ACI-style tool registry:
  - `shell_tool.py` — `shell.exec` with timeout, output truncation, safety checks
  - `file_tools.py` — `open` (windowed viewer), `search_file` (50-match limit), `search_dir` (50-match limit), `edit` (with Python syntax guardrail)
  - `submit_tool.py` — `submit` to end episode
- `core/tools/__init__.py` — `ToolRegistry`, `ToolResult`, `create_default_registry()`
- `core/tvc_nodes.py` — `make_async_execute_node` now uses action parser + tool registry instead of raw `subprocess.run(..., shell=True, ...)`
- `core/tvc_graph.py` — Forwards `tool_registry` to execute node
- `core/harness.py` — Owns `ToolRegistry` lifecycle, passes to graph builder
- `tests/core/test_action_parser.py` — 13 tests
- `tests/core/test_tools.py` — 21 tests
- `tests/benchmarks/test_real_adapter.py` — 3 tests for real dataset loading

### 5. Real Benchmark Dataset Integration (This Session)
- Downloaded SWE-bench Lite dev set (23 instances) from HuggingFace
- `benchmarks/data/swe-bench-lite-dev.json` — Real instances in JSON format
- `benchmarks/swe_bench_adapter.py` — Updated to:
  - Load instances from JSON via `_load_instances()`
  - Clone repos and checkout base commits via `_setup_repo()`
  - Run harness against real codebases
- `run_real_benchmark.py` — End-to-end runner script
  - `--instance` — Run single instance (default: `sqlfluff__sqlfluff-1625`)
  - `--mock` — Use mock model client for demonstration
  - `--full-suite` — Run full dev set
- `benchmarks/eval_orchestrator.py` — `dataset_paths` parameter for real data

**Current test status:** 87 passed, 2 skipped.

---

## Remaining Work

### Phase 2: Production Hardening (Next Priority)
**Goal:** Make the harness robust enough for real benchmark runs.

**P0 — Critical:**
1. **Sandboxed Execution**
   - Containerize execute node (Docker or Firecracker)
   - Mount workspace read/write, isolate from host
   - Add resource limits (CPU, memory, disk, network)
   - Deliverable: `core/sandbox.py` + tests

**P1 — Important:**
2. **Ollama Integration Improvements**
   - Add streaming support for long-running generation
   - Add retry logic with exponential backoff for transient failures
   - Add request/response logging for debugging
   - Test against actual Ollama endpoint running kimi-k2.6

3. **Error Handling**
   - Add structured error types instead of generic `Exception`
   - Handle model refusal / empty responses
   - Handle subprocess timeouts
   - Handle graph checkpoint corruption

**P2 — Nice to Have:**
4. **Execute Node Hardening**
   - Add timeout configuration (currently hardcoded 60s)
   - Add working directory validation
   - Sanitize shell commands (prevent injection)
   - Handle long-running commands gracefully

---

### Phase 3: Real Benchmark Integration (In Progress)
**Goal:** Move from mock evaluation to actual benchmark datasets.

**Completed:**
1. **SWE-bench** — Dataset downloaded, adapter loads real instances, runner works

**Remaining:**
2. **SWE-bench** — Patch application and test execution in cloned repos
3. **Terminal-Bench 2.0** — Find/download dataset
4. **BrowseComp** — Integrate web browsing tool (playwright / selenium)
5. **GAIA** — Integrate with GAIA dataset

---

### Phase 4: Skill Learning & Memory
**Goal:** Make the harness self-improving by learning from successes and failures.

**Tasks:**
1. **Skill extraction from traces**
   - After a successful task, automatically extract reusable skill
   - Compress thought traces using existing zstd compression
   - Tag skills with task patterns and outcomes

2. **Skill retrieval improvements**
   - Add semantic search using embeddings (currently keyword-only)
   - Implement skill ranking by validation success rate
   - Add skill versioning

3. **Hierarchical memory**
   - Implement L1-L4 memory tiers as documented in architecture
   - Add insight extraction from task history
   - Archive old sessions to cold storage

---

### Phase 5: Subagent-Driven Execution
**Goal:** Use subagent pattern for complex multi-step tasks.

**Tasks:**
1. Implement subagent dispatch within execute node
2. Add subagent result aggregation
3. Add parent-child session tracking
4. Update IPC protocol for subagent messages

---

## Quick Reference

### Running Tests
```bash
pytest tests/core/ tests/benchmarks/ -q
```

### Running Real Benchmark
```bash
# Single instance with mock model (demonstration)
python run_real_benchmark.py --mock --instance sqlfluff__sqlfluff-1625

# Full suite with real Ollama model
python run_real_benchmark.py --full-suite
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
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="kimi-k2.6"
export OLLAMA_API_KEY=""  # set if using reverse proxy
export OLLAMA_TIMEOUT_SECONDS="300"
```

---

## Architecture Notes for Next Session

- **TVC StateGraph:** plan → execute → verify → [end | correct → plan]
- **Action Parser:** DISCUSSION + COMMAND format, markdown code fences, retry on malformed
- **Tool Registry:** `shell.exec`, `open`, `search_file`, `search_dir`, `edit`, `submit`
- **Checkpointing:** AsyncSqliteSaver persists state between retries (optional — falls back to in-memory)
- **IPC:** MessagePack over Unix sockets (local) or TCP (remote)
- **Skill Store:** SQLite with zstd-compressed traces, tag-based retrieval
- **Model Client:** httpx.AsyncClient → Ollama `/api/generate`, Bearer auth optional
- **Benchmark Data:** SWE-bench Lite dev set at `benchmarks/data/swe-bench-lite-dev.json`

## Design Decisions to Revisit

1. **Mock model client** — `run_real_benchmark.py` defaults to `MockModelClient` when `--mock` is passed. Production runs should use `OllamaClient()` with Ollama running kimi-k2.6.
2. **Tool output truncation** — `ShellExecTool` truncates output at 10KB to prevent context window flooding. This may hide important error messages.
3. **No Docker sandbox** — Tools still run on the host filesystem. Before running untrusted code, implement `core/sandbox.py` with container isolation.
