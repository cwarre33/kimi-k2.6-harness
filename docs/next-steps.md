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

**Current test status:** 52/52 core + benchmark tests passing.

---

## Remaining Work

### Phase 1: Research Existing Harnesses (Next Priority)
**Goal:** Understand how SWE-agent, AutoCodeRover, and Devin-style harnesses structure their TVC-like loops, tool use, and skill memory. Inform our design with proven patterns.

**Deliverable:** `docs/research/harness_comparison.md`

**Tasks:**
1. Read SWE-agent paper and codebase (https://github.com/princeton-nlp/SWE-agent)
   - How do they structure the agent loop?
   - What tools do they expose to the model?
   - How is the action parser implemented?
2. Read AutoCodeRover paper and codebase (https://github.com/nus-apr/auto-code-rover)
   - How do they combine AST analysis with LLM reasoning?
   - What is their skill/memory representation?
3. Skim Devin-style agent architectures (if open implementations exist)
4. Document comparison matrix:
   - Loop structure (TVC vs ReAct vs custom)
   - Tool set (shell, file, search, browser, etc.)
   - Memory/skill representation
   - Failure recovery strategy
   - Benchmark performance on SWE-bench

### Phase 2: Production Hardening
**Goal:** Make the harness robust enough for real benchmark runs.

**Tasks:**
1. **Ollama integration improvements**
   - Add streaming support for long-running generation
   - Add retry logic with exponential backoff for transient failures
   - Add request/response logging for debugging
   - Test against actual Ollama endpoint running kimi-k2.6

2. **Execute node hardening**
   - Add timeout configuration (currently hardcoded 60s)
   - Add working directory validation
   - Sanitize shell commands (prevent injection)
   - Handle long-running commands gracefully
   - Support multiple tool types (not just shell.exec)

3. **Error handling**
   - Add structured error types instead of generic `Exception`
   - Handle model refusal / empty responses
   - Handle subprocess timeouts
   - Handle graph checkpoint corruption

### Phase 3: Real Benchmark Integration
**Goal:** Move from mock evaluation to actual benchmark datasets.

**Tasks:**
1. **SWE-bench**
   - Integrate with actual SWE-bench dataset (JSON format)
   - Implement patch application and test execution
   - Add Docker/container isolation for test runs
   - Store results in reproducible format

2. **Terminal-Bench 2.0**
   - Find/download dataset
   - Implement terminal command evaluation
   - Match expected output format

3. **BrowseComp**
   - Integrate web browsing tool ( playwright / selenium )
   - Implement multi-step web navigation
   - Evaluate information extraction accuracy

4. **GAIA**
   - Integrate with GAIA dataset
   - Support multi-modal inputs if needed
   - Evaluate answer correctness

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

### Running Specific Test
```bash
pytest tests/core/test_ollama_client.py -v
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
- **Checkpointing:** AsyncSqliteSaver persists state between retries
- **IPC:** MessagePack over Unix sockets (local) or TCP (remote)
- **Skill Store:** SQLite with zstd-compressed traces, tag-based retrieval
- **Model Client:** httpx.AsyncClient → Ollama `/api/generate`, Bearer auth optional

## Design Decisions to Revisit

1. **Shell execution in execute node** — currently runs `subprocess.run(..., shell=True, ...)` with the raw LLM output. This is a security boundary that needs hardening before running untrusted code.
2. **Model prompt design** — plan/execute/correct prompts are hand-written and may need tuning for kimi-k2.6 specifically.
3. **Mock vs real benchmarks** — all adapters currently default to mock mode when harness is None. Need a clean way to switch to real evaluation.
