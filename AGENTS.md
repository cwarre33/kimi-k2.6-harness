# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

Kimi-K2.6 Autonomous Harness — an AI agent orchestration system implementing a Trial-Verify-Correct (TVC) loop for autonomous code repair/task execution. Uses LangGraph state graphs, async SQLite for skill storage and checkpointing, and communicates via MessagePack IPC. See `docs/next-steps.md` for architecture details and roadmap.

### Running tests

All tests use mocks/stubs and require no external services:

```bash
pytest tests/core/ tests/benchmarks/ -v
```

### Running the harness

`main.py` has a known issue: it calls `Harness()` without the required `skill_db_path` argument. To run the harness end-to-end with mocked Ollama, instantiate `Harness` directly with explicit paths (see `tests/core/test_harness.py` for examples).

Real Ollama usage requires an Ollama server at `localhost:11434` with the `kimi-k2.6` model. Env vars: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_API_KEY`, `OLLAMA_TIMEOUT_SECONDS`.

### Linting

No linter is configured in the repo. `ruff check .` works well; existing code has minor unused-import warnings (pre-existing, not regressions).

### Key gotcha: langgraph-checkpoint-sqlite

The `langgraph` pip package does **not** include the SQLite checkpoint module. You must also install `langgraph-checkpoint-sqlite` for `from langgraph.checkpoint.sqlite import SqliteSaver` to resolve.

### SWE-bench Lite report runner

`benchmarks/run_swe_lite_report.py` runs the SWE-bench Lite dev suite and produces SOTA comparison reports. Without the HuggingFace `datasets` package, it falls back to a placeholder instance that always fails the gate (exit code 1) — this is correct behavior. Run with:

```bash
python3 -m benchmarks.run_swe_lite_report --output-dir benchmarks/results/runs/<timestamp>
```

### No dependency manifest

This repo has no `requirements.txt`, `pyproject.toml`, or lockfile. The update script installs packages directly via pip. If new imports are added, the update script must be updated accordingly.
