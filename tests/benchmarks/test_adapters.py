"""Tests for benchmark adapters and evaluation orchestrator."""

from pathlib import Path

import pytest

from benchmarks.eval_orchestrator import EvalOrchestrator
from benchmarks.sota_scores import load_sota_scores
from benchmarks.swe_bench_adapter import SWEBenchAdapter


def test_load_sota_scores_has_swe_bench():
    scores = load_sota_scores()
    assert "swe_bench_verified" in scores
    assert scores["swe_bench_verified"]["top_open_source"]["score"] > 0.0


def test_load_sota_scores_has_terminal_bench():
    scores = load_sota_scores()
    assert "terminal_bench_2_0" in scores


def test_adapter_initializes_with_repo():
    adapter = SWEBenchAdapter(repo_path="/tmp/test-repo")
    assert adapter.repo_path == Path("/tmp/test-repo")


def test_orchestrator_loads_targets():
    orch = EvalOrchestrator()
    targets = orch.load_targets()
    assert "swe_bench_verified" in targets
    assert targets["swe_bench_verified"]["our_target"]["score"] == 0.60


def test_orchestrator_blocks_when_score_below_target():
    orch = EvalOrchestrator()
    comparison = orch.compare_to_sota("swe_bench_verified", 0.55)
    assert comparison["passed_target"] is False
    assert comparison["recommendation"] == "BLOCKED — score below target"


def test_orchestrator_proceeds_when_score_above_target():
    orch = EvalOrchestrator()
    comparison = orch.compare_to_sota("swe_bench_verified", 0.65)
    assert comparison["passed_target"] is True
    assert comparison["recommendation"] == "PROCEED"
