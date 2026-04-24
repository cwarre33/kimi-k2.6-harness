"""Tests for benchmark adapters and evaluation orchestrator."""

from pathlib import Path

import pytest

from benchmarks.eval_orchestrator import EvalOrchestrator
from benchmarks.sota_scores import load_sota_scores
from benchmarks.swe_bench_adapter import SWEBenchAdapter
from benchmarks.terminal_bench_adapter import TerminalBenchAdapter
from core.harness import Harness


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


@pytest.mark.asyncio
async def test_adapter_evaluates_instance_with_harness(tmp_path):
    skill_db = tmp_path / "skills.db"
    checkpoint_db = tmp_path / "checkpoints.sqlite"
    harness = Harness(
        skill_db_path=str(skill_db),
        checkpoint_db_path=str(checkpoint_db),
    )
    await harness.initialize()
    try:
        adapter = SWEBenchAdapter(repo_path=str(tmp_path), harness=harness)
        result = await adapter.evaluate_instance(
            instance_id="test-instance-001",
            patch="",
        )
        assert "instance_id" in result
        assert result["instance_id"] == "test-instance-001"
        assert "resolved" in result
        assert isinstance(result["resolved"], bool)
    finally:
        await harness.shutdown()


@pytest.mark.asyncio
async def test_terminal_bench_mock_score_computed():
    adapter = TerminalBenchAdapter(repo_path=".", mock_mode=True)
    instances = [
        {"instance_id": "t1", "expected_resolved": True},
        {"instance_id": "t2", "expected_resolved": False},
        {"instance_id": "t3", "expected_resolved": True},
    ]
    result = await adapter.evaluate_dataset(instances)
    assert result["benchmark"] == "terminal_bench_2_0"
    assert result["total"] == 3
    assert result["resolved"] == 2
    assert result["score"] == pytest.approx(2 / 3)
