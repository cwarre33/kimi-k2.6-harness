"""Tests for the SWE-bench Lite report runner."""

import json

import pytest

from benchmarks import run_swe_lite_report


class FakeHarness:
    def __init__(self, resolved_by_id):
        self.resolved_by_id = resolved_by_id

    async def run_task(self, task_id, task_description, repo_path):
        resolved = self.resolved_by_id[task_id]
        return {
            "verification_outcome": "success" if resolved else "failure",
            "verification_details": f"details for {task_id}",
        }


@pytest.mark.asyncio
async def test_report_runner_writes_instance_summary_and_eval_report(tmp_path):
    instances = [
        {"instance_id": "suite/001", "problem_statement": "Fix one"},
        {"instance_id": "suite/002", "problem_statement": "Fix two"},
    ]
    harness = FakeHarness({"suite/001": True, "suite/002": True})

    result = await run_swe_lite_report.run_report(
        output_dir=tmp_path,
        workers=2,
        report_name="eval_report.json",
        instances=instances,
        harness=harness,
    )

    summary = json.loads(result.summary_path.read_text())
    report = json.loads(result.report_path.read_text())

    assert result.exit_code == 0
    assert summary["benchmark"] == "swe_bench_verified"
    assert summary["suite"] == "swe_bench_lite_dev"
    assert summary["total"] == 2
    assert summary["resolved"] == 2
    assert summary["score"] == pytest.approx(1.0)
    assert (tmp_path / "suite__001.json").exists()
    assert (tmp_path / "suite__002.json").exists()
    assert report["overall_passed"] is True
    assert report["comparison"]["our_score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_report_runner_exits_nonzero_when_below_target(tmp_path):
    instances = [
        {"instance_id": "suite/001", "problem_statement": "Fix one"},
        {"instance_id": "suite/002", "problem_statement": "Fix two"},
    ]
    harness = FakeHarness({"suite/001": True, "suite/002": False})

    result = await run_swe_lite_report.run_report(
        output_dir=tmp_path,
        workers=1,
        report_name="eval_report.json",
        instances=instances,
        harness=harness,
    )

    report = json.loads(result.report_path.read_text())

    assert result.exit_code == 1
    assert report["overall_passed"] is False
    assert report["comparison"]["passed_target"] is False


@pytest.mark.asyncio
async def test_report_runner_fails_gate_when_dataset_is_unavailable(tmp_path):
    result = await run_swe_lite_report.run_report(
        output_dir=tmp_path,
        workers=1,
        report_name="eval_report.json",
        instances=run_swe_lite_report._default_instances(),
        harness=FakeHarness({"swe-bench-lite-dev-unavailable": True}),
    )

    summary = json.loads(result.summary_path.read_text())

    assert result.exit_code == 1
    assert summary["resolved"] == 0
    assert summary["results"][0]["resolved"] is False
    assert "dataset is not installed" in summary["results"][0]["test_output"]
