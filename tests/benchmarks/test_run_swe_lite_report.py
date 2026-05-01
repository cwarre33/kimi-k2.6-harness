"""Tests for the SWE-bench Lite report runner."""

import json

import pytest

from benchmarks import run_swe_lite_report


class FakeHarness:
    def __init__(self, resolved_by_id):
        self.resolved_by_id = resolved_by_id
        self.task_descriptions = {}

    async def run_task(self, task_id, task_description, repo_path):
        self.task_descriptions[task_id] = task_description
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


def test_loader_normalizes_swe_bench_lite_dev_rows():
    def fake_load_dataset(dataset_name, split):
        assert dataset_name == "princeton-nlp/SWE-bench_Lite"
        assert split == "dev"
        return [
            {
                "repo": "django/django",
                "instance_id": "django__django-123",
                "base_commit": "abc123",
                "patch": "diff --git a/file.py b/file.py",
                "test_patch": "diff --git a/test.py b/test.py",
                "problem_statement": "Fix the failing Django issue.",
                "hints_text": "Look at the queryset path.",
                "created_at": "2024-01-01T00:00:00Z",
                "version": "4.2",
                "FAIL_TO_PASS": '["tests.test_issue"]',
                "PASS_TO_PASS": '["tests.test_regression"]',
                "environment_setup_commit": "def456",
            }
        ]

    instances = run_swe_lite_report.load_swe_bench_lite_dev_instances(
        load_dataset_fn=fake_load_dataset
    )

    assert instances == [
        {
            "repo": "django/django",
            "instance_id": "django__django-123",
            "base_commit": "abc123",
            "patch": "diff --git a/file.py b/file.py",
            "test_patch": "diff --git a/test.py b/test.py",
            "problem_statement": "Fix the failing Django issue.",
            "hints_text": "Look at the queryset path.",
            "created_at": "2024-01-01T00:00:00Z",
            "version": "4.2",
            "fail_to_pass": ["tests.test_issue"],
            "pass_to_pass": ["tests.test_regression"],
            "environment_setup_commit": "def456",
        }
    ]


@pytest.mark.asyncio
async def test_report_runner_passes_problem_statement_to_harness(tmp_path):
    instances = [
        {
            "instance_id": "suite/001",
            "problem_statement": "Use this real issue text.",
        }
    ]
    harness = FakeHarness({"suite/001": True})

    await run_swe_lite_report.run_report(
        output_dir=tmp_path,
        workers=1,
        report_name="eval_report.json",
        instances=instances,
        harness=harness,
    )

    assert harness.task_descriptions["suite/001"] == "Use this real issue text."
