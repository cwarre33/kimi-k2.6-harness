"""Tests for real SWE-bench adapter with dataset loading."""

import json
from pathlib import Path

import pytest

from benchmarks.swe_bench_adapter import SWEBenchAdapter


class TestSWEBenchAdapterRealData:
    """Test suite for SWE-bench adapter with real dataset."""

    def test_load_instances_from_json(self, tmp_path):
        dataset = [
            {
                "instance_id": "test__test-1",
                "repo": "test/repo",
                "base_commit": "abc123",
                "patch": "diff --git a/test.py",
                "problem_statement": "Fix the bug",
            }
        ]
        dataset_path = tmp_path / "dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        adapter = SWEBenchAdapter(
            repo_path=str(tmp_path),
            dataset_path=str(dataset_path),
        )
        instances = adapter._load_instances()

        assert len(instances) == 1
        assert instances[0]["instance_id"] == "test__test-1"
        assert instances[0]["repo"] == "test/repo"

    def test_load_real_dataset(self):
        """Test loading the actual SWE-bench Lite dev dataset."""
        dataset_path = Path("benchmarks/data/swe-bench-lite-dev.json")
        if not dataset_path.exists():
            pytest.skip("Real dataset not downloaded")

        adapter = SWEBenchAdapter(
            repo_path=".",
            dataset_path=str(dataset_path),
        )
        instances = adapter._load_instances()

        assert len(instances) == 23
        assert all("instance_id" in inst for inst in instances)
        assert all("repo" in inst for inst in instances)
        assert all("base_commit" in inst for inst in instances)
        assert all("problem_statement" in inst for inst in instances)

    @pytest.mark.asyncio
    async def test_evaluate_instance_not_found(self, tmp_path):
        from unittest.mock import AsyncMock

        dataset = [
            {
                "instance_id": "test__test-1",
                "repo": "test/repo",
                "base_commit": "abc123",
                "patch": "",
                "problem_statement": "Fix the bug",
            }
        ]
        dataset_path = tmp_path / "dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        mock_harness = AsyncMock()
        adapter = SWEBenchAdapter(
            repo_path=str(tmp_path),
            harness=mock_harness,
            dataset_path=str(dataset_path),
        )

        result = await adapter.evaluate_instance("nonexistent", "")
        assert result["resolved"] is False
        assert "not found" in result["test_output"]
