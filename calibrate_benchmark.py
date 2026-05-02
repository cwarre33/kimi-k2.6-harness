"""Calibrate the benchmark infrastructure using ground-truth patches.

This script evaluates SWE-bench instances by applying the ground-truth patches
and running the test suites. It verifies that:
1. Repo cloning works
2. Patch application works
3. Test execution works
4. The scoring infrastructure is correct

Usage:
    python calibrate_benchmark.py [--instance INSTANCE_ID]
"""

import argparse
import asyncio
import json
import logging
import tempfile
from pathlib import Path

from benchmarks.swe_bench_adapter import SWEBenchAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def calibrate_instance(instance_id: str):
    """Run ground-truth evaluation on a single instance."""
    dataset_path = Path("benchmarks/data/swe-bench-lite-dev.json")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repos"
        repo_path.mkdir()

        adapter = SWEBenchAdapter(
            repo_path=str(repo_path),
            harness=None,
            mock_mode=False,
            dataset_path=str(dataset_path),
        )

        result = await adapter.evaluate_instance_with_ground_truth(instance_id)
        return result


async def calibrate_full_suite():
    """Run ground-truth evaluation on all instances."""
    dataset_path = Path("benchmarks/data/swe-bench-lite-dev.json")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "repos"
        repo_path.mkdir()

        adapter = SWEBenchAdapter(
            repo_path=str(repo_path),
            harness=None,
            mock_mode=False,
            dataset_path=str(dataset_path),
        )

        instances = adapter._load_instances()
        logger.info(f"Calibrating {len(instances)} instances...")

        resolved_count = 0
        results = []

        for inst in instances:
            instance_id = inst["instance_id"]
            logger.info(f"Calibrating {instance_id}...")
            result = await adapter.evaluate_instance_with_ground_truth(instance_id)
            results.append(result)
            if result["resolved"]:
                resolved_count += 1
                logger.info(f"  PASS")
            else:
                logger.info(f"  FAIL: {result['test_output'][:200]}")

        score = resolved_count / len(instances) if instances else 0.0
        logger.info(f"\n{'='*60}")
        logger.info(f"Calibration Results:")
        logger.info(f"  Total: {len(instances)}")
        logger.info(f"  Resolved: {resolved_count}")
        logger.info(f"  Score: {score:.2%}")
        logger.info(f"  Expected: ~100% (ground-truth patches)")
        logger.info(f"{'='*60}")

        # Save report
        report_path = Path("benchmarks/results/calibration_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({
                "mode": "calibration",
                "total": len(instances),
                "resolved": resolved_count,
                "score": score,
                "results": results,
            }, f, indent=2)
        logger.info(f"Report saved to {report_path}")

        return score


def main():
    parser = argparse.ArgumentParser(description="Calibrate SWE-bench infrastructure")
    parser.add_argument("--instance", default="sqlfluff__sqlfluff-1625", help="Instance ID to calibrate")
    parser.add_argument("--full-suite", action="store_true", help="Run all instances")
    args = parser.parse_args()

    if args.full_suite:
        score = asyncio.run(calibrate_full_suite())
    else:
        result = asyncio.run(calibrate_instance(args.instance))
        if result:
            logger.info(f"Result: resolved={result['resolved']}")
            logger.info(f"Output: {result['test_output'][:500]}")


if __name__ == "__main__":
    main()
