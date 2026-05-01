"""Run SWE-bench Lite dev evaluations and write comparison reports."""

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from benchmarks.eval_orchestrator import EvalOrchestrator
from benchmarks.swe_bench_adapter import SWEBenchAdapter
from core.harness import Harness


SUITE_NAME = "swe_bench_lite_dev"
BENCHMARK_NAME = "swe_bench_verified"


@dataclass(frozen=True)
class ReportResult:
    """Paths and exit code produced by a report run."""

    summary_path: Path
    report_path: Path
    exit_code: int


def _safe_instance_filename(instance_id: str) -> str:
    """Return a filesystem-safe JSON filename for a SWE-bench instance."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", instance_id).strip("._")
    return f"{safe or 'instance'}.json"


def _default_instances() -> List[Dict[str, Any]]:
    """Return a minimal full-suite placeholder when no dataset package is present."""
    return [
        {
            "instance_id": "swe-bench-lite-dev-unavailable",
            "problem_statement": (
                "SWE-bench Lite dev dataset is not installed in this environment."
            ),
            "patch": "",
            "force_unresolved_reason": (
                "SWE-bench Lite dev dataset is not installed in this environment."
            ),
        }
    ]


def load_swe_bench_lite_dev_instances() -> List[Dict[str, Any]]:
    """Load SWE-bench Lite dev instances if the datasets package is available."""
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError:
        return _default_instances()

    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="dev")
    return [dict(instance) for instance in dataset]


async def _evaluate_instances(
    *,
    adapter: SWEBenchAdapter,
    instances: Iterable[Dict[str, Any]],
    output_dir: Path,
    workers: int,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, workers))

    async def evaluate(instance: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            instance_id = instance["instance_id"]
            forced_reason = instance.get("force_unresolved_reason")
            if forced_reason:
                result = {
                    "instance_id": instance_id,
                    "resolved": False,
                    "test_output": forced_reason,
                }
            else:
                result = await adapter.evaluate_instance(
                    instance_id,
                    instance.get("patch", ""),
                )
            record = {
                "benchmark": BENCHMARK_NAME,
                "suite": SUITE_NAME,
                "instance_id": instance_id,
                "resolved": bool(result["resolved"]),
                "test_output": result.get("test_output", ""),
            }
            instance_path = output_dir / _safe_instance_filename(instance_id)
            with instance_path.open("w") as f:
                json.dump(record, f, indent=2)
                f.write("\n")
            return record

    return await asyncio.gather(*(evaluate(instance) for instance in instances))


def _write_summary(output_dir: Path, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    resolved = sum(1 for result in results if result["resolved"])
    summary = {
        "benchmark": BENCHMARK_NAME,
        "suite": SUITE_NAME,
        "total": total,
        "resolved": resolved,
        "score": resolved / total if total else 0.0,
        "results": results,
    }
    with (output_dir / "full_suite_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


async def run_report(
    *,
    output_dir: Path,
    workers: int,
    report_name: str,
    instances: Optional[List[Dict[str, Any]]] = None,
    harness: Optional[Any] = None,
) -> ReportResult:
    """Run SWE-bench Lite dev and write per-instance, summary, and eval reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = instances if instances is not None else load_swe_bench_lite_dev_instances()
    owns_harness = harness is None
    harness = harness or Harness(
        skill_db_path=str(output_dir / "skills.db"),
        checkpoint_db_path=str(output_dir / "checkpoints.sqlite"),
    )

    if owns_harness:
        await harness.initialize()

    try:
        adapter = SWEBenchAdapter(repo_path=".", harness=harness)
        results = await _evaluate_instances(
            adapter=adapter,
            instances=instances,
            output_dir=output_dir,
            workers=workers,
        )
    finally:
        if owns_harness:
            await harness.shutdown()

    summary = _write_summary(output_dir, results)
    orchestrator = EvalOrchestrator(output_dir=str(output_dir))
    comparison = orchestrator.compare_to_sota(BENCHMARK_NAME, summary["score"])
    report = {
        "benchmark": BENCHMARK_NAME,
        "suite": SUITE_NAME,
        "summary_path": str(output_dir / "full_suite_summary.json"),
        "comparison": comparison,
        "overall_passed": comparison["passed_target"],
    }
    report_path = orchestrator.save_report(report, report_name)
    return ReportResult(
        summary_path=output_dir / "full_suite_summary.json",
        report_path=report_path,
        exit_code=0 if report["overall_passed"] else 1,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SWE-bench Lite dev suite and produce SOTA comparison reports."
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-name", default="eval_report.json")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    result = asyncio.run(
        run_report(
            output_dir=args.output_dir,
            workers=args.workers,
            report_name=args.report_name,
        )
    )
    print(f"SUMMARY: {result.summary_path}")
    print(f"REPORT: {result.report_path}")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
