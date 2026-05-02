"""Run SWE-bench Lite dev suite and generate a SOTA comparison report.

This is designed to be automation-friendly (CI / Cursor Automations):
- Runs the full suite via `run_real_benchmark.py`
- Reads the produced `full_suite_summary.json`
- Compares against SOTA/targets in `benchmarks/sota_scores.yaml`
- Writes a single report JSON to the output directory
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from benchmarks.eval_orchestrator import EvalOrchestrator


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench Lite dev suite and save eval report"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name (defaults to OLLAMA_MODEL env / repo default)",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Use Ollama Cloud (requires OLLAMA_API_KEY)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent workers for suite (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory to write per-instance results and reports",
    )
    parser.add_argument(
        "--report-name",
        default="eval_report.json",
        help="Filename for the comparison report JSON",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running benchmark; only generate report from existing summary",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        cmd = [sys.executable, "run_real_benchmark.py", "--full-suite"]
        if args.model:
            cmd += ["--model", args.model]
        if args.cloud:
            cmd += ["--cloud"]
        cmd += ["--workers", str(args.workers), "--output-dir", str(output_dir)]

        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            # Benchmark suite failures should fail the automation run.
            return completed.returncode

    summary_path = output_dir / "full_suite_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Expected summary at {summary_path}. Run without --skip-run."
        )

    summary = _load_json(summary_path)
    our_score = float(summary.get("score", 0.0))

    orchestrator = EvalOrchestrator(output_dir=str(output_dir))
    comparison = orchestrator.compare_to_sota("swe_bench_lite_dev", our_score)
    report: Dict[str, Any] = {
        "benchmark": "swe_bench_lite_dev",
        "our_score": our_score,
        "resolved": int(summary.get("resolved", 0)),
        "total": int(summary.get("total", 0)),
        "comparison": comparison,
        "summary_path": str(summary_path),
    }
    orchestrator.save_report(report, filename=args.report_name)

    # Gate: fail if below target. (Keeps automations honest.)
    return 0 if comparison.get("passed_target") else 2


if __name__ == "__main__":
    raise SystemExit(main())

