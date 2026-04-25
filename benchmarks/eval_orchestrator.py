"""Evaluation orchestrator with SOTA comparison and production gate."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.sota_scores import load_sota_scores

logger = logging.getLogger(__name__)


class EvalOrchestrator:
    """Runs benchmark suites and produces comparison reports."""

    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sota = load_sota_scores()

    def load_targets(self) -> Dict[str, Any]:
        """Return loaded SOTA targets."""
        return self._sota

    def compare_to_sota(
        self, benchmark_name: str, our_score: float
    ) -> Dict[str, Any]:
        """Compare our score against published SOTA."""
        benchmark = self._sota.get(benchmark_name, {})
        target = benchmark.get("our_target", {}).get("score", 0.0)
        top_open = benchmark.get("top_open_source", {}).get("score", 0.0)

        passed = our_score >= target
        gap_to_sota = top_open - our_score

        return {
            "benchmark": benchmark_name,
            "our_score": our_score,
            "target_score": target,
            "top_open_source": top_open,
            "passed_target": passed,
            "gap_to_sota": gap_to_sota,
            "recommendation": (
                "PROCEED" if passed else "BLOCKED — score below target"
            ),
        }

    def save_report(
        self, report: Dict[str, Any], filename: str = "eval_report.json"
    ) -> Path:
        """Persist report to disk."""
        path = self.output_dir / filename
        with path.open("w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Evaluation report saved to {path}")
        return path

    async def run_full_suite(
        self, harness=None, mock_instances: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """Run all benchmark adapters and produce unified report."""
        from benchmarks.swe_bench_adapter import SWEBenchAdapter
        from benchmarks.terminal_bench_adapter import TerminalBenchAdapter
        from benchmarks.browsecomp_adapter import BrowseCompAdapter
        from benchmarks.gaia_adapter import GAIAAdapter

        report = {
            "harness_version": "0.1.0",
            "benchmarks": {},
            "overall_passed": False,
        }

        mock_instances = mock_instances or {}

        # SWE-bench Verified
        swe = SWEBenchAdapter(
            repo_path=".",
            harness=harness if not mock_instances.get("swe_bench_verified") else None,
            mock_mode=bool(mock_instances.get("swe_bench_verified")),
        )
        swe_result = await swe.evaluate_dataset(
            mock_instances.get("swe_bench_verified", [])
        )
        report["benchmarks"]["swe_bench_verified"] = self.compare_to_sota(
            "swe_bench_verified", swe_result["score"]
        )

        # Terminal-Bench 2.0
        terminal = TerminalBenchAdapter(
            repo_path=".",
            harness=harness if not mock_instances.get("terminal_bench_2_0") else None,
            mock_mode=bool(mock_instances.get("terminal_bench_2_0")),
        )
        terminal_result = await terminal.evaluate_dataset(
            mock_instances.get("terminal_bench_2_0", [])
        )
        report["benchmarks"]["terminal_bench_2_0"] = self.compare_to_sota(
            "terminal_bench_2_0", terminal_result["score"]
        )

        # BrowseComp
        browse = BrowseCompAdapter(
            repo_path=".",
            harness=harness if not mock_instances.get("browsecomp") else None,
            mock_mode=bool(mock_instances.get("browsecomp")),
        )
        browse_result = await browse.evaluate_dataset(
            mock_instances.get("browsecomp", [])
        )
        report["benchmarks"]["browsecomp"] = self.compare_to_sota(
            "browsecomp", browse_result["score"]
        )

        # GAIA
        gaia = GAIAAdapter(
            repo_path=".",
            harness=harness if not mock_instances.get("gaia") else None,
            mock_mode=bool(mock_instances.get("gaia")),
        )
        gaia_result = await gaia.evaluate_dataset(
            mock_instances.get("gaia", [])
        )
        report["benchmarks"]["gaia"] = self.compare_to_sota(
            "gaia", gaia_result["score"]
        )

        report["overall_passed"] = all(
            b["passed_target"] for b in report["benchmarks"].values()
        )

        self.save_report(report)
        return report
