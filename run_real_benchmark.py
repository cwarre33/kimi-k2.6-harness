"""Run a real SWE-bench benchmark instance end-to-end.

This script demonstrates the full benchmark pipeline:
1. Load real SWE-bench Lite instances
2. Run the TVC harness with tool-use against a real repo
3. Evaluate results and produce a report

Usage:
    python run_real_benchmark.py

Requirements:
    - Ollama running with kimi-k2.6 (or use --mock for demonstration)
    - Git installed for repo cloning
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from core.harness import Harness
from core.ollama_client import OllamaClient
from benchmarks.swe_bench_adapter import SWEBenchAdapter
from benchmarks.eval_orchestrator import EvalOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class MockModelClient:
    """Mock model client that simulates tool-use for demonstration.

    In production, replace with OllamaClient(base_url="...", model="kimi-k2.6").
    """

    def __init__(self, commands: list[str] = None):
        self.commands = commands or [
            "DISCUSSION: I'll explore the repository structure first.\n\n```bash\nshell.exec ls -la\n```",
            "DISCUSSION: Let me look at the main source files.\n\n```bash\nopen README.md\n```",
            "DISCUSSION: Now I'll check the test file to understand the expected behavior.\n\n```bash\nshell.exec cat test/cli/commands_test.py | head -20\n```",
            "DISCUSSION: I need to search for the relevant code.\n\n```bash\nsearch_dir 'Avoid using aliases' src/\n```",
            "DISCUSSION: I found the issue. Let me view the file.\n\n```bash\nopen src/sqlfluff/rules/L031.py 211\n```",
            "DISCUSSION: I can see the bug. I'll fix the description text.\n\n```bash\nshell.exec sed -i 's/Avoid using aliases in join condition/Avoid aliases in from clauses and join conditions./' src/sqlfluff/rules/L031.py\n```",
            "DISCUSSION: Let me verify the fix by running the tests.\n\n```bash\nshell.exec python -m pytest test/cli/commands_test.py::test__cli__command_directed -xvs\n```",
            "DISCUSSION: The tests pass. I'll submit the fix.\n\n```bash\nsubmit\n```",
        ]
        self.index = 0

    async def generate(self, prompt: str, system: str = None, options: dict = None) -> str:
        if self.index < len(self.commands):
            cmd = self.commands[self.index]
            self.index += 1
            return cmd
        return "DISCUSSION: Task complete.\n\n```bash\nsubmit\n```"

    async def close(self):
        pass


async def run_single_instance(instance_id: str, mock: bool = False, model: str = "llama3.2:3b", cloud: bool = False):
    """Run a single SWE-bench instance end-to-end."""
    dataset_path = Path("benchmarks/data/swe-bench-lite-dev.json")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        logger.error("Download it first:")
        logger.error("  python -c \"import pandas as pd; df = pd.read_parquet('https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite/resolve/main/data/dev-00000-of-00001.parquet'); df.to_json('benchmarks/data/swe-bench-lite-dev.json', orient='records')\"")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_db = Path(tmpdir) / "skills.db"
        checkpoint_db = Path(tmpdir) / "checkpoints.sqlite"
        repo_path = Path(tmpdir) / "repos"
        repo_path.mkdir()

        if mock:
            model_client = MockModelClient()
        elif cloud:
            api_key = os.getenv("OLLAMA_API_KEY", "")
            if not api_key:
                logger.error("OLLAMA_API_KEY environment variable required for cloud mode")
                sys.exit(1)
            model_client = OllamaClient(
                base_url="https://ollama.com",
                model=model,
                api_key=api_key,
            )
            logger.info(f"Using Ollama Cloud model: {model}")
        else:
            model_client = OllamaClient(model=model)

        harness = Harness(
            skill_db_path=str(skill_db),
            checkpoint_db_path=str(checkpoint_db),
            ollama_client=model_client,
        )
        await harness.initialize()

        try:
            adapter = SWEBenchAdapter(
                repo_path=str(repo_path),
                harness=harness,
                mock_mode=False,
                dataset_path=str(dataset_path),
            )

            instances = adapter._load_instances()
            logger.info(f"Loaded {len(instances)} instances from dataset")

            target = next((i for i in instances if i["instance_id"] == instance_id), None)
            if target is None:
                logger.error(f"Instance {instance_id} not found")
                logger.info(f"Available instances: {[i['instance_id'] for i in instances]}")
                sys.exit(1)

            logger.info(f"\n{'='*60}")
            logger.info(f"Running benchmark on: {target['instance_id']}")
            logger.info(f"Repo: {target['repo']}")
            logger.info(f"Base commit: {target['base_commit']}")
            logger.info(f"{'='*60}\n")

            result = await adapter.evaluate_instance(
                target["instance_id"],
                target.get("patch", ""),
            )

            logger.info(f"\n{'='*60}")
            logger.info(f"Result: resolved={result['resolved']}")
            output = result.get('test_output') or ""
            logger.info(f"Output: {output[:500]}")
            patch_len = len(result.get('model_patch', ''))
            logger.info(f"Model patch size: {patch_len} chars")
            tool_count = len(result.get('tool_history', []))
            logger.info(f"Tool calls: {tool_count}")
            logger.info(f"{'='*60}")

            # Save report
            report_path = Path("benchmarks/results") / f"{target['instance_id']}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Report saved to {report_path}")

            return result

        finally:
            await harness.shutdown()


async def run_full_suite(mock: bool = False, model: str = "llama3.2:3b", cloud: bool = False, workers: int = 4, output_dir: str = "benchmarks/results"):
    """Run the full SWE-bench Lite dev suite with concurrent workers."""
    dataset_path = Path("benchmarks/data/swe-bench-lite-dev.json")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_db = Path(tmpdir) / "skills.db"
        checkpoint_db = Path(tmpdir) / "checkpoints.sqlite"
        repo_path = Path(tmpdir) / "repos"
        repo_path.mkdir()

        if mock:
            model_client = MockModelClient()
        elif cloud:
            api_key = os.getenv("OLLAMA_API_KEY", "")
            if not api_key:
                logger.error("OLLAMA_API_KEY environment variable required for cloud mode")
                sys.exit(1)
            model_client = OllamaClient(
                base_url="https://ollama.com",
                model=model,
                api_key=api_key,
            )
            logger.info(f"Using Ollama Cloud model: {model}")
        else:
            model_client = OllamaClient(model=model)

        harness = Harness(
            skill_db_path=str(skill_db),
            checkpoint_db_path=str(checkpoint_db),
            ollama_client=model_client,
        )
        await harness.initialize()

        try:
            adapter = SWEBenchAdapter(
                repo_path=str(repo_path),
                harness=harness,
                mock_mode=False,
                dataset_path=str(dataset_path),
            )
            instances = adapter._load_instances()
            logger.info(f"Loaded {len(instances)} instances, running with {workers} concurrent workers")

            semaphore = asyncio.Semaphore(workers)
            results_dir = Path(output_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            async def run_one(inst: dict) -> dict:
                async with semaphore:
                    instance_id = inst["instance_id"]
                    logger.info(f"[START] {instance_id}")
                    try:
                        result = await adapter.evaluate_instance(
                            instance_id,
                            inst.get("patch", ""),
                        )
                    except Exception as exc:
                        logger.exception(f"[ERROR] {instance_id}: {exc}")
                        result = {
                            "instance_id": instance_id,
                            "resolved": False,
                            "test_output": str(exc),
                            "model_patch": "",
                            "tool_history": [],
                        }
                    # Save per-instance result immediately
                    out_path = results_dir / f"{instance_id}.json"
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)
                    logger.info(f"[DONE] {instance_id} resolved={result.get('resolved', False)}")
                    return result

            results = await asyncio.gather(*[run_one(i) for i in instances])

            resolved_count = sum(1 for r in results if r.get("resolved", False))
            total = len(instances)
            score = resolved_count / total if total else 0.0

            logger.info(f"\n{'='*60}")
            logger.info(f"Full Suite Results: {resolved_count}/{total} resolved ({score:.1%})")
            for r in results:
                status = "PASS" if r.get("resolved") else "FAIL"
                logger.info(f"  [{status}] {r['instance_id']}")
            logger.info(f"{'='*60}")

            # Save summary report
            summary = {
                "benchmark": "swe_bench_lite_dev",
                "total": total,
                "resolved": resolved_count,
                "score": score,
                "results": results,
            }
            summary_path = results_dir / "full_suite_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary saved to {summary_path}")

            return summary

        finally:
            await harness.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Run real SWE-bench benchmark")
    parser.add_argument("--instance", default="sqlfluff__sqlfluff-1625", help="Instance ID to run")
    parser.add_argument("--mock", action="store_true", help="Use mock model client")
    parser.add_argument("--full-suite", action="store_true", help="Run full suite")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--cloud", action="store_true", help="Use Ollama Cloud (requires OLLAMA_API_KEY)")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent instances for full suite (default: 4)")
    parser.add_argument("--output-dir", default="benchmarks/results", help="Directory to save per-instance results")
    args = parser.parse_args()

    if args.full_suite:
        result = asyncio.run(run_full_suite(mock=args.mock, model=args.model, cloud=args.cloud, workers=args.workers, output_dir=args.output_dir))
    else:
        result = asyncio.run(run_single_instance(args.instance, mock=args.mock, model=args.model, cloud=args.cloud))

    # Exit with non-zero if benchmark failed
    if isinstance(result, dict) and not result.get("resolved", False):
        if not args.mock:
            sys.exit(1)


if __name__ == "__main__":
    main()
