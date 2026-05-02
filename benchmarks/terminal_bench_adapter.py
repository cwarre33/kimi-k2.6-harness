"""Terminal-Bench evaluation adapter."""

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TerminalBenchAdapter:
    """Maps TVC loop output to Terminal-Bench evaluation protocol."""

    def __init__(self, repo_path: str, harness=None, mock_mode: bool = False):
        self.repo_path = Path(repo_path)
        self.harness = harness
        self.mock_mode = mock_mode
        self.results: List[Dict[str, Any]] = []

    async def evaluate_instance(
        self, instance_id: str, command: str
    ) -> Dict[str, Any]:
        """Evaluate a single Terminal-Bench instance."""
        logger.info(f"Evaluating Terminal-Bench instance {instance_id}")

        if self.mock_mode:
            return {
                "instance_id": instance_id,
                "resolved": True,
                "test_output": "Mock evaluation",
            }

        if self.harness is None:
            return {
                "instance_id": instance_id,
                "resolved": False,
                "test_output": "Harness not wired — stub evaluation",
            }

        state = await self.harness.run_task(
            task_id=instance_id,
            task_description=f"Execute terminal command for {instance_id}",
            repo_path=str(self.repo_path),
        )

        resolved = state.get("verification_outcome") == "success"
        return {
            "instance_id": instance_id,
            "resolved": resolved,
            "test_output": state.get("verification_details", ""),
        }

    async def evaluate_dataset(
        self, instances: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluate a dataset of Terminal-Bench instances."""
        resolved_count = 0
        total = len(instances) if instances else 0

        if instances is None:
            return {
                "benchmark": "terminal_bench_2_0",
                "total": 0,
                "resolved": 0,
                "score": 0.0,
                "results": [],
            }

        for inst in instances:
            if self.mock_mode:
                expected = inst.get("expected_resolved", False)
                result = {
                    "instance_id": inst["instance_id"],
                    "resolved": expected,
                    "test_output": "Mock evaluation",
                }
            else:
                result = await self.evaluate_instance(
                    inst["instance_id"], inst.get("command", "")
                )
            self.results.append(result)
            if result["resolved"]:
                resolved_count += 1

        score = resolved_count / total if total > 0 else 0.0
        return {
            "benchmark": "terminal_bench_2_0",
            "total": total,
            "resolved": resolved_count,
            "score": score,
            "results": self.results,
        }
