"""SWE-bench evaluation adapter."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _model_error_output(state: Dict[str, Any]) -> Optional[str]:
    """Return model-error text when the harness only reported a model failure."""
    details = state.get("verification_details")
    haystack = [str(details or "")]
    haystack.extend(
        str(entry.get("output", "")) for entry in state.get("tool_history", [])
    )
    for text in haystack:
        if "Model error:" in text:
            return text
    return None


class SWEBenchAdapter:
    """Maps TVC loop output to SWE-bench evaluation protocol."""

    def __init__(self, repo_path: str, harness=None, mock_mode: bool = False):
        self.repo_path = Path(repo_path)
        self.harness = harness
        self.mock_mode = mock_mode
        self.results: List[Dict[str, Any]] = []

    async def evaluate_instance(
        self, instance_id: str, patch: str
    ) -> Dict[str, Any]:
        """Evaluate a single SWE-bench instance."""
        logger.info(f"Evaluating SWE-bench instance {instance_id}")

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
            task_description=f"Fix the issue described in {instance_id}",
            repo_path=str(self.repo_path),
        )

        model_error = _model_error_output(state)
        resolved = state.get("verification_outcome") == "success" and not model_error
        return {
            "instance_id": instance_id,
            "resolved": resolved,
            "test_output": model_error or state.get("verification_details", ""),
        }

    async def evaluate_dataset(
        self, instances: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a dataset of SWE-bench instances."""
        resolved_count = 0
        total = len(instances)

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
                    inst["instance_id"], inst.get("patch", "")
                )
            self.results.append(result)
            if result["resolved"]:
                resolved_count += 1

        score = resolved_count / total if total > 0 else 0.0
        return {
            "benchmark": "swe_bench_verified",
            "total": total,
            "resolved": resolved_count,
            "score": score,
            "results": self.results,
        }
