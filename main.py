"""Entrypoint for the Kimi-K2.6 autonomous harness.

Enforces the benchmark gate: halts if any benchmark score is below target.
"""

import asyncio
import logging
import sys

from benchmarks.eval_orchestrator import EvalOrchestrator
from core.harness import Harness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run evaluation gate and either proceed or exit."""
    logger.info("Starting Kimi-K2.6 harness evaluation gate")

    harness = Harness()
    await harness.initialize()
    try:
        orchestrator = EvalOrchestrator()
        report = await orchestrator.run_full_suite(harness=harness)

        if not report["overall_passed"]:
            logger.error("Benchmark gate FAILED. Halting harness.")
            for name, result in report["benchmarks"].items():
                logger.error(f"  {name}: {result['recommendation']}")
            sys.exit(1)

        logger.info("Benchmark gate PASSED. Harness cleared for production.")
        return report
    finally:
        await harness.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
