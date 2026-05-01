"""Tests for the Harness orchestrator."""

import httpx
import pytest
from core.harness import Harness
from core.tvc_state import VerificationOutcome


@pytest.mark.asyncio
async def test_harness_runs_task_to_success(tmp_path):
    skill_db = tmp_path / "skills.db"
    checkpoint_db = tmp_path / "checkpoints.sqlite"
    harness = Harness(
        skill_db_path=str(skill_db),
        checkpoint_db_path=str(checkpoint_db),
    )
    await harness.initialize()
    try:
        result = await harness.run_task(
            task_id="task-success-001",
            task_description="Run a successful task",
            repo_path=str(tmp_path),
        )
        assert result["verification_outcome"] == VerificationOutcome.SUCCESS
        assert result["task_id"] == "task-success-001"
    finally:
        await harness.shutdown()


@pytest.mark.asyncio
async def test_harness_uses_pre_stored_skills(tmp_path):
    skill_db = tmp_path / "skills.db"
    checkpoint_db = tmp_path / "checkpoints.sqlite"
    harness = Harness(
        skill_db_path=str(skill_db),
        checkpoint_db_path=str(checkpoint_db),
    )
    await harness.initialize()

    skill_id = await harness.skill_store.store_skill(
        canonical_name="recursion-fix",
        task_pattern="Fix recursion bug",
        tool_sequence=[{"tool": "shell.exec", "args": {"command": "echo fixed"}}],
        thought_trace="I need to fix the recursion bug by adding a base case.",
        code_artifact=None,
        context_requirements={},
        tags=["recursion", "fix"],
    )
    await harness.skill_store.validate_skill(skill_id, "success", session_id="sess-001")

    try:
        result = await harness.run_task(
            task_id="task-skill-001",
            task_description="Fix recursion bug",
            repo_path=str(tmp_path),
        )
        assert result["verification_outcome"] == VerificationOutcome.SUCCESS
        assert skill_id in result["injected_skills"]
    finally:
        await harness.shutdown()


@pytest.mark.asyncio
async def test_harness_runs_task_with_model(tmp_path):
    async def handler(request):
        await request.aread()
        return httpx.Response(200, json={"response": "echo hello", "done": True})

    transport = httpx.MockTransport(handler)
    from core.ollama_client import OllamaClient

    client = OllamaClient()
    client._client = httpx.AsyncClient(
        base_url="http://localhost:11434",
        headers=client._client.headers,
        transport=transport,
    )

    harness = Harness(
        skill_db_path=str(tmp_path / "skills.db"),
        checkpoint_db_path=str(tmp_path / "checkpoints.sqlite"),
        ollama_client=client,
    )
    await harness.initialize()
    try:
        result = await harness.run_task(
            task_id="task-001",
            task_description="Say hello",
            repo_path=str(tmp_path),
        )
        assert result["verification_outcome"] in ("success", "failure")
        assert len(result["tool_history"]) > 0
    finally:
        await harness.shutdown()
