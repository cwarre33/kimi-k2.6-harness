"""Tests for TVC loop and LangGraph integration."""

import asyncio
import tempfile

import pytest
import pytest_asyncio

from core.tvc_state import TVCState, VerificationOutcome
from core.tvc_nodes import (
    plan_node,
    verify_node,
    make_async_plan_node,
    make_async_execute_node,
    make_async_verify_node,
    make_async_correct_node,
)
from core.tvc_graph import build_tvc_graph
from core.skill_store import SqliteSkillStore
from core.ipc_bus import IPCBus, IPCRole
from core.ipc_protocol import EventType


@pytest_asyncio.fixture
async def skill_store(tmp_path):
    db_path = tmp_path / "skills.db"
    store = SqliteSkillStore(str(db_path))
    await store.initialize()
    yield store
    await store.close()


def test_state_creation():
    state = TVCState(
        task_id="task-001",
        task_description="Fix recursion bug in utils.py",
        reasoning_plan="",
        tool_history=[],
        injected_skills=[],
        verification_outcome=VerificationOutcome.PENDING,
        verification_details=None,
        failure_analysis="",
        failure_count=0,
        max_retries=3,
        session_id="sess-001",
    )
    assert state["task_id"] == "task-001"
    assert state["failure_count"] == 0
    assert state["verification_outcome"] == VerificationOutcome.PENDING


def test_plan_node_populates_reasoning():
    state = TVCState(
        task_id="task-002",
        task_description="Refactor auth module",
        reasoning_plan="",
        tool_history=[],
        injected_skills=[],
        verification_outcome=VerificationOutcome.PENDING,
        verification_details=None,
        failure_analysis="",
        failure_count=0,
        max_retries=3,
        session_id="sess-002",
    )
    result = plan_node(state)
    assert "Refactor auth module" in result["reasoning_plan"]
    assert result["injected_skills"] == []


def test_verify_node_detects_failure():
    state = TVCState(
        task_id="task-003",
        task_description="Deploy to staging",
        reasoning_plan="",
        tool_history=[{"tool": "deploy", "exit_code": 1, "status": "success"}],
        injected_skills=[],
        verification_outcome=VerificationOutcome.PENDING,
        verification_details=None,
        failure_analysis="",
        failure_count=0,
        max_retries=3,
        session_id="sess-003",
    )
    result = verify_node(state)
    assert result["verification_outcome"] == VerificationOutcome.FAILURE


def test_graph_runs_to_success():
    graph = build_tvc_graph()
    state = TVCState(
        task_id="task-success",
        task_description="Run a successful task",
        reasoning_plan="",
        tool_history=[{"tool": "deploy", "exit_code": 0, "status": "success"}],
        injected_skills=[],
        verification_outcome=VerificationOutcome.PENDING,
        verification_details=None,
        failure_analysis="",
        failure_count=0,
        max_retries=3,
        session_id="sess-success",
    )
    result = graph.invoke(state, config={"configurable": {"thread_id": "thread-success"}})
    assert result["verification_outcome"] == VerificationOutcome.SUCCESS


def test_graph_retries_on_failure_then_gives_up():
    graph = build_tvc_graph()
    state = TVCState(
        task_id="task-fail",
        task_description="Run a failing task",
        reasoning_plan="",
        tool_history=[{"tool": "deploy", "exit_code": 1, "status": "success"}],
        injected_skills=[],
        verification_outcome=VerificationOutcome.PENDING,
        verification_details=None,
        failure_analysis="",
        failure_count=0,
        max_retries=2,
        session_id="sess-fail",
    )
    result = graph.invoke(state, config={"configurable": {"thread_id": "thread-fail"}})
    assert result["verification_outcome"] == VerificationOutcome.FAILURE
    assert result["failure_count"] == 2


def test_checkpoint_survives_restart():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "checkpoints.sqlite")
        graph = build_tvc_graph(checkpoint_db_path=db_path)
        state = TVCState(
            task_id="task-checkpoint",
            task_description="Checkpoint test",
            reasoning_plan="",
            tool_history=[{"tool": "shell.exec", "status": "completed", "exit_code": 0}],
            injected_skills=[],
            verification_outcome=VerificationOutcome.PENDING,
            verification_details=None,
            failure_analysis="",
            failure_count=0,
            max_retries=3,
            session_id="sess-001",
        )
        result = graph.invoke(state, config={"configurable": {"thread_id": "thread-1"}})
        assert result["verification_outcome"] == VerificationOutcome.SUCCESS
        assert os.path.exists(db_path)
        graph.checkpointer.conn.close()


# ---- Async node tests ----


@pytest.mark.asyncio
async def test_async_plan_node_injects_skills(skill_store):
    skill_id = await skill_store.store_skill(
        canonical_name="recursion-fix",
        task_pattern="Fix recursion bug",
        tool_sequence=[{"tool": "shell.exec", "args": {"command": "echo fixed"}}],
        thought_trace="I need to fix the recursion bug by adding a base case.",
        code_artifact=None,
        context_requirements={},
        tags=["recursion", "fix"],
    )
    await skill_store.validate_skill(skill_id, "success", session_id="sess-001")

    node = make_async_plan_node(skill_store)
    state = TVCState(
        task_id="task-001",
        task_description="recursion",
        reasoning_plan="",
        tool_history=[],
        injected_skills=[],
        verification_outcome=VerificationOutcome.PENDING,
        verification_details=None,
        failure_analysis="",
        failure_count=0,
        max_retries=3,
        session_id="sess-001",
    )
    result = await node(state)

    assert skill_id in result["injected_skills"]
    assert "recursion-fix" in result["reasoning_plan"]


@pytest.mark.asyncio
async def test_async_execute_node_emits_ipc_event():
    socket_path = tempfile.mktemp(suffix=".sock")
    server = IPCBus(IPCRole.AUDITOR, socket_path=socket_path)
    addr = await server.start_server()

    client = IPCBus(IPCRole.CONTROLLER, socket_path=socket_path)
    await client.connect(addr)

    try:
        node = make_async_execute_node(client)
        state = TVCState(
            task_id="task-001",
            task_description="Run a task",
            reasoning_plan="",
            tool_history=[],
            injected_skills=[],
            verification_outcome=VerificationOutcome.PENDING,
            verification_details=None,
            failure_analysis="",
            failure_count=0,
            max_retries=3,
            session_id="sess-001",
        )
        result = await node(state)

        assert len(result["tool_history"]) == 1
        assert result["tool_history"][0]["tool"] == "placeholder"

        async def get_next_non_heartbeat():
            async for msg in server.iter_messages():
                if msg.event_type != EventType.SESSION_HEARTBEAT:
                    return msg

        received = await asyncio.wait_for(get_next_non_heartbeat(), timeout=5.0)
        assert received.event_type == EventType.TOOL_INVOCATION
        assert received.session_id == "sess-001"
    finally:
        await client.disconnect()
        await server.stop_server()
