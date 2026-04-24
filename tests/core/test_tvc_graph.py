"""Tests for TVC loop and LangGraph integration."""

import pytest

from core.tvc_state import TVCState, VerificationOutcome
from core.tvc_nodes import plan_node, verify_node
from core.tvc_graph import build_tvc_graph


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
    result = graph.invoke(state)
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
    result = graph.invoke(state)
    assert result["verification_outcome"] == VerificationOutcome.FAILURE
    assert result["failure_count"] == 2
