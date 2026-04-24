"""Tests for TVC loop and LangGraph integration."""

import pytest

from core.tvc_state import TVCState, VerificationOutcome
from core.tvc_nodes import plan_node, verify_node


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
