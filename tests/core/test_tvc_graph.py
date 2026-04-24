"""Tests for TVC loop and LangGraph integration."""

import pytest

from core.tvc_state import TVCState, VerificationOutcome


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
