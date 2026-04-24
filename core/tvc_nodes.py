"""Trial-Verify-Correct (TVC) graph nodes."""

from typing import Any, Dict
from core.tvc_state import TVCState, VerificationOutcome


def plan_node(state: TVCState) -> TVCState:
    """Generate a reasoning plan from the task description."""
    state["reasoning_plan"] = f"Plan: {state['task_description']}"
    state["injected_skills"] = []
    return state


def execute_node(state: TVCState) -> TVCState:
    """Append a placeholder tool execution to the tool history."""
    placeholder: Dict[str, Any] = {
        "tool": "placeholder",
        "status": "success",
        "exit_code": 0,
        "output": "Placeholder execution completed.",
    }
    state["tool_history"].append(placeholder)
    return state


def verify_node(state: TVCState) -> TVCState:
    """Check tool_history for errors and set verification outcome."""
    errors = []
    for entry in state["tool_history"]:
        if entry.get("status") == "error" or entry.get("exit_code", 0) != 0:
            errors.append(str(entry))

    if errors:
        state["verification_outcome"] = VerificationOutcome.FAILURE
        state["verification_details"] = "; ".join(errors)
    else:
        state["verification_outcome"] = VerificationOutcome.SUCCESS
        state["verification_details"] = None

    return state


def correct_node(state: TVCState) -> TVCState:
    """Increment failure count, set analysis, and reset outcome to pending."""
    state["failure_count"] = state["failure_count"] + 1
    state["failure_analysis"] = "Failure detected; retry required."
    state["verification_outcome"] = VerificationOutcome.PENDING
    return state
