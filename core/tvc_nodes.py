"""Trial-Verify-Correct (TVC) graph nodes."""

import time
import uuid
from typing import Any, Dict, Optional

from core.ipc_bus import IPCBus
from core.ipc_protocol import EventType, IPCMessage
from core.skill_store import SqliteSkillStore
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


def _evaluate_verification(state: TVCState) -> list[str]:
    """Check tool_history for errors, update verification fields, and return errors."""
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

    return errors


def verify_node(state: TVCState) -> TVCState:
    """Check tool_history for errors and set verification outcome."""
    _evaluate_verification(state)
    return state


def correct_node(state: TVCState) -> TVCState:
    """Increment failure count, set analysis, and reset outcome to pending."""
    state["failure_count"] = state["failure_count"] + 1
    state["failure_analysis"] = "Failure detected; retry required."
    state["verification_outcome"] = VerificationOutcome.PENDING
    return state


# ---- Async variants with dependency injection ----


def make_async_plan_node(skill_store: Optional[SqliteSkillStore] = None):
    """Return an async plan node that injects skills from *skill_store*."""
    async def async_plan_node(state: TVCState) -> TVCState:
        state["reasoning_plan"] = f"Plan: {state['task_description']}"
        state["injected_skills"] = []
        if skill_store is not None:
            matches = await skill_store.retrieve_skills(
                query_text=state["task_description"],
                query_embedding=None,
            )
            state["injected_skills"] = [m.skill_id for m in matches]
            for m in matches:
                state["reasoning_plan"] += f"\n- {m.canonical_name} ({m.skill_id})"
        return state
    return async_plan_node


def make_async_execute_node(ipc_bus: Optional[IPCBus] = None):
    """Return an async execute node that emits a TOOL_INVOCATION event."""
    async def async_execute_node(state: TVCState) -> TVCState:
        placeholder: Dict[str, Any] = {
            "tool": "placeholder",
            "status": "success",
            "exit_code": 0,
            "output": "Placeholder execution completed.",
        }
        state["tool_history"].append(placeholder)
        if ipc_bus is not None:
            msg = IPCMessage(
                msg_id=str(uuid.uuid4()),
                timestamp_ns=time.time_ns(),
                event_type=EventType.TOOL_INVOCATION,
                session_id=state["session_id"],
                payload={"tool": "placeholder", "status": "success"},
            )
            await ipc_bus.send_message(msg)
        return state
    return async_execute_node


def make_async_verify_node(ipc_bus: Optional[IPCBus] = None):
    """Return an async verify node that emits CHECKPOINT or LOOP_WARNING."""
    async def async_verify_node(state: TVCState) -> TVCState:
        errors = _evaluate_verification(state)

        if ipc_bus is not None:
            if errors:
                event_type = EventType.LOOP_WARNING
                payload = {"details": state["verification_details"], "errors": errors}
            else:
                event_type = EventType.CHECKPOINT
                payload = {"status": "verified"}

            msg = IPCMessage(
                msg_id=str(uuid.uuid4()),
                timestamp_ns=time.time_ns(),
                event_type=event_type,
                session_id=state["session_id"],
                payload=payload,
            )
            await ipc_bus.send_message(msg)
        return state
    return async_verify_node


def make_async_correct_node(skill_store: Optional[SqliteSkillStore] = None):
    """Return an async correct node that injects failure-analysis skills."""
    async def async_correct_node(state: TVCState) -> TVCState:
        state["failure_count"] = state["failure_count"] + 1
        state["failure_analysis"] = "Failure detected; retry required."
        state["verification_outcome"] = VerificationOutcome.PENDING

        if skill_store is not None:
            details = state.get("verification_details")
            if details:
                matches = await skill_store.retrieve_skills(
                    query_text=details,
                    query_embedding=None,
                )
                for m in matches:
                    if m.skill_id not in state["injected_skills"]:
                        state["injected_skills"].append(m.skill_id)
        return state
    return async_correct_node
