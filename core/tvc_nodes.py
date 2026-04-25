"""Trial-Verify-Correct (TVC) graph nodes."""

import time
import uuid
from typing import Any, Dict, Optional

from core.ollama_client import OllamaClient
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


def make_async_plan_node(
    skill_store: Optional[SqliteSkillStore] = None,
    model_client: Optional[OllamaClient] = None,
):
    """Return an async plan node that injects skills and optionally calls an LLM."""
    async def async_plan_node(state: TVCState) -> TVCState:
        task = state["task_description"]
        state["injected_skills"] = []

        if skill_store is not None:
            matches = await skill_store.retrieve_skills(
                query_text=task,
                query_embedding=None,
            )
            state["injected_skills"] = [m.skill_id for m in matches]

        if model_client is not None:
            skills_text = "\n".join(
                f"- {s}" for s in state["injected_skills"]
            )
            prompt = (
                f"Task: {task}\n"
                f"Skills available:\n{skills_text}\n\n"
                "Generate a concise step-by-step reasoning plan to solve this task."
            )
            try:
                plan = await model_client.generate(prompt)
            except Exception as exc:
                plan = f"Plan: {task}\nError calling model: {exc}"
            state["reasoning_plan"] = plan
        else:
            state["reasoning_plan"] = f"Plan: {task}"
            if skill_store is not None:
                for m in matches:
                    state["reasoning_plan"] += f"\n- {m.canonical_name} ({m.skill_id})"

        return state
    return async_plan_node


def make_async_execute_node(
    ipc_bus: Optional[IPCBus] = None,
    model_client: Optional[OllamaClient] = None,
):
    """Return an async execute node that optionally generates shell commands via LLM."""
    async def async_execute_node(state: TVCState) -> TVCState:
        if model_client is not None:
            prompt = (
                f"Plan:\n{state['reasoning_plan']}\n\n"
                "What is the single next shell command to execute? "
                "Reply with ONLY the command, no explanation."
            )
            try:
                command = await model_client.generate(prompt)
            except Exception as exc:
                command = f"echo 'Model error: {exc}'"

            import subprocess
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=state.get("repo_path", "."),
                )
                entry = {
                    "tool": "shell.exec",
                    "command": command,
                    "status": "success" if proc.returncode == 0 else "error",
                    "exit_code": proc.returncode,
                    "output": proc.stdout + proc.stderr,
                }
            except Exception as exc:
                entry = {
                    "tool": "shell.exec",
                    "command": command,
                    "status": "error",
                    "exit_code": -1,
                    "output": str(exc),
                }
        else:
            entry = {
                "tool": "placeholder",
                "status": "success",
                "exit_code": 0,
                "output": "Placeholder execution completed.",
            }

        state["tool_history"].append(entry)

        if ipc_bus is not None and ipc_bus.is_connected():
            msg = IPCMessage(
                msg_id=str(uuid.uuid4()),
                timestamp_ns=time.time_ns(),
                event_type=EventType.TOOL_INVOCATION,
                session_id=state["session_id"],
                payload={"tool": entry["tool"], "status": entry["status"]},
            )
            await ipc_bus.send_message(msg)
        return state
    return async_execute_node


def make_async_verify_node(ipc_bus: Optional[IPCBus] = None):
    """Return an async verify node that emits CHECKPOINT or LOOP_WARNING."""
    async def async_verify_node(state: TVCState) -> TVCState:
        errors = _evaluate_verification(state)

        if ipc_bus is not None and ipc_bus.is_connected():
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


def make_async_correct_node(
    skill_store: Optional[SqliteSkillStore] = None,
    model_client: Optional[OllamaClient] = None,
):
    """Return an async correct node that analyzes failures via LLM."""
    async def async_correct_node(state: TVCState) -> TVCState:
        state["failure_count"] = state["failure_count"] + 1

        if model_client is not None:
            prompt = (
                f"Task: {state['task_description']}\n"
                f"Plan: {state['reasoning_plan']}\n"
                f"Error: {state.get('verification_details', 'unknown')}\n\n"
                "Analyze the failure and suggest a corrected approach."
            )
            try:
                analysis = await model_client.generate(prompt)
            except Exception as exc:
                analysis = f"Failure detected; retry required. Model error: {exc}"
            state["failure_analysis"] = analysis
        else:
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
