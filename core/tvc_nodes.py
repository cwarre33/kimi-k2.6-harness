"""Trial-Verify-Correct (TVC) graph nodes."""

import logging
import time
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from core.action_parser import ActionParser, ParseError
from core.ollama_client import OllamaClient
from core.ipc_bus import IPCBus
from core.ipc_protocol import EventType, IPCMessage
from core.skill_store import SqliteSkillStore
from core.tools import create_default_registry, ToolRegistry
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
    # Transient harness signals and exploration errors don't fail the episode
    _SKIP_TOOLS = {"__retry__", "parse_error", "search_file", "search_dir", "shell.exec"}
    errors = []
    for entry in state["tool_history"]:
        if entry.get("tool") in _SKIP_TOOLS:
            continue
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
            system = (
                "You are an expert software engineer. "
                "Your task is to fix bugs in code repositories. "
                "You have access to tools that let you explore and edit files. "
                "Think step by step and be thorough."
            )
            prompt = (
                f"Task: {task}\n\n"
                f"Skills available:\n{skills_text}\n\n"
                "Generate a concise step-by-step reasoning plan to solve this task. "
                "Be specific about which files you need to examine and what changes you need to make."
            )
            try:
                plan = await model_client.generate(prompt, system=system)
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


def _build_tool_history_summary(tool_history: list[dict], max_entries: int = 8) -> str:
    """Build a summary of recent tool executions for the prompt.

    File-viewing tools get up to 8 kB so the model can act on the full file
    for typical source files (~200 lines).  Other tools stay heavily truncated.
    """
    if not tool_history:
        return "No actions taken yet."

    # Detect repeated view_file calls on the same path (dedup warning)
    viewed_files: dict[str, int] = {}
    for i, entry in enumerate(tool_history):
        if entry.get("tool") in ("view_file", "open"):
            path = entry.get("command", "").split()[-1] if entry.get("command") else ""
            if path:
                viewed_files[path] = viewed_files.get(path, 0) + 1

    lines = []
    for entry in tool_history[-max_entries:]:
        tool = entry.get("tool", "unknown")
        status = entry.get("status", "unknown")
        output = entry.get("output", "")
        cmd = entry.get("command", "")

        if tool == "submit":
            lines.append("- submit: Episode ended.")
            continue

        if tool in ("view_file", "open"):
            limit = 10000
            path = cmd.split()[-1] if cmd else ""
            dupe_count = viewed_files.get(path, 1)
            out_preview = output[:limit]
            if len(output) > limit:
                out_preview += f"\n... [file truncated at {limit} chars — use shell.exec with sed/tail for remaining lines] ..."
            header = f"- {tool}: {cmd} | status={status}"
            if dupe_count > 1:
                header += f"  *** WARNING: you have viewed '{path}' {dupe_count} times. Do NOT view it again. Use shell.exec with sed/tail for specific sections if needed. ***"
            lines.append(f"{header}\n{out_preview}")
        else:
            if tool == "shell.exec":
                # Preserve newlines so the model can copy exact code for replace_string
                limit = 2000
                out_preview = output[:limit]
                if len(output) > limit:
                    out_preview += "\n... [truncated]"
                lines.append(f"- {tool}: {cmd} | status={status}\n{out_preview}")
                continue
            limit = 400 if tool in ("search_file", "search_dir") else 120
            out_preview = output[:limit].replace("\n", " ")
            if len(output) > limit:
                out_preview += " ..."
            if cmd:
                lines.append(f"- {tool}: {cmd} | status={status} | output={out_preview}")
            else:
                lines.append(f"- {tool}: status={status} | output={out_preview}")

    return "\n".join(lines)


def make_async_execute_node(
    ipc_bus: Optional[IPCBus] = None,
    model_client: Optional[OllamaClient] = None,
    tool_registry: Optional[ToolRegistry] = None,
):
    """Return an async execute node that runs an inner loop of tool-use until submit."""
    registry = tool_registry or create_default_registry()
    parser = ActionParser(max_retries=3)

    async def async_execute_node(state: TVCState) -> TVCState:
        if model_client is None:
            entry = {
                "tool": "placeholder",
                "status": "success",
                "exit_code": 0,
                "output": "Placeholder execution completed.",
            }
            state["tool_history"].append(entry)
            return state

        tools_text = registry.get_system_prompt()
        max_steps = 20
        max_timeout_retries = 2

        system = (
            "You are an expert software engineer fixing a bug in a code repository.\n"
            "You have file and shell tools available.\n"
            "Rules:\n"
            "1. Find the bug location with search_file or search_dir.\n"
            "2. Use view_file ONCE on the source file that needs fixing.\n"
            "   For long files use shell.exec with sed -n 'N,Mp' to read specific line ranges.\n"
            "3. Use replace_string to fix ONLY source files — NEVER edit test files.\n"
            "4. Call submit immediately after making the fix. Do NOT run tests.\n"
            "5. Be decisive — do NOT view the same file twice, search repeatedly, or look at tests.\n"
            "6. Every step counts; use the minimum tools needed.\n"
            "7. After viewing the source file, make the fix immediately.\n"
            "Format:\n"
            "DISCUSSION: brief reasoning\n"
            "```bash\n"
            "replace_string <path> <old_text> <new_text>\n"
            "```"
        )

        for step in range(max_steps):
            history_summary = _build_tool_history_summary(state.get("tool_history", []))

            prompt = (
                f"Task:\n{state['task_description']}\n\n"
                f"Plan:\n{state['reasoning_plan']}\n\n"
                f"Recent actions:\n{history_summary}\n\n"
                f"{tools_text}\n\n"
                "What is the single next action to take? "
                "Reply with a DISCUSSION block and a COMMAND block."
            )

            raw_response = ""
            timeout_retries = 0
            while timeout_retries <= max_timeout_retries:
                try:
                    raw_response = await model_client.generate(prompt, system=system)
                    break
                except TimeoutError as exc:
                    timeout_retries += 1
                    if timeout_retries <= max_timeout_retries:
                        logger.warning(f"Step {step}: model timeout (retry {timeout_retries}/{max_timeout_retries})")
                        continue
                    logger.error(f"Step {step}: model timeout exhausted after {max_timeout_retries} retries")
                    exc_text = str(exc).replace("\n", " ").replace("'", '"')
                    raw_response = f"DISCUSSION: Model generation timed out after all retries.\n\n```bash\nshell.exec echo \"Timeout: {exc_text}\"\n```"
                except Exception as exc:
                    exc_text = str(exc).replace("\n", " ").replace("'", '"')
                    raw_response = f"DISCUSSION: Model error\n\n```bash\nshell.exec echo \"Model error: {exc_text}\"\n```"
                    break

            # Parse the action
            try:
                action = parser.parse(raw_response)
                logger.info(f"Step {step}: parsed action={action.command_type} | cmd={action.command[:60]}")
            except ParseError as exc:
                logger.warning(f"Step {step}: parse error: {exc}")
                entry = {
                    "tool": "parse_error",
                    "command": raw_response[:200],
                    "status": "error",
                    "exit_code": -1,
                    "output": str(exc),
                }
                state["tool_history"].append(entry)
                if ipc_bus is not None and ipc_bus.is_connected():
                    msg = IPCMessage(
                        msg_id=str(uuid.uuid4()),
                        timestamp_ns=time.time_ns(),
                        event_type=EventType.TOOL_INVOCATION,
                        session_id=state["session_id"],
                        payload={"tool": "parse_error", "status": "error"},
                    )
                    await ipc_bus.send_message(msg)
                continue

            # Handle retry signal
            if action.command_type == "__retry__":
                entry = {
                    "tool": "__retry__",
                    "command": action.command,
                    "status": "error",
                    "exit_code": -1,
                    "output": action.arguments.get("error", "Parse failed"),
                }
                state["tool_history"].append(entry)
                continue

            # Dispatch to tool
            tool = registry.get(action.command_type)
            if tool is None:
                entry = {
                    "tool": action.command_type,
                    "command": action.command,
                    "status": "error",
                    "exit_code": -1,
                    "output": f"Unknown tool: {action.command_type}",
                }
            else:
                result = await tool.run(
                    action.arguments,
                    cwd=state.get("repo_path", "."),
                )
                entry = {
                    "tool": action.command_type,
                    "command": action.command,
                    "status": result.status,
                    "exit_code": result.exit_code,
                    "output": result.output,
                    "metadata": result.metadata,
                }

            state["tool_history"].append(entry)
            logger.info(
                f"Step {step}: {entry['tool']} | status={entry['status']} | "
                f"output={entry['output'][:100].replace(chr(10), ' ')}"
            )

            if ipc_bus is not None and ipc_bus.is_connected():
                msg = IPCMessage(
                    msg_id=str(uuid.uuid4()),
                    timestamp_ns=time.time_ns(),
                    event_type=EventType.TOOL_INVOCATION,
                    session_id=state["session_id"],
                    payload={"tool": entry["tool"], "status": entry["status"]},
                )
                await ipc_bus.send_message(msg)

            # Stop inner loop on submit
            if action.command_type == "submit":
                break

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
            history_summary = _build_tool_history_summary(state.get("tool_history", []))
            prompt = (
                f"Task: {state['task_description']}\n"
                f"Plan: {state['reasoning_plan']}\n"
                f"Actions taken:\n{history_summary}\n\n"
                f"Error: {state.get('verification_details', 'unknown')}\n\n"
                "Analyze the failure and suggest a corrected approach. "
                "What went wrong and what should you do differently?"
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
