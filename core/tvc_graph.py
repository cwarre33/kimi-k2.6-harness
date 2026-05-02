"""LangGraph StateGraph builder for the Trial-Verify-Correct (TVC) loop."""

import os
import sqlite3
from typing import Optional

import aiosqlite
from langgraph.graph import StateGraph, END

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:
    SqliteSaver = None  # type: ignore
    AsyncSqliteSaver = None  # type: ignore

from core.tvc_state import TVCState, VerificationOutcome
from core.tvc_nodes import (
    plan_node,
    execute_node,
    verify_node,
    correct_node,
    make_async_plan_node,
    make_async_execute_node,
    make_async_verify_node,
    make_async_correct_node,
)
from core.tvc_config import CHECKPOINT_DB_PATH
from core.skill_store import SqliteSkillStore
from core.ipc_bus import IPCBus
from core.ollama_client import OllamaClient
from core.tools import ToolRegistry


def should_continue(state: TVCState) -> str:
    """Route to 'end' if SUCCESS or max retries exceeded, else 'retry'."""
    if (
        state["verification_outcome"] == VerificationOutcome.SUCCESS
        or state["failure_count"] >= state["max_retries"]
    ):
        return "end"
    return "retry"


def build_tvc_graph(checkpoint_db_path: str = CHECKPOINT_DB_PATH):
    """Build and compile the TVC StateGraph."""
    builder = StateGraph(TVCState)

    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("verify", verify_node)
    builder.add_node("correct", correct_node)

    builder.set_entry_point("plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "verify")
    builder.add_conditional_edges(
        "verify",
        should_continue,
        {"end": END, "retry": "correct"},
    )
    builder.add_edge("correct", "plan")

    if checkpoint_db_path and SqliteSaver is not None:
        db_dir = os.path.dirname(checkpoint_db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


async def build_async_tvc_graph(
    checkpoint_db_path: str = CHECKPOINT_DB_PATH,
    skill_store: Optional[SqliteSkillStore] = None,
    ipc_bus: Optional[IPCBus] = None,
    model_client: Optional[OllamaClient] = None,
    tool_registry: Optional[ToolRegistry] = None,
):
    """Build and compile the async TVC StateGraph."""
    builder = StateGraph(TVCState)

    builder.add_node("plan", make_async_plan_node(skill_store, model_client))
    builder.add_node("execute", make_async_execute_node(ipc_bus, model_client, tool_registry))
    builder.add_node("verify", make_async_verify_node(ipc_bus))
    builder.add_node("correct", make_async_correct_node(skill_store, model_client))

    builder.set_entry_point("plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "verify")
    builder.add_conditional_edges(
        "verify",
        should_continue,
        {"end": END, "retry": "correct"},
    )
    builder.add_edge("correct", "plan")

    if checkpoint_db_path and AsyncSqliteSaver is not None:
        db_dir = os.path.dirname(checkpoint_db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = await aiosqlite.connect(checkpoint_db_path)
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()
