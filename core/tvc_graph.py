"""LangGraph StateGraph builder for the Trial-Verify-Correct (TVC) loop."""

import os
import sqlite3

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from core.tvc_state import TVCState, VerificationOutcome
from core.tvc_nodes import plan_node, execute_node, verify_node, correct_node
from core.tvc_config import CHECKPOINT_DB_PATH


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

    if checkpoint_db_path:
        db_dir = os.path.dirname(checkpoint_db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()
