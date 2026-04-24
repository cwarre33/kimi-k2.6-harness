"""LangGraph StateGraph builder for the Trial-Verify-Correct (TVC) loop."""

from langgraph.graph import StateGraph, END

from core.tvc_state import TVCState, VerificationOutcome
from core.tvc_nodes import plan_node, execute_node, verify_node, correct_node


def should_continue(state: TVCState) -> str:
    """Route to 'end' if SUCCESS or max retries exceeded, else 'retry'."""
    if (
        state["verification_outcome"] == VerificationOutcome.SUCCESS
        or state["failure_count"] >= state["max_retries"]
    ):
        return "end"
    return "retry"


def build_tvc_graph() -> StateGraph:
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

    return builder.compile()
