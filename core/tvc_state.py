"""Typed state definition for the Trial-Verify-Correct (TVC) graph."""

from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum


class VerificationOutcome(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class TVCState(TypedDict):
    task_id: str
    task_description: str
    reasoning_plan: str
    tool_history: List[Dict[str, Any]]
    injected_skills: List[str]
    verification_outcome: VerificationOutcome
    verification_details: Optional[str]
    failure_analysis: str
    failure_count: int
    max_retries: int
    session_id: str
