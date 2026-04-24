"""Top-level orchestrator for the Kimi-K2.6 autonomous harness."""

import os
import uuid
from typing import Any, Dict, List, Optional

from core.skill_store import SqliteSkillStore
from core.ipc_bus import IPCBus, IPCRole
from core.tvc_graph import build_async_tvc_graph
from core.tvc_state import TVCState, VerificationOutcome
from core.tvc_config import CHECKPOINT_DB_PATH


class Harness:
    """Orchestrates the Skill Store, IPC Bus, and TVC loop."""

    def __init__(
        self,
        skill_db_path: str,
        socket_path: Optional[str] = None,
        checkpoint_db_path: Optional[str] = None,
    ):
        self.skill_db_path = skill_db_path
        self.socket_path = socket_path
        self.checkpoint_db_path = checkpoint_db_path or CHECKPOINT_DB_PATH
        self.skill_store = SqliteSkillStore(self.skill_db_path)
        self.ipc_bus = IPCBus(IPCRole.CONTROLLER, socket_path=self.socket_path)

    async def initialize(self) -> None:
        """Open the skill store database and start the IPC bus server."""
        await self.skill_store.initialize()
        await self.ipc_bus.start_server()

    async def shutdown(self) -> None:
        """Close the skill store and stop the IPC bus server."""
        await self.skill_store.close()
        await self.ipc_bus.stop_server()

    async def run_task(
        self, task_id: str, task_description: str, repo_path: str
    ) -> Dict[str, Any]:
        """Build an async TVC graph, invoke it, validate skills, and return the final state."""
        graph = await build_async_tvc_graph(
            skill_store=self.skill_store,
            ipc_bus=self.ipc_bus,
            checkpoint_db_path=self.checkpoint_db_path,
        )

        state = TVCState(
            task_id=task_id,
            task_description=task_description,
            reasoning_plan="",
            tool_history=[],
            injected_skills=[],
            verification_outcome=VerificationOutcome.PENDING,
            verification_details=None,
            failure_analysis="",
            failure_count=0,
            max_retries=3,
            session_id=str(uuid.uuid4()),
        )

        try:
            result = await graph.ainvoke(
                state, config={"configurable": {"thread_id": task_id}}
            )
        finally:
            if hasattr(graph, "checkpointer") and hasattr(graph.checkpointer, "conn"):
                await graph.checkpointer.conn.close()

        outcome = str(result["verification_outcome"])
        session_id = result["session_id"]
        for skill_id in result.get("injected_skills", []):
            await self.skill_store.validate_skill(skill_id, outcome, session_id)

        return dict(result)
