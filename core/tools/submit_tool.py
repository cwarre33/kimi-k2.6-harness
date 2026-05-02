"""Submit tool to end an episode and return the result."""

from typing import Any, Dict

from . import Tool, ToolResult


class SubmitTool(Tool):
    """End the task and submit the final patch or answer."""

    name = "submit"
    description = "End the episode and submit your solution. Usage: submit"

    async def run(self, arguments: Dict[str, Any], cwd: str = ".") -> ToolResult:
        return ToolResult(
            status="success",
            output="Episode ended. Solution submitted.",
            exit_code=0,
            metadata={"submitted": True},
        )
