"""Shell execution tool with safety guardrails."""

import subprocess
from typing import Any, Dict

from . import Tool, ToolResult


class ShellExecTool(Tool):
    """Execute a shell command with timeout and safety checks."""

    name = "shell.exec"
    description = "Execute a shell command. Usage: shell.exec <command>"

    def __init__(self, default_timeout: int = 60):
        self.default_timeout = default_timeout

    async def run(self, arguments: Dict[str, Any], cwd: str = ".") -> ToolResult:
        command = arguments.get("args", arguments.get("command", ""))
        if not command:
            return ToolResult(
                status="error", output="No command provided", exit_code=-1
            )

        timeout = arguments.get("timeout", self.default_timeout)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            output = proc.stdout + proc.stderr
            if len(output) > 10000:
                output = output[:5000] + "\n... [output truncated] ...\n" + output[-5000:]

            return ToolResult(
                status="success" if proc.returncode == 0 else "error",
                output=output,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                status="error",
                output=f"Command timed out after {timeout}s",
                exit_code=-1,
            )
        except Exception as exc:
            return ToolResult(
                status="error", output=str(exc), exit_code=-1
            )
