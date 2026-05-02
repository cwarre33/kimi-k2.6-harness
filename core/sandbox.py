"""Docker sandbox for isolated tool execution."""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """Raised when sandbox operations fail."""


class DockerSandbox:
    """Execute commands inside an isolated Docker container.

    Each task gets its own container with the repo mounted read-write.
    The container is cleaned up when the task completes or the harness shuts down.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        cpu_limit: Optional[str] = None,
        memory_limit: Optional[str] = None,
        network: bool = False,
        working_dir: str = "/workspace",
    ):
        self.image = image
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.network = network
        self.working_dir = working_dir
        self.container_id: Optional[str] = None
        self._repo_path: Optional[str] = None

    async def start(self, repo_path: str) -> None:
        """Start a Docker container and mount the repo."""
        if self.container_id is not None:
            raise SandboxError("Sandbox already started")

        self._repo_path = repo_path
        abs_repo = Path(repo_path).resolve()

        cmd = [
            "docker", "run", "-d",
            "--rm",
            "-v", f"{abs_repo}:{self.working_dir}",
            "-w", self.working_dir,
        ]

        if not self.network:
            cmd.append("--network=none")
        if self.cpu_limit:
            cmd.extend(["--cpus", self.cpu_limit])
        if self.memory_limit:
            cmd.extend(["--memory", self.memory_limit])

        cmd.append(self.image)
        cmd.append("sleep")
        cmd.append("3600")

        logger.info(f"Starting sandbox container: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise SandboxError(
                f"Failed to start container: {stderr.decode().strip()}"
            )

        self.container_id = stdout.decode().strip()
        logger.info(f"Sandbox container started: {self.container_id[:12]}")

    async def exec(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 60,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a command inside the container.

        Returns a dict with keys: stdout, stderr, exit_code, status.
        """
        if self.container_id is None:
            raise SandboxError("Sandbox not started — call start() first")

        exec_cmd = ["docker", "exec"]
        if env:
            for key, value in env.items():
                exec_cmd.extend(["-e", f"{key}={value}"])

        work_dir = cwd or self.working_dir
        exec_cmd.extend(["-w", work_dir, self.container_id])
        exec_cmd.extend(["sh", "-c", command])

        logger.debug(f"Sandbox exec: {command[:200]}")
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *exec_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=timeout + 5,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "exit_code": -1,
                "status": "error",
            }

        exit_code = proc.returncode or 0
        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "exit_code": exit_code,
            "status": "success" if exit_code == 0 else "error",
        }

    async def copy_in(self, host_path: str, container_path: str) -> None:
        """Copy a file from host into the container."""
        if self.container_id is None:
            raise SandboxError("Sandbox not started")

        proc = await asyncio.create_subprocess_exec(
            "docker", "cp", host_path, f"{self.container_id}:{container_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise SandboxError(f"docker cp failed: {stderr.decode().strip()}")

    async def copy_out(self, container_path: str, host_path: str) -> None:
        """Copy a file from container to host."""
        if self.container_id is None:
            raise SandboxError("Sandbox not started")

        proc = await asyncio.create_subprocess_exec(
            "docker", "cp", f"{self.container_id}:{container_path}", host_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise SandboxError(f"docker cp failed: {stderr.decode().strip()}")

    async def stop(self) -> None:
        """Stop and remove the container."""
        if self.container_id is None:
            return

        logger.info(f"Stopping sandbox container: {self.container_id[:12]}")
        proc = await asyncio.create_subprocess_exec(
            "docker", "stop", "-t", "5", self.container_id,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        self.container_id = None
        logger.info("Sandbox container stopped")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()


class SandboxShellExecTool:
    """Shell execution tool that runs inside a DockerSandbox."""

    name = "shell.exec"
    description = "Execute a shell command inside the sandbox. Usage: shell.exec <command>"

    def __init__(
        self,
        sandbox: DockerSandbox,
        default_timeout: int = 60,
    ):
        self.sandbox = sandbox
        self.default_timeout = default_timeout

    async def run(self, arguments: Dict[str, Any], cwd: str = ".") -> "ToolResult":
        from . import ToolResult

        command = arguments.get("args", arguments.get("command", ""))
        if not command:
            return ToolResult(
                status="error", output="No command provided", exit_code=-1
            )

        timeout = arguments.get("timeout", self.default_timeout)

        try:
            result = await self.sandbox.exec(command, cwd=cwd, timeout=timeout)
            output = result["stdout"] + result["stderr"]
            if len(output) > 10000:
                output = output[:5000] + "\n... [output truncated] ...\n" + output[-5000:]

            return ToolResult(
                status=result["status"],
                output=output,
                exit_code=result["exit_code"],
            )
        except Exception as exc:
            return ToolResult(
                status="error", output=str(exc), exit_code=-1
            )
