"""Tests for Docker sandbox."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from core.sandbox import DockerSandbox, SandboxError


class TestDockerSandbox:
    """Test suite for DockerSandbox."""

    @pytest.fixture(scope="class")
    def docker_available(self):
        import subprocess
        try:
            result = subprocess.run(
                ["docker", "info"], check=True, capture_output=True, text=True, timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @pytest.mark.asyncio
    async def test_start_and_stop(self, docker_available):
        if not docker_available:
            pytest.skip("Docker not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = DockerSandbox(image="python:3.12-slim")
            await sandbox.start(tmpdir)
            assert sandbox.container_id is not None
            await sandbox.stop()
            assert sandbox.container_id is None

    @pytest.mark.asyncio
    async def test_exec_echo(self, docker_available):
        if not docker_available:
            pytest.skip("Docker not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = DockerSandbox(image="python:3.12-slim")
            await sandbox.start(tmpdir)
            try:
                result = await sandbox.exec("echo hello")
                assert result["status"] == "success"
                assert result["exit_code"] == 0
                assert "hello" in result["stdout"]
            finally:
                await sandbox.stop()

    @pytest.mark.asyncio
    async def test_exec_timeout(self, docker_available):
        if not docker_available:
            pytest.skip("Docker not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = DockerSandbox(image="python:3.12-slim")
            await sandbox.start(tmpdir)
            try:
                result = await sandbox.exec("sleep 10", timeout=1)
                assert result["status"] == "error"
                assert "timed out" in result["stderr"]
            finally:
                await sandbox.stop()

    @pytest.mark.asyncio
    async def test_exec_write_and_read_file(self, docker_available):
        if not docker_available:
            pytest.skip("Docker not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = DockerSandbox(image="python:3.12-slim")
            await sandbox.start(tmpdir)
            try:
                await sandbox.exec("echo 'test content' > /workspace/testfile.txt")
                result = await sandbox.exec("cat /workspace/testfile.txt")
                assert "test content" in result["stdout"]
            finally:
                await sandbox.stop()

    @pytest.mark.asyncio
    async def test_exec_cwd(self, docker_available):
        if not docker_available:
            pytest.skip("Docker not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = DockerSandbox(image="python:3.12-slim")
            await sandbox.start(tmpdir)
            try:
                result = await sandbox.exec("pwd", cwd="/tmp")
                assert "/tmp" in result["stdout"]
            finally:
                await sandbox.stop()

    @pytest.mark.asyncio
    async def test_copy_in_and_out(self, docker_available):
        if not docker_available:
            pytest.skip("Docker not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            host_file = Path(tmpdir) / "host.txt"
            host_file.write_text("from host")

            sandbox = DockerSandbox(image="python:3.12-slim")
            await sandbox.start(tmpdir)
            try:
                await sandbox.copy_in(str(host_file), "/workspace/copied.txt")
                result = await sandbox.exec("cat /workspace/copied.txt")
                assert "from host" in result["stdout"]

                out_path = Path(tmpdir) / "out.txt"
                await sandbox.copy_out("/workspace/copied.txt", str(out_path))
                assert out_path.read_text() == "from host"
            finally:
                await sandbox.stop()

    @pytest.mark.asyncio
    async def test_no_network(self, docker_available):
        if not docker_available:
            pytest.skip("Docker not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = DockerSandbox(image="python:3.12-slim", network=False)
            await sandbox.start(tmpdir)
            try:
                result = await sandbox.exec("curl -s https://example.com", timeout=5)
                assert result["status"] == "error"
            finally:
                await sandbox.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self, docker_available):
        if not docker_available:
            pytest.skip("Docker not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            async with DockerSandbox(image="python:3.12-slim") as sandbox:
                await sandbox.start(tmpdir)
                result = await sandbox.exec("echo hello")
                assert result["status"] == "success"
            assert sandbox.container_id is None

    @pytest.mark.asyncio
    async def test_double_start_raises(self, docker_available):
        if not docker_available:
            pytest.skip("Docker not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = DockerSandbox(image="python:3.12-slim")
            await sandbox.start(tmpdir)
            try:
                with pytest.raises(SandboxError, match="already started"):
                    await sandbox.start(tmpdir)
            finally:
                await sandbox.stop()

    @pytest.mark.asyncio
    async def test_exec_without_start_raises(self, docker_available):
        if not docker_available:
            pytest.skip("Docker not available")

        sandbox = DockerSandbox(image="python:3.12-slim")
        with pytest.raises(SandboxError, match="not started"):
            await sandbox.exec("echo hello")
