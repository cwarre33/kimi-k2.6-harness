"""Tests for ACI-style tools."""

import tempfile
from pathlib import Path

import pytest

from core.tools.shell_tool import ShellExecTool
from core.tools.file_tools import OpenTool, SearchFileTool, SearchDirTool, EditTool
from core.tools.submit_tool import SubmitTool


class TestShellExecTool:
    """Test suite for ShellExecTool."""

    @pytest.mark.asyncio
    async def test_run_echo(self):
        tool = ShellExecTool()
        result = await tool.run({"args": "echo hello"}, cwd=".")
        assert result.status == "success"
        assert result.exit_code == 0
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_run_invalid_command(self):
        tool = ShellExecTool()
        result = await tool.run({"args": "not_a_real_command_12345"}, cwd=".")
        assert result.status == "error"
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_run_timeout(self):
        tool = ShellExecTool(default_timeout=1)
        result = await tool.run({"args": "sleep 5"}, cwd=".")
        assert result.status == "error"
        assert "timed out" in result.output

    @pytest.mark.asyncio
    async def test_empty_command(self):
        tool = ShellExecTool()
        result = await tool.run({"args": ""}, cwd=".")
        assert result.status == "error"
        assert "No command provided" in result.output

    @pytest.mark.asyncio
    async def test_truncation(self):
        tool = ShellExecTool()
        result = await tool.run({"args": "python -c \"print('x' * 20000)\""}, cwd=".")
        assert "truncated" in result.output


class TestOpenTool:
    """Test suite for OpenTool."""

    @pytest.mark.asyncio
    async def test_open_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            for i in range(1, 21):
                f.write(f"line {i}\n")
            path = f.name

        tool = OpenTool()
        result = await tool.run({"path": path}, cwd=".")

        assert result.status == "success"
        assert "line 1" in result.output
        assert result.metadata["total_lines"] == 20

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_open_with_line(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            for i in range(1, 201):
                f.write(f"line {i}\n")
            path = f.name

        tool = OpenTool()
        result = await tool.run({"path": path, "line": 100}, cwd=".")

        assert result.status == "success"
        assert result.metadata["current_line"] == 100
        assert result.metadata["window_start"] < 100
        assert result.metadata["window_end"] > 100

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_open_missing_file(self):
        tool = OpenTool()
        result = await tool.run({"path": "/does/not/exist"}, cwd=".")
        assert result.status == "error"
        assert "File not found" in result.output

    @pytest.mark.asyncio
    async def test_open_no_path(self):
        tool = OpenTool()
        result = await tool.run({}, cwd=".")
        assert result.status == "error"
        assert "No path provided" in result.output


class TestSearchFileTool:
    """Test suite for SearchFileTool."""

    @pytest.mark.asyncio
    async def test_search_found(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def hello():\n    pass\n")
            path = f.name

        tool = SearchFileTool()
        result = await tool.run({"term": "hello", "file": path}, cwd=".")

        assert result.status == "success"
        assert "hello" in result.output
        assert result.metadata["match_count"] >= 1

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_search_not_found(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def foo():\n    pass\n")
            path = f.name

        tool = SearchFileTool()
        result = await tool.run({"term": "bar", "file": path}, cwd=".")

        assert result.status == "success"
        assert "No matches" in result.output

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_search_max_matches(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            for _ in range(60):
                f.write("target word\n")
            path = f.name

        tool = SearchFileTool()
        result = await tool.run({"term": "target", "file": path}, cwd=".")

        assert result.status == "success"
        assert "refine your search" in result.output

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_search_missing_file(self):
        tool = SearchFileTool()
        result = await tool.run(
            {"term": "foo", "file": "/does/not/exist"}, cwd="."
        )
        assert result.status == "error"
        assert "File not found" in result.output


class TestSearchDirTool:
    """Test suite for SearchDirTool."""

    @pytest.mark.asyncio
    async def test_search_dir_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.py").write_text("hello world\n")
            Path(tmpdir, "b.py").write_text("hello again\n")

            tool = SearchDirTool()
            result = await tool.run({"term": "hello", "dir": tmpdir}, cwd=".")

            assert result.status == "success"
            assert "a.py" in result.output
            assert "b.py" in result.output

    @pytest.mark.asyncio
    async def test_search_dir_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = SearchDirTool()
            result = await tool.run({"term": "xyz", "dir": tmpdir}, cwd=".")

            assert result.status == "success"
            assert "No matches" in result.output

    @pytest.mark.asyncio
    async def test_search_dir_missing(self):
        tool = SearchDirTool()
        result = await tool.run(
            {"term": "foo", "dir": "/does/not/exist"}, cwd="."
        )
        assert result.status == "error"
        assert "Directory not found" in result.output


class TestEditTool:
    """Test suite for EditTool."""

    @pytest.mark.asyncio
    async def test_edit_success(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("line1\nline2\nline3\n")
            path = f.name

        tool = EditTool()
        result = await tool.run(
            {"path": path, "start_line": 2, "end_line": 2, "replacement": "edited"},
            cwd=".",
        )

        assert result.status == "success"
        content = Path(path).read_text()
        assert "edited" in content
        assert "line2" not in content

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_edit_syntax_guardrail(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("x = 1\ny = 2\n")
            path = f.name

        tool = EditTool()
        result = await tool.run(
            {
                "path": path,
                "start_line": 1,
                "end_line": 1,
                "replacement": "def broken(:",
            },
            cwd=".",
        )

        assert result.status == "error"
        assert "Syntax error" in result.output

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_edit_invalid_range(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("line1\n")
            path = f.name

        tool = EditTool()
        result = await tool.run(
            {"path": path, "start_line": 5, "end_line": 10, "replacement": "x"},
            cwd=".",
        )

        assert result.status == "error"
        assert "Invalid line range" in result.output

        Path(path).unlink()

    @pytest.mark.asyncio
    async def test_edit_missing_file(self):
        tool = EditTool()
        result = await tool.run(
            {"path": "/does/not/exist", "start_line": 1, "end_line": 1},
            cwd=".",
        )
        assert result.status == "error"
        assert "File not found" in result.output


class TestSubmitTool:
    """Test suite for SubmitTool."""

    @pytest.mark.asyncio
    async def test_submit(self):
        tool = SubmitTool()
        result = await tool.run({}, cwd=".")
        assert result.status == "success"
        assert result.metadata["submitted"] is True
