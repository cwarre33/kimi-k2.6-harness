"""File viewer, search, and edit tools."""

import difflib
import re
from pathlib import Path
from typing import Any, Dict

from . import Tool, ToolResult


class ViewFileTool(Tool):
    """View a complete file without pagination."""

    name = "view_file"
    description = (
        "View the entire contents of a file. "
        "Use this when you need to see the complete file to find a bug. "
        "Usage: view_file <path>"
    )

    MAX_LINES = 500

    async def run(self, arguments: Dict[str, Any], cwd: str = ".") -> ToolResult:
        path = arguments.get("path", "")
        if not path:
            return ToolResult(
                status="error", output="No path provided", exit_code=-1
            )

        file_path = Path(cwd) / path
        if not file_path.exists():
            return ToolResult(
                status="error",
                output=f"File not found: {path}",
                exit_code=-1,
            )

        try:
            text = file_path.read_text()
        except Exception as exc:
            return ToolResult(
                status="error", output=str(exc), exit_code=-1
            )

        lines = text.splitlines()
        total = len(lines)

        if total > self.MAX_LINES:
            # Show first and last parts with ellipsis
            head_lines = lines[: self.MAX_LINES // 2]
            tail_lines = lines[-self.MAX_LINES // 2 :]
            output_lines = head_lines + [f"\n... {total - self.MAX_LINES} lines omitted ...\n"] + tail_lines
        else:
            output_lines = lines

        output = "\n".join(output_lines)
        if total > self.MAX_LINES:
            output += f"\n\n[File: {path} | {total} lines total | showing first and last {self.MAX_LINES // 2} lines]"
        else:
            output += f"\n\n[File: {path} | {total} lines total]"

        return ToolResult(
            status="success",
            output=output,
            exit_code=0,
            metadata={"total_lines": total},
        )


class OpenTool(Tool):
    """Open a file and display a window of lines."""

    name = "open"
    description = (
        "Open a file with line numbers. Shows a 100-line window centered on the given line. "
        "Usage: open <path> [line_number]. "
        "If the file is longer than 100 lines, request a different line_number to scroll."
    )

    WINDOW_SIZE = 100

    async def run(self, arguments: Dict[str, Any], cwd: str = ".") -> ToolResult:
        path = arguments.get("path", "")
        if not path:
            return ToolResult(
                status="error", output="No path provided", exit_code=-1
            )

        file_path = Path(cwd) / path
        if not file_path.exists():
            return ToolResult(
                status="error",
                output=f"File not found: {path}",
                exit_code=-1,
            )

        try:
            lines = file_path.read_text().splitlines()
        except Exception as exc:
            return ToolResult(
                status="error", output=str(exc), exit_code=-1
            )

        total = len(lines)
        line_num = arguments.get("line", 1)
        if line_num < 1:
            line_num = 1
        if line_num > total:
            line_num = total

        # Center window on line_num
        half_window = self.WINDOW_SIZE // 2
        start = max(0, line_num - half_window - 1)
        end = min(total, line_num + half_window)

        output_lines = []
        output_lines.append(
            f"[File: {path} | Lines {start + 1}-{end} of {total}]"
        )
        for i in range(start, end):
            marker = " >" if i == line_num - 1 else "  "
            output_lines.append(f"{marker}{i + 1:4d} {lines[i]}")

        if end < total:
            output_lines.append(f"  ... {total - end} more lines ...")
        if start > 0:
            output_lines.insert(1, f"  ... {start} lines above ...")

        return ToolResult(
            status="success",
            output="\n".join(output_lines),
            exit_code=0,
            metadata={
                "total_lines": total,
                "window_start": start + 1,
                "window_end": end,
                "current_line": line_num,
            },
        )


class SearchFileTool(Tool):
    """Search for a term within a specific file."""

    name = "search_file"
    description = (
        "Search for a term in a file. "
        "Usage: search_file <term> [file_path]"
    )

    MAX_MATCHES = 50

    async def run(self, arguments: Dict[str, Any], cwd: str = ".") -> ToolResult:
        term = arguments.get("term", "")
        if not term:
            return ToolResult(
                status="error", output="No search term provided", exit_code=-1
            )

        file_path = arguments.get("file", "")
        if not file_path:
            return ToolResult(
                status="error", output="No file path provided", exit_code=-1
            )

        target = Path(cwd) / file_path
        if not target.exists():
            return ToolResult(
                status="error",
                output=f"File not found: {file_path}",
                exit_code=-1,
            )

        try:
            text = target.read_text()
            lines = text.splitlines()
        except Exception as exc:
            return ToolResult(
                status="error", output=str(exc), exit_code=-1
            )

        matches = []
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                matches.append(f"{i:4d}: {line}")
            if len(matches) >= self.MAX_MATCHES:
                matches.append(
                    f"[Warning: more than {self.MAX_MATCHES} matches, refine your search]"
                )
                break

        if not matches:
            return ToolResult(
                status="success",
                output=f"No matches for '{term}' in {file_path}",
                exit_code=0,
            )

        return ToolResult(
            status="success",
            output=f"Matches for '{term}' in {file_path}:\n"
            + "\n".join(matches),
            exit_code=0,
            metadata={"match_count": len(matches)},
        )


class SearchDirTool(Tool):
    """Search for a term across files in a directory."""

    name = "search_dir"
    description = (
        "Search for a term across files in a directory. "
        "Usage: search_dir <term> [dir_path]"
    )

    MAX_MATCHES = 50

    async def run(self, arguments: Dict[str, Any], cwd: str = ".") -> ToolResult:
        term = arguments.get("term", "")
        if not term:
            return ToolResult(
                status="error", output="No search term provided", exit_code=-1
            )

        dir_path = arguments.get("dir", cwd)
        target = Path(cwd) / dir_path
        if not target.exists():
            return ToolResult(
                status="error",
                output=f"Directory not found: {dir_path}",
                exit_code=-1,
            )

        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = []

        try:
            for file_path in target.rglob("*"):
                if not file_path.is_file():
                    continue
                try:
                    text = file_path.read_text()
                    for i, line in enumerate(text.splitlines(), 1):
                        if pattern.search(line):
                            rel = file_path.relative_to(target)
                            matches.append(f"{rel}:{i}: {line}")
                        if len(matches) >= self.MAX_MATCHES:
                            matches.append(
                                "[Warning: more than "
                                f"{self.MAX_MATCHES} matches, refine your search]"
                            )
                            break
                    if len(matches) >= self.MAX_MATCHES:
                        break
                except (UnicodeDecodeError, PermissionError):
                    continue
        except Exception as exc:
            return ToolResult(
                status="error", output=str(exc), exit_code=-1
            )

        if not matches:
            return ToolResult(
                status="success",
                output=f"No matches for '{term}' in {dir_path}",
                exit_code=0,
            )

        return ToolResult(
            status="success",
            output=f"Matches for '{term}' in {dir_path}:\n"
            + "\n".join(matches),
            exit_code=0,
            metadata={"match_count": len(matches)},
        )


class ReplaceStringTool(Tool):
    """Replace a string in a file with another string."""

    name = "replace_string"
    description = (
        "Replace a specific string in a file with another string. "
        "This is the preferred tool for simple text fixes. "
        "Usage: replace_string <path> <old_string> <new_string>"
    )

    async def run(self, arguments: Dict[str, Any], cwd: str = ".") -> ToolResult:
        path = arguments.get("path", "")
        old = arguments.get("old_string", "")
        new = arguments.get("new_string", "")

        if not path:
            return ToolResult(
                status="error", output="No file path provided", exit_code=-1
            )
        if not old:
            return ToolResult(
                status="error", output="No old_string provided", exit_code=-1
            )

        file_path = Path(cwd) / path
        if not file_path.exists():
            return ToolResult(
                status="error",
                output=f"File not found: {path}",
                exit_code=-1,
            )

        # Reject edits to test files
        parts = [p.lower() for p in file_path.parts]
        if any(p in parts for p in ("test", "tests", "fixtures")):
            return ToolResult(
                status="error",
                output="Edits to test/fixture files are not allowed. Edit the source code instead.",
                exit_code=-1,
            )

        try:
            text = file_path.read_text()
        except Exception as exc:
            return ToolResult(
                status="error", output=str(exc), exit_code=-1
            )

        if old not in text:
            # Try to find the closest match to help the model fix its replace_string
            closest = difflib.get_close_matches(old, [text], n=1, cutoff=0.3)
            hint = ""
            if closest:
                hint = f"\n\nDid you mean:\n{closest[0][:500]}"
            return ToolResult(
                status="error",
                output=f"String not found in file: {old[:80]}{hint}",
                exit_code=-1,
            )

        # Replace only the first occurrence
        new_text = text.replace(old, new, 1)

        # Basic syntax check for Python files
        if file_path.suffix == ".py":
            try:
                compile(new_text, str(file_path), "exec")
            except SyntaxError as exc:
                return ToolResult(
                    status="error",
                    output=(
                        f"Syntax error introduced by replace: {exc}\n"
                        f"Replace reverted."
                    ),
                    exit_code=-1,
                )

        file_path.write_text(new_text)

        return ToolResult(
            status="success",
            output=f"Replaced string in {path}",
            exit_code=0,
            metadata={"path": path, "replacements": 1},
        )


class EditTool(Tool):
    """Edit a range of lines in a file."""

    name = "edit"
    description = (
        "Replace lines in a file. "
        "Usage: edit <start_line>:<end_line> <replacement_text>"
    )

    async def run(self, arguments: Dict[str, Any], cwd: str = ".") -> ToolResult:
        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")
        replacement = arguments.get("replacement", "")
        path = arguments.get("path", "")

        if not path:
            return ToolResult(
                status="error", output="No file path provided", exit_code=-1
            )
        if start_line is None or end_line is None:
            return ToolResult(
                status="error",
                output="Missing start_line or end_line",
                exit_code=-1,
            )

        file_path = Path(cwd) / path
        if not file_path.exists():
            return ToolResult(
                status="error",
                output=f"File not found: {path}",
                exit_code=-1,
            )

        # Reject edits to test files
        parts = [p.lower() for p in file_path.parts]
        if any(p in parts for p in ("test", "tests", "fixtures")):
            return ToolResult(
                status="error",
                output="Edits to test/fixture files are not allowed. Edit the source code instead.",
                exit_code=-1,
            )

        try:
            lines = file_path.read_text().splitlines()
        except Exception as exc:
            return ToolResult(
                status="error", output=str(exc), exit_code=-1
            )

        total = len(lines)
        if start_line < 1 or end_line > total or start_line > end_line:
            return ToolResult(
                status="error",
                output=(
                    f"Invalid line range "
                    f"{start_line}:{end_line} (file has {total} lines)"
                ),
                exit_code=-1,
            )

        # Perform the edit
        replacement_lines = replacement.splitlines()
        new_lines = (
            lines[: start_line - 1]
            + replacement_lines
            + lines[end_line:]
        )

        # Basic syntax check for Python files
        if file_path.suffix == ".py":
            new_text = "\n".join(new_lines)
            try:
                compile(new_text, str(file_path), "exec")
            except SyntaxError as exc:
                return ToolResult(
                    status="error",
                    output=(
                        f"Syntax error introduced by edit: {exc}\n"
                        f"Edit reverted."
                    ),
                    exit_code=-1,
                )

        file_path.write_text("\n".join(new_lines) + "\n")

        return ToolResult(
            status="success",
            output=(
                f"Edited {path} lines {start_line}:{end_line} "
                f"(replaced with {len(replacement_lines)} lines)"
            ),
            exit_code=0,
            metadata={
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "replacement_lines": len(replacement_lines),
            },
        )
