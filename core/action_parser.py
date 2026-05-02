"""Structured action parser for LLM outputs."""

import re
import shlex
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ParsedAction:
    """Represents a parsed action from an LLM response."""

    discussion: str
    command: str
    command_type: str  # e.g., 'shell.exec', 'open', 'edit', 'submit'
    arguments: dict
    raw: str


class ActionParser:
    """Parse LLM responses into structured actions.

    Enforces a DISCUSSION + COMMAND format inspired by SWE-agent.
    Supports markdown code fences and plain command output.
    """

    FORMAT_ERROR_TEMPLATE = (
        "Your response was malformed. Please use this exact format:\n\n"
        "DISCUSSION: \u003cyour reasoning here\u003e\n\n"
        "```bash\n"
        "\u003ctool_name\u003e \u003carguments\u003e\n"
        "```"
    )

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def parse(self, raw_response: str) -> ParsedAction:
        """Parse a raw LLM response into a structured action.

        On first malformed response, returns a __retry__ action so the caller
        can ask the model to retry. After max_retries exceeded, raises ParseError.

        NOTE: This method is stateless; retry_count is tracked by the caller.
        For single-shot parsing without retry semantics, use parse_once().
        """
        try:
            return self._try_parse(raw_response)
        except ParseError:
            # Return retry signal for caller to handle
            return ParsedAction(
                discussion="",
                command="",
                command_type="__retry__",
                arguments={"error": "Failed to parse action"},
                raw=raw_response,
            )

    def parse_once(self, raw_response: str) -> ParsedAction:
        """Parse without retry fallback — raises ParseError on failure."""
        return self._try_parse(raw_response)

    def _try_parse(self, raw: str) -> ParsedAction:
        """Attempt to extract DISCUSSION and COMMAND from raw text."""
        if not raw or not raw.strip():
            raise ParseError("Empty response")

        # Extract discussion
        discussion = ""
        discussion_match = re.search(
            r"DISCUSSION:\s*(.+?)(?=\n\n|\n```|$)", raw, re.DOTALL | re.IGNORECASE
        )
        if discussion_match:
            discussion = discussion_match.group(1).strip()

        # Extract command from markdown code fence
        command = ""
        fence_match = re.search(r"```(?:\w+)?\n(.*?)\n?```", raw, re.DOTALL)
        if fence_match:
            command = fence_match.group(1).strip()
        else:
            # Fallback 1: look for COMMAND: prefix
            cmd_match = re.search(r"^\s*COMMAND:\s*(.+)$", raw, re.MULTILINE | re.IGNORECASE)
            if cmd_match:
                command = cmd_match.group(1).strip()
            else:
                # Fallback 2: take the last non-empty line as command
                lines = [line.strip() for line in raw.splitlines() if line.strip()]
                if lines:
                    # Skip DISCUSSION line if present
                    if not lines[-1].upper().startswith("DISCUSSION:"):
                        command = lines[-1]

        if not command:
            raise ParseError("No command found in response")

        # Parse command into type and arguments
        cmd_type, args = self._parse_command(command)

        return ParsedAction(
            discussion=discussion,
            command=command,
            command_type=cmd_type,
            arguments=args,
            raw=raw,
        )

    def _parse_command(self, command: str) -> Tuple[str, dict]:
        """Parse a command string into type and arguments."""
        try:
            parts = shlex.split(command)
        except ValueError:
            # Fallback for unclosed quotes
            parts = command.split(None, 1)

        if not parts:
            raise ParseError("Empty command")

        cmd_type = parts[0].strip()
        args_parts = parts[1:]  # remaining tokens after the command name

        arguments: dict = {"args": " ".join(args_parts)}

        # Parse named arguments for specific tools
        if cmd_type == "open" and args_parts:
            # open <path> [line_number]
            arguments["path"] = args_parts[0]
            if len(args_parts) > 1:
                try:
                    arguments["line"] = int(args_parts[1])
                except ValueError:
                    pass

        elif cmd_type == "view_file" and args_parts:
            # view_file <path>
            arguments["path"] = args_parts[0]

        elif cmd_type == "edit" and args_parts:
            # edit <start_line>:<end_line> <replacement_text>
            # Re-join remaining parts for regex matching
            args_str = " ".join(args_parts)
            match = re.match(r"(\d+):(\d+)\s+(.+)", args_str, re.DOTALL)
            if match:
                arguments["start_line"] = int(match.group(1))
                arguments["end_line"] = int(match.group(2))
                arguments["replacement"] = match.group(3)

        elif cmd_type == "search_file" and args_parts:
            # search_file <term> [file]
            arguments["term"] = args_parts[0]
            if len(args_parts) > 1:
                arguments["file"] = args_parts[1]

        elif cmd_type == "search_dir" and args_parts:
            # search_dir <term> [dir]
            arguments["term"] = args_parts[0]
            if len(args_parts) > 1:
                arguments["dir"] = args_parts[1]

        elif cmd_type == "replace_string" and len(args_parts) >= 3:
            # replace_string <path> <old_string> <new_string>
            arguments["path"] = args_parts[0]
            arguments["old_string"] = args_parts[1]
            arguments["new_string"] = args_parts[2]

        elif cmd_type == "shell.exec" and args_parts:
            arguments["command"] = " ".join(args_parts)

        return cmd_type, arguments

    def get_retry_prompt(self, error: str) -> str:
        """Return a prompt to send back to the model on parse failure."""
        return f"{self.FORMAT_ERROR_TEMPLATE}\n\nError: {error}"


class ParseError(Exception):
    """Raised when an LLM response cannot be parsed into a valid action."""
