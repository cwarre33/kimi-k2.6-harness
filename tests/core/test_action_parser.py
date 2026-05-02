"""Tests for the action parser."""

import pytest

from core.action_parser import ActionParser, ParsedAction, ParseError


class TestActionParser:
    """Test suite for ActionParser."""

    def test_parse_valid_discussion_and_command(self):
        raw = (
            "DISCUSSION: I need to check the current directory contents.\n\n"
            "```bash\n"
            "shell.exec ls -la\n"
            "```"
        )
        parser = ActionParser()
        action = parser.parse(raw)

        assert action.discussion == "I need to check the current directory contents."
        assert action.command == "shell.exec ls -la"
        assert action.command_type == "shell.exec"
        assert action.arguments["command"] == "ls -la"

    def test_parse_without_discussion(self):
        raw = "```bash\nopen main.py 10\n```"
        parser = ActionParser()
        action = parser.parse(raw)

        assert action.command == "open main.py 10"
        assert action.command_type == "open"
        assert action.arguments["path"] == "main.py"
        assert action.arguments["line"] == 10

    def test_parse_plain_command_fallback(self):
        raw = "shell.exec echo hello"
        parser = ActionParser()
        action = parser.parse(raw)

        assert action.command == "shell.exec echo hello"
        assert action.command_type == "shell.exec"

    def test_parse_edit_command(self):
        raw = "```bash\nedit 5:10 replacement text here\n```"
        parser = ActionParser()
        action = parser.parse(raw)

        assert action.command_type == "edit"
        assert action.arguments["start_line"] == 5
        assert action.arguments["end_line"] == 10
        assert action.arguments["replacement"] == "replacement text here"

    def test_parse_search_file(self):
        raw = "```bash\nsearch_file foo main.py\n```"
        parser = ActionParser()
        action = parser.parse(raw)

        assert action.command_type == "search_file"
        assert action.arguments["term"] == "foo"
        assert action.arguments["file"] == "main.py"

    def test_parse_empty_response_returns_retry(self):
        parser = ActionParser()
        action = parser.parse("")
        assert action.command_type == "__retry__"
        assert "error" in action.arguments

    def test_parse_no_command_returns_retry(self):
        parser = ActionParser()
        action = parser.parse("DISCUSSION: Just thinking here.")
        assert action.command_type == "__retry__"

    def test_parse_once_empty_raises(self):
        parser = ActionParser()
        with pytest.raises(ParseError, match="Empty response"):
            parser.parse_once("")

    def test_parse_once_no_command_raises(self):
        parser = ActionParser()
        with pytest.raises(ParseError, match="No command found"):
            parser.parse_once("DISCUSSION: Just thinking here.")

    def test_parse_submit(self):
        raw = "```bash\nsubmit\n```"
        parser = ActionParser()
        action = parser.parse(raw)

        assert action.command_type == "submit"

    def test_get_retry_prompt(self):
        parser = ActionParser()
        prompt = parser.get_retry_prompt("missing command block")
        assert "DISCUSSION:" in prompt
        assert "missing command block" in prompt

    def test_parse_search_dir(self):
        raw = "```bash\nsearch_dir bar src/\n```"
        parser = ActionParser()
        action = parser.parse(raw)

        assert action.command_type == "search_dir"
        assert action.arguments["term"] == "bar"
        assert action.arguments["dir"] == "src/"

    def test_parse_multiline_discussion(self):
        raw = (
            "DISCUSSION: Line one.\nLine two.\n\n"
            "```bash\nshell.exec pwd\n```"
        )
        parser = ActionParser()
        action = parser.parse(raw)

        assert "Line one." in action.discussion
        assert action.command == "shell.exec pwd"
