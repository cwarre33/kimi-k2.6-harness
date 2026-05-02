"""Tool registry and base classes for ACI-style tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Result of a tool invocation."""

    status: str  # "success" or "error"
    output: str
    exit_code: int = 0
    metadata: Optional[Dict[str, Any]] = None


class Tool(ABC):
    """Base class for all tools."""

    name: str
    description: str

    @abstractmethod
    async def run(self, arguments: Dict[str, Any], cwd: str = ".") -> ToolResult:
        """Execute the tool with the given arguments."""


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}

    def get_system_prompt(self) -> str:
        """Generate a system prompt describing all available tools."""
        lines = ["Available tools:"]
        for name, description in self.list_tools().items():
            lines.append(f"- {name}: {description}")
        lines.append("\nFormat your response as:")
        lines.append("DISCUSSION: <your reasoning>")
        lines.append("```bash")
        lines.append("<tool_name> <arguments>")
        lines.append("```")
        return "\n".join(lines)


def create_default_registry() -> ToolRegistry:
    """Create a registry with all default tools."""
    from .file_tools import OpenTool, ViewFileTool, SearchFileTool, SearchDirTool, EditTool, ReplaceStringTool
    from .shell_tool import ShellExecTool
    from .submit_tool import SubmitTool

    registry = ToolRegistry()
    registry.register(ShellExecTool())
    registry.register(OpenTool())
    registry.register(ViewFileTool())
    registry.register(SearchFileTool())
    registry.register(SearchDirTool())
    registry.register(EditTool())
    registry.register(ReplaceStringTool())
    registry.register(SubmitTool())
    return registry
