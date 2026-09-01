"""
agent/tools.py — Tool registry for AuraLite Agent.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ToolResult:
    output: str
    error: Optional[str] = None
    success: bool = True

    @classmethod
    def ok(cls, text: str) -> "ToolResult":
        return cls(output=text, error=None, success=True)

    @classmethod
    def fail(cls, msg: str) -> "ToolResult":
        return cls(output="", error=msg, success=False)

    def to_text(self) -> str:
        if self.error:
            return f"[ERROR] {self.error}"
        return self.output


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict
    handler: Callable
    example: str = ""


# ── Tool implementations ────────────────────────────────────────────────────

def _tool_python(sandbox, code: str) -> ToolResult:
    from .sandbox import SandboxResult
    result: SandboxResult = sandbox.run_python(code)
    out = result.combined_output()
    if result.blocked:
        return ToolResult.fail(out)
    return ToolResult(output=out, success=result.success)


def _tool_shell(sandbox, command: str) -> ToolResult:
    from .sandbox import SandboxResult
    result: SandboxResult = sandbox.run_shell(command)
    out = result.combined_output()
    if result.blocked:
        return ToolResult.fail(out)
    return ToolResult(output=out, success=result.success)


def _tool_write_file(sandbox, filename: str, content: str) -> ToolResult:
    result = sandbox.write_file(filename, content)
    if result.error:
        return ToolResult.fail(result.error)
    return ToolResult.ok(result.stdout)


def _tool_read_file(sandbox, filename: str) -> ToolResult:
    result = sandbox.read_file(filename)
    if result.error:
        return ToolResult.fail(result.error)
    return ToolResult.ok(result.stdout)


def _tool_list_files(sandbox, directory: str = ".") -> ToolResult:
    result = sandbox.list_files(directory)
    if result.error:
        return ToolResult.fail(result.error)
    return ToolResult.ok(result.stdout)


def _tool_install(sandbox, package: str) -> ToolResult:
    result = sandbox.install_package(package)
    out = result.combined_output()
    if result.blocked:
        return ToolResult.fail(out)
    return ToolResult(output=out or "Installed.", success=result.success)


def _tool_web_search(sandbox, query: str) -> ToolResult:
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from web_tools import build_web_context
        ctx = build_web_context(query, max_results=4, max_chars=2000)
        return ToolResult.ok(ctx if ctx else "No results found.")
    except Exception as e:
        return ToolResult.fail(str(e))


def _tool_calculate(sandbox, expression: str) -> ToolResult:
    code = f"import math, cmath\nprint(eval({expression!r}, {{'__builtins__': {{}}, 'math': math, 'cmath': cmath}}))"
    result = sandbox.run_python(code)
    if result.returncode != 0 or result.stderr:
        return ToolResult.fail(f"Calculation error: {result.stderr or result.stdout}")
    return ToolResult.ok(result.stdout.strip())


# ── Registry ─────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, ToolDef] = {
    "python": ToolDef(
        name="python",
        description="Execute Python code in a sandboxed subprocess.",
        parameters={"code": {"type": "string", "description": "Python code to execute"}},
        handler=_tool_python,
        example='<tool name="python">print("Hello!")</tool>',
    ),
    "shell": ToolDef(
        name="shell",
        description="Run a whitelisted shell command (ls, git, curl, etc.).",
        parameters={"command": {"type": "string", "description": "Shell command string"}},
        handler=_tool_shell,
        example='<tool name="shell">ls -la</tool>',
    ),
    "write_file": ToolDef(
        name="write_file",
        description="Write text content to a file inside the sandbox workspace.",
        parameters={
            "filename": {"type": "string", "description": "Relative filename"},
            "content": {"type": "string", "description": "File content"},
        },
        handler=_tool_write_file,
        example='<tool name="write_file" filename="hello.py">print("hi")</tool>',
    ),
    "read_file": ToolDef(
        name="read_file",
        description="Read a file from the sandbox workspace.",
        parameters={"filename": {"type": "string", "description": "Relative filename"}},
        handler=_tool_read_file,
        example='<tool name="read_file">hello.py</tool>',
    ),
    "list_files": ToolDef(
        name="list_files",
        description="List files in a sandbox directory.",
        parameters={"directory": {"type": "string", "description": "Directory path, default '.'"}},
        handler=_tool_list_files,
        example='<tool name="list_files">.</tool>',
    ),
    "install": ToolDef(
        name="install",
        description="Install a Python package with pip.",
        parameters={"package": {"type": "string", "description": "Package name"}},
        handler=_tool_install,
        example='<tool name="install">requests</tool>',
    ),
    "web_search": ToolDef(
        name="web_search",
        description="Search the web and return relevant snippets.",
        parameters={"query": {"type": "string", "description": "Search query"}},
        handler=_tool_web_search,
        example='<tool name="web_search">Python asyncio tutorial</tool>',
    ),
    "calculate": ToolDef(
        name="calculate",
        description="Evaluate a mathematical expression safely.",
        parameters={"expression": {"type": "string", "description": "Math expression"}},
        handler=_tool_calculate,
        example='<tool name="calculate">2**32 + math.pi</tool>',
    ),
}


# ── XML parser ────────────────────────────────────────────────────────────────

_TOOL_OPEN_RE  = re.compile(r'<tool\s+name=["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']([^>]*)>', re.DOTALL)
_TOOL_CLOSE_RE = re.compile(r'</tool>', re.DOTALL)
_ATTR_RE       = re.compile(r'([a-zA-Z_]\w*)\s*=\s*["\']([^"\']*)["\']')


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract all <tool ...>...</tool> blocks from model output."""
    calls = []
    pos = 0
    while pos < len(text):
        m_open = _TOOL_OPEN_RE.search(text, pos)
        if not m_open:
            break
        tool_name = m_open.group(1)
        attr_str  = m_open.group(2)
        body_start = m_open.end()

        m_close = _TOOL_CLOSE_RE.search(text, body_start)
        if not m_close:
            break

        body = text[body_start:m_close.start()].strip()
        args: dict[str, str] = {}
        for am in _ATTR_RE.finditer(attr_str):
            k, v = am.group(1), am.group(2)
            if k != "name":
                args[k] = v

        tool_def = TOOL_REGISTRY.get(tool_name)
        if tool_def:
            param_names = list(tool_def.parameters.keys())
            # Find the first parameter not yet set from attributes
            primary = next((p for p in param_names if p not in args), param_names[0] if param_names else "content")
        else:
            primary = "content"

        if primary not in args and body:
            args[primary] = body

        calls.append({"name": tool_name, "args": args, "body": body})
        pos = m_close.end()

    return calls


def dispatch_tool(tool_name: str, args: dict[str, Any], sandbox) -> ToolResult:
    """Dispatch a parsed tool call to its handler."""
    tool_def = TOOL_REGISTRY.get(tool_name)
    if tool_def is None:
        return ToolResult.fail(f"Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}")
    try:
        return tool_def.handler(sandbox, **args)
    except TypeError as e:
        return ToolResult.fail(f"Tool argument error for '{tool_name}': {e}")
    except Exception as e:
        return ToolResult.fail(f"Tool '{tool_name}' raised an exception: {e}")


def build_system_prompt(enabled_tools: list[str] | None = None) -> str:
    """Build the system prompt describing available tools."""
    tools = enabled_tools or list(TOOL_REGISTRY.keys())
    lines = [
        "You are AuraLite Agent — an AI assistant with access to a sandboxed environment.",
        "You can use the following tools by emitting XML-style tags in your response:",
        "",
    ]
    for name in tools:
        td = TOOL_REGISTRY.get(name)
        if not td:
            continue
        lines.append(f'  <tool name="{name}"> ... </tool>')
        lines.append(f"    → {td.description}")
        if td.example:
            lines.append(f"    Example: {td.example.split(chr(10))[0]}")
        lines.append("")

    lines += [
        "Rules:",
        "  1. Use tools when needed — run code, search the web, write files.",
        "  2. After a tool runs, you will see its output between [TOOL RESULT] tags.",
        "  3. Always explain what you are doing before using a tool.",
        "  4. If a tool fails, read the error and try again with a fix.",
        "  5. Sandbox has a timeout per execution — keep code short and focused.",
        "  6. You are isolated: no access to the host filesystem outside the sandbox.",
    ]
    return "\n".join(lines)
