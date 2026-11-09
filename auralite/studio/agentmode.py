"""The Chat tab's agent mode (2.26): the model may call tools - run Python, commands, read
and write files in a temporary folder, search the web, calculate - and reads their results
before it answers. The classic Chat tab's agent mode, for the Studio; no toolkit here.

The tools run on this computer, as the user, in a temporary folder
(:class:`auralite.agent.Sandbox`): that limits the damage of a mistake, it is not a
security boundary - :data:`WARNING` says so before agent mode starts.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any

__all__ = ["DEFAULTS", "STEPS", "TIMEOUTS", "WARNING", "AgentOptions", "ToolRow", "form_defaults",
           "make_agent", "make_sandbox", "options_from_form", "python_for_tools", "step_lines",
           "tool_rows"]

#: The classic Chat tab's agent settings.
DEFAULTS = {"max_iterations": 8, "max_tokens_per_step": 512, "timeout": 15.0,
            "temperature": 0.7, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1}
#: The steps an answer may take, and the seconds a tool may run (the windows' limits).
STEPS = (1, 30)
TIMEOUTS = (1, 600)
#: Tools that need a Python interpreter of their own.
PYTHON_TOOLS = ("python", "install")
WARNING = ("In agent mode the model can run Python code and commands on this computer, "
           "with your permissions, in a temporary folder that is deleted afterwards; it can "
           "also read files elsewhere and use the network. That limits the damage of a "
           "mistake - it is not a security boundary. Turn on only the tools you need, and "
           "use agent mode with models you trust.")


def form_defaults() -> tuple[int, float]:
    """The steps and the seconds per tool a window's agent fields start with: the
    ``AURALITE_AGENT_MAX_ITERATIONS`` and ``AURALITE_AGENT_TOOL_TIMEOUT`` variables (as
    for :class:`auralite.agent.AuraLiteAgent`) within the fields' limits, else
    :data:`DEFAULTS`."""
    steps, seconds = int(DEFAULTS["max_iterations"]), float(DEFAULTS["timeout"])
    try:
        steps = int(os.environ.get("AURALITE_AGENT_MAX_ITERATIONS", "").strip() or steps)
    except ValueError:
        pass
    try:
        seconds = float(os.environ.get("AURALITE_AGENT_TOOL_TIMEOUT", "").strip() or seconds)
    except ValueError:
        pass
    if seconds != seconds:                    # NaN
        seconds = float(DEFAULTS["timeout"])
    return (min(max(steps, STEPS[0]), STEPS[1]),
            min(max(seconds, float(TIMEOUTS[0])), float(TIMEOUTS[1])))


@dataclass(frozen=True)
class ToolRow:
    """A tool as the tool list shows it."""

    name: str
    description: str
    available: bool
    note: str = ""          # why it is not available, or what it runs with


def python_for_tools() -> str | None:
    """The Python the Python tools run: this one, or inside the app (which has no Python
    of its own: its program is the app) a Python found on the PATH; None when there is
    none (the Microsoft Store's stand-in, which only opens the Store, does not count)."""
    from auralite import bundle
    import sys
    if not bundle.packaged():
        return sys.executable
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found and "windowsapps" not in found.lower().replace("/", os.sep):
            return found
    return None


def tool_rows(python: str | None | bool = False) -> list[ToolRow]:
    """Every tool of :data:`auralite.agent.TOOL_REGISTRY`, and whether it can run here
    (`python`: the interpreter of :func:`python_for_tools`; ``False`` finds it)."""
    from auralite.agent import TOOL_REGISTRY
    from auralite import bundle
    interpreter = python_for_tools() if python is False else python
    rows = []
    for name, tool in TOOL_REGISTRY.items():
        available, note = True, ""
        if name in PYTHON_TOOLS:
            if not interpreter:
                available = False
                note = ("needs Python: the app has none of its own - install Python, or "
                        "AuraLite with pip")
            elif bundle.packaged():
                note = f"runs {interpreter}"
        rows.append(ToolRow(name, str(tool.description), available, note))
    return rows


@dataclass
class AgentOptions:
    """What a window lets a user choose about the agent."""

    tools: tuple[str, ...] = ()
    max_iterations: int = 8
    timeout: float = 15.0
    max_tokens_per_step: int = 512
    temperature: float = 0.7
    chat_template: str = "chatml"


def options_from_form(tools: Any, steps: Any = DEFAULTS["max_iterations"],
                      timeout: Any = DEFAULTS["timeout"], **more: Any) -> AgentOptions:
    """2.26: the agent's options from a form's fields (the classic Chat tab types them):
    the tools turned on, the steps an answer may take and the seconds a tool may run.
    ValueError says which field is out of its range; `more` sets the other options."""
    try:
        steps_value = int(str(steps).strip())
    except ValueError:
        steps_value = 0
    if not STEPS[0] <= steps_value <= STEPS[1]:
        raise ValueError(f"the steps are a whole number from {STEPS[0]} to {STEPS[1]}")
    try:
        seconds = float(str(timeout).strip())
    except ValueError:
        seconds = 0.0
    if not TIMEOUTS[0] <= seconds <= TIMEOUTS[1]:
        raise ValueError(f"the seconds a tool may run are from {TIMEOUTS[0]} to {TIMEOUTS[1]}")
    chosen = tuple(str(name) for name in tools or ())
    if not chosen:
        raise ValueError("turn on at least one tool the agent can use here")
    return AgentOptions(tools=chosen, max_iterations=steps_value, timeout=seconds, **more)


def make_sandbox(timeout: float = 15.0, python: str | None | bool = False) -> Any:
    """The temporary folder the tools work in (:class:`auralite.agent.Sandbox`), with the
    Python of :func:`python_for_tools`. A chat keeps one, so a file the agent writes
    is there for the next message too."""
    from auralite.agent import Sandbox
    interpreter = python_for_tools() if python is False else python
    return Sandbox(timeout=float(timeout),
                   python_executable=interpreter if isinstance(interpreter, str) else None)


def make_agent(engine: Any, options: AgentOptions, python: str | None | bool = False,
               sandbox: Any = None) -> Any:
    """An :class:`auralite.agent.AuraLiteAgent` for the model in `engine` with `options`,
    working in `sandbox` (a new one when None); the tools that cannot run here are left
    out (ValueError when none is left)."""
    from auralite.agent import AuraLiteAgent
    interpreter = python_for_tools() if python is False else python
    runnable = {row.name for row in tool_rows(interpreter) if row.available}
    tools = [name for name in options.tools if name in runnable]
    if not tools:
        raise ValueError("turn on at least one tool the agent can use here")
    if sandbox is None:
        sandbox = make_sandbox(options.timeout, interpreter)
    else:
        sandbox.timeout = float(options.timeout)
    return AuraLiteAgent(
        engine, sandbox=sandbox, enabled_tools=tools,
        max_iterations=max(1, int(options.max_iterations)),
        max_tokens_per_step=max(16, int(options.max_tokens_per_step)),
        temperature=float(options.temperature), top_p=DEFAULTS["top_p"],
        top_k=int(DEFAULTS["top_k"]), repetition_penalty=DEFAULTS["repetition_penalty"],
        chat_template=options.chat_template or "chatml",
        timeout_per_tool=float(options.timeout))


def step_lines(step: Any, limit: int = 300) -> list[tuple[str, str]]:
    """The sandbox log's lines for one agent step: ``(text, kind)`` with kind ``"tool"``
    (a call), ``"ok"`` or ``"error"`` (its result, at most `limit` characters)."""
    lines: list[tuple[str, str]] = []
    for call in getattr(step, "tool_calls", None) or []:
        name = call.get("name", "?") if isinstance(call, dict) else str(call)
        lines.append((f"→ Tool: {name}", "tool"))
    for result in getattr(step, "tool_results", None) or []:
        try:
            text = str(result.to_text())
        except Exception as exc:          # a result that cannot say what it is
            text = f"[ERROR] {exc}"
        lines.append((text[:limit], "ok" if getattr(result, "success", False) else "error"))
    return lines
