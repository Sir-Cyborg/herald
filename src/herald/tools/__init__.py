"""Tools the assistant can call, and how to add your own.

To add a tool, write a function with type hints and a docstring and mark it with ``@tool``::

    from herald.tools import tool

    @tool(triggers=["add", "plus", "sum"])
    def add(a: int, b: int) -> str:
        \"\"\"Add two numbers.

        Args:
            a: The first number.
            b: The second number.
        \"\"\"
        return str(a + b)

``triggers`` are the words that make the tool available: it is only offered to the model when the
user's message contains one of them (ignoring case and accents), because a small model calls any
tool it is offered on almost every message. A tool without triggers is always offered.

Put it in a ``.py`` file in the ``tools/`` directory of the project (``HERALD_TOOLS_DIR``) and
it is found at startup. Built-in tools are modules of ``herald.tools.builtin``. A tool that must
speak or act later takes a keyword parameter annotated ``ToolContext``. See ``herald.tools.loader``
for the loading rules and the security note.

``Scheduler`` (``herald.tools.scheduler``) is also available as ``herald.tools.Scheduler``; it is
imported only when asked for.
"""

from __future__ import annotations

from typing import Any

from herald.tools.context import ToolContext
from herald.tools.decorator import tool
from herald.tools.loader import LoadedTool, LoadReport, load_tools
from herald.tools.registry import Tool, ToolRegistry

__all__ = [
    "LoadReport",
    "LoadedTool",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "load_tools",
    "tool",
]


def __getattr__(name: str) -> Any:
    if name == "Scheduler":  # lazy: keeps this package importable without the scheduler module
        from herald.tools.scheduler import Scheduler

        return Scheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
