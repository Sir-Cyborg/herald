"""What the application gives a tool that needs to act later or speak on its own."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from herald.tools.scheduler import Scheduler


@dataclass(frozen=True)
class ToolContext:
    """Services for tools. A tool asks for it by annotating a parameter with ``ToolContext``.

    ``say`` shows or speaks a message right now. It is safe to call from any thread (a timer
    fires on the scheduler's thread) and must not raise. ``scheduler`` runs a function later.
    """

    say: Callable[[str], None]
    scheduler: Scheduler
