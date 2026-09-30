"""Run a callback once, some time from now. Standard library only.

Tools use this for anything that happens later, such as a timer that speaks when it ends::

    scheduler = Scheduler()
    task = scheduler.schedule(300, lambda: print("five minutes are up"), label="tea")
    scheduler.pending()  # [task]
    scheduler.cancel(task.id)

Each task is a daemon :class:`threading.Timer`, so a forgotten scheduler never keeps the
program alive. Timers use the monotonic clock: they do not count the time a laptop spends asleep.
``ScheduledTask.due`` is wall-clock time, meant for showing "time left" to a person.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledTask:
    """A pending task, as returned by :meth:`Scheduler.schedule` and :meth:`Scheduler.pending`."""

    id: int
    label: str  # free text for humans and logs, e.g. what a timer will say
    delay: float  # seconds that were asked for
    due: float  # when it fires, as wall-clock epoch seconds (``time.time()``)

    def seconds_left(self) -> float:
        """Seconds until the task fires; 0 once it is due, never negative."""
        return max(0.0, self.due - time.time())


class Scheduler:
    """Fires callbacks after a delay, each one on its own daemon thread.

    Thread-safe. A callback runs on a timer thread, not on the caller's thread, and with no lock
    held, so it may itself call ``schedule``. A callback that raises is logged and forgotten.
    At most ``max_tasks`` tasks can be pending at once.
    """

    def __init__(self, max_tasks: int = 20) -> None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be at least 1")
        self._max_tasks = max_tasks
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[ScheduledTask, threading.Timer]] = {}
        self._next_id = 1
        self._closed = False

    def schedule(
        self, delay_seconds: float, callback: Callable[[], None], *, label: str = ""
    ) -> ScheduledTask:
        """Call ``callback()`` once, ``delay_seconds`` from now, and return the new task.

        Raises ``ValueError`` for a delay that is not a positive, finite, reasonable number, and
        ``RuntimeError`` when ``max_tasks`` are already pending or after :meth:`shutdown`.
        """
        if not math.isfinite(delay_seconds) or delay_seconds <= 0:
            raise ValueError(f"delay must be a positive number of seconds, got {delay_seconds}")
        if delay_seconds > threading.TIMEOUT_MAX:
            raise ValueError(f"delay is too long: {delay_seconds} seconds")
        with self._lock:
            if self._closed:
                raise RuntimeError("scheduler is shut down")
            if len(self._pending) >= self._max_tasks:
                raise RuntimeError(f"too many pending timers (max {self._max_tasks})")
            task = ScheduledTask(
                id=self._next_id,
                label=label,
                delay=float(delay_seconds),
                due=time.time() + delay_seconds,
            )
            self._next_id += 1
            timer = threading.Timer(delay_seconds, self._fire, args=(task.id, callback))
            timer.name = f"herald-timer-{task.id}"
            timer.daemon = True
            self._pending[task.id] = (task, timer)
            timer.start()
        return task

    def cancel(self, task_id: int) -> bool:
        """Cancel a pending task. Returns False if it already fired or was cancelled."""
        with self._lock:
            entry = self._pending.pop(task_id, None)
        if entry is None:
            return False
        entry[1].cancel()
        return True

    def pending(self) -> list[ScheduledTask]:
        """The tasks that have not fired yet, soonest first."""
        with self._lock:
            tasks = [task for task, _ in self._pending.values()]
        return sorted(tasks, key=lambda task: (task.due, task.id))

    def shutdown(self) -> int:
        """Cancel every pending task and refuse new ones. Returns how many were cancelled.

        Safe to call more than once; later calls cancel nothing and return 0.
        """
        with self._lock:
            self._closed = True
            entries = list(self._pending.values())
            self._pending.clear()
        for _, timer in entries:
            timer.cancel()
        return len(entries)

    def _fire(self, task_id: int, callback: Callable[[], None]) -> None:
        with self._lock:
            entry = self._pending.pop(task_id, None)
        if entry is None:  # cancelled just as the timer went off
            return
        try:
            callback()
        except Exception:
            logger.exception("Scheduled task %d (%r) failed", task_id, entry[0].label)
