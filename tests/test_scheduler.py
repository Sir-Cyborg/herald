"""The scheduler with real (tiny) timers. Waits are on events with a timeout, never on sleeps."""

import threading
import time

import pytest

from herald.tools import scheduler as scheduler_module
from herald.tools.scheduler import ScheduledTask, Scheduler

WAIT = 2.0  # generous upper bound for a timer that should fire within milliseconds
LONG = 60.0  # a delay that never elapses during a test: the timer is cancelled at the end


@pytest.fixture
def sched():
    scheduler = Scheduler()
    yield scheduler
    scheduler.shutdown()  # no timer outlives its test


class TestFiring:
    def test_the_callback_runs_once_and_the_task_leaves_pending(self, sched):
        fired, calls = threading.Event(), []

        def callback():
            calls.append(1)
            fired.set()

        task = sched.schedule(0.01, callback, label="tea")

        assert fired.wait(WAIT)
        assert calls == [1]
        assert sched.pending() == []  # already gone while its callback runs
        assert sched.cancel(task.id) is False

    def test_tasks_fire_in_due_order(self, sched):
        order, done = [], threading.Event()

        def record(name):
            def callback():
                order.append(name)
                if len(order) == 3:
                    done.set()

            return callback

        sched.schedule(0.15, record("last"))
        sched.schedule(0.05, record("first"))
        sched.schedule(0.10, record("middle"))

        assert done.wait(WAIT)
        assert order == ["first", "middle", "last"]

    def test_a_callback_may_schedule_another_task(self, sched):
        second = threading.Event()
        sched.schedule(0.01, lambda: sched.schedule(0.01, second.set))
        assert second.wait(WAIT)

    def test_a_failing_callback_is_logged_and_later_tasks_still_fire(self, sched, mocker):
        logged, later = threading.Event(), threading.Event()
        mocker.patch.object(
            scheduler_module.logger, "exception", side_effect=lambda *a, **k: logged.set()
        )

        def boom():
            raise RuntimeError("callback failed")

        sched.schedule(0.01, boom, label="broken")
        sched.schedule(0.05, later.set)

        assert logged.wait(WAIT)
        assert later.wait(WAIT)
        assert sched.pending() == []

    def test_timer_threads_are_daemons(self, sched):
        task = sched.schedule(LONG, lambda: None)
        (thread,) = [t for t in threading.enumerate() if t.name == f"herald-timer-{task.id}"]
        assert thread.daemon  # a forgotten scheduler must not block interpreter exit


class TestCancel:
    def test_a_cancelled_task_does_not_fire(self, sched):
        cancelled, later = [], threading.Event()
        task = sched.schedule(0.05, lambda: cancelled.append(1))
        sched.schedule(0.2, later.set)

        assert sched.cancel(task.id) is True
        assert later.wait(WAIT)  # the cancelled one was due well before this
        assert cancelled == []
        assert [t.id for t in sched.pending()] == []

    def test_cancel_reports_whether_the_task_was_pending(self, sched):
        task = sched.schedule(LONG, lambda: None)
        assert sched.cancel(task.id) is True
        assert sched.cancel(task.id) is False
        assert sched.cancel(12345) is False

    def test_cancel_only_affects_its_own_task(self, sched):
        keep = sched.schedule(LONG, lambda: None)
        drop = sched.schedule(LONG, lambda: None)
        sched.cancel(drop.id)
        assert sched.pending() == [keep]


class TestPending:
    def test_tasks_are_listed_soonest_first_with_their_details(self, sched):
        before = time.time()
        late = sched.schedule(LONG * 2, lambda: None, label="late")
        soon = sched.schedule(LONG, lambda: None, label="soon")

        assert sched.pending() == [soon, late]
        assert (soon.label, soon.delay) == ("soon", LONG)
        assert before + LONG <= soon.due <= time.time() + LONG
        assert 0 < soon.seconds_left() <= LONG

    def test_the_label_defaults_to_empty(self, sched):
        assert sched.schedule(LONG, lambda: None).label == ""

    def test_ids_increase_and_are_not_reused(self, sched):
        first = sched.schedule(LONG, lambda: None)
        sched.cancel(first.id)
        second = sched.schedule(LONG, lambda: None)
        third = sched.schedule(LONG, lambda: None)
        assert first.id < second.id < third.id

    def test_pending_is_a_copy(self, sched):
        sched.schedule(LONG, lambda: None)
        sched.pending().clear()
        assert len(sched.pending()) == 1

    def test_seconds_left_is_never_negative(self):
        overdue = ScheduledTask(id=1, label="", delay=1.0, due=time.time() - 10)
        assert overdue.seconds_left() == 0.0


class TestLimits:
    def test_no_more_than_max_tasks_can_be_pending(self):
        sched = Scheduler(max_tasks=2)
        try:
            sched.schedule(LONG, lambda: None)
            sched.schedule(LONG, lambda: None)
            with pytest.raises(RuntimeError, match=r"too many pending timers \(max 2\)"):
                sched.schedule(LONG, lambda: None)
        finally:
            sched.shutdown()

    def test_a_cancelled_task_frees_its_slot(self):
        sched = Scheduler(max_tasks=1)
        try:
            task = sched.schedule(LONG, lambda: None)
            sched.cancel(task.id)
            sched.schedule(LONG, lambda: None)
        finally:
            sched.shutdown()

    def test_a_fired_task_frees_its_slot(self):
        sched = Scheduler(max_tasks=1)
        fired = threading.Event()
        try:
            sched.schedule(0.01, fired.set)
            assert fired.wait(WAIT)
            sched.schedule(LONG, lambda: None)
        finally:
            sched.shutdown()

    def test_the_limit_holds_when_many_threads_schedule_at_once(self):
        sched = Scheduler(max_tasks=10)
        results, start = [], threading.Barrier(30)

        def worker():
            start.wait(WAIT)
            try:
                sched.schedule(LONG, lambda: None)
                results.append(True)
            except RuntimeError:
                results.append(False)

        try:
            threads = [threading.Thread(target=worker) for _ in range(30)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(WAIT)
            assert results.count(True) == 10 and results.count(False) == 20
            assert len(sched.pending()) == 10
        finally:
            sched.shutdown()

    def test_max_tasks_must_be_positive(self):
        with pytest.raises(ValueError, match="max_tasks"):
            Scheduler(max_tasks=0)

    @pytest.mark.parametrize(
        "delay",
        [0, -1, 0.0, float("nan"), float("inf"), float("-inf"), threading.TIMEOUT_MAX * 2],
    )
    def test_invalid_delays_are_rejected(self, sched, delay):
        with pytest.raises(ValueError, match="delay"):
            sched.schedule(delay, lambda: None)
        assert sched.pending() == []


class TestShutdown:
    def test_cancels_everything_and_returns_how_many(self, sched):
        victim, ran = Scheduler(), []
        victim.schedule(0.05, lambda: ran.append("soon"))
        victim.schedule(LONG, lambda: ran.append("late"))

        assert victim.shutdown() == 2
        assert victim.pending() == []

        past_the_first = threading.Event()
        sched.schedule(0.2, past_the_first.set)  # due well after the cancelled 0.05 s task
        assert past_the_first.wait(WAIT)
        assert ran == []

    def test_is_idempotent(self):
        sched = Scheduler()
        sched.schedule(LONG, lambda: None)
        assert sched.shutdown() == 1
        assert sched.shutdown() == 0

    def test_nothing_can_be_scheduled_afterwards(self):
        sched = Scheduler()
        sched.shutdown()
        with pytest.raises(RuntimeError, match="scheduler is shut down"):
            sched.schedule(1, lambda: None)
        assert sched.pending() == []

    def test_shutting_down_an_empty_scheduler(self):
        assert Scheduler().shutdown() == 0
