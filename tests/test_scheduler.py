# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.scheduler — task scheduling and queue module."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from tritium_lib.scheduler import (
    Task,
    TaskQueue,
    TaskResult,
    TaskStatus,
    Scheduler,
    ScheduleType,
)
from tritium_lib.scheduler.builtin import (
    prune_stale_targets,
    generate_daily_report,
    check_sensor_health,
    rotate_logs,
    _do_prune_stale_targets,
    _do_generate_daily_report,
    _do_check_sensor_health,
    _do_rotate_logs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop() -> str:
    return "ok"


def _add(a: int, b: int) -> int:
    return a + b


def _fail() -> None:
    raise RuntimeError("intentional failure")


def _slow(duration: float = 0.2) -> str:
    time.sleep(duration)
    return "done"


_counter_lock = threading.Lock()
_counter = 0


def _increment() -> int:
    global _counter
    with _counter_lock:
        _counter += 1
        return _counter


def _reset_counter() -> None:
    global _counter
    with _counter_lock:
        _counter = 0


# Mock tracker for built-in prune tests
@dataclass
class _MockTarget:
    target_id: str
    source: str = "ble"
    alliance: str = "neutral"
    last_seen: float = field(default_factory=time.monotonic)


class _MockTracker:
    def __init__(self, targets: list[_MockTarget] | None = None) -> None:
        self._targets = {t.target_id: t for t in (targets or [])}

    def all_targets(self) -> list[_MockTarget]:
        return list(self._targets.values())

    def remove_target(self, tid: str) -> None:
        self._targets.pop(tid, None)


# Mock event store
class _MockEventStore:
    def __init__(self, count: int = 42) -> None:
        self._count = count
        self._deleted = 0

    def count(self) -> int:
        return self._count

    def delete_before(self, cutoff: float) -> int:
        self._deleted = 5
        return 5


# Mock health monitor
class _MockHealthMonitor:
    def __init__(self, overall: str = "up", components: list | None = None) -> None:
        self._overall = overall
        self._components = components or []

    def check_all(self) -> Any:
        return _MockSystemStatus(self._overall, self._components)


@dataclass
class _MockComponentHealth:
    name: str
    status: str

    @property
    def value(self) -> str:
        return self.status


@dataclass
class _MockSystemStatus:
    overall: str
    components: list


# ===========================================================================
# Task tests
# ===========================================================================

class TestTask:
    def test_create_interval_task(self) -> None:
        t = Task(name="test", func=_noop, schedule_type="interval", interval_seconds=10)
        assert t.name == "test"
        assert t.schedule_type == ScheduleType.INTERVAL
        assert t.interval_seconds == 10
        assert t.status == TaskStatus.PENDING
        assert t.run_count == 0

    def test_create_cron_task(self) -> None:
        t = Task(name="cron", func=_noop, schedule_type="cron", cron_hour=3, cron_minute=30)
        assert t.schedule_type == ScheduleType.CRON
        assert t.cron_hour == 3
        assert t.cron_minute == 30

    def test_create_one_shot_task(self) -> None:
        t = Task(name="shot", func=_noop, schedule_type="one_shot", delay_seconds=5.0)
        assert t.schedule_type == ScheduleType.ONE_SHOT
        assert t.delay_seconds == 5.0

    def test_execute_success(self) -> None:
        t = Task(name="add", func=_add, args=(3, 4))
        result = t.execute()
        assert result.success is True
        assert result.result == 7
        assert result.error == ""
        assert result.duration >= 0
        assert t.status == TaskStatus.COMPLETED
        assert t.run_count == 1
        assert t.fail_count == 0
        assert t.last_result is result

    def test_execute_failure(self) -> None:
        t = Task(name="bad", func=_fail)
        result = t.execute()
        assert result.success is False
        assert "RuntimeError" in result.error
        assert t.status == TaskStatus.FAILED
        assert t.run_count == 1
        assert t.fail_count == 1

    def test_execute_with_retries(self) -> None:
        call_count = 0
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "finally"

        t = Task(name="retry", func=flaky, max_retries=3)
        result = t.execute()
        assert result.success is True
        assert result.result == "finally"
        assert call_count == 3

    def test_execute_exhausts_retries(self) -> None:
        t = Task(name="doomed", func=_fail, max_retries=2)
        result = t.execute()
        assert result.success is False
        assert t.fail_count == 1

    def test_task_with_kwargs(self) -> None:
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        t = Task(name="greet", func=greet, kwargs={"name": "World", "greeting": "Hi"})
        result = t.execute()
        assert result.success is True
        assert result.result == "Hi, World!"

    def test_to_dict(self) -> None:
        t = Task(name="test", func=_noop, description="A test task")
        d = t.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "A test task"
        assert d["schedule_type"] == "interval"
        assert d["status"] == "pending"
        assert d["last_result"] is None

    def test_to_dict_after_execution(self) -> None:
        t = Task(name="test", func=_noop)
        t.execute()
        d = t.to_dict()
        assert d["status"] == "completed"
        assert d["run_count"] == 1
        assert d["last_result"] is not None
        assert d["last_result"]["success"] is True

    def test_disabled_task_not_auto_skipped(self) -> None:
        """Task.execute() always runs — enabled is for the Scheduler to check."""
        t = Task(name="disabled", func=_noop, enabled=False)
        result = t.execute()
        assert result.success is True


# ===========================================================================
# TaskResult tests
# ===========================================================================

class TestTaskResult:
    def test_duration(self) -> None:
        r = TaskResult(task_name="t", success=True, started_at=100.0, finished_at=102.5)
        assert r.duration == pytest.approx(2.5)

    def test_to_dict(self) -> None:
        r = TaskResult(
            task_name="t", success=False, started_at=1.0, finished_at=2.0,
            error="boom",
        )
        d = r.to_dict()
        assert d["task_name"] == "t"
        assert d["success"] is False
        assert d["error"] == "boom"
        assert d["duration"] == pytest.approx(1.0)


# ===========================================================================
# Scheduler tests
# ===========================================================================

class TestScheduler:
    def test_add_and_get_task(self) -> None:
        s = Scheduler()
        t = Task(name="a", func=_noop)
        s.add_task(t)
        assert s.get_task("a") is t
        assert s.task_count() == 1

    def test_add_duplicate_raises(self) -> None:
        s = Scheduler()
        s.add_task(Task(name="dup", func=_noop))
        with pytest.raises(ValueError, match="already registered"):
            s.add_task(Task(name="dup", func=_noop))

    def test_remove_task(self) -> None:
        s = Scheduler()
        s.add_task(Task(name="rm", func=_noop))
        removed = s.remove_task("rm")
        assert removed is not None
        assert removed.name == "rm"
        assert removed.status == TaskStatus.CANCELLED
        assert s.task_count() == 0

    def test_remove_nonexistent(self) -> None:
        s = Scheduler()
        assert s.remove_task("ghost") is None

    def test_enable_disable(self) -> None:
        s = Scheduler()
        t = Task(name="toggle", func=_noop, enabled=True)
        s.add_task(t)
        assert s.disable_task("toggle") is True
        assert t.enabled is False
        assert s.enable_task("toggle") is True
        assert t.enabled is True
        assert s.enable_task("missing") is False
        assert s.disable_task("missing") is False

    def test_run_now(self) -> None:
        s = Scheduler()
        s.add_task(Task(name="now", func=_add, args=(10, 20)))
        result = s.run_now("now")
        assert result is not None
        assert result.success is True
        assert result.result == 30

    def test_run_now_nonexistent(self) -> None:
        s = Scheduler()
        assert s.run_now("nope") is None

    def test_start_stop(self) -> None:
        s = Scheduler()
        assert s.running is False
        s.start()
        assert s.running is True
        s.stop()
        assert s.running is False

    def test_start_idempotent(self) -> None:
        s = Scheduler()
        s.start()
        s.start()  # Should not raise
        assert s.running is True
        s.stop()

    def test_stop_idempotent(self) -> None:
        s = Scheduler()
        s.stop()  # Not running — should not raise
        assert s.running is False

    def test_interval_task_executes(self) -> None:
        _reset_counter()
        s = Scheduler()
        s.add_task(Task(name="inc", func=_increment, interval_seconds=0.05))
        s.start()
        time.sleep(0.25)
        s.stop()
        assert _counter >= 2, f"Expected >= 2 executions, got {_counter}"

    def test_one_shot_task_executes_once(self) -> None:
        _reset_counter()
        s = Scheduler()
        s.add_task(Task(
            name="once", func=_increment,
            schedule_type="one_shot", delay_seconds=0.05,
        ))
        s.start()
        time.sleep(0.3)
        s.stop()
        assert _counter == 1, f"Expected exactly 1 execution, got {_counter}"

    def test_disabled_task_not_scheduled(self) -> None:
        _reset_counter()
        s = Scheduler()
        s.add_task(Task(name="off", func=_increment, interval_seconds=0.05, enabled=False))
        s.start()
        time.sleep(0.2)
        s.stop()
        assert _counter == 0

    def test_results_recorded(self) -> None:
        s = Scheduler()
        s.add_task(Task(name="rec", func=_noop, interval_seconds=0.05))
        s.start()
        time.sleep(0.2)
        s.stop()
        results = s.results
        assert len(results) >= 1
        assert all(r.task_name == "rec" for r in results)
        assert all(r.success is True for r in results)

    def test_to_dict(self) -> None:
        s = Scheduler()
        s.add_task(Task(name="a", func=_noop, description="Task A"))
        d = s.to_dict()
        assert d["running"] is False
        assert d["task_count"] == 1
        assert "a" in d["tasks"]
        assert d["tasks"]["a"]["description"] == "Task A"

    def test_add_task_while_running(self) -> None:
        _reset_counter()
        s = Scheduler()
        s.start()
        s.add_task(Task(name="late", func=_increment, interval_seconds=0.05))
        time.sleep(0.2)
        s.stop()
        assert _counter >= 1

    def test_tasks_property_returns_copy(self) -> None:
        s = Scheduler()
        s.add_task(Task(name="x", func=_noop))
        tasks = s.tasks
        tasks["y"] = Task(name="y", func=_noop)  # Should not affect scheduler
        assert s.task_count() == 1


# ===========================================================================
# TaskQueue tests
# ===========================================================================

class TestTaskQueue:
    def test_submit_and_execute(self) -> None:
        q = TaskQueue(num_workers=1)
        q.start()
        assert q.submit(Task(name="t1", func=_noop, schedule_type="one_shot"))
        q.drain(timeout=2.0)
        q.stop()
        assert q.completed_count >= 1
        results = q.results
        assert any(r.task_name == "t1" and r.success for r in results)

    def test_submit_func_convenience(self) -> None:
        q = TaskQueue(num_workers=1)
        q.start()
        assert q.submit_func("add", _add, args=(5, 6))
        q.drain(timeout=2.0)
        q.stop()
        results = q.results
        assert any(r.result == 11 for r in results)

    def test_submit_when_stopped(self) -> None:
        q = TaskQueue()
        assert q.submit(Task(name="nope", func=_noop)) is False

    def test_multiple_workers(self) -> None:
        results_lock = threading.Lock()
        results: list[str] = []

        def record(name: str) -> str:
            time.sleep(0.05)
            with results_lock:
                results.append(name)
            return name

        q = TaskQueue(num_workers=3)
        q.start()
        for i in range(6):
            q.submit_func(f"task-{i}", record, kwargs={"name": f"t{i}"})
        q.drain(timeout=3.0)
        q.stop()
        assert len(results) == 6

    def test_fifo_order_single_worker(self) -> None:
        order: list[int] = []
        order_lock = threading.Lock()

        def record_order(n: int) -> None:
            with order_lock:
                order.append(n)

        q = TaskQueue(num_workers=1)
        q.start()
        for i in range(5):
            q.submit_func(f"o-{i}", record_order, kwargs={"n": i})
        q.drain(timeout=2.0)
        q.stop()
        assert order == [0, 1, 2, 3, 4]

    def test_start_stop_idempotent(self) -> None:
        q = TaskQueue()
        q.start()
        q.start()  # Should not raise
        q.stop()
        q.stop()  # Should not raise

    def test_running_property(self) -> None:
        q = TaskQueue()
        assert q.running is False
        q.start()
        assert q.running is True
        q.stop()
        assert q.running is False

    def test_to_dict(self) -> None:
        q = TaskQueue(num_workers=2)
        d = q.to_dict()
        assert d["running"] is False
        assert d["num_workers"] == 2
        assert d["pending"] == 0
        assert d["completed"] == 0

    def test_failed_task_in_queue(self) -> None:
        q = TaskQueue(num_workers=1)
        q.start()
        q.submit(Task(name="bad", func=_fail, schedule_type="one_shot"))
        q.drain(timeout=2.0)
        q.stop()
        results = q.results
        assert any(r.task_name == "bad" and not r.success for r in results)


# ===========================================================================
# Built-in task tests
# ===========================================================================

class TestBuiltinPruneStaleTargets:
    def test_factory_creates_interval_task(self) -> None:
        tracker = _MockTracker()
        task = prune_stale_targets(tracker, interval=15.0, stale_seconds=60.0)
        assert task.name == "prune_stale_targets"
        assert task.schedule_type == ScheduleType.INTERVAL
        assert task.interval_seconds == 15.0
        assert "60" in task.description

    def test_prunes_stale_targets(self) -> None:
        old_target = _MockTarget(target_id="ble_old", last_seen=time.monotonic() - 200)
        fresh_target = _MockTarget(target_id="ble_new", last_seen=time.monotonic())
        tracker = _MockTracker([old_target, fresh_target])

        result = _do_prune_stale_targets(tracker, stale_seconds=100)
        assert result["pruned_count"] == 1
        assert "ble_old" in result["pruned_ids"]
        # Fresh target should remain
        remaining = tracker.all_targets()
        assert len(remaining) == 1
        assert remaining[0].target_id == "ble_new"

    def test_no_prune_when_all_fresh(self) -> None:
        targets = [
            _MockTarget(target_id="a", last_seen=time.monotonic()),
            _MockTarget(target_id="b", last_seen=time.monotonic()),
        ]
        tracker = _MockTracker(targets)
        result = _do_prune_stale_targets(tracker, stale_seconds=100)
        assert result["pruned_count"] == 0
        assert len(tracker.all_targets()) == 2


class TestBuiltinGenerateDailyReport:
    def test_factory_creates_cron_task(self) -> None:
        task = generate_daily_report(
            _MockTracker(), _MockEventStore(), hour=6, minute=30,
        )
        assert task.name == "generate_daily_report"
        assert task.schedule_type == ScheduleType.CRON
        assert task.cron_hour == 6
        assert task.cron_minute == 30

    def test_generates_report(self) -> None:
        targets = [
            _MockTarget(target_id="a", source="ble", alliance="friendly"),
            _MockTarget(target_id="b", source="yolo", alliance="hostile"),
            _MockTarget(target_id="c", source="ble", alliance="friendly"),
        ]
        tracker = _MockTracker(targets)
        store = _MockEventStore(count=100)

        result = _do_generate_daily_report(tracker, store)
        assert result["target_count"] == 3
        assert result["targets_by_source"]["ble"] == 2
        assert result["targets_by_source"]["yolo"] == 1
        assert result["targets_by_alliance"]["friendly"] == 2
        assert result["event_count_24h"] == 100
        assert "generated_at" in result


class TestBuiltinCheckSensorHealth:
    def test_factory_creates_interval_task(self) -> None:
        task = check_sensor_health(_MockHealthMonitor(), interval=45.0)
        assert task.name == "check_sensor_health"
        assert task.schedule_type == ScheduleType.INTERVAL
        assert task.interval_seconds == 45.0

    def test_checks_health(self) -> None:
        comps = [
            _MockComponentHealth(name="tracker", status="up"),
            _MockComponentHealth(name="fusion", status="degraded"),
        ]
        monitor = _MockHealthMonitor(overall="degraded", components=comps)
        result = _do_check_sensor_health(monitor)
        assert result["overall"] == "degraded"
        assert "tracker" in result["components"]
        assert "checked_at" in result

    def test_handles_no_check_all(self) -> None:
        """If monitor has no check_all, returns unknown."""
        result = _do_check_sensor_health(object())
        assert result["overall"] == "unknown"


class TestBuiltinRotateLogs:
    def test_factory_creates_cron_task(self) -> None:
        task = rotate_logs(_MockEventStore(), max_age_hours=48, hour=5, minute=15)
        assert task.name == "rotate_logs"
        assert task.schedule_type == ScheduleType.CRON
        assert task.cron_hour == 5
        assert task.cron_minute == 15

    def test_rotates_old_events(self) -> None:
        store = _MockEventStore()
        result = _do_rotate_logs(store, max_age_hours=24)
        assert result["rotated_count"] == 5
        assert result["max_age_hours"] == 24
        assert "cutoff_timestamp" in result


# ===========================================================================
# Thread safety tests
# ===========================================================================

class TestThreadSafety:
    def test_concurrent_submit_to_queue(self) -> None:
        q = TaskQueue(num_workers=3)
        q.start()
        barrier = threading.Barrier(4)
        submitted = []
        submit_lock = threading.Lock()

        def submit_batch(start: int) -> None:
            barrier.wait()
            for i in range(10):
                ok = q.submit_func(f"t-{start}-{i}", _noop)
                with submit_lock:
                    submitted.append(ok)

        threads = [threading.Thread(target=submit_batch, args=(n * 10,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        q.drain(timeout=5.0)
        q.stop()

        assert all(submitted)
        assert q.completed_count == 40

    def test_concurrent_run_now(self) -> None:
        s = Scheduler()
        results_list: list[TaskResult | None] = []
        results_lock = threading.Lock()

        s.add_task(Task(name="shared", func=_noop))

        def run_it() -> None:
            r = s.run_now("shared")
            with results_lock:
                results_list.append(r)

        threads = [threading.Thread(target=run_it) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results_list) == 10
        assert all(r is not None and r.success for r in results_list)


# ===========================================================================
# Integration-like test
# ===========================================================================

class TestIntegration:
    def test_scheduler_with_builtin_tasks(self) -> None:
        """Smoke test: register all built-in tasks, run them, check results."""
        tracker = _MockTracker([
            _MockTarget(target_id="old", last_seen=time.monotonic() - 500),
            _MockTarget(target_id="new", last_seen=time.monotonic()),
        ])
        store = _MockEventStore(count=10)
        monitor = _MockHealthMonitor(overall="up")

        sched = Scheduler()
        sched.add_task(prune_stale_targets(tracker, interval=60, stale_seconds=100))
        sched.add_task(generate_daily_report(tracker, store, hour=12))
        sched.add_task(check_sensor_health(monitor, interval=60))
        sched.add_task(rotate_logs(store, max_age_hours=24, hour=4))

        assert sched.task_count() == 4

        # Run each task manually
        for name in ["prune_stale_targets", "generate_daily_report",
                      "check_sensor_health", "rotate_logs"]:
            result = sched.run_now(name)
            assert result is not None
            assert result.success is True

        assert sched.results == sched.results  # Stable copy
        assert len(sched.results) == 4

    def test_queue_drains_all(self) -> None:
        """Submit many tasks to a queue and verify all complete."""
        q = TaskQueue(num_workers=2)
        q.start()
        for i in range(20):
            q.submit_func(f"job-{i}", _add, args=(i, i))
        q.drain(timeout=5.0)
        q.stop()
        assert q.completed_count == 20
        # Verify results
        for r in q.results:
            assert r.success is True
