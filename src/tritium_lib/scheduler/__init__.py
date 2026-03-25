# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.scheduler — task scheduling and queue for recurring operations.

Provides a pure-Python, thread-safe scheduler for recurring operations such as
data collection, report generation, sensor health checks, and log rotation.
No external dependencies (no APScheduler) — built on threading.Timer and
standard library primitives.

Schedule types:
  - **interval**: run every N seconds
  - **cron**: run at specific hour/minute each day
  - **one-shot**: run once after a delay

Components:
  - Task          — a unit of work (function + schedule + metadata)
  - Scheduler     — manages recurring tasks with cron-like scheduling
  - TaskQueue     — FIFO queue for one-time tasks with worker threads
  - TaskResult    — result of a completed task execution

Built-in tasks (in ``tritium_lib.scheduler.builtin``):
  - prune_stale_targets  — remove old targets from tracker
  - generate_daily_report — daily situation report
  - check_sensor_health  — periodic sensor heartbeat check
  - rotate_logs          — archive old event data

Usage
-----
    from tritium_lib.scheduler import Scheduler, Task, TaskQueue

    # Create a scheduler
    sched = Scheduler()

    # Add an interval task — runs every 30 seconds
    sched.add_task(Task(
        name="heartbeat",
        func=my_heartbeat_fn,
        schedule_type="interval",
        interval_seconds=30,
    ))

    # Add a cron task — runs daily at 03:00
    sched.add_task(Task(
        name="daily_cleanup",
        func=cleanup,
        schedule_type="cron",
        cron_hour=3,
        cron_minute=0,
    ))

    # Start all tasks
    sched.start()

    # One-time task queue
    q = TaskQueue(num_workers=2)
    q.start()
    q.submit(Task(name="one-off", func=do_something, schedule_type="one_shot"))
    q.stop()

    # Shut down scheduler
    sched.stop()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class ScheduleType(str, Enum):
    """Supported schedule types."""
    INTERVAL = "interval"
    CRON = "cron"
    ONE_SHOT = "one_shot"


class TaskStatus(str, Enum):
    """Lifecycle status of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_name: str
    success: bool
    started_at: float
    finished_at: float
    result: Any = None
    error: str = ""

    @property
    def duration(self) -> float:
        """Execution duration in seconds."""
        return self.finished_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "task_name": self.task_name,
            "success": self.success,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
        }


@dataclass
class Task:
    """A unit of work with scheduling metadata.

    Parameters
    ----------
    name:
        Human-readable task name (unique within a scheduler).
    func:
        Callable to execute. Receives ``*args`` and ``**kwargs``.
    schedule_type:
        One of "interval", "cron", or "one_shot".
    interval_seconds:
        For interval tasks — how often to run (seconds).
    cron_hour:
        For cron tasks — hour of day (0-23). None means every hour.
    cron_minute:
        For cron tasks — minute of hour (0-59). Defaults to 0.
    delay_seconds:
        For one-shot tasks — delay before execution (seconds).
    args:
        Positional arguments passed to ``func``.
    kwargs:
        Keyword arguments passed to ``func``.
    enabled:
        If False, the scheduler will skip this task.
    max_retries:
        Number of times to retry on failure (0 = no retries).
    description:
        Human-readable description of what the task does.
    """
    name: str
    func: Callable[..., Any]
    schedule_type: str | ScheduleType = ScheduleType.INTERVAL
    interval_seconds: float = 60.0
    cron_hour: int | None = None
    cron_minute: int = 0
    delay_seconds: float = 0.0
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    max_retries: int = 0
    description: str = ""

    # Runtime state (managed by scheduler)
    status: TaskStatus = TaskStatus.PENDING
    run_count: int = 0
    fail_count: int = 0
    last_run: float = 0.0
    last_result: TaskResult | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schedule_type, str):
            self.schedule_type = ScheduleType(self.schedule_type)

    def execute(self) -> TaskResult:
        """Run the task function and return a TaskResult."""
        self.status = TaskStatus.RUNNING
        started = time.time()
        retries = 0
        last_error = ""

        while retries <= self.max_retries:
            try:
                result = self.func(*self.args, **self.kwargs)
                finished = time.time()
                self.status = TaskStatus.COMPLETED
                self.run_count += 1
                self.last_run = finished
                tr = TaskResult(
                    task_name=self.name,
                    success=True,
                    started_at=started,
                    finished_at=finished,
                    result=result,
                )
                self.last_result = tr
                return tr
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                retries += 1
                if retries <= self.max_retries:
                    logger.warning(
                        "Task %r failed (attempt %d/%d): %s",
                        self.name, retries, self.max_retries + 1, last_error,
                    )

        finished = time.time()
        self.status = TaskStatus.FAILED
        self.fail_count += 1
        self.run_count += 1
        self.last_run = finished
        tr = TaskResult(
            task_name=self.name,
            success=False,
            started_at=started,
            finished_at=finished,
            error=last_error,
        )
        self.last_result = tr
        return tr

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "name": self.name,
            "schedule_type": self.schedule_type.value,
            "interval_seconds": self.interval_seconds,
            "cron_hour": self.cron_hour,
            "cron_minute": self.cron_minute,
            "delay_seconds": self.delay_seconds,
            "enabled": self.enabled,
            "max_retries": self.max_retries,
            "description": self.description,
            "status": self.status.value,
            "run_count": self.run_count,
            "fail_count": self.fail_count,
            "last_run": self.last_run,
            "last_result": self.last_result.to_dict() if self.last_result else None,
        }


# ---------------------------------------------------------------------------
# Scheduler — manages recurring tasks
# ---------------------------------------------------------------------------

class Scheduler:
    """Thread-safe scheduler for recurring tasks.

    Manages interval, cron, and one-shot tasks using daemon threads.
    All tasks run in background threads and are cleaned up on ``stop()``.

    The scheduler is reusable: after ``stop()`` you can ``start()`` again.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._running = False
        self._cron_thread: threading.Thread | None = None
        self._cron_stop = threading.Event()
        self._results: list[TaskResult] = []
        self._results_lock = threading.Lock()

    @property
    def running(self) -> bool:
        """True if the scheduler is active."""
        return self._running

    @property
    def tasks(self) -> dict[str, Task]:
        """Copy of registered tasks keyed by name."""
        with self._lock:
            return dict(self._tasks)

    @property
    def results(self) -> list[TaskResult]:
        """Copy of completed task results (most recent first)."""
        with self._results_lock:
            return list(self._results)

    def add_task(self, task: Task) -> None:
        """Register a task. If the scheduler is running, the task starts immediately."""
        with self._lock:
            if task.name in self._tasks:
                raise ValueError(f"Task {task.name!r} already registered")
            self._tasks[task.name] = task
            if self._running and task.enabled:
                self._start_task(task)

    def remove_task(self, name: str) -> Task | None:
        """Remove a task by name. Returns the removed task, or None."""
        with self._lock:
            task = self._tasks.pop(name, None)
            if task:
                self._cancel_timer(name)
                task.status = TaskStatus.CANCELLED
            return task

    def get_task(self, name: str) -> Task | None:
        """Look up a task by name."""
        with self._lock:
            return self._tasks.get(name)

    def enable_task(self, name: str) -> bool:
        """Enable a disabled task. Returns True if found and enabled."""
        with self._lock:
            task = self._tasks.get(name)
            if not task:
                return False
            task.enabled = True
            if self._running:
                self._start_task(task)
            return True

    def disable_task(self, name: str) -> bool:
        """Disable a task (stops scheduling). Returns True if found."""
        with self._lock:
            task = self._tasks.get(name)
            if not task:
                return False
            task.enabled = False
            self._cancel_timer(name)
            return True

    def start(self) -> None:
        """Start the scheduler — begins executing all enabled tasks."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._cron_stop.clear()

            # Start interval and one-shot tasks
            for task in self._tasks.values():
                if task.enabled:
                    self._start_task(task)

            # Start cron checker thread
            has_cron = any(
                t.schedule_type == ScheduleType.CRON and t.enabled
                for t in self._tasks.values()
            )
            if has_cron:
                self._start_cron_thread()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the scheduler — cancels all pending timers and waits for completion."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._cron_stop.set()

            # Cancel all timers
            for name in list(self._timers):
                self._cancel_timer(name)

        # Wait for cron thread
        if self._cron_thread and self._cron_thread.is_alive():
            self._cron_thread.join(timeout=timeout)
        self._cron_thread = None

    def run_now(self, name: str) -> TaskResult | None:
        """Execute a task immediately (regardless of schedule). Returns result."""
        with self._lock:
            task = self._tasks.get(name)
        if not task:
            return None
        result = task.execute()
        self._record_result(result)
        return result

    # -- Internal helpers --------------------------------------------------

    def _record_result(self, result: TaskResult) -> None:
        """Store a task result (thread-safe, bounded to 1000 entries)."""
        with self._results_lock:
            self._results.insert(0, result)
            if len(self._results) > 1000:
                self._results = self._results[:1000]

    def _start_task(self, task: Task) -> None:
        """Start scheduling a task (must hold self._lock)."""
        if task.schedule_type == ScheduleType.INTERVAL:
            self._schedule_interval(task)
        elif task.schedule_type == ScheduleType.ONE_SHOT:
            self._schedule_one_shot(task)
        elif task.schedule_type == ScheduleType.CRON:
            # Cron tasks are handled by the cron checker thread
            if not self._cron_thread or not self._cron_thread.is_alive():
                self._start_cron_thread()

    def _schedule_interval(self, task: Task) -> None:
        """Schedule an interval task to fire after interval_seconds."""
        def _run() -> None:
            if not self._running or not task.enabled:
                return
            result = task.execute()
            self._record_result(result)
            logger.debug("Task %r completed: success=%s", task.name, result.success)
            # Reschedule
            with self._lock:
                if self._running and task.enabled and task.name in self._tasks:
                    self._schedule_interval(task)

        timer = threading.Timer(task.interval_seconds, _run)
        timer.daemon = True
        self._cancel_timer(task.name)
        self._timers[task.name] = timer
        timer.start()

    def _schedule_one_shot(self, task: Task) -> None:
        """Schedule a one-shot task to fire after delay_seconds."""
        def _run() -> None:
            if not self._running or not task.enabled:
                return
            result = task.execute()
            self._record_result(result)
            logger.debug("One-shot task %r completed: success=%s", task.name, result.success)
            with self._lock:
                self._timers.pop(task.name, None)

        timer = threading.Timer(task.delay_seconds, _run)
        timer.daemon = True
        self._cancel_timer(task.name)
        self._timers[task.name] = timer
        timer.start()

    def _start_cron_thread(self) -> None:
        """Start the background thread that checks for cron task execution."""
        self._cron_thread = threading.Thread(
            target=self._cron_loop, name="scheduler-cron", daemon=True,
        )
        self._cron_thread.start()

    def _cron_loop(self) -> None:
        """Background loop that checks every 30 seconds for cron tasks to execute."""
        last_runs: dict[str, str] = {}  # task_name -> "YYYY-MM-DD HH:MM"
        while not self._cron_stop.is_set():
            now = time.localtime()
            now_key = time.strftime("%Y-%m-%d %H:%M", now)

            with self._lock:
                cron_tasks = [
                    t for t in self._tasks.values()
                    if t.schedule_type == ScheduleType.CRON and t.enabled
                ]

            for task in cron_tasks:
                # Check if this minute matches the cron spec
                hour_match = task.cron_hour is None or task.cron_hour == now.tm_hour
                minute_match = task.cron_minute == now.tm_min

                if hour_match and minute_match:
                    # Only run once per matching minute
                    if last_runs.get(task.name) != now_key:
                        last_runs[task.name] = now_key
                        result = task.execute()
                        self._record_result(result)
                        logger.debug(
                            "Cron task %r executed at %s: success=%s",
                            task.name, now_key, result.success,
                        )

            # Sleep 30 seconds or until stopped
            self._cron_stop.wait(30.0)

    def _cancel_timer(self, name: str) -> None:
        """Cancel and remove a timer by task name (must hold self._lock)."""
        timer = self._timers.pop(name, None)
        if timer is not None:
            timer.cancel()

    def task_count(self) -> int:
        """Number of registered tasks."""
        with self._lock:
            return len(self._tasks)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable scheduler status."""
        with self._lock:
            tasks = {name: t.to_dict() for name, t in self._tasks.items()}
        return {
            "running": self._running,
            "task_count": len(tasks),
            "tasks": tasks,
            "recent_results": [r.to_dict() for r in self.results[:20]],
        }


# ---------------------------------------------------------------------------
# TaskQueue — FIFO queue for one-time tasks with worker threads
# ---------------------------------------------------------------------------

class TaskQueue:
    """Thread-safe FIFO queue for one-time task execution.

    Runs tasks using a pool of worker threads. Tasks are submitted and
    executed in order (FIFO). Results are stored for later retrieval.

    Parameters
    ----------
    num_workers:
        Number of worker threads to spawn (default 2).
    max_size:
        Maximum queue size (0 = unlimited).
    """

    def __init__(self, num_workers: int = 2, max_size: int = 0) -> None:
        self._queue: queue.Queue[Task | None] = queue.Queue(maxsize=max_size)
        self._num_workers = max(1, num_workers)
        self._workers: list[threading.Thread] = []
        self._running = False
        self._results: list[TaskResult] = []
        self._results_lock = threading.Lock()
        self._pending_count = 0
        self._count_lock = threading.Lock()

    @property
    def running(self) -> bool:
        """True if the queue is processing tasks."""
        return self._running

    @property
    def results(self) -> list[TaskResult]:
        """Copy of completed results (most recent first)."""
        with self._results_lock:
            return list(self._results)

    @property
    def pending(self) -> int:
        """Approximate number of pending tasks."""
        return self._queue.qsize()

    @property
    def completed_count(self) -> int:
        """Number of completed task executions."""
        with self._results_lock:
            return len(self._results)

    def start(self) -> None:
        """Start worker threads."""
        if self._running:
            return
        self._running = True
        self._workers = []
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"task-queue-worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop all workers. Waits for currently running tasks to finish."""
        if not self._running:
            return
        self._running = False
        # Send poison pills
        for _ in self._workers:
            self._queue.put(None)
        # Wait for workers
        for w in self._workers:
            w.join(timeout=timeout)
        self._workers = []

    def submit(self, task: Task) -> bool:
        """Add a task to the queue. Returns False if the queue is full."""
        if not self._running:
            return False
        try:
            self._queue.put_nowait(task)
            with self._count_lock:
                self._pending_count += 1
            return True
        except queue.Full:
            return False

    def submit_func(
        self,
        name: str,
        func: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> bool:
        """Convenience: submit a function as a one-shot task."""
        task = Task(
            name=name,
            func=func,
            schedule_type=ScheduleType.ONE_SHOT,
            args=args,
            kwargs=kwargs or {},
        )
        return self.submit(task)

    def drain(self, timeout: float = 10.0) -> list[TaskResult]:
        """Wait until all submitted tasks are processed, up to timeout.

        Returns all results collected during the drain.
        """
        deadline = time.monotonic() + timeout
        initial_count = self.completed_count
        while time.monotonic() < deadline:
            if self._queue.empty():
                # Brief pause to let any in-progress task finish
                time.sleep(0.05)
                if self._queue.empty():
                    break
            time.sleep(0.02)
        with self._results_lock:
            return list(self._results[: len(self._results) - initial_count])

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                # Poison pill — exit
                break
            result = task.execute()
            with self._results_lock:
                self._results.insert(0, result)
                if len(self._results) > 1000:
                    self._results = self._results[:1000]
            with self._count_lock:
                self._pending_count = max(0, self._pending_count - 1)
            logger.debug(
                "Queue task %r completed: success=%s", task.name, result.success,
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable queue status."""
        return {
            "running": self._running,
            "num_workers": self._num_workers,
            "pending": self.pending,
            "completed": self.completed_count,
            "recent_results": [r.to_dict() for r in self.results[:20]],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ScheduleType",
    "TaskStatus",
    "TaskResult",
    "Task",
    "Scheduler",
    "TaskQueue",
]
