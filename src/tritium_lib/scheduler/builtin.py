# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Built-in scheduled tasks for common Tritium operations.

These factory functions create pre-configured Task objects that integrate
with the Tritium tracking, reporting, monitoring, and event subsystems.

Usage
-----
    from tritium_lib.scheduler import Scheduler
    from tritium_lib.scheduler.builtin import (
        prune_stale_targets,
        generate_daily_report,
        check_sensor_health,
        rotate_logs,
    )
    from tritium_lib.tracking import TargetTracker
    from tritium_lib.store.event_store import EventStore
    from tritium_lib.monitoring import HealthMonitor

    tracker = TargetTracker()
    event_store = EventStore(":memory:")
    monitor = HealthMonitor()

    sched = Scheduler()
    sched.add_task(prune_stale_targets(tracker, interval=30))
    sched.add_task(generate_daily_report(tracker, event_store, hour=3))
    sched.add_task(check_sensor_health(monitor, interval=60))
    sched.add_task(rotate_logs(event_store, max_age_hours=72, hour=4))
    sched.start()
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from tritium_lib.scheduler import Task, ScheduleType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# prune_stale_targets
# ---------------------------------------------------------------------------

def _do_prune_stale_targets(
    tracker: Any,
    stale_seconds: float,
) -> dict[str, Any]:
    """Remove targets older than stale_seconds from the tracker.

    Returns a summary dict with the count and IDs of pruned targets.
    """
    now = time.monotonic()
    pruned: list[str] = []

    # Use the tracker's public API to iterate and remove stale targets
    all_targets = tracker.all_targets() if hasattr(tracker, "all_targets") else []
    for target in all_targets:
        age = now - getattr(target, "last_seen", now)
        if age > stale_seconds:
            tid = getattr(target, "target_id", None) or getattr(target, "id", None)
            if tid:
                if hasattr(tracker, "remove_target"):
                    tracker.remove_target(tid)
                    pruned.append(tid)

    if pruned:
        logger.info("Pruned %d stale targets (age > %ds): %s",
                     len(pruned), int(stale_seconds), pruned[:10])
    return {"pruned_count": len(pruned), "pruned_ids": pruned}


def prune_stale_targets(
    tracker: Any,
    interval: float = 30.0,
    stale_seconds: float = 120.0,
) -> Task:
    """Create a task that prunes stale targets from a TargetTracker.

    Parameters
    ----------
    tracker:
        A TargetTracker instance (or any object with ``all_targets()`` and
        ``remove_target(tid)`` methods).
    interval:
        How often to run the prune check, in seconds.
    stale_seconds:
        Targets older than this are considered stale.
    """
    return Task(
        name="prune_stale_targets",
        func=_do_prune_stale_targets,
        schedule_type=ScheduleType.INTERVAL,
        interval_seconds=interval,
        kwargs={"tracker": tracker, "stale_seconds": stale_seconds},
        description=f"Remove targets not seen for {stale_seconds}s",
    )


# ---------------------------------------------------------------------------
# generate_daily_report
# ---------------------------------------------------------------------------

def _do_generate_daily_report(
    tracker: Any,
    event_store: Any,
) -> dict[str, Any]:
    """Generate a daily situation report from current tracking state.

    Returns a summary dict with report metadata.
    """
    # Gather target stats
    targets = tracker.all_targets() if hasattr(tracker, "all_targets") else []
    target_count = len(targets)

    # Count by alliance/source
    by_source: dict[str, int] = {}
    by_alliance: dict[str, int] = {}
    for t in targets:
        src = getattr(t, "source", "unknown")
        alliance = getattr(t, "alliance", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        by_alliance[alliance] = by_alliance.get(alliance, 0) + 1

    # Count events in last 24 hours
    event_count = 0
    if hasattr(event_store, "count"):
        event_count = event_store.count()
    elif hasattr(event_store, "get_events"):
        cutoff = time.time() - 86400
        events = event_store.get_events(since=cutoff)
        event_count = len(events) if events else 0

    report = {
        "generated_at": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "target_count": target_count,
        "targets_by_source": by_source,
        "targets_by_alliance": by_alliance,
        "event_count_24h": event_count,
    }

    logger.info(
        "Daily report: %d targets, %d events in 24h",
        target_count, event_count,
    )
    return report


def generate_daily_report(
    tracker: Any,
    event_store: Any,
    hour: int = 3,
    minute: int = 0,
) -> Task:
    """Create a cron task that generates a daily situation report.

    Parameters
    ----------
    tracker:
        A TargetTracker instance.
    event_store:
        An EventStore instance for querying recent events.
    hour:
        Hour of day to run (0-23). Default 3 (03:00).
    minute:
        Minute of hour to run (0-59). Default 0.
    """
    return Task(
        name="generate_daily_report",
        func=_do_generate_daily_report,
        schedule_type=ScheduleType.CRON,
        cron_hour=hour,
        cron_minute=minute,
        kwargs={"tracker": tracker, "event_store": event_store},
        description=f"Generate daily situation report at {hour:02d}:{minute:02d}",
    )


# ---------------------------------------------------------------------------
# check_sensor_health
# ---------------------------------------------------------------------------

def _do_check_sensor_health(
    monitor: Any,
) -> dict[str, Any]:
    """Check health of all registered sensor components.

    Returns a summary dict with component statuses.
    """
    if hasattr(monitor, "check_all"):
        status = monitor.check_all()
        components = {}
        if hasattr(status, "components"):
            for comp in status.components:
                name = getattr(comp, "name", "unknown")
                st = getattr(comp, "status", "unknown")
                components[name] = str(st.value) if hasattr(st, "value") else str(st)

        overall = getattr(status, "overall", "unknown")
        if hasattr(overall, "value"):
            overall = overall.value

        result = {
            "overall": str(overall),
            "components": components,
            "checked_at": time.time(),
        }

        if str(overall) != "up":
            logger.warning("Sensor health check: overall=%s, components=%s",
                           overall, components)
        return result

    return {"overall": "unknown", "components": {}, "checked_at": time.time()}


def check_sensor_health(
    monitor: Any,
    interval: float = 60.0,
) -> Task:
    """Create a task that periodically checks sensor/component health.

    Parameters
    ----------
    monitor:
        A HealthMonitor instance (or any object with ``check_all()``).
    interval:
        How often to check health, in seconds.
    """
    return Task(
        name="check_sensor_health",
        func=_do_check_sensor_health,
        schedule_type=ScheduleType.INTERVAL,
        interval_seconds=interval,
        kwargs={"monitor": monitor},
        description=f"Check sensor health every {interval}s",
    )


# ---------------------------------------------------------------------------
# rotate_logs
# ---------------------------------------------------------------------------

def _do_rotate_logs(
    event_store: Any,
    max_age_hours: float,
) -> dict[str, Any]:
    """Archive or delete event data older than max_age_hours.

    Returns a summary dict with the count of rotated events.
    """
    cutoff = time.time() - (max_age_hours * 3600)
    rotated = 0

    if hasattr(event_store, "delete_before"):
        rotated = event_store.delete_before(cutoff)
    elif hasattr(event_store, "prune"):
        rotated = event_store.prune(before=cutoff)
    elif hasattr(event_store, "get_events") and hasattr(event_store, "delete_event"):
        # Fallback: iterate and delete individually
        events = event_store.get_events()
        if events:
            for evt in events:
                ts = getattr(evt, "timestamp", None) or getattr(evt, "created_at", 0)
                if ts < cutoff:
                    eid = getattr(evt, "id", None) or getattr(evt, "event_id", None)
                    if eid and hasattr(event_store, "delete_event"):
                        event_store.delete_event(eid)
                        rotated += 1

    if rotated:
        logger.info("Log rotation: removed %d events older than %dh",
                     rotated, int(max_age_hours))

    return {
        "rotated_count": rotated,
        "max_age_hours": max_age_hours,
        "cutoff_timestamp": cutoff,
    }


def rotate_logs(
    event_store: Any,
    max_age_hours: float = 72.0,
    hour: int = 4,
    minute: int = 0,
) -> Task:
    """Create a cron task that rotates (deletes) old event log data.

    Parameters
    ----------
    event_store:
        An EventStore instance.
    max_age_hours:
        Delete events older than this many hours (default 72 = 3 days).
    hour:
        Hour of day to run (0-23). Default 4 (04:00).
    minute:
        Minute of hour to run (0-59). Default 0.
    """
    return Task(
        name="rotate_logs",
        func=_do_rotate_logs,
        schedule_type=ScheduleType.CRON,
        cron_hour=hour,
        cron_minute=minute,
        kwargs={"event_store": event_store, "max_age_hours": max_age_hours},
        description=f"Rotate event logs older than {max_age_hours}h at {hour:02d}:{minute:02d}",
    )


__all__ = [
    "prune_stale_targets",
    "generate_daily_report",
    "check_sensor_health",
    "rotate_logs",
]
