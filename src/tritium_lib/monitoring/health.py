# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""HealthMonitor — system health monitoring for the Tritium tracking pipeline.

Provides a pluggable health-check framework where each subsystem registers
a callable that returns its current ComponentHealth.  HealthMonitor
aggregates these into a SystemStatus with an overall "up", "degraded",
or "down" verdict.

Monitored components (typical):
  - TargetTracker: target count, update latency, stale targets
  - FusionEngine:  fusion rate, correlation confidence, queue depth
  - EventBus:      event throughput, subscriber count, queue overflow
  - Stores:        disk usage, query latency, record count

Usage::

    from tritium_lib.monitoring import HealthMonitor, ComponentHealth, MetricsCollector

    metrics = MetricsCollector()
    monitor = HealthMonitor(metrics=metrics)

    # Register health checks
    monitor.register("tracker", check_tracker)
    monitor.register("fusion", check_fusion)

    # Run all checks
    status = monitor.check_all()
    print(status.overall)        # "up" | "degraded" | "down"
    print(status.components)     # dict of ComponentHealth
    print(status.to_dict())      # JSON-serializable snapshot
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .metrics import MetricsCollector

logger = logging.getLogger("tritium.monitoring")


class ComponentStatus(str, Enum):
    """Health status for a single component."""

    UP = "up"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


# Type alias for a health check callable
HealthCheck = Callable[[], "ComponentHealth"]


@dataclass
class ComponentHealth:
    """Health report for a single subsystem component.

    Attributes
    ----------
    name:
        Human-readable component name (e.g., "tracker", "fusion").
    status:
        Current status — "up", "degraded", "down", or "unknown".
    message:
        Optional description of the current state.
    details:
        Arbitrary key-value pairs with component-specific metrics.
    last_checked:
        Monotonic timestamp of the last health check.
    error:
        Error message if the check failed.
    """

    name: str
    status: ComponentStatus = ComponentStatus.UNKNOWN
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    last_checked: float = field(default_factory=time.time)
    error: str = ""

    @property
    def is_healthy(self) -> bool:
        """True if status is UP."""
        return self.status == ComponentStatus.UP

    @property
    def is_degraded(self) -> bool:
        """True if status is DEGRADED."""
        return self.status == ComponentStatus.DEGRADED

    @property
    def is_down(self) -> bool:
        """True if status is DOWN."""
        return self.status == ComponentStatus.DOWN

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "name": self.name,
            "status": self.status.value if isinstance(self.status, ComponentStatus) else str(self.status),
            "message": self.message,
            "details": self.details,
            "last_checked": self.last_checked,
            "error": self.error,
        }


@dataclass
class SystemStatus:
    """Aggregated health status for the entire Tritium system.

    Attributes
    ----------
    overall:
        Overall system status.  Rules:
        - "up" if ALL components are up
        - "degraded" if any component is degraded but none are down
        - "down" if any component is down
    components:
        Per-component health reports.
    timestamp:
        When this status was generated.
    """

    overall: ComponentStatus = ComponentStatus.UNKNOWN
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def component_count(self) -> int:
        """Number of registered components."""
        return len(self.components)

    @property
    def healthy_count(self) -> int:
        """Number of components with status UP."""
        return sum(1 for c in self.components.values() if c.status == ComponentStatus.UP)

    @property
    def degraded_count(self) -> int:
        """Number of components with status DEGRADED."""
        return sum(1 for c in self.components.values() if c.status == ComponentStatus.DEGRADED)

    @property
    def down_count(self) -> int:
        """Number of components with status DOWN."""
        return sum(1 for c in self.components.values() if c.status == ComponentStatus.DOWN)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation of full system status."""
        return {
            "overall": self.overall.value if isinstance(self.overall, ComponentStatus) else str(self.overall),
            "timestamp": self.timestamp,
            "component_count": self.component_count,
            "healthy_count": self.healthy_count,
            "degraded_count": self.degraded_count,
            "down_count": self.down_count,
            "components": {
                name: ch.to_dict() for name, ch in self.components.items()
            },
        }


class HealthMonitor:
    """Orchestrates health checks across all Tritium subsystems.

    Thread-safe.  Components register health-check callables that return
    ComponentHealth.  ``check_all()`` runs every registered check and
    aggregates into a SystemStatus.

    Parameters
    ----------
    metrics:
        Optional MetricsCollector for recording check latency and status
        gauges automatically.
    check_timeout:
        Maximum seconds a single health check is allowed to take.
        Checks exceeding this are marked DEGRADED.
    stale_threshold:
        Seconds after which a cached check result is considered stale
        and the component is marked UNKNOWN.
    """

    def __init__(
        self,
        metrics: MetricsCollector | None = None,
        check_timeout: float = 5.0,
        stale_threshold: float = 60.0,
    ) -> None:
        self._lock = threading.RLock()
        self._checks: dict[str, HealthCheck] = {}
        self._last_results: dict[str, ComponentHealth] = {}
        self._metrics = metrics
        self._check_timeout = check_timeout
        self._stale_threshold = stale_threshold

    # -- Registration ----------------------------------------------------------

    def register(self, name: str, check: HealthCheck) -> None:
        """Register a health check for a named component.

        Parameters
        ----------
        name:
            Unique component name (e.g., "tracker", "fusion", "event_bus").
        check:
            Callable that returns a ComponentHealth instance.
        """
        with self._lock:
            self._checks[name] = check

    def unregister(self, name: str) -> bool:
        """Remove a health check.  Returns True if it existed."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                self._last_results.pop(name, None)
                return True
            return False

    @property
    def registered_components(self) -> list[str]:
        """List of registered component names."""
        with self._lock:
            return list(self._checks.keys())

    # -- Checking --------------------------------------------------------------

    def check(self, name: str) -> ComponentHealth:
        """Run the health check for a single component.

        Returns the ComponentHealth result.  If the check raises an
        exception, returns a DOWN ComponentHealth with the error message.
        """
        with self._lock:
            check_fn = self._checks.get(name)
        if check_fn is None:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNKNOWN,
                message=f"No health check registered for '{name}'",
            )

        start = time.monotonic()
        try:
            result = check_fn()
            elapsed = time.monotonic() - start

            # Ensure the result has the right name
            if result.name != name:
                result.name = name
            result.last_checked = time.time()

            # Flag slow checks as degraded
            if elapsed > self._check_timeout and result.status == ComponentStatus.UP:
                result.status = ComponentStatus.DEGRADED
                result.message = (
                    f"Health check took {elapsed:.2f}s "
                    f"(timeout={self._check_timeout:.1f}s)"
                )

            # Record metrics
            if self._metrics is not None:
                self._metrics.record_latency(f"health.{name}.check_time", elapsed)
                status_val = {"up": 1.0, "degraded": 0.5, "down": 0.0}.get(
                    result.status.value, -1.0
                )
                self._metrics.set_gauge(f"health.{name}.status", status_val)

        except Exception as exc:
            elapsed = time.monotonic() - start
            result = ComponentHealth(
                name=name,
                status=ComponentStatus.DOWN,
                message="Health check raised an exception",
                error=str(exc),
                last_checked=time.time(),
            )
            if self._metrics is not None:
                self._metrics.record_latency(f"health.{name}.check_time", elapsed)
                self._metrics.set_gauge(f"health.{name}.status", 0.0)

        with self._lock:
            self._last_results[name] = result
        return result

    def check_all(self) -> SystemStatus:
        """Run all registered health checks and return aggregated status.

        Components are checked sequentially in registration order.
        The overall status is:
          - UP if all components are UP
          - DEGRADED if any are DEGRADED but none DOWN
          - DOWN if any are DOWN
          - UNKNOWN if no components are registered
        """
        with self._lock:
            names = list(self._checks.keys())

        if not names:
            return SystemStatus(
                overall=ComponentStatus.UNKNOWN,
                timestamp=time.time(),
            )

        components: dict[str, ComponentHealth] = {}
        for name in names:
            components[name] = self.check(name)

        overall = self._compute_overall(components)

        status = SystemStatus(
            overall=overall,
            components=components,
            timestamp=time.time(),
        )

        # Record overall gauge
        if self._metrics is not None:
            status_val = {"up": 1.0, "degraded": 0.5, "down": 0.0}.get(
                overall.value, -1.0
            )
            self._metrics.set_gauge("health.system.overall", status_val)

        return status

    def get_last_result(self, name: str) -> ComponentHealth | None:
        """Return the most recent health check result for a component.

        Returns None if the component has never been checked.
        """
        with self._lock:
            return self._last_results.get(name)

    def get_last_status(self) -> SystemStatus:
        """Return a SystemStatus from cached results (no new checks).

        Useful for dashboards that poll frequently — avoids running
        the actual checks.  Components whose cached result is older
        than ``stale_threshold`` are marked UNKNOWN.
        """
        with self._lock:
            names = list(self._checks.keys())
            components: dict[str, ComponentHealth] = {}
            now = time.time()
            for name in names:
                cached = self._last_results.get(name)
                if cached is None:
                    components[name] = ComponentHealth(
                        name=name,
                        status=ComponentStatus.UNKNOWN,
                        message="Never checked",
                    )
                elif (now - cached.last_checked) > self._stale_threshold:
                    components[name] = ComponentHealth(
                        name=name,
                        status=ComponentStatus.UNKNOWN,
                        message=f"Stale (last checked {now - cached.last_checked:.0f}s ago)",
                        details=cached.details,
                        last_checked=cached.last_checked,
                    )
                else:
                    components[name] = cached

        overall = self._compute_overall(components) if components else ComponentStatus.UNKNOWN
        return SystemStatus(
            overall=overall,
            components=components,
            timestamp=now,
        )

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _compute_overall(components: dict[str, ComponentHealth]) -> ComponentStatus:
        """Derive overall status from individual component statuses."""
        if not components:
            return ComponentStatus.UNKNOWN

        statuses = {c.status for c in components.values()}

        if ComponentStatus.DOWN in statuses:
            return ComponentStatus.DOWN
        if ComponentStatus.DEGRADED in statuses:
            return ComponentStatus.DEGRADED
        if statuses == {ComponentStatus.UP}:
            return ComponentStatus.UP
        if ComponentStatus.UP in statuses:
            # Mix of UP and UNKNOWN — degraded
            return ComponentStatus.DEGRADED
        return ComponentStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Built-in health check factories for common Tritium components
# ---------------------------------------------------------------------------

def make_tracker_check(
    tracker: Any,
    stale_seconds: float = 60.0,
    max_stale_ratio: float = 0.5,
) -> HealthCheck:
    """Create a health check for a TargetTracker.

    Parameters
    ----------
    tracker:
        A TargetTracker instance (or any object with ``get_all()``
        returning TrackedTarget-like objects with ``last_seen``).
    stale_seconds:
        Targets not updated in this many seconds are considered stale.
    max_stale_ratio:
        If more than this fraction of targets are stale, status is DEGRADED.
    """

    def _check() -> ComponentHealth:
        try:
            targets = tracker.get_all()
        except Exception as exc:
            return ComponentHealth(
                name="tracker",
                status=ComponentStatus.DOWN,
                error=str(exc),
                message="Failed to retrieve targets",
            )

        total = len(targets)
        now = time.monotonic()
        stale_count = sum(
            1 for t in targets if (now - t.last_seen) > stale_seconds
        )
        active_count = total - stale_count

        details = {
            "target_count": total,
            "active_count": active_count,
            "stale_count": stale_count,
            "stale_threshold_seconds": stale_seconds,
        }

        if total == 0:
            return ComponentHealth(
                name="tracker",
                status=ComponentStatus.UP,
                message="No targets tracked",
                details=details,
            )

        stale_ratio = stale_count / total
        if stale_ratio > max_stale_ratio:
            return ComponentHealth(
                name="tracker",
                status=ComponentStatus.DEGRADED,
                message=f"{stale_count}/{total} targets stale "
                        f"({stale_ratio:.0%} > {max_stale_ratio:.0%} threshold)",
                details=details,
            )

        return ComponentHealth(
            name="tracker",
            status=ComponentStatus.UP,
            message=f"{active_count} active targets, {stale_count} stale",
            details=details,
        )

    return _check


def make_fusion_check(
    fusion_engine: Any,
    min_fusion_rate: float = 0.0,
) -> HealthCheck:
    """Create a health check for a FusionEngine.

    Parameters
    ----------
    fusion_engine:
        A FusionEngine instance (or any object with ``get_fused_targets()``
        and optionally ``metrics`` attribute).
    min_fusion_rate:
        Minimum fusions/hour.  Below this, status is DEGRADED.
        0 disables the rate check.
    """

    def _check() -> ComponentHealth:
        details: dict[str, Any] = {}
        try:
            fused = fusion_engine.get_fused_targets()
            details["fused_target_count"] = len(fused)
        except Exception as exc:
            return ComponentHealth(
                name="fusion",
                status=ComponentStatus.DOWN,
                error=str(exc),
                message="Failed to query fused targets",
            )

        # Check metrics if available
        metrics = getattr(fusion_engine, "metrics", None)
        if metrics is not None:
            try:
                status = metrics.get_status()
                details["total_fusions"] = status.get("total_fusions", 0)
                details["hourly_rate"] = status.get("hourly_rate", 0.0)
                details["pending_feedback"] = status.get("total_pending", 0)
            except Exception:
                pass

        # Evaluate health
        hourly_rate = details.get("hourly_rate", 0.0)
        if min_fusion_rate > 0 and hourly_rate < min_fusion_rate:
            return ComponentHealth(
                name="fusion",
                status=ComponentStatus.DEGRADED,
                message=f"Low fusion rate: {hourly_rate:.1f}/hr "
                        f"(min={min_fusion_rate:.1f}/hr)",
                details=details,
            )

        return ComponentHealth(
            name="fusion",
            status=ComponentStatus.UP,
            message=f"{details.get('fused_target_count', 0)} fused targets",
            details=details,
        )

    return _check


def make_event_bus_check(
    event_bus: Any,
    max_queue_depth: int = 10000,
) -> HealthCheck:
    """Create a health check for an EventBus.

    Parameters
    ----------
    event_bus:
        An EventBus or QueueEventBus instance.
    max_queue_depth:
        Queue depth above which status is DEGRADED.
    """

    def _check() -> ComponentHealth:
        details: dict[str, Any] = {}

        try:
            # Count subscribers
            subs = getattr(event_bus, "_subscribers", {})
            total_subs = sum(len(v) for v in subs.values())
            topic_count = len(subs)
            details["subscriber_count"] = total_subs
            details["topic_count"] = topic_count
        except Exception as exc:
            return ComponentHealth(
                name="event_bus",
                status=ComponentStatus.DOWN,
                error=str(exc),
                message="Failed to inspect event bus",
            )

        # Check queue depth for QueueEventBus
        q = getattr(event_bus, "_queue", None)
        if q is not None:
            try:
                depth = q.qsize()
                details["queue_depth"] = depth
                if depth > max_queue_depth:
                    return ComponentHealth(
                        name="event_bus",
                        status=ComponentStatus.DEGRADED,
                        message=f"Queue depth {depth} exceeds {max_queue_depth}",
                        details=details,
                    )
            except Exception:
                pass

        return ComponentHealth(
            name="event_bus",
            status=ComponentStatus.UP,
            message=f"{total_subs} subscribers on {topic_count} topics",
            details=details,
        )

    return _check


def make_store_check(
    store: Any,
    name: str = "store",
    max_query_latency: float = 1.0,
) -> HealthCheck:
    """Create a health check for a BaseStore (SQLite-backed store).

    Parameters
    ----------
    store:
        A BaseStore subclass instance.
    name:
        Component name for the health report.
    max_query_latency:
        Maximum acceptable seconds for a simple query.
        If exceeded, status is DEGRADED.
    """

    def _check() -> ComponentHealth:
        details: dict[str, Any] = {}

        # Probe the database with a lightweight query
        conn = getattr(store, "_conn", None)
        if conn is None:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.DOWN,
                message="No database connection",
            )

        try:
            start = time.monotonic()
            # Simple integrity check
            cursor = conn.execute("SELECT 1")
            cursor.fetchone()
            latency = time.monotonic() - start
            details["query_latency"] = round(latency, 6)
        except Exception as exc:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.DOWN,
                error=str(exc),
                message="Database query failed",
            )

        # Get table row counts
        try:
            lock = getattr(store, "_lock", None)
            tables_cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in tables_cursor.fetchall()]
            record_counts: dict[str, int] = {}
            for table in tables:
                try:
                    count_cursor = conn.execute(f"SELECT COUNT(*) FROM [{table}]")
                    record_counts[table] = count_cursor.fetchone()[0]
                except Exception:
                    record_counts[table] = -1
            details["tables"] = record_counts
            details["total_records"] = sum(
                v for v in record_counts.values() if v >= 0
            )
        except Exception:
            pass

        # Get database file size
        db_path = getattr(store, "_db_path", None)
        if db_path and db_path != ":memory:":
            try:
                from pathlib import Path

                p = Path(db_path)
                if p.exists():
                    size_bytes = p.stat().st_size
                    details["db_size_bytes"] = size_bytes
                    details["db_size_mb"] = round(size_bytes / (1024 * 1024), 2)
            except Exception:
                pass

        if details.get("query_latency", 0) > max_query_latency:
            return ComponentHealth(
                name=name,
                status=ComponentStatus.DEGRADED,
                message=f"Query latency {details['query_latency']:.3f}s "
                        f"exceeds {max_query_latency:.1f}s threshold",
                details=details,
            )

        return ComponentHealth(
            name=name,
            status=ComponentStatus.UP,
            message=f"{details.get('total_records', 0)} records across "
                    f"{len(details.get('tables', {}))} tables",
            details=details,
        )

    return _check
