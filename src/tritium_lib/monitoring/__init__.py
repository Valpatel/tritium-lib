# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.monitoring — system health monitoring for the tracking pipeline.

Provides real-time health checks and performance metrics for all Tritium
subsystems without any external dependencies (no Prometheus, no Grafana).

Components:
  - HealthMonitor    — orchestrates component health checks
  - ComponentHealth  — health status for a single component
  - SystemStatus     — overall system health aggregation
  - MetricsCollector — collects performance metrics (latency, throughput, queue depth)

Quick start::

    from tritium_lib.monitoring import HealthMonitor, MetricsCollector

    metrics = MetricsCollector()
    monitor = HealthMonitor(metrics=metrics)

    # Register a component health check
    monitor.register("tracker", lambda: ComponentHealth(
        name="tracker", status="up", target_count=42,
    ))

    # Get system-wide status
    status = monitor.check_all()
    print(status.overall)  # "up", "degraded", or "down"
"""

from .health import (
    ComponentHealth,
    ComponentStatus,
    HealthCheck,
    HealthMonitor,
    SystemStatus,
)
from .metrics import (
    MetricSample,
    MetricsCollector,
    MetricWindow,
)

__all__ = [
    "ComponentHealth",
    "ComponentStatus",
    "HealthCheck",
    "HealthMonitor",
    "MetricSample",
    "MetricsCollector",
    "MetricWindow",
    "SystemStatus",
]
