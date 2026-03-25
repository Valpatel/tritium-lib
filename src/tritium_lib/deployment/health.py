# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""HealthCheck — verify all components of a Tritium deployment are running.

Performs local-only checks: process existence, port availability, file
presence. No SSH or network calls are made.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from .config import DeploymentConfig
from .status import ComponentStatus, StatusLevel


@dataclass
class HealthReport:
    """Aggregated health report for an entire deployment.

    Attributes
    ----------
    deployment_name:
        Name of the deployment that was checked.
    component_statuses:
        Status of each component keyed by component name.
    overall:
        Overall status: "healthy", "degraded", or "unhealthy".
    checked_at:
        Timestamp of the health check run.
    errors:
        List of error strings encountered during checks.
    """

    deployment_name: str
    component_statuses: dict[str, ComponentStatus] = field(default_factory=dict)
    overall: str = "unknown"
    checked_at: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """True if overall status is healthy."""
        return self.overall == "healthy"

    @property
    def component_count(self) -> int:
        """Number of components checked."""
        return len(self.component_statuses)

    @property
    def running_count(self) -> int:
        """Number of components currently running."""
        return sum(
            1 for s in self.component_statuses.values() if s.is_running
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "deployment_name": self.deployment_name,
            "component_statuses": {
                k: v.to_dict() for k, v in self.component_statuses.items()
            },
            "overall": self.overall,
            "checked_at": self.checked_at,
            "errors": list(self.errors),
            "component_count": self.component_count,
            "running_count": self.running_count,
        }


class HealthCheck:
    """Verify all components of a Tritium deployment.

    Performs local-only checks:
      - PID file existence
      - Port file markers
      - Data directory existence
      - Log directory existence

    No actual network connections are made — this is purely a data-model
    and filesystem-based checker suitable for local deployments.

    Parameters
    ----------
    config:
        The deployment configuration to check against.
    pid_dir:
        Directory where PID files are stored.
        Default: /var/run/tritium
    """

    def __init__(
        self,
        config: DeploymentConfig,
        pid_dir: str = "/var/run/tritium",
    ) -> None:
        self.config = config
        self.pid_dir = pid_dir

    def check_component(self, component_name: str) -> ComponentStatus:
        """Check health of a single component by name.

        Returns a ComponentStatus with the determined status level.
        """
        comp = self.config.get_component(component_name)
        if comp is None:
            return ComponentStatus(
                name=component_name,
                status=StatusLevel.UNKNOWN,
                error_message=f"Component '{component_name}' not in config",
            )

        # Check PID file
        pid_file = os.path.join(self.pid_dir, f"{component_name}.pid")
        pid = 0
        if os.path.isfile(pid_file):
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
            except (ValueError, OSError):
                pid = 0

        # Check if process is actually running (local only)
        process_running = False
        if pid > 0:
            try:
                # os.kill with signal 0 checks existence without killing
                os.kill(pid, 0)
                process_running = True
            except (OSError, PermissionError):
                process_running = False

        # Determine status
        if process_running:
            status = StatusLevel.RUNNING
            error = ""
        elif pid > 0:
            status = StatusLevel.ERROR
            error = f"PID {pid} exists in file but process not running"
        else:
            status = StatusLevel.STOPPED
            error = "No PID file found"

        return ComponentStatus(
            name=component_name,
            status=status,
            pid=pid,
            port=comp.port,
            version=comp.version,
            error_message=error,
        )

    def check_all(self) -> HealthReport:
        """Check health of all enabled components.

        Returns a HealthReport with individual component statuses
        and an aggregated overall status.
        """
        report = HealthReport(deployment_name=self.config.name)

        for comp in self.config.enabled_components:
            try:
                status = self.check_component(comp.name)
                report.component_statuses[comp.name] = status
            except Exception as exc:
                report.errors.append(f"{comp.name}: {exc}")
                report.component_statuses[comp.name] = ComponentStatus(
                    name=comp.name,
                    status=StatusLevel.ERROR,
                    error_message=str(exc),
                )

        # Determine overall status
        if not report.component_statuses:
            report.overall = "unknown"
        else:
            running = report.running_count
            total = report.component_count
            if running == total:
                report.overall = "healthy"
            elif running > 0:
                report.overall = "degraded"
            else:
                report.overall = "unhealthy"

        return report

    def check_data_dir(self) -> bool:
        """Check that the configured data directory exists."""
        return os.path.isdir(self.config.data_dir)

    def check_log_dir(self) -> bool:
        """Check that the configured log directory exists."""
        return os.path.isdir(self.config.log_dir)
