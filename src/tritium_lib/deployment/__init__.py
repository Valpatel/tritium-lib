# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.deployment — utilities for deploying and managing Tritium instances.

Provides data models and local utilities for deployment planning, health
verification, system requirements checking, backup management, and log
collection across all Tritium components.

Components:
  - DeploymentConfig     — target host, credentials, components to deploy
  - DeploymentComponent  — individual component specification
  - ComponentStatus      — runtime status of a deployed component
  - HealthCheck          — verify all components are running
  - SystemRequirements   — check system meets minimum requirements
  - BackupManager        — create/restore backups of tracking data
  - BackupManifest       — metadata about a backup archive
  - LogCollector         — collect logs from all components
  - LogEntry             — single structured log entry

Quick start::

    from tritium_lib.deployment import (
        DeploymentConfig, HealthCheck, SystemRequirements,
        BackupManager, LogCollector,
    )

    # Define a deployment
    config = DeploymentConfig(
        name="field-station-1",
        host="192.168.1.100",
        components=["sc", "lib", "mqtt"],
    )

    # Check system requirements
    reqs = SystemRequirements()
    result = reqs.check_local()
    print(result.meets_minimum)  # True/False

    # Health check
    hc = HealthCheck(config)
    status = hc.check_all()
    print(status)  # {component: ComponentStatus, ...}

    # Backup
    bm = BackupManager(backup_dir="/backups")
    manifest = bm.create_backup(source_dir="/data/tritium")
    bm.restore_backup(manifest.backup_id, target_dir="/data/tritium-restored")

    # Collect logs
    lc = LogCollector(log_dirs=["/var/log/tritium"])
    entries = lc.collect(since_hours=24)
"""

from .config import DeploymentComponent, DeploymentConfig
from .status import ComponentStatus
from .health import HealthCheck
from .requirements import SystemRequirements, RequirementsResult
from .backup import BackupManager, BackupManifest
from .logs import LogCollector, LogEntry

__all__ = [
    "BackupManager",
    "BackupManifest",
    "ComponentStatus",
    "DeploymentComponent",
    "DeploymentConfig",
    "HealthCheck",
    "LogCollector",
    "LogEntry",
    "RequirementsResult",
    "SystemRequirements",
]
