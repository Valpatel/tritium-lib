# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Audit entry model — a single immutable audit record."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AuditSeverity(str, Enum):
    """Severity level for audit entries."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AuditEntry:
    """A single audit record for compliance and forensics.

    Captures who performed what action, on which resource, when, and
    from where.  Immutable once created — audit records must not be
    modified after the fact.

    Attributes
    ----------
    entry_id : str
        Unique identifier for this entry (UUID4).
    timestamp : float
        Unix epoch time when the action occurred.
    actor : str
        Who performed the action (e.g. ``"user:admin"``, ``"plugin:acoustic"``).
    action : str
        What was done (e.g. ``"target_accessed"``).  Prefer ``AuditAction`` values.
    resource : str
        Affected resource type (e.g. ``"target"``, ``"zone"``, ``"config"``).
    resource_id : str
        Identifier of the affected resource (e.g. ``"ble_AA:BB:CC"``).
    details : str
        Human-readable description of the action.
    severity : str
        One of debug/info/warning/error/critical.
    source_ip : str
        IP address of the originating request.
    metadata : dict
        Additional structured data (old/new values, context, etc.).
    db_id : int
        Database row ID (0 if not yet persisted).
    """

    entry_id: str = ""
    timestamp: float = 0.0
    actor: str = ""
    action: str = ""
    resource: str = ""
    resource_id: str = ""
    details: str = ""
    severity: str = "info"
    source_ip: str = ""
    metadata: dict = field(default_factory=dict)
    db_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary for JSON export."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "resource_id": self.resource_id,
            "details": self.details,
            "severity": self.severity,
            "source_ip": self.source_ip,
            "metadata": dict(self.metadata),
            "db_id": self.db_id,
        }

    @staticmethod
    def create(
        actor: str,
        action: str,
        resource: str = "",
        resource_id: str = "",
        details: str = "",
        severity: str = "info",
        source_ip: str = "",
        metadata: Optional[dict] = None,
        timestamp: Optional[float] = None,
    ) -> AuditEntry:
        """Factory to create a new audit entry with auto-generated ID and timestamp."""
        return AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=timestamp if timestamp is not None else time.time(),
            actor=actor,
            action=action,
            resource=resource,
            resource_id=resource_id,
            details=details,
            severity=severity,
            source_ip=source_ip,
            metadata=metadata or {},
        )
