# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Persistent audit trail for compliance and forensics.

Provides a high-level API for recording, querying, and managing
audit records across the Tritium ecosystem.  Built on SQLite with
automatic rotation and predefined compliance-oriented action types.

Usage
-----
    trail = AuditTrail("/var/lib/tritium/audit.db")

    # Record actions with typed helpers
    trail.record_target_accessed("user:analyst", "ble_AA:BB:CC",
                                  details="Viewed dossier")
    trail.record_config_changed("user:admin", "mqtt.broker_url",
                                 old_value="localhost", new_value="10.0.0.1")

    # Query with builder
    query = AuditQuery().by_actor("user:admin").by_action(AuditAction.CONFIG_CHANGED)
    results = trail.search(query)

    # Export for compliance reports
    records = trail.export(start_time=epoch_start, end_time=epoch_end)
"""

from __future__ import annotations

from .trail import AuditTrail
from .entry import AuditEntry, AuditSeverity
from .query import AuditQuery
from .actions import AuditAction

__all__ = [
    "AuditTrail",
    "AuditEntry",
    "AuditSeverity",
    "AuditQuery",
    "AuditAction",
]
