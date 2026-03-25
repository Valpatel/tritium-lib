# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fluent query builder for searching and filtering audit records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AuditQuery:
    """Fluent builder for audit trail queries.

    Usage
    -----
        query = (AuditQuery()
                 .by_actor("user:admin")
                 .by_action("config_changed")
                 .since(start_ts)
                 .until(end_ts)
                 .with_severity("warning")
                 .with_resource("zone")
                 .with_resource_id("zone_alpha")
                 .limit(50)
                 .offset(10))
    """

    actor: Optional[str] = None
    action: Optional[str] = None
    severity: Optional[str] = None
    resource: Optional[str] = None
    resource_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    source_ip: Optional[str] = None
    keyword: Optional[str] = None
    max_results: int = 100
    skip: int = 0

    def by_actor(self, actor: str) -> AuditQuery:
        """Filter by actor (who performed the action)."""
        self.actor = actor
        return self

    def by_action(self, action: str) -> AuditQuery:
        """Filter by action type."""
        self.action = action
        return self

    def with_severity(self, severity: str) -> AuditQuery:
        """Filter by severity level."""
        self.severity = severity
        return self

    def with_resource(self, resource: str) -> AuditQuery:
        """Filter by resource type."""
        self.resource = resource
        return self

    def with_resource_id(self, resource_id: str) -> AuditQuery:
        """Filter by specific resource ID."""
        self.resource_id = resource_id
        return self

    def since(self, timestamp: float) -> AuditQuery:
        """Only entries at or after this timestamp."""
        self.start_time = timestamp
        return self

    def until(self, timestamp: float) -> AuditQuery:
        """Only entries at or before this timestamp."""
        self.end_time = timestamp
        return self

    def from_ip(self, ip: str) -> AuditQuery:
        """Filter by source IP address."""
        self.source_ip = ip
        return self

    def containing(self, keyword: str) -> AuditQuery:
        """Filter entries whose details contain a keyword (case-insensitive)."""
        self.keyword = keyword
        return self

    def limit(self, n: int) -> AuditQuery:
        """Maximum number of results to return."""
        self.max_results = n
        return self

    def offset(self, n: int) -> AuditQuery:
        """Skip the first N results."""
        self.skip = n
        return self

    def build_sql(self) -> tuple[str, list]:
        """Build SQL WHERE clause and parameters.

        Returns a ``(where_clause, params)`` tuple.  The where clause
        includes the ``WHERE`` keyword when filters are present, and
        is an empty string when no filters are applied.
        """
        conditions: list[str] = []
        params: list = []

        if self.actor is not None:
            conditions.append("actor = ?")
            params.append(self.actor)
        if self.action is not None:
            conditions.append("action = ?")
            params.append(self.action)
        if self.severity is not None:
            conditions.append("severity = ?")
            params.append(self.severity)
        if self.resource is not None:
            conditions.append("resource = ?")
            params.append(self.resource)
        if self.resource_id is not None:
            conditions.append("resource_id = ?")
            params.append(self.resource_id)
        if self.start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(self.start_time)
        if self.end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(self.end_time)
        if self.source_ip is not None:
            conditions.append("ip_address = ?")
            params.append(self.source_ip)
        if self.keyword is not None:
            conditions.append("detail LIKE ?")
            params.append(f"%{self.keyword}%")

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        return where, params
