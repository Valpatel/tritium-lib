# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Notification system — shared model and manager for alerts across the ecosystem.

Provides a Notification dataclass and a thread-safe NotificationManager that
collects, stores, queries, and broadcasts notifications. Used by tritium-sc
(WebSocket broadcast to UI) and tritium-edge (fleet server alerts).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Callable


@dataclass
class Notification:
    """A single notification from any plugin or subsystem."""
    id: str
    title: str
    message: str
    severity: str  # info, warning, critical
    source: str  # plugin name or subsystem
    timestamp: float
    read: bool = False
    entity_id: str | None = None  # optional link to target/dossier

    def to_dict(self) -> dict:
        return asdict(self)


class NotificationManager:
    """Collects notifications and provides query/mark-read API.

    Thread-safe. Optionally broadcasts new notifications via a callback
    (e.g., WebSocket push, MQTT publish).

    Args:
        broadcast: Optional callback invoked with {"type": "notification:new", "data": ...}
            for each new notification.
        max_notifications: Maximum stored notifications (oldest evicted first).
    """

    def __init__(
        self,
        broadcast: Callable[[dict], None] | None = None,
        max_notifications: int = 500,
    ) -> None:
        self._lock = threading.Lock()
        self._notifications: list[Notification] = []
        self._max = max_notifications
        self._broadcast = broadcast

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        title: str,
        message: str,
        severity: str = "info",
        source: str = "system",
        entity_id: str | None = None,
    ) -> str:
        """Create and store a notification. Returns the notification id."""
        if severity not in ("info", "warning", "critical"):
            severity = "info"

        nid = uuid.uuid4().hex[:12]
        notif = Notification(
            id=nid,
            title=title,
            message=message,
            severity=severity,
            source=source,
            timestamp=time.time(),
            entity_id=entity_id,
        )

        with self._lock:
            self._notifications.insert(0, notif)
            if len(self._notifications) > self._max:
                self._notifications = self._notifications[: self._max]

        # Broadcast
        if self._broadcast is not None:
            try:
                self._broadcast({
                    "type": "notification:new",
                    "data": notif.to_dict(),
                })
            except Exception:
                pass  # Best-effort broadcast

        return nid

    def get_unread(self) -> list[dict]:
        """Return all unread notifications as dicts."""
        with self._lock:
            return [n.to_dict() for n in self._notifications if not n.read]

    def get_all(
        self, limit: int = 100, since: float | None = None
    ) -> list[dict]:
        """Return notifications as dicts, newest first."""
        with self._lock:
            result = self._notifications
            if since is not None:
                result = [n for n in result if n.timestamp >= since]
            return [n.to_dict() for n in result[:limit]]

    def mark_read(self, notification_id: str) -> bool:
        """Mark a single notification as read. Returns True if found."""
        with self._lock:
            for n in self._notifications:
                if n.id == notification_id:
                    n.read = True
                    return True
        return False

    def mark_all_read(self) -> int:
        """Mark all notifications as read. Returns count marked."""
        count = 0
        with self._lock:
            for n in self._notifications:
                if not n.read:
                    n.read = True
                    count += 1
        return count

    def count_unread(self) -> int:
        """Return the number of unread notifications."""
        with self._lock:
            return sum(1 for n in self._notifications if not n.read)

    def clear(self) -> int:
        """Remove all notifications. Returns count removed."""
        with self._lock:
            count = len(self._notifications)
            self._notifications.clear()
            return count
