# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Notification rule models for configurable event routing.

Defines NotificationRule — the contract for deciding which events
generate notifications, how they are routed (WebSocket, MQTT, email),
and how often they can fire (cooldown).  Used by both the command center
notification engine and fleet server alert pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class NotificationChannel(str, Enum):
    """Supported notification delivery channels."""
    WEBSOCKET = "websocket"    # Real-time push to connected dashboards
    MQTT = "mqtt"              # Publish to MQTT notification topic
    EMAIL = "email"            # Email delivery (stub — requires SMTP config)
    WEBHOOK = "webhook"        # HTTP POST to external URL
    LOG = "log"                # Write to persistent notification log


class NotificationSeverity(str, Enum):
    """Severity levels for filtering which events trigger notifications."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric rank for comparison (higher = more severe)."""
        return {
            "debug": 0,
            "info": 1,
            "warning": 2,
            "error": 3,
            "critical": 4,
        }[self.value]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, NotificationSeverity):
            return NotImplemented
        return self.rank >= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, NotificationSeverity):
            return NotImplemented
        return self.rank > other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, NotificationSeverity):
            return NotImplemented
        return self.rank <= other.rank

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, NotificationSeverity):
            return NotImplemented
        return self.rank < other.rank


@dataclass
class NotificationRule:
    """A rule that maps events to notification channels.

    When an event matching ``trigger_event`` occurs with severity at or
    above ``severity_filter``, a notification is sent to all listed
    ``channels``.  The ``cooldown_seconds`` prevents flooding by
    suppressing duplicate notifications within the cooldown window.

    Attributes
    ----------
    rule_id:
        Unique identifier for this rule.
    name:
        Human-readable name for the rule.
    trigger_event:
        Event type pattern to match.  Exact match or wildcard ``*`` for
        all events.  Examples: ``"node_offline"``, ``"battery_low"``,
        ``"ble_new_device"``, ``"*"``.
    severity_filter:
        Minimum severity to trigger the notification.  Events below this
        level are ignored.
    channels:
        List of delivery channels for this notification.
    cooldown_seconds:
        Minimum seconds between successive firings of this rule for the
        same event source.  0 = no cooldown (fire every time).
    template:
        Message template with ``{placeholders}`` for event data.
        Supported placeholders: ``{event}``, ``{severity}``, ``{source}``,
        ``{message}``, ``{timestamp}``, ``{device_id}``.
    enabled:
        Whether this rule is currently active.
    device_filter:
        Optional list of device IDs to scope this rule to.  Empty list
        means all devices.
    created_at:
        When this rule was created.
    updated_at:
        When this rule was last modified.
    fire_count:
        Total number of times this rule has fired.
    last_fired_at:
        Timestamp of the most recent firing.
    """
    rule_id: str
    name: str = ""
    trigger_event: str = "*"
    severity_filter: NotificationSeverity = NotificationSeverity.INFO
    channels: list[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.WEBSOCKET])
    cooldown_seconds: int = 60
    template: str = "[{severity}] {event}: {message}"
    enabled: bool = True
    device_filter: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    fire_count: int = 0
    last_fired_at: Optional[datetime] = None

    def matches_event(self, event_type: str, severity: NotificationSeverity) -> bool:
        """Check if this rule should fire for the given event.

        Parameters
        ----------
        event_type:
            The event type string (e.g. ``"node_offline"``).
        severity:
            The severity of the event.

        Returns True if the rule is enabled, the event matches, and
        the severity meets or exceeds the filter.
        """
        if not self.enabled:
            return False
        if severity < self.severity_filter:
            return False
        if self.trigger_event == "*":
            return True
        return self.trigger_event == event_type

    def matches_device(self, device_id: str) -> bool:
        """Check if this rule applies to the given device.

        Returns True if ``device_filter`` is empty (applies to all)
        or if the device_id is in the filter list.
        """
        if not self.device_filter:
            return True
        return device_id in self.device_filter

    def is_cooled_down(self, now: Optional[datetime] = None) -> bool:
        """Check if the cooldown period has elapsed since last firing.

        Parameters
        ----------
        now:
            Current time.  Defaults to ``datetime.now(timezone.utc)``.

        Returns True if the rule can fire again (cooldown expired or
        never fired).
        """
        if self.cooldown_seconds <= 0:
            return True
        if self.last_fired_at is None:
            return True
        if now is None:
            now = datetime.now(timezone.utc)
        elapsed = (now - self.last_fired_at).total_seconds()
        return elapsed >= self.cooldown_seconds

    def render_message(self, **kwargs: str) -> str:
        """Render the notification message using the template.

        Parameters
        ----------
        **kwargs:
            Values for template placeholders (event, severity, source,
            message, timestamp, device_id, etc.).

        Returns the rendered message string.  Unknown placeholders are
        left as-is.
        """
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def record_firing(self, now: Optional[datetime] = None) -> None:
        """Record that this rule has fired.

        Updates ``fire_count``, ``last_fired_at``, and ``updated_at``.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        self.fire_count += 1
        self.last_fired_at = now
        self.updated_at = now

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON/REST transport."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "trigger_event": self.trigger_event,
            "severity_filter": self.severity_filter.value,
            "channels": [c.value for c in self.channels],
            "cooldown_seconds": self.cooldown_seconds,
            "template": self.template,
            "enabled": self.enabled,
            "device_filter": self.device_filter,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "fire_count": self.fire_count,
            "last_fired_at": self.last_fired_at.isoformat() if self.last_fired_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NotificationRule:
        """Deserialize from a plain dict."""
        channels = [
            NotificationChannel(c) if isinstance(c, str) else c
            for c in data.get("channels", ["websocket"])
        ]
        severity = data.get("severity_filter", "info")
        if isinstance(severity, str):
            severity = NotificationSeverity(severity)

        rule = cls(
            rule_id=data["rule_id"],
            name=data.get("name", ""),
            trigger_event=data.get("trigger_event", "*"),
            severity_filter=severity,
            channels=channels,
            cooldown_seconds=data.get("cooldown_seconds", 60),
            template=data.get("template", "[{severity}] {event}: {message}"),
            enabled=data.get("enabled", True),
            device_filter=data.get("device_filter", []),
            fire_count=data.get("fire_count", 0),
        )

        if data.get("created_at"):
            rule.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            rule.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("last_fired_at"):
            rule.last_fired_at = datetime.fromisoformat(data["last_fired_at"])

        return rule


# ---------------------------------------------------------------------------
# Default rules — sensible out-of-the-box notification config
# ---------------------------------------------------------------------------

DEFAULT_RULES: list[NotificationRule] = [
    NotificationRule(
        rule_id="default-critical",
        name="Critical alerts to all channels",
        trigger_event="*",
        severity_filter=NotificationSeverity.CRITICAL,
        channels=[NotificationChannel.WEBSOCKET, NotificationChannel.MQTT, NotificationChannel.LOG],
        cooldown_seconds=30,
        template="CRITICAL: {event} — {message}",
    ),
    NotificationRule(
        rule_id="default-node-offline",
        name="Node offline alert",
        trigger_event="node_offline",
        severity_filter=NotificationSeverity.WARNING,
        channels=[NotificationChannel.WEBSOCKET, NotificationChannel.LOG],
        cooldown_seconds=300,
        template="Node {device_id} went offline: {message}",
    ),
    NotificationRule(
        rule_id="default-battery-low",
        name="Low battery warning",
        trigger_event="battery_low",
        severity_filter=NotificationSeverity.WARNING,
        channels=[NotificationChannel.WEBSOCKET],
        cooldown_seconds=600,
        template="Battery low on {device_id}: {message}",
    ),
    NotificationRule(
        rule_id="default-new-target",
        name="New target detected",
        trigger_event="target_new",
        severity_filter=NotificationSeverity.INFO,
        channels=[NotificationChannel.WEBSOCKET],
        cooldown_seconds=10,
        template="New target: {message}",
    ),
]
