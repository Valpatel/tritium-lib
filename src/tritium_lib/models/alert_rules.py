# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Alert rule models for generating alerts/notifications from system events.

AlertRule defines a rule that evaluates system events against conditions
and generates alerts with configurable severity, channels, message templates,
and cooldown periods.  This is distinct from automation rules (which execute
actions) — alert rules specifically generate notifications and alerts.

Integrates with NotificationRule for delivery routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .notification_rules import NotificationChannel, NotificationSeverity


class AlertTrigger(str, Enum):
    """Events that can trigger an alert rule."""
    TARGET_NEW = "target_new"
    TARGET_LOST = "target_lost"
    TARGET_ENTER_ZONE = "target_enter_zone"
    TARGET_EXIT_ZONE = "target_exit_zone"
    TARGET_LOITER = "target_loiter"
    TARGET_SPEED = "target_speed"
    DEVICE_OFFLINE = "device_offline"
    DEVICE_ONLINE = "device_online"
    DEVICE_BATTERY_LOW = "device_battery_low"
    DEVICE_ERROR = "device_error"
    SENSOR_MOTION = "sensor_motion"
    SENSOR_ACOUSTIC = "sensor_acoustic"
    THREAT_DETECTED = "threat_detected"
    GEOFENCE_BREACH = "geofence_breach"
    CORRELATION_EVENT = "correlation_event"
    CUSTOM = "custom"


class ConditionOperator(str, Enum):
    """Operators for evaluating alert conditions."""
    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN_LIST = "in"
    REGEX = "regex"


@dataclass
class AlertCondition:
    """A single condition that must be met for an alert to fire.

    Attributes
    ----------
    field:
        The event data field to evaluate (e.g. ``"rssi"``, ``"alliance"``).
    operator:
        Comparison operator.
    value:
        Expected value to compare against.
    """
    field: str
    operator: ConditionOperator = ConditionOperator.EQUALS
    value: Any = None

    def evaluate(self, event_data: dict) -> bool:
        """Evaluate this condition against event data.

        Returns True if the condition is satisfied.
        """
        actual = event_data.get(self.field)
        if actual is None:
            return False

        op = self.operator
        expected = self.value

        if op == ConditionOperator.EQUALS:
            return actual == expected
        elif op == ConditionOperator.NOT_EQUALS:
            return actual != expected
        elif op == ConditionOperator.GREATER_THAN:
            return _numeric(actual) > _numeric(expected)
        elif op == ConditionOperator.LESS_THAN:
            return _numeric(actual) < _numeric(expected)
        elif op == ConditionOperator.GREATER_EQUAL:
            return _numeric(actual) >= _numeric(expected)
        elif op == ConditionOperator.LESS_EQUAL:
            return _numeric(actual) <= _numeric(expected)
        elif op == ConditionOperator.CONTAINS:
            return str(expected) in str(actual)
        elif op == ConditionOperator.NOT_CONTAINS:
            return str(expected) not in str(actual)
        elif op == ConditionOperator.IN_LIST:
            return actual in (expected if isinstance(expected, (list, tuple, set)) else [expected])
        elif op == ConditionOperator.REGEX:
            import re
            try:
                return bool(re.search(str(expected), str(actual)))
            except re.error:
                return False
        return False

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AlertCondition:
        op = data.get("operator", "eq")
        if isinstance(op, str):
            op = ConditionOperator(op)
        return cls(
            field=data["field"],
            operator=op,
            value=data.get("value"),
        )


@dataclass
class AlertRule:
    """A rule that evaluates system events and generates alerts.

    When an event matching ``trigger`` occurs and all ``conditions`` are
    satisfied, an alert of the given ``severity`` is generated and sent
    to the listed ``channels``.  The ``cooldown_seconds`` prevents alert
    flooding by suppressing duplicate firings within the cooldown window.

    Attributes
    ----------
    rule_id:
        Unique identifier for this rule.
    name:
        Human-readable name.
    trigger:
        The event type that activates this rule.
    conditions:
        List of conditions that must ALL be true for the rule to fire.
        Empty list means the rule fires on any matching trigger.
    severity:
        Alert severity level.
    channels:
        Notification delivery channels.
    message_template:
        Message template with ``{placeholders}`` for event data.
        Supported: ``{trigger}``, ``{severity}``, ``{target_id}``,
        ``{device_id}``, ``{zone_id}``, ``{message}``, ``{timestamp}``.
    cooldown_seconds:
        Minimum seconds between firings. 0 = no cooldown.
    enabled:
        Whether this rule is active.
    zone_filter:
        Optional list of zone IDs to scope this rule to.
    target_filter:
        Optional target alliance filter (e.g. ``["hostile", "unknown"]``).
    tags:
        Arbitrary tags for grouping/filtering rules.
    created_at:
        Creation timestamp.
    updated_at:
        Last modification timestamp.
    fire_count:
        Total number of times fired.
    last_fired_at:
        Timestamp of most recent firing.
    """
    rule_id: str
    name: str = ""
    trigger: AlertTrigger = AlertTrigger.CUSTOM
    conditions: list[AlertCondition] = field(default_factory=list)
    severity: NotificationSeverity = NotificationSeverity.WARNING
    channels: list[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.WEBSOCKET]
    )
    message_template: str = "[{severity}] {trigger}: {message}"
    cooldown_seconds: int = 60
    enabled: bool = True
    zone_filter: list[str] = field(default_factory=list)
    target_filter: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    fire_count: int = 0
    last_fired_at: Optional[datetime] = None

    def matches(self, trigger: str, event_data: dict) -> bool:
        """Check if this rule should fire for the given event.

        Parameters
        ----------
        trigger:
            The event trigger type string.
        event_data:
            Dictionary of event data fields.

        Returns True if enabled, trigger matches, and all conditions pass.
        """
        if not self.enabled:
            return False
        if self.trigger.value != trigger and self.trigger != AlertTrigger.CUSTOM:
            return False
        # Zone filter
        if self.zone_filter:
            zone_id = event_data.get("zone_id", "")
            if zone_id and zone_id not in self.zone_filter:
                return False
        # Target alliance filter
        if self.target_filter:
            alliance = event_data.get("alliance", "")
            if alliance and alliance not in self.target_filter:
                return False
        # Evaluate all conditions
        for condition in self.conditions:
            if not condition.evaluate(event_data):
                return False
        return True

    def is_cooled_down(self, now: Optional[datetime] = None) -> bool:
        """Check if cooldown has elapsed since last firing."""
        if self.cooldown_seconds <= 0:
            return True
        if self.last_fired_at is None:
            return True
        if now is None:
            now = datetime.now(timezone.utc)
        elapsed = (now - self.last_fired_at).total_seconds()
        return elapsed >= self.cooldown_seconds

    def render_message(self, **kwargs: str) -> str:
        """Render the alert message using the template.

        Parameters
        ----------
        **kwargs:
            Values for template placeholders.

        Returns the rendered message string.
        """
        result = self.message_template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def record_firing(self, now: Optional[datetime] = None) -> None:
        """Record that this rule has fired."""
        if now is None:
            now = datetime.now(timezone.utc)
        self.fire_count += 1
        self.last_fired_at = now
        self.updated_at = now

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON/REST transport."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "trigger": self.trigger.value,
            "conditions": [c.to_dict() for c in self.conditions],
            "severity": self.severity.value,
            "channels": [c.value for c in self.channels],
            "message_template": self.message_template,
            "cooldown_seconds": self.cooldown_seconds,
            "enabled": self.enabled,
            "zone_filter": self.zone_filter,
            "target_filter": self.target_filter,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "fire_count": self.fire_count,
            "last_fired_at": self.last_fired_at.isoformat() if self.last_fired_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AlertRule:
        """Deserialize from plain dict."""
        trigger = data.get("trigger", "custom")
        if isinstance(trigger, str):
            trigger = AlertTrigger(trigger)

        severity = data.get("severity", "warning")
        if isinstance(severity, str):
            severity = NotificationSeverity(severity)

        channels = [
            NotificationChannel(c) if isinstance(c, str) else c
            for c in data.get("channels", ["websocket"])
        ]

        conditions = [
            AlertCondition.from_dict(c)
            for c in data.get("conditions", [])
        ]

        rule = cls(
            rule_id=data["rule_id"],
            name=data.get("name", ""),
            trigger=trigger,
            conditions=conditions,
            severity=severity,
            channels=channels,
            message_template=data.get("message_template", "[{severity}] {trigger}: {message}"),
            cooldown_seconds=data.get("cooldown_seconds", 60),
            enabled=data.get("enabled", True),
            zone_filter=data.get("zone_filter", []),
            target_filter=data.get("target_filter", []),
            tags=data.get("tags", []),
            fire_count=data.get("fire_count", 0),
        )

        if data.get("created_at"):
            rule.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            rule.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("last_fired_at"):
            rule.last_fired_at = datetime.fromisoformat(data["last_fired_at"])

        return rule


def _numeric(v: Any) -> float:
    """Convert value to float for numeric comparison."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Default alert rules — sensible out-of-the-box alert config
# ---------------------------------------------------------------------------

DEFAULT_ALERT_RULES: list[AlertRule] = [
    AlertRule(
        rule_id="alert-hostile-detected",
        name="Hostile target detected",
        trigger=AlertTrigger.THREAT_DETECTED,
        severity=NotificationSeverity.CRITICAL,
        channels=[NotificationChannel.WEBSOCKET, NotificationChannel.MQTT, NotificationChannel.LOG],
        message_template="HOSTILE DETECTED: {message} at {zone_id}",
        cooldown_seconds=30,
        target_filter=["hostile"],
    ),
    AlertRule(
        rule_id="alert-geofence-breach",
        name="Geofence breach",
        trigger=AlertTrigger.GEOFENCE_BREACH,
        severity=NotificationSeverity.WARNING,
        channels=[NotificationChannel.WEBSOCKET, NotificationChannel.LOG],
        message_template="GEOFENCE BREACH: {target_id} entered {zone_id}",
        cooldown_seconds=60,
    ),
    AlertRule(
        rule_id="alert-device-offline",
        name="Edge device offline",
        trigger=AlertTrigger.DEVICE_OFFLINE,
        severity=NotificationSeverity.WARNING,
        channels=[NotificationChannel.WEBSOCKET, NotificationChannel.LOG],
        message_template="DEVICE OFFLINE: {device_id} — {message}",
        cooldown_seconds=300,
    ),
    AlertRule(
        rule_id="alert-battery-critical",
        name="Critical battery level",
        trigger=AlertTrigger.DEVICE_BATTERY_LOW,
        conditions=[AlertCondition(field="battery_level", operator=ConditionOperator.LESS_THAN, value=10)],
        severity=NotificationSeverity.ERROR,
        channels=[NotificationChannel.WEBSOCKET, NotificationChannel.MQTT],
        message_template="BATTERY CRITICAL: {device_id} at {battery_level}%",
        cooldown_seconds=600,
    ),
    AlertRule(
        rule_id="alert-target-loiter",
        name="Target loitering",
        trigger=AlertTrigger.TARGET_LOITER,
        conditions=[AlertCondition(field="duration_seconds", operator=ConditionOperator.GREATER_THAN, value=300)],
        severity=NotificationSeverity.INFO,
        channels=[NotificationChannel.WEBSOCKET],
        message_template="LOITER: {target_id} stationary for {duration_seconds}s in {zone_id}",
        cooldown_seconds=120,
    ),
]
