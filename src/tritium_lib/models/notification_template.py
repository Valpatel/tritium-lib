# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Notification template models for templated alert generation.

A NotificationTemplate defines reusable alert patterns with placeholder
substitution.  When an event fires, the matching template renders
the title and body with live data, routes to the configured channels,
and respects cooldown to prevent flooding.

Works alongside NotificationRule (notification_rules.py) — rules decide
*when* to fire; templates decide *what* to say.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class NotificationTemplateChannel(str, Enum):
    """Delivery channels for templated notifications."""

    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    EMAIL = "email"


class NotificationTemplateSeverity(str, Enum):
    """Severity levels for notification templates."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric rank for comparison (higher = more severe)."""
        return {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}[
            self.value
        ]


class NotificationTemplate(BaseModel):
    """A reusable notification template for generating alerts.

    Templates contain title and body strings with ``{placeholder}``
    syntax.  At render time, placeholders are replaced with live
    event data (target_id, device_id, value, etc.).

    Attributes:
        template_id: Unique identifier for this template.
        name: Human-readable name.
        event_type: Event type this template matches (e.g. ``"node_offline"``).
        title_template: Title string with ``{placeholders}``.
        body_template: Body string with ``{placeholders}``.
        severity: Default severity level for alerts from this template.
        channels: Which delivery channels to route notifications to.
        cooldown_seconds: Minimum seconds between firings for the same
            source.  0 = no cooldown.
        enabled: Whether this template is active.
        created_at: When this template was created.
        updated_at: When this template was last modified.
    """

    template_id: str = ""
    name: str = ""
    event_type: str = ""
    title_template: str = ""
    body_template: str = ""
    severity: NotificationTemplateSeverity = NotificationTemplateSeverity.INFO
    channels: list[NotificationTemplateChannel] = Field(
        default_factory=lambda: [NotificationTemplateChannel.WEBSOCKET],
    )
    cooldown_seconds: int = Field(default=60, ge=0)
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"frozen": False}

    def model_post_init(self, __context: object) -> None:
        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

    def render_title(self, **kwargs: str) -> str:
        """Render the title template with placeholder values.

        Unknown placeholders are left as-is.
        """
        result = self.title_template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def render_body(self, **kwargs: str) -> str:
        """Render the body template with placeholder values.

        Unknown placeholders are left as-is.
        """
        result = self.body_template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def render(self, **kwargs: str) -> dict[str, str]:
        """Render both title and body, returning a dict.

        Returns ``{"title": ..., "body": ...}``.
        """
        return {
            "title": self.render_title(**kwargs),
            "body": self.render_body(**kwargs),
        }

    def matches_event(self, event_type: str) -> bool:
        """Check if this template applies to the given event type."""
        if not self.enabled:
            return False
        return self.event_type == event_type

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON/REST transport."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "event_type": self.event_type,
            "title_template": self.title_template,
            "body_template": self.body_template,
            "severity": self.severity.value,
            "channels": [c.value for c in self.channels],
            "cooldown_seconds": self.cooldown_seconds,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NotificationTemplate:
        """Deserialize from a plain dict."""
        channels = [
            NotificationTemplateChannel(c) if isinstance(c, str) else c
            for c in data.get("channels", ["websocket"])
        ]
        severity = data.get("severity", "info")
        if isinstance(severity, str):
            severity = NotificationTemplateSeverity(severity)

        return cls(
            template_id=data.get("template_id", ""),
            name=data.get("name", ""),
            event_type=data.get("event_type", ""),
            title_template=data.get("title_template", ""),
            body_template=data.get("body_template", ""),
            severity=severity,
            channels=channels,
            cooldown_seconds=data.get("cooldown_seconds", 60),
            enabled=data.get("enabled", True),
        )


# ---------------------------------------------------------------------------
# Built-in notification templates
# ---------------------------------------------------------------------------

BUILTIN_NOTIFICATION_TEMPLATES: list[NotificationTemplate] = [
    NotificationTemplate(
        template_id="tpl-node-offline",
        name="Node Offline",
        event_type="node_offline",
        title_template="Node {device_id} Offline",
        body_template="Device {device_id} has gone offline. Last seen: {last_seen}.",
        severity=NotificationTemplateSeverity.WARNING,
        channels=[
            NotificationTemplateChannel.WEBSOCKET,
            NotificationTemplateChannel.MQTT,
        ],
        cooldown_seconds=300,
    ),
    NotificationTemplate(
        template_id="tpl-battery-low",
        name="Battery Low",
        event_type="battery_low",
        title_template="Low Battery: {device_id}",
        body_template="Device {device_id} battery at {battery_pct}%. Consider charging.",
        severity=NotificationTemplateSeverity.WARNING,
        channels=[NotificationTemplateChannel.WEBSOCKET],
        cooldown_seconds=600,
    ),
    NotificationTemplate(
        template_id="tpl-new-target",
        name="New Target Detected",
        event_type="target_new",
        title_template="New Target: {target_id}",
        body_template="New {classification} target detected by {source}: {target_id}.",
        severity=NotificationTemplateSeverity.INFO,
        channels=[NotificationTemplateChannel.WEBSOCKET],
        cooldown_seconds=10,
    ),
    NotificationTemplate(
        template_id="tpl-geofence-breach",
        name="Geofence Breach",
        event_type="geofence_breach",
        title_template="Geofence Breach: {target_id}",
        body_template="Target {target_id} has {direction} zone {zone_name}.",
        severity=NotificationTemplateSeverity.ERROR,
        channels=[
            NotificationTemplateChannel.WEBSOCKET,
            NotificationTemplateChannel.MQTT,
            NotificationTemplateChannel.EMAIL,
        ],
        cooldown_seconds=30,
    ),
    NotificationTemplate(
        template_id="tpl-ble-first-seen",
        name="BLE Device First Seen",
        event_type="ble:first_seen",
        title_template="New Device First Seen",
        body_template="New device first seen: {name}, RSSI {rssi}dBm.",
        severity=NotificationTemplateSeverity.INFO,
        channels=[NotificationTemplateChannel.WEBSOCKET],
        cooldown_seconds=5,
    ),
    NotificationTemplate(
        template_id="tpl-convoy-detected",
        name="Convoy Detected",
        event_type="convoy_detected",
        title_template="Convoy Detected: {member_count} targets",
        body_template=(
            "Convoy of {member_count} targets detected moving together. "
            "Suspicious score: {suspicious_score}."
        ),
        severity=NotificationTemplateSeverity.WARNING,
        channels=[
            NotificationTemplateChannel.WEBSOCKET,
            NotificationTemplateChannel.MQTT,
        ],
        cooldown_seconds=60,
    ),
]
