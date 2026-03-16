# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for NotificationTemplate model."""

from tritium_lib.models.notification_template import (
    BUILTIN_NOTIFICATION_TEMPLATES,
    NotificationTemplate,
    NotificationTemplateChannel,
    NotificationTemplateSeverity,
)


def test_default_creation():
    t = NotificationTemplate(template_id="t1", name="Test")
    assert t.template_id == "t1"
    assert t.name == "Test"
    assert t.enabled is True
    assert t.cooldown_seconds == 60
    assert t.severity == NotificationTemplateSeverity.INFO
    assert t.channels == [NotificationTemplateChannel.WEBSOCKET]
    assert t.created_at is not None
    assert t.updated_at is not None


def test_render_title():
    t = NotificationTemplate(
        template_id="t1",
        title_template="Alert: {device_id} is {status}",
    )
    result = t.render_title(device_id="node-1", status="offline")
    assert result == "Alert: node-1 is offline"


def test_render_body():
    t = NotificationTemplate(
        template_id="t1",
        body_template="Device {device_id} battery at {pct}%.",
    )
    result = t.render_body(device_id="node-2", pct="15")
    assert result == "Device node-2 battery at 15%."


def test_render_both():
    t = NotificationTemplate(
        template_id="t1",
        title_template="Title: {x}",
        body_template="Body: {x}",
    )
    out = t.render(x="hello")
    assert out == {"title": "Title: hello", "body": "Body: hello"}


def test_render_unknown_placeholders_left():
    t = NotificationTemplate(
        template_id="t1",
        title_template="Hello {name}, {unknown}!",
    )
    result = t.render_title(name="world")
    assert result == "Hello world, {unknown}!"


def test_matches_event():
    t = NotificationTemplate(
        template_id="t1",
        event_type="node_offline",
        enabled=True,
    )
    assert t.matches_event("node_offline") is True
    assert t.matches_event("battery_low") is False


def test_matches_event_disabled():
    t = NotificationTemplate(
        template_id="t1",
        event_type="node_offline",
        enabled=False,
    )
    assert t.matches_event("node_offline") is False


def test_severity_rank():
    assert NotificationTemplateSeverity.DEBUG.rank == 0
    assert NotificationTemplateSeverity.CRITICAL.rank == 4


def test_channels_enum():
    assert NotificationTemplateChannel.WEBSOCKET.value == "websocket"
    assert NotificationTemplateChannel.MQTT.value == "mqtt"
    assert NotificationTemplateChannel.EMAIL.value == "email"


def test_to_dict():
    t = NotificationTemplate(
        template_id="t1",
        name="Test",
        event_type="battery_low",
        title_template="Low: {device_id}",
        body_template="At {pct}%",
        severity=NotificationTemplateSeverity.WARNING,
        channels=[NotificationTemplateChannel.WEBSOCKET, NotificationTemplateChannel.EMAIL],
        cooldown_seconds=120,
        enabled=True,
    )
    d = t.to_dict()
    assert d["template_id"] == "t1"
    assert d["severity"] == "warning"
    assert d["channels"] == ["websocket", "email"]
    assert d["cooldown_seconds"] == 120
    assert d["created_at"] is not None


def test_from_dict():
    data = {
        "template_id": "t2",
        "name": "Round-trip",
        "event_type": "target_new",
        "title_template": "New: {target_id}",
        "body_template": "Detected {target_id}",
        "severity": "error",
        "channels": ["mqtt", "email"],
        "cooldown_seconds": 30,
        "enabled": False,
    }
    t = NotificationTemplate.from_dict(data)
    assert t.template_id == "t2"
    assert t.severity == NotificationTemplateSeverity.ERROR
    assert t.channels == [NotificationTemplateChannel.MQTT, NotificationTemplateChannel.EMAIL]
    assert t.cooldown_seconds == 30
    assert t.enabled is False


def test_round_trip():
    original = NotificationTemplate(
        template_id="rt1",
        name="Roundtrip",
        event_type="geofence_breach",
        title_template="Breach: {target_id}",
        body_template="Zone {zone} breached by {target_id}",
        severity=NotificationTemplateSeverity.CRITICAL,
        channels=[
            NotificationTemplateChannel.WEBSOCKET,
            NotificationTemplateChannel.MQTT,
            NotificationTemplateChannel.EMAIL,
        ],
        cooldown_seconds=15,
    )
    d = original.to_dict()
    restored = NotificationTemplate.from_dict(d)
    assert restored.template_id == original.template_id
    assert restored.event_type == original.event_type
    assert restored.severity == original.severity
    assert len(restored.channels) == 3


def test_builtin_templates():
    assert len(BUILTIN_NOTIFICATION_TEMPLATES) >= 4
    for tpl in BUILTIN_NOTIFICATION_TEMPLATES:
        assert tpl.template_id != ""
        assert tpl.event_type != ""
        assert tpl.title_template != ""
        assert tpl.body_template != ""
        assert len(tpl.channels) >= 1


def test_multiple_channels():
    t = NotificationTemplate(
        template_id="mc",
        channels=[
            NotificationTemplateChannel.WEBSOCKET,
            NotificationTemplateChannel.MQTT,
            NotificationTemplateChannel.EMAIL,
        ],
    )
    assert len(t.channels) == 3
