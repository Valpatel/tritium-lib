# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for collaboration models."""

import time
from tritium_lib.models.collaboration import (
    ChatMessageType,
    DrawingType,
    MapDrawing,
    OperatorAction,
    OperatorChatMessage,
    SharedWorkspace,
    WorkspaceEvent,
    WorkspaceEventType,
)


def test_workspace_event_type_enum():
    assert WorkspaceEventType.ENTITY_ADDED == "entity_added"
    assert WorkspaceEventType.OPERATOR_JOINED == "operator_joined"
    assert WorkspaceEventType.STATUS_CHANGED == "status_changed"


def test_drawing_type_enum():
    assert DrawingType.FREEHAND == "freehand"
    assert DrawingType.GEOFENCE == "geofence"
    assert DrawingType.MEASUREMENT == "measurement"


def test_chat_message_type_enum():
    assert ChatMessageType.TEXT == "text"
    assert ChatMessageType.ALERT == "alert"
    assert ChatMessageType.COMMAND == "command"


def test_operator_action_defaults():
    action = OperatorAction(
        operator_id="op1",
        action_type=WorkspaceEventType.ENTITY_ADDED,
    )
    assert action.operator_id == "op1"
    assert action.action_type == WorkspaceEventType.ENTITY_ADDED
    assert len(action.action_id) > 0
    assert action.timestamp > 0
    assert action.details == {}


def test_workspace_event_creation():
    event = WorkspaceEvent(
        workspace_id="ws-123",
        event_type=WorkspaceEventType.ANNOTATION_ADDED,
        operator_id="op1",
        operator_name="Alice",
        entity_id="ble_aa:bb:cc",
        data={"note": "Suspicious activity"},
    )
    assert event.workspace_id == "ws-123"
    assert event.event_type == WorkspaceEventType.ANNOTATION_ADDED
    assert event.entity_id == "ble_aa:bb:cc"
    assert event.data["note"] == "Suspicious activity"


def test_shared_workspace_defaults():
    ws = SharedWorkspace(investigation_id="inv-001")
    assert ws.investigation_id == "inv-001"
    assert len(ws.workspace_id) > 0
    assert ws.active_operators == []
    assert ws.recent_events == []
    assert ws.version == 0
    assert ws.created_at > 0


def test_map_drawing_creation():
    drawing = MapDrawing(
        drawing_type=DrawingType.CIRCLE,
        operator_id="op2",
        operator_name="Bob",
        coordinates=[[40.7128, -74.0060]],
        radius=50.0,
        color="#ff2a6d",
        label="Perimeter",
        persistent=True,
    )
    assert drawing.drawing_type == DrawingType.CIRCLE
    assert drawing.radius == 50.0
    assert drawing.color == "#ff2a6d"
    assert drawing.persistent is True
    assert len(drawing.drawing_id) > 0


def test_map_drawing_defaults():
    drawing = MapDrawing(
        drawing_type=DrawingType.FREEHAND,
        operator_id="op1",
    )
    assert drawing.color == "#00f0ff"
    assert drawing.line_width == 2.0
    assert drawing.opacity == 0.8
    assert drawing.layer == "default"
    assert drawing.persistent is False


def test_operator_chat_message():
    msg = OperatorChatMessage(
        operator_id="op1",
        operator_name="Alice",
        content="Target moving south",
        channel="tactical",
    )
    assert msg.content == "Target moving south"
    assert msg.channel == "tactical"
    assert msg.message_type == ChatMessageType.TEXT
    assert len(msg.message_id) > 0


def test_chat_message_alert_type():
    msg = OperatorChatMessage(
        operator_id="op1",
        content="Contact!",
        message_type=ChatMessageType.ALERT,
    )
    assert msg.message_type == ChatMessageType.ALERT


def test_workspace_event_serialization():
    event = WorkspaceEvent(
        workspace_id="ws-001",
        event_type=WorkspaceEventType.ENTITY_ADDED,
        operator_id="op1",
        data={"entity_id": "ble_aa:bb:cc"},
    )
    d = event.model_dump()
    assert d["workspace_id"] == "ws-001"
    assert d["event_type"] == "entity_added"
    assert d["data"]["entity_id"] == "ble_aa:bb:cc"


def test_shared_workspace_serialization():
    ws = SharedWorkspace(
        investigation_id="inv-001",
        title="Test Investigation",
        active_operators=["op1", "op2"],
        version=5,
    )
    d = ws.model_dump()
    assert d["investigation_id"] == "inv-001"
    assert d["title"] == "Test Investigation"
    assert len(d["active_operators"]) == 2
    assert d["version"] == 5


def test_imports_from_init():
    """Verify collaboration models are importable from the models package."""
    from tritium_lib.models import (
        ChatMessageType,
        DrawingType,
        MapDrawing,
        OperatorAction,
        OperatorChatMessage,
        SharedWorkspace,
        WorkspaceEvent,
        WorkspaceEventType,
    )
    assert SharedWorkspace is not None
    assert WorkspaceEvent is not None
