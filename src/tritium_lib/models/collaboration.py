# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Collaborative workspace models for multi-operator investigation editing.

SharedWorkspace tracks an investigation being collaboratively edited by
multiple operators.  WorkspaceEvent records individual changes (add entity,
annotate, change status).  OperatorAction captures who did what and when.
OperatorChatMessage represents text messages between operators.
MapDrawing represents collaborative map annotations shared in real time.
"""

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorkspaceEventType(str, Enum):
    """Types of changes that can occur in a shared workspace."""
    ENTITY_ADDED = "entity_added"
    ENTITY_REMOVED = "entity_removed"
    ANNOTATION_ADDED = "annotation_added"
    STATUS_CHANGED = "status_changed"
    TITLE_CHANGED = "title_changed"
    OPERATOR_JOINED = "operator_joined"
    OPERATOR_LEFT = "operator_left"
    CURSOR_MOVED = "cursor_moved"
    SELECTION_CHANGED = "selection_changed"


class DrawingType(str, Enum):
    """Types of map drawings that can be collaboratively shared."""
    FREEHAND = "freehand"
    LINE = "line"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    MEASUREMENT = "measurement"
    GEOFENCE = "geofence"
    TEXT = "text"
    ARROW = "arrow"


class ChatMessageType(str, Enum):
    """Types of operator chat messages."""
    TEXT = "text"
    ALERT = "alert"
    SYSTEM = "system"
    COMMAND = "command"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class OperatorAction(BaseModel):
    """Record of a single operator action in a workspace."""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    operator_id: str
    operator_name: str = ""
    action_type: WorkspaceEventType
    timestamp: float = Field(default_factory=time.time)
    details: dict[str, Any] = Field(default_factory=dict)


class WorkspaceEvent(BaseModel):
    """An event broadcast to all operators viewing a shared workspace.

    Contains the action that occurred plus the workspace context so
    receivers can apply the change to their local view.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    workspace_id: str
    event_type: WorkspaceEventType
    operator_id: str
    operator_name: str = ""
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)
    # Entity ID affected by this event (if applicable)
    entity_id: Optional[str] = None
    # Previous value for undo support
    previous_value: Optional[Any] = None


class SharedWorkspace(BaseModel):
    """A collaborative investigation workspace.

    Tracks which operators are currently viewing the workspace and
    maintains the event history for late-joining operators to catch up.
    """
    workspace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    investigation_id: str
    title: str = ""
    created_at: float = Field(default_factory=time.time)
    # Operator IDs currently connected
    active_operators: list[str] = Field(default_factory=list)
    # Recent events for catch-up (capped ring buffer in implementation)
    recent_events: list[WorkspaceEvent] = Field(default_factory=list)
    # Version counter for conflict detection
    version: int = 0


class MapDrawing(BaseModel):
    """A collaborative map drawing shared between operators in real time."""
    drawing_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    drawing_type: DrawingType
    operator_id: str
    operator_name: str = ""
    color: str = "#00f0ff"
    # GeoJSON-style coordinates: [[lng, lat], ...]
    coordinates: list[list[float]] = Field(default_factory=list)
    # Properties specific to drawing type
    radius: Optional[float] = None  # For circles, in meters
    text: Optional[str] = None  # For text annotations
    label: Optional[str] = None
    line_width: float = 2.0
    opacity: float = 0.8
    # Timestamps
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    # Layer grouping
    layer: str = "default"
    # Transient vs persistent
    persistent: bool = False


class OperatorChatMessage(BaseModel):
    """A text message between operators for coordination."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    operator_id: str
    operator_name: str = ""
    message_type: ChatMessageType = ChatMessageType.TEXT
    content: str
    timestamp: float = Field(default_factory=time.time)
    # Optional channel/room for future multi-channel support
    channel: str = "general"
    # Reference to investigation or workspace
    workspace_id: Optional[str] = None
