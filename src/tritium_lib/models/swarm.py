# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Swarm coordination models for multi-robot operations.

Defines formations, commands, and member roles for coordinating groups of
robots (rovers, drones, turrets) as a cohesive swarm unit.

Formations:
  - LINE: robots in a straight line perpendicular to heading
  - WEDGE: V-shaped formation (lead unit at point)
  - CIRCLE: defensive perimeter around a center point
  - DIAMOND: 4-unit diamond with lead, flanks, and rear
  - COLUMN: single file along heading direction
  - STAGGERED: offset line for wider coverage

Commands:
  - ADVANCE: move formation toward waypoint
  - RETREAT: reverse formation away from threat
  - SPREAD: increase inter-unit spacing
  - CONVERGE: decrease spacing / collapse to center
  - HOLD: maintain current formation and position
  - PATROL: cycle through waypoints in formation
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SwarmFormationType(str, Enum):
    """Available swarm formation patterns."""
    LINE = "line"
    WEDGE = "wedge"
    CIRCLE = "circle"
    DIAMOND = "diamond"
    COLUMN = "column"
    STAGGERED = "staggered"


class SwarmCommandType(str, Enum):
    """Available swarm movement commands."""
    ADVANCE = "advance"
    RETREAT = "retreat"
    SPREAD = "spread"
    CONVERGE = "converge"
    HOLD = "hold"
    PATROL = "patrol"


class SwarmRole(str, Enum):
    """Roles within a swarm formation."""
    LEAD = "lead"
    FLANK_LEFT = "flank_left"
    FLANK_RIGHT = "flank_right"
    REAR = "rear"
    CENTER = "center"
    SCOUT = "scout"
    SUPPORT = "support"


class SwarmMemberStatus(str, Enum):
    """Status of a swarm member."""
    ACTIVE = "active"
    REPOSITIONING = "repositioning"
    DISABLED = "disabled"
    DISCONNECTED = "disconnected"
    RETURNING = "returning"


class SwarmMember(BaseModel):
    """A single member of a swarm unit.

    Each member has a role within the formation and maintains its own
    position relative to the formation center.
    """
    member_id: str
    device_id: str = ""
    asset_type: str = "rover"  # rover, drone, turret, etc.
    role: SwarmRole = SwarmRole.SUPPORT
    status: SwarmMemberStatus = SwarmMemberStatus.ACTIVE

    # Current position
    position_x: float = 0.0
    position_y: float = 0.0
    heading: float = 0.0

    # Formation-relative offset from center
    formation_offset_x: float = 0.0
    formation_offset_y: float = 0.0

    # Telemetry
    battery: float = 1.0
    speed: float = 0.0
    last_seen: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Capabilities
    has_camera: bool = False
    has_weapon: bool = False
    has_sensor: bool = False
    max_speed: float = 2.0  # m/s


class SwarmFormation(BaseModel):
    """A formation configuration for a swarm.

    Defines how members are arranged relative to the formation center,
    spacing between units, and the formation heading.
    """
    formation_type: SwarmFormationType = SwarmFormationType.LINE
    heading: float = 0.0  # degrees, formation facing direction
    spacing: float = 5.0  # meters between units
    center_x: float = 0.0
    center_y: float = 0.0

    # Per-member offsets computed from formation type and spacing
    member_offsets: dict[str, tuple[float, float]] = Field(default_factory=dict)

    def compute_offsets(self, member_ids: list[str]) -> dict[str, tuple[float, float]]:
        """Compute formation offsets for each member based on formation type.

        Args:
            member_ids: List of member IDs to assign positions.

        Returns:
            Dict mapping member_id -> (offset_x, offset_y) relative to center.
        """
        import math
        n = len(member_ids)
        offsets: dict[str, tuple[float, float]] = {}

        if n == 0:
            return offsets

        rad = math.radians(self.heading)
        cos_h = math.cos(rad)
        sin_h = math.sin(rad)

        if self.formation_type == SwarmFormationType.LINE:
            # Perpendicular to heading
            for i, mid in enumerate(member_ids):
                offset = (i - (n - 1) / 2.0) * self.spacing
                ox = -offset * sin_h
                oy = offset * cos_h
                offsets[mid] = (round(ox, 2), round(oy, 2))

        elif self.formation_type == SwarmFormationType.WEDGE:
            # V-shape: lead at front, others behind and to sides
            offsets[member_ids[0]] = (0.0, 0.0)  # lead at point
            for i, mid in enumerate(member_ids[1:], 1):
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                ox = -row * self.spacing * cos_h + side * row * self.spacing * sin_h * 0.5
                oy = -row * self.spacing * sin_h - side * row * self.spacing * cos_h * 0.5
                offsets[mid] = (round(ox, 2), round(oy, 2))

        elif self.formation_type == SwarmFormationType.CIRCLE:
            # Equal spacing around a circle
            radius = self.spacing * max(1, n) / (2 * math.pi) if n > 1 else 0
            for i, mid in enumerate(member_ids):
                angle = 2 * math.pi * i / n
                ox = radius * math.cos(angle)
                oy = radius * math.sin(angle)
                offsets[mid] = (round(ox, 2), round(oy, 2))

        elif self.formation_type == SwarmFormationType.DIAMOND:
            # Diamond: front, left, right, rear (+ extras behind)
            positions = [
                (self.spacing, 0),       # front
                (0, -self.spacing),      # left
                (0, self.spacing),       # right
                (-self.spacing, 0),      # rear
            ]
            for i, mid in enumerate(member_ids):
                if i < len(positions):
                    fx, fy = positions[i]
                else:
                    # Extra members line up behind
                    fx = -self.spacing * (1 + (i - 3))
                    fy = 0
                # Rotate by heading
                ox = fx * cos_h - fy * sin_h
                oy = fx * sin_h + fy * cos_h
                offsets[mid] = (round(ox, 2), round(oy, 2))

        elif self.formation_type == SwarmFormationType.COLUMN:
            # Single file along heading
            for i, mid in enumerate(member_ids):
                dist = -i * self.spacing  # behind lead
                ox = dist * cos_h
                oy = dist * sin_h
                offsets[mid] = (round(ox, 2), round(oy, 2))

        elif self.formation_type == SwarmFormationType.STAGGERED:
            # Offset line for wider coverage
            for i, mid in enumerate(member_ids):
                row = i // 2
                side = 1 if i % 2 == 0 else -1
                fx = -row * self.spacing
                fy = side * self.spacing * 0.5
                ox = fx * cos_h - fy * sin_h
                oy = fx * sin_h + fy * cos_h
                offsets[mid] = (round(ox, 2), round(oy, 2))

        self.member_offsets = offsets
        return offsets


class SwarmCommand(BaseModel):
    """A command issued to a swarm unit.

    Commands modify formation, position, or behavior of the entire swarm.
    Individual member tasking goes through separate channels.
    """
    command_id: str = ""
    swarm_id: str = ""
    command_type: SwarmCommandType = SwarmCommandType.HOLD
    formation: Optional[SwarmFormationType] = None

    # Target waypoint for ADVANCE/RETREAT/PATROL
    waypoint_x: float = 0.0
    waypoint_y: float = 0.0
    waypoints: list[tuple[float, float]] = Field(default_factory=list)

    # Spacing override
    spacing: Optional[float] = None

    # Speed limit for movement commands
    max_speed: float = 2.0

    # Priority (higher = more urgent)
    priority: int = 5

    # Command metadata
    issued_by: str = ""
    issued_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    expires_at: Optional[datetime] = None
    description: str = ""


class SwarmStatus(BaseModel):
    """Current status of a swarm unit."""
    swarm_id: str
    name: str = ""
    formation: SwarmFormationType = SwarmFormationType.LINE
    center_x: float = 0.0
    center_y: float = 0.0
    heading: float = 0.0
    spacing: float = 5.0
    member_count: int = 0
    active_members: int = 0
    current_command: Optional[SwarmCommandType] = None
    avg_battery: float = 1.0
    members: list[SwarmMember] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
