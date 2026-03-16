# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Target group model — operator-defined collections of tracked targets.

Operators can create named groups (e.g., "Building A devices", "Patrol route
suspects") and add/remove targets. Groups persist and can be filtered on the
tactical map. Each group has a color and icon for visual distinction.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class TargetGroup(BaseModel):
    """A named collection of tracked targets.

    Attributes:
        group_id: Unique identifier for the group.
        name: Human-readable group name.
        description: Optional description of the group's purpose.
        target_ids: Set of target IDs belonging to this group.
        created_by: Operator who created the group.
        color: Display color for the group on the map (hex).
        icon: Icon identifier for map markers.
        created_at: When the group was created.
        updated_at: When the group was last modified.
    """

    group_id: str = ""
    name: str = ""
    description: str = ""
    target_ids: list[str] = Field(default_factory=list)
    created_by: str = "operator"
    color: str = "#00f0ff"
    icon: str = "group"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def model_post_init(self, __context) -> None:
        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

    @property
    def target_count(self) -> int:
        """Number of targets in this group."""
        return len(self.target_ids)

    def add_target(self, target_id: str) -> bool:
        """Add a target to the group. Returns True if added, False if already present."""
        if target_id not in self.target_ids:
            self.target_ids.append(target_id)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def remove_target(self, target_id: str) -> bool:
        """Remove a target from the group. Returns True if removed, False if not found."""
        if target_id in self.target_ids:
            self.target_ids.remove(target_id)
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def has_target(self, target_id: str) -> bool:
        """Check if a target is in this group."""
        return target_id in self.target_ids


class TargetGroupSummary(BaseModel):
    """Lightweight summary of a target group for listing endpoints."""

    group_id: str = ""
    name: str = ""
    description: str = ""
    target_count: int = 0
    color: str = "#00f0ff"
    icon: str = "group"
    created_by: str = "operator"

    @classmethod
    def from_group(cls, group: TargetGroup) -> TargetGroupSummary:
        return cls(
            group_id=group.group_id,
            name=group.name,
            description=group.description,
            target_count=group.target_count,
            color=group.color,
            icon=group.icon,
            created_by=group.created_by,
        )
