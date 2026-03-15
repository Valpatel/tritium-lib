# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Quick action model — tactical actions taken on targets.

Records operator actions like investigate, watch, classify, track
for audit logging and operational awareness.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class QuickActionType(str, Enum):
    """Types of quick tactical actions an operator can take on a target."""
    INVESTIGATE = "investigate"
    WATCH = "watch"
    CLASSIFY = "classify"
    TRACK = "track"
    DISMISS = "dismiss"
    ESCALATE = "escalate"
    ANNOTATE = "annotate"


class QuickAction(BaseModel):
    """A tactical action taken on a target by an operator.

    Logged for audit trail and operational awareness. Each action
    captures what was done, to which target, by whom, and when.
    """
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: QuickActionType
    target_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    operator: str = "system"
    timestamp: float = Field(default_factory=time.time)
    notes: str = ""

    def to_event(self) -> dict[str, Any]:
        """Convert to event bus payload."""
        return {
            "event_type": "quick_action",
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "target_id": self.target_id,
            "params": self.params,
            "operator": self.operator,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


class QuickActionLog(BaseModel):
    """Collection of quick actions with query helpers."""
    actions: list[QuickAction] = Field(default_factory=list)
    max_size: int = 1000

    def add(self, action: QuickAction) -> None:
        """Add an action, trimming oldest if at capacity."""
        self.actions.append(action)
        if len(self.actions) > self.max_size:
            self.actions = self.actions[-self.max_size:]

    def for_target(self, target_id: str) -> list[QuickAction]:
        """Get all actions for a specific target."""
        return [a for a in self.actions if a.target_id == target_id]

    def by_type(self, action_type: QuickActionType) -> list[QuickAction]:
        """Get all actions of a specific type."""
        return [a for a in self.actions if a.action_type == action_type]

    def recent(self, limit: int = 50) -> list[QuickAction]:
        """Get most recent actions."""
        return list(reversed(self.actions[-limit:]))
