# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Device lifecycle management models.

Standardizes device state tracking across the fleet. Each device moves
through a lifecycle: provisioning -> active -> maintenance -> retired.
State transitions are recorded as events for audit and fleet health.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DeviceState(str, Enum):
    """Lifecycle state of a fleet device."""
    PROVISIONING = "provisioning"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    RETIRED = "retired"
    ERROR = "error"


# Valid state transitions: from_state -> set of allowed to_states
VALID_TRANSITIONS: dict[DeviceState, set[DeviceState]] = {
    DeviceState.PROVISIONING: {DeviceState.ACTIVE, DeviceState.ERROR, DeviceState.RETIRED},
    DeviceState.ACTIVE: {DeviceState.MAINTENANCE, DeviceState.RETIRED, DeviceState.ERROR},
    DeviceState.MAINTENANCE: {DeviceState.ACTIVE, DeviceState.RETIRED, DeviceState.ERROR},
    DeviceState.RETIRED: {DeviceState.PROVISIONING},  # Can be re-provisioned
    DeviceState.ERROR: {DeviceState.MAINTENANCE, DeviceState.PROVISIONING, DeviceState.RETIRED},
}


def is_valid_transition(from_state: DeviceState, to_state: DeviceState) -> bool:
    """Check if a state transition is allowed."""
    allowed = VALID_TRANSITIONS.get(from_state, set())
    return to_state in allowed


class DeviceLifecycleEvent(BaseModel):
    """Record of a device state transition."""
    device_id: str
    from_state: DeviceState
    to_state: DeviceState
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""
    operator: str = ""  # Who initiated the transition (user, system, auto)


class DeviceProvisioningConfig(BaseModel):
    """Configuration applied when a device is provisioned."""
    device_id: str
    device_name: str = ""
    device_group: str = ""
    firmware_target: str = ""  # Target firmware version
    config_template: str = ""  # Config template to apply
    site_id: str = "home"
    location_description: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None
    provisioned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provisioned_by: str = "system"


class DeviceLifecycleStatus(BaseModel):
    """Current lifecycle status of a device."""
    device_id: str
    state: DeviceState = DeviceState.PROVISIONING
    state_since: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provisioning_config: Optional[DeviceProvisioningConfig] = None
    transition_count: int = 0
    last_error: str = ""
    history: list[DeviceLifecycleEvent] = Field(default_factory=list)


class FleetLifecycleSummary(BaseModel):
    """Aggregate lifecycle state counts for the fleet."""
    total: int = 0
    provisioning: int = 0
    active: int = 0
    maintenance: int = 0
    retired: int = 0
    error: int = 0

    @classmethod
    def from_states(cls, states: list[DeviceState]) -> FleetLifecycleSummary:
        """Build summary from a list of device states."""
        summary = cls(total=len(states))
        for s in states:
            if s == DeviceState.PROVISIONING:
                summary.provisioning += 1
            elif s == DeviceState.ACTIVE:
                summary.active += 1
            elif s == DeviceState.MAINTENANCE:
                summary.maintenance += 1
            elif s == DeviceState.RETIRED:
                summary.retired += 1
            elif s == DeviceState.ERROR:
                summary.error += 1
        return summary
