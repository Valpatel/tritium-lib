# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""User and operator models for multi-user session management.

Defines the User, UserRole, Permission, and UserSession models used by
tritium-sc's session management system. Multiple operators can connect
simultaneously with different roles and permission sets.

Roles:
    admin      — full system access, user management
    commander  — tactical control, target engagement, mission management
    analyst    — intelligence, investigations, dossiers, enrichment
    operator   — device management, fleet control, sensor config
    observer   — read-only view of the operating picture
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class UserRole(str, Enum):
    """Operator role within the Tritium system."""
    ADMIN = "admin"
    COMMANDER = "commander"
    ANALYST = "analyst"
    OPERATOR = "operator"
    OBSERVER = "observer"


class Permission(str, Enum):
    """Granular permissions for restricting panel/API access."""
    # Target operations
    TARGETS_VIEW = "targets.view"
    TARGETS_EDIT = "targets.edit"
    TARGETS_ENGAGE = "targets.engage"

    # Mission operations
    MISSIONS_VIEW = "missions.view"
    MISSIONS_MANAGE = "missions.manage"

    # Fleet operations
    FLEET_VIEW = "fleet.view"
    FLEET_COMMAND = "fleet.command"
    FLEET_OTA = "fleet.ota"

    # Intelligence
    INTEL_VIEW = "intel.view"
    INTEL_CLASSIFY = "intel.classify"
    INTEL_INVESTIGATE = "intel.investigate"

    # Camera / sensors
    SENSORS_VIEW = "sensors.view"
    SENSORS_CONFIGURE = "sensors.configure"

    # System administration
    SYSTEM_CONFIG = "system.config"
    SYSTEM_USERS = "system.users"
    SYSTEM_AUDIT = "system.audit"

    # Automation / rules
    AUTOMATION_VIEW = "automation.view"
    AUTOMATION_MANAGE = "automation.manage"

    # Briefings / reports
    BRIEFINGS_VIEW = "briefings.view"
    BRIEFINGS_GENERATE = "briefings.generate"

    # Amy / AI
    AMY_CHAT = "amy.chat"
    AMY_COMMAND = "amy.command"


# Default permission sets per role
ROLE_PERMISSIONS: dict[UserRole, set[Permission]] = {
    UserRole.ADMIN: set(Permission),  # all permissions
    UserRole.COMMANDER: {
        Permission.TARGETS_VIEW,
        Permission.TARGETS_EDIT,
        Permission.TARGETS_ENGAGE,
        Permission.MISSIONS_VIEW,
        Permission.MISSIONS_MANAGE,
        Permission.FLEET_VIEW,
        Permission.FLEET_COMMAND,
        Permission.INTEL_VIEW,
        Permission.INTEL_CLASSIFY,
        Permission.SENSORS_VIEW,
        Permission.SENSORS_CONFIGURE,
        Permission.AUTOMATION_VIEW,
        Permission.AUTOMATION_MANAGE,
        Permission.BRIEFINGS_VIEW,
        Permission.BRIEFINGS_GENERATE,
        Permission.AMY_CHAT,
        Permission.AMY_COMMAND,
    },
    UserRole.ANALYST: {
        Permission.TARGETS_VIEW,
        Permission.TARGETS_EDIT,
        Permission.MISSIONS_VIEW,
        Permission.FLEET_VIEW,
        Permission.INTEL_VIEW,
        Permission.INTEL_CLASSIFY,
        Permission.INTEL_INVESTIGATE,
        Permission.SENSORS_VIEW,
        Permission.BRIEFINGS_VIEW,
        Permission.BRIEFINGS_GENERATE,
        Permission.AMY_CHAT,
        Permission.SYSTEM_AUDIT,
    },
    UserRole.OPERATOR: {
        Permission.TARGETS_VIEW,
        Permission.FLEET_VIEW,
        Permission.FLEET_COMMAND,
        Permission.FLEET_OTA,
        Permission.SENSORS_VIEW,
        Permission.SENSORS_CONFIGURE,
        Permission.AUTOMATION_VIEW,
        Permission.BRIEFINGS_VIEW,
        Permission.AMY_CHAT,
    },
    UserRole.OBSERVER: {
        Permission.TARGETS_VIEW,
        Permission.MISSIONS_VIEW,
        Permission.FLEET_VIEW,
        Permission.INTEL_VIEW,
        Permission.SENSORS_VIEW,
        Permission.BRIEFINGS_VIEW,
    },
}


@dataclass
class User:
    """A Tritium operator account.

    Attributes:
        user_id: Unique identifier for this user.
        username: Login name.
        display_name: Human-readable display name.
        role: Operator role (admin, commander, analyst, operator, observer).
        permissions: Explicit permission overrides (if empty, uses role defaults).
        active_since: When the user account was created.
        last_action: Timestamp of the user's most recent action.
        email: Optional contact email.
        color: Hex color for cursor sharing and UI identification.
    """
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    display_name: str = ""
    role: UserRole = UserRole.OBSERVER
    permissions: set[str] = field(default_factory=set)
    active_since: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_action: Optional[datetime] = None
    email: str = ""
    color: str = "#00f0ff"  # cyan default

    def has_permission(self, perm: Permission | str) -> bool:
        """Check if this user has a specific permission.

        Uses explicit permissions if set, otherwise falls back to role defaults.
        """
        perm_str = perm.value if isinstance(perm, Permission) else perm

        # Explicit overrides
        if self.permissions:
            return perm_str in self.permissions

        # Role defaults
        role_perms = ROLE_PERMISSIONS.get(self.role, set())
        return any(p.value == perm_str for p in role_perms)

    def get_effective_permissions(self) -> set[str]:
        """Get the full set of effective permissions for this user."""
        if self.permissions:
            return set(self.permissions)
        role_perms = ROLE_PERMISSIONS.get(self.role, set())
        return {p.value for p in role_perms}

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON transport."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role.value,
            "permissions": sorted(self.get_effective_permissions()),
            "active_since": self.active_since.isoformat(),
            "last_action": self.last_action.isoformat() if self.last_action else None,
            "email": self.email,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: dict) -> User:
        """Deserialize from a dict."""
        return cls(
            user_id=data.get("user_id", str(uuid.uuid4())),
            username=data.get("username", ""),
            display_name=data.get("display_name", ""),
            role=UserRole(data.get("role", "observer")),
            permissions=set(data.get("permissions", [])),
            active_since=datetime.fromisoformat(data["active_since"]) if data.get("active_since") else datetime.now(timezone.utc),
            last_action=datetime.fromisoformat(data["last_action"]) if data.get("last_action") else None,
            email=data.get("email", ""),
            color=data.get("color", "#00f0ff"),
        )


@dataclass
class UserSession:
    """An active operator session.

    Tracks a logged-in operator's state including their panel layout
    preferences, notification settings, and cursor position.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    username: str = ""
    display_name: str = ""
    role: UserRole = UserRole.OBSERVER
    color: str = "#00f0ff"
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: str = ""
    user_agent: str = ""

    # Panel layout preferences (stored per-session)
    panel_layout: dict = field(default_factory=dict)
    notification_prefs: dict = field(default_factory=dict)

    # Cursor position for real-time sharing
    cursor_lat: Optional[float] = None
    cursor_lng: Optional[float] = None

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Serialize to dict for API responses."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role.value,
            "color": self.color,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "ip_address": self.ip_address,
            "cursor_lat": self.cursor_lat,
            "cursor_lng": self.cursor_lng,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UserSession:
        """Deserialize from a dict."""
        return cls(
            session_id=data.get("session_id", str(uuid.uuid4())),
            user_id=data.get("user_id", ""),
            username=data.get("username", ""),
            display_name=data.get("display_name", ""),
            role=UserRole(data.get("role", "observer")),
            color=data.get("color", "#00f0ff"),
            connected_at=datetime.fromisoformat(data["connected_at"]) if data.get("connected_at") else datetime.now(timezone.utc),
            last_activity=datetime.fromisoformat(data["last_activity"]) if data.get("last_activity") else datetime.now(timezone.utc),
            ip_address=data.get("ip_address", ""),
            cursor_lat=data.get("cursor_lat"),
            cursor_lng=data.get("cursor_lng"),
        )
