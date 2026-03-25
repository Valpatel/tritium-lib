# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fine-grained access control lists for Tritium resources.

Provides in-memory ACL management with optional JSON persistence.
Supports user-level and role-level permissions on typed resources.

Usage:
    from tritium_lib.auth.acl import ACLManager, Permission, ResourceType

    acl = ACLManager()
    acl.grant_role("operator", ResourceType.CAMERA, "*", {Permission.READ, Permission.OPERATE})
    acl.assign_role("alice", "operator")
    assert acl.check_access("alice", ResourceType.CAMERA, "cam_01", Permission.READ)
"""

import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Permission(str, Enum):
    """Actions that can be granted on a resource."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OPERATE = "operate"


class ResourceType(str, Enum):
    """Categories of resources that can be protected."""

    TARGET = "target"
    ZONE = "zone"
    CAMERA = "camera"
    REPORT = "report"
    INVESTIGATION = "investigation"


@dataclass
class ACLEntry:
    """A single access control entry.

    Maps a principal (user or role) to a specific resource and its
    granted permissions.

    Attributes:
        principal: User ID or role name.
        principal_type: Either "user" or "role".
        resource_type: The category of resource.
        resource_id: Specific resource ID, or "*" for all resources of that type.
        permissions: Set of granted permissions.
    """

    principal: str
    principal_type: str  # "user" or "role"
    resource_type: ResourceType
    resource_id: str  # "*" means all resources of this type
    permissions: set[Permission] = field(default_factory=set)


# Built-in role definitions
BUILTIN_ROLES: dict[str, dict[ResourceType, set[Permission]]] = {
    "admin": {
        rt: {Permission.READ, Permission.WRITE, Permission.ADMIN, Permission.OPERATE}
        for rt in ResourceType
    },
    "operator": {
        rt: {Permission.READ, Permission.OPERATE}
        for rt in ResourceType
    },
    "viewer": {
        rt: {Permission.READ}
        for rt in ResourceType
    },
}


class ACLManager:
    """Manages access control entries for users and roles.

    Thread-safe. Supports user-level grants, role-level grants,
    role assignment to users, built-in roles, and JSON persistence.

    Resolution order for check_access:
        1. Explicit user grant on (resource_type, resource_id)
        2. Explicit user grant on (resource_type, "*")
        3. Role grants (assigned roles checked in order)
        4. Built-in role definitions
        5. Deny by default
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # user -> set of role names
        self._user_roles: dict[str, set[str]] = {}
        # (principal, principal_type, resource_type, resource_id) -> ACLEntry
        self._entries: dict[tuple[str, str, ResourceType, str], ACLEntry] = {}
        # Custom role definitions: role_name -> {resource_type -> permissions}
        self._custom_roles: dict[str, dict[ResourceType, set[Permission]]] = {}

    # --- Role management ---

    def assign_role(self, user: str, role: str) -> None:
        """Assign a role to a user."""
        with self._lock:
            self._user_roles.setdefault(user, set()).add(role)

    def revoke_role(self, user: str, role: str) -> None:
        """Remove a role from a user."""
        with self._lock:
            if user in self._user_roles:
                self._user_roles[user].discard(role)

    def get_user_roles(self, user: str) -> set[str]:
        """Return the set of roles assigned to a user."""
        with self._lock:
            return set(self._user_roles.get(user, set()))

    def define_role(
        self,
        role: str,
        resource_type: ResourceType,
        permissions: set[Permission],
    ) -> None:
        """Define or update a custom role's permissions for a resource type."""
        with self._lock:
            self._custom_roles.setdefault(role, {})[resource_type] = set(permissions)

    # --- Grant / revoke on specific resources ---

    def grant(
        self,
        principal: str,
        principal_type: str,
        resource_type: ResourceType,
        resource_id: str,
        permissions: set[Permission],
    ) -> None:
        """Grant permissions to a principal on a resource.

        Args:
            principal: User ID or role name.
            principal_type: "user" or "role".
            resource_type: Category of the resource.
            resource_id: Specific ID or "*" for all.
            permissions: Set of permissions to grant.
        """
        if principal_type not in ("user", "role"):
            raise ValueError(f"principal_type must be 'user' or 'role', got '{principal_type}'")
        with self._lock:
            key = (principal, principal_type, resource_type, resource_id)
            if key in self._entries:
                self._entries[key].permissions |= permissions
            else:
                self._entries[key] = ACLEntry(
                    principal=principal,
                    principal_type=principal_type,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    permissions=set(permissions),
                )

    def grant_user(
        self,
        user: str,
        resource_type: ResourceType,
        resource_id: str,
        permissions: set[Permission],
    ) -> None:
        """Convenience: grant permissions to a user on a resource."""
        self.grant(user, "user", resource_type, resource_id, permissions)

    def grant_role(
        self,
        role: str,
        resource_type: ResourceType,
        resource_id: str,
        permissions: set[Permission],
    ) -> None:
        """Convenience: grant permissions to a role on a resource."""
        self.grant(role, "role", resource_type, resource_id, permissions)

    def revoke(
        self,
        principal: str,
        principal_type: str,
        resource_type: ResourceType,
        resource_id: str,
        permissions: Optional[set[Permission]] = None,
    ) -> None:
        """Revoke permissions from a principal on a resource.

        If permissions is None, removes the entire entry.
        Otherwise, removes only the specified permissions.
        """
        with self._lock:
            key = (principal, principal_type, resource_type, resource_id)
            if key not in self._entries:
                return
            if permissions is None:
                del self._entries[key]
            else:
                self._entries[key].permissions -= permissions
                if not self._entries[key].permissions:
                    del self._entries[key]

    # --- Access checking ---

    def check_access(
        self,
        user: str,
        resource_type: ResourceType,
        resource_id: str,
        permission: Permission,
    ) -> bool:
        """Check whether a user has a specific permission on a resource.

        Resolution order:
            1. Explicit user grant on (resource_type, resource_id)
            2. Explicit user wildcard grant on (resource_type, "*")
            3. Assigned roles — explicit role grants, then custom role defs, then built-in defs
            4. Deny
        """
        with self._lock:
            # 1. Exact user entry
            key_exact = (user, "user", resource_type, resource_id)
            entry = self._entries.get(key_exact)
            if entry and permission in entry.permissions:
                return True

            # 2. Wildcard user entry
            key_wild = (user, "user", resource_type, "*")
            entry = self._entries.get(key_wild)
            if entry and permission in entry.permissions:
                return True

            # 3. Check roles
            roles = self._user_roles.get(user, set())
            for role in roles:
                # 3a. Explicit role grant on exact resource
                rk_exact = (role, "role", resource_type, resource_id)
                entry = self._entries.get(rk_exact)
                if entry and permission in entry.permissions:
                    return True

                # 3b. Explicit role grant on wildcard resource
                rk_wild = (role, "role", resource_type, "*")
                entry = self._entries.get(rk_wild)
                if entry and permission in entry.permissions:
                    return True

                # 3c. Custom role definition
                custom = self._custom_roles.get(role, {})
                if resource_type in custom and permission in custom[resource_type]:
                    return True

                # 3d. Built-in role definition
                builtin = BUILTIN_ROLES.get(role, {})
                if resource_type in builtin and permission in builtin[resource_type]:
                    return True

            return False

    # --- Query helpers ---

    def get_entries(
        self,
        principal: Optional[str] = None,
        resource_type: Optional[ResourceType] = None,
    ) -> list[ACLEntry]:
        """Return ACL entries, optionally filtered by principal or resource type."""
        with self._lock:
            results = []
            for entry in self._entries.values():
                if principal and entry.principal != principal:
                    continue
                if resource_type and entry.resource_type != resource_type:
                    continue
                results.append(ACLEntry(
                    principal=entry.principal,
                    principal_type=entry.principal_type,
                    resource_type=entry.resource_type,
                    resource_id=entry.resource_id,
                    permissions=set(entry.permissions),
                ))
            return results

    def get_user_permissions(
        self,
        user: str,
        resource_type: ResourceType,
        resource_id: str,
    ) -> set[Permission]:
        """Return all effective permissions a user has on a resource."""
        result: set[Permission] = set()
        for perm in Permission:
            if self.check_access(user, resource_type, resource_id, perm):
                result.add(perm)
        return result

    # --- Persistence ---

    def save(self, path: str | Path) -> None:
        """Save all ACL state to a JSON file."""
        path = Path(path)
        with self._lock:
            data = {
                "user_roles": {
                    user: sorted(roles) for user, roles in self._user_roles.items()
                },
                "entries": [
                    {
                        "principal": e.principal,
                        "principal_type": e.principal_type,
                        "resource_type": e.resource_type.value,
                        "resource_id": e.resource_id,
                        "permissions": sorted(p.value for p in e.permissions),
                    }
                    for e in self._entries.values()
                ],
                "custom_roles": {
                    role: {
                        rt.value: sorted(p.value for p in perms)
                        for rt, perms in defs.items()
                    }
                    for role, defs in self._custom_roles.items()
                },
            }
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Load ACL state from a JSON file, replacing current state."""
        path = Path(path)
        raw = json.loads(path.read_text())
        with self._lock:
            self._user_roles.clear()
            self._entries.clear()
            self._custom_roles.clear()

            for user, roles in raw.get("user_roles", {}).items():
                self._user_roles[user] = set(roles)

            for item in raw.get("entries", []):
                rt = ResourceType(item["resource_type"])
                perms = {Permission(p) for p in item["permissions"]}
                key = (item["principal"], item["principal_type"], rt, item["resource_id"])
                self._entries[key] = ACLEntry(
                    principal=item["principal"],
                    principal_type=item["principal_type"],
                    resource_type=rt,
                    resource_id=item["resource_id"],
                    permissions=perms,
                )

            for role, defs in raw.get("custom_roles", {}).items():
                self._custom_roles[role] = {
                    ResourceType(rt): {Permission(p) for p in perms}
                    for rt, perms in defs.items()
                }

    def clear(self) -> None:
        """Remove all entries, role assignments, and custom roles."""
        with self._lock:
            self._user_roles.clear()
            self._entries.clear()
            self._custom_roles.clear()


def check_permission(
    acl: ACLManager,
    user: str,
    resource_type: ResourceType,
    resource_id: str,
    permission: Permission,
) -> bool:
    """Module-level convenience function for access checks.

    Equivalent to ``acl.check_access(user, resource_type, resource_id, permission)``.
    """
    return acl.check_access(user, resource_type, resource_id, permission)
