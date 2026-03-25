# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.auth.acl — fine-grained access control."""

import json
import tempfile
from pathlib import Path

import pytest

from tritium_lib.auth.acl import (
    ACLEntry,
    ACLManager,
    BUILTIN_ROLES,
    Permission,
    ResourceType,
    check_permission,
)


# ---------------------------------------------------------------------------
# Enum basics
# ---------------------------------------------------------------------------

class TestEnums:
    def test_permission_values(self):
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.ADMIN.value == "admin"
        assert Permission.OPERATE.value == "operate"

    def test_resource_type_values(self):
        assert ResourceType.TARGET.value == "target"
        assert ResourceType.ZONE.value == "zone"
        assert ResourceType.CAMERA.value == "camera"
        assert ResourceType.REPORT.value == "report"
        assert ResourceType.INVESTIGATION.value == "investigation"

    def test_permission_count(self):
        assert len(Permission) == 4

    def test_resource_type_count(self):
        assert len(ResourceType) == 5


# ---------------------------------------------------------------------------
# ACLEntry dataclass
# ---------------------------------------------------------------------------

class TestACLEntry:
    def test_create_entry(self):
        entry = ACLEntry(
            principal="alice",
            principal_type="user",
            resource_type=ResourceType.CAMERA,
            resource_id="cam_01",
            permissions={Permission.READ, Permission.OPERATE},
        )
        assert entry.principal == "alice"
        assert entry.principal_type == "user"
        assert entry.resource_type == ResourceType.CAMERA
        assert entry.resource_id == "cam_01"
        assert Permission.READ in entry.permissions
        assert Permission.OPERATE in entry.permissions

    def test_entry_default_permissions(self):
        entry = ACLEntry(
            principal="bob",
            principal_type="role",
            resource_type=ResourceType.TARGET,
            resource_id="*",
        )
        assert entry.permissions == set()


# ---------------------------------------------------------------------------
# Built-in roles
# ---------------------------------------------------------------------------

class TestBuiltinRoles:
    def test_admin_has_all_permissions(self):
        for rt in ResourceType:
            assert BUILTIN_ROLES["admin"][rt] == {
                Permission.READ, Permission.WRITE,
                Permission.ADMIN, Permission.OPERATE,
            }

    def test_operator_has_read_and_operate(self):
        for rt in ResourceType:
            assert BUILTIN_ROLES["operator"][rt] == {
                Permission.READ, Permission.OPERATE,
            }

    def test_viewer_has_read_only(self):
        for rt in ResourceType:
            assert BUILTIN_ROLES["viewer"][rt] == {Permission.READ}

    def test_viewer_cannot_write(self):
        for rt in ResourceType:
            assert Permission.WRITE not in BUILTIN_ROLES["viewer"][rt]
            assert Permission.ADMIN not in BUILTIN_ROLES["viewer"][rt]


# ---------------------------------------------------------------------------
# ACLManager — role assignment
# ---------------------------------------------------------------------------

class TestRoleAssignment:
    def test_assign_and_get_roles(self):
        acl = ACLManager()
        acl.assign_role("alice", "operator")
        acl.assign_role("alice", "viewer")
        assert acl.get_user_roles("alice") == {"operator", "viewer"}

    def test_revoke_role(self):
        acl = ACLManager()
        acl.assign_role("alice", "admin")
        acl.assign_role("alice", "viewer")
        acl.revoke_role("alice", "admin")
        assert acl.get_user_roles("alice") == {"viewer"}

    def test_revoke_nonexistent_role(self):
        acl = ACLManager()
        acl.assign_role("alice", "viewer")
        acl.revoke_role("alice", "admin")  # should not raise
        assert acl.get_user_roles("alice") == {"viewer"}

    def test_get_roles_unknown_user(self):
        acl = ACLManager()
        assert acl.get_user_roles("nobody") == set()


# ---------------------------------------------------------------------------
# ACLManager — built-in role access
# ---------------------------------------------------------------------------

class TestBuiltinRoleAccess:
    def test_admin_can_do_everything(self):
        acl = ACLManager()
        acl.assign_role("root", "admin")
        for rt in ResourceType:
            for perm in Permission:
                assert acl.check_access("root", rt, "any_id", perm)

    def test_viewer_can_read(self):
        acl = ACLManager()
        acl.assign_role("guest", "viewer")
        assert acl.check_access("guest", ResourceType.TARGET, "t1", Permission.READ)

    def test_viewer_cannot_write(self):
        acl = ACLManager()
        acl.assign_role("guest", "viewer")
        assert not acl.check_access("guest", ResourceType.TARGET, "t1", Permission.WRITE)

    def test_operator_can_operate(self):
        acl = ACLManager()
        acl.assign_role("ops", "operator")
        assert acl.check_access("ops", ResourceType.CAMERA, "cam_1", Permission.OPERATE)

    def test_operator_cannot_admin(self):
        acl = ACLManager()
        acl.assign_role("ops", "operator")
        assert not acl.check_access("ops", ResourceType.CAMERA, "cam_1", Permission.ADMIN)

    def test_no_role_means_deny(self):
        acl = ACLManager()
        assert not acl.check_access("nobody", ResourceType.TARGET, "t1", Permission.READ)


# ---------------------------------------------------------------------------
# ACLManager — explicit grants
# ---------------------------------------------------------------------------

class TestExplicitGrants:
    def test_user_grant_specific_resource(self):
        acl = ACLManager()
        acl.grant_user("alice", ResourceType.CAMERA, "cam_01", {Permission.READ})
        assert acl.check_access("alice", ResourceType.CAMERA, "cam_01", Permission.READ)
        assert not acl.check_access("alice", ResourceType.CAMERA, "cam_02", Permission.READ)

    def test_user_wildcard_grant(self):
        acl = ACLManager()
        acl.grant_user("bob", ResourceType.TARGET, "*", {Permission.READ, Permission.WRITE})
        assert acl.check_access("bob", ResourceType.TARGET, "any_target", Permission.READ)
        assert acl.check_access("bob", ResourceType.TARGET, "any_target", Permission.WRITE)
        assert not acl.check_access("bob", ResourceType.TARGET, "any_target", Permission.ADMIN)

    def test_role_grant_on_resource(self):
        acl = ACLManager()
        acl.grant_role("analyst", ResourceType.INVESTIGATION, "inv_42", {Permission.READ, Permission.WRITE})
        acl.assign_role("charlie", "analyst")
        assert acl.check_access("charlie", ResourceType.INVESTIGATION, "inv_42", Permission.WRITE)
        assert not acl.check_access("charlie", ResourceType.INVESTIGATION, "inv_99", Permission.WRITE)

    def test_grant_merges_permissions(self):
        acl = ACLManager()
        acl.grant_user("dave", ResourceType.ZONE, "z1", {Permission.READ})
        acl.grant_user("dave", ResourceType.ZONE, "z1", {Permission.WRITE})
        assert acl.check_access("dave", ResourceType.ZONE, "z1", Permission.READ)
        assert acl.check_access("dave", ResourceType.ZONE, "z1", Permission.WRITE)

    def test_revoke_specific_permission(self):
        acl = ACLManager()
        acl.grant_user("eve", ResourceType.REPORT, "r1", {Permission.READ, Permission.WRITE})
        acl.revoke("eve", "user", ResourceType.REPORT, "r1", {Permission.WRITE})
        assert acl.check_access("eve", ResourceType.REPORT, "r1", Permission.READ)
        assert not acl.check_access("eve", ResourceType.REPORT, "r1", Permission.WRITE)

    def test_revoke_all_permissions_removes_entry(self):
        acl = ACLManager()
        acl.grant_user("frank", ResourceType.TARGET, "t1", {Permission.READ})
        acl.revoke("frank", "user", ResourceType.TARGET, "t1")
        assert not acl.check_access("frank", ResourceType.TARGET, "t1", Permission.READ)
        assert acl.get_entries(principal="frank") == []

    def test_revoke_nonexistent_entry(self):
        acl = ACLManager()
        acl.revoke("ghost", "user", ResourceType.TARGET, "t1")  # should not raise

    def test_invalid_principal_type_raises(self):
        acl = ACLManager()
        with pytest.raises(ValueError, match="principal_type"):
            acl.grant("x", "group", ResourceType.TARGET, "t1", {Permission.READ})


# ---------------------------------------------------------------------------
# ACLManager — custom role definitions
# ---------------------------------------------------------------------------

class TestCustomRoles:
    def test_define_custom_role(self):
        acl = ACLManager()
        acl.define_role("camera_viewer", ResourceType.CAMERA, {Permission.READ})
        acl.assign_role("alice", "camera_viewer")
        assert acl.check_access("alice", ResourceType.CAMERA, "cam_1", Permission.READ)
        assert not acl.check_access("alice", ResourceType.CAMERA, "cam_1", Permission.WRITE)
        # Custom role does not grant access to other resource types
        assert not acl.check_access("alice", ResourceType.TARGET, "t1", Permission.READ)

    def test_custom_role_overrides_nothing_for_other_types(self):
        acl = ACLManager()
        acl.define_role("zone_ops", ResourceType.ZONE, {Permission.OPERATE})
        acl.assign_role("bob", "zone_ops")
        assert acl.check_access("bob", ResourceType.ZONE, "z1", Permission.OPERATE)
        assert not acl.check_access("bob", ResourceType.REPORT, "r1", Permission.OPERATE)


# ---------------------------------------------------------------------------
# ACLManager — resolution priority
# ---------------------------------------------------------------------------

class TestResolutionPriority:
    def test_user_grant_takes_effect_without_role(self):
        """User with no roles but explicit grant should have access."""
        acl = ACLManager()
        acl.grant_user("alice", ResourceType.TARGET, "t1", {Permission.WRITE})
        assert acl.check_access("alice", ResourceType.TARGET, "t1", Permission.WRITE)

    def test_explicit_grant_plus_role(self):
        """User has viewer role (read) + explicit write on one resource."""
        acl = ACLManager()
        acl.assign_role("bob", "viewer")
        acl.grant_user("bob", ResourceType.CAMERA, "cam_1", {Permission.WRITE})
        assert acl.check_access("bob", ResourceType.CAMERA, "cam_1", Permission.READ)
        assert acl.check_access("bob", ResourceType.CAMERA, "cam_1", Permission.WRITE)
        # Write only on cam_1, not cam_2
        assert not acl.check_access("bob", ResourceType.CAMERA, "cam_2", Permission.WRITE)


# ---------------------------------------------------------------------------
# ACLManager — query helpers
# ---------------------------------------------------------------------------

class TestQueryHelpers:
    def test_get_entries_all(self):
        acl = ACLManager()
        acl.grant_user("a", ResourceType.TARGET, "t1", {Permission.READ})
        acl.grant_user("b", ResourceType.CAMERA, "c1", {Permission.WRITE})
        assert len(acl.get_entries()) == 2

    def test_get_entries_by_principal(self):
        acl = ACLManager()
        acl.grant_user("a", ResourceType.TARGET, "t1", {Permission.READ})
        acl.grant_user("b", ResourceType.CAMERA, "c1", {Permission.WRITE})
        entries = acl.get_entries(principal="a")
        assert len(entries) == 1
        assert entries[0].principal == "a"

    def test_get_entries_by_resource_type(self):
        acl = ACLManager()
        acl.grant_user("a", ResourceType.TARGET, "t1", {Permission.READ})
        acl.grant_user("a", ResourceType.CAMERA, "c1", {Permission.READ})
        entries = acl.get_entries(resource_type=ResourceType.CAMERA)
        assert len(entries) == 1
        assert entries[0].resource_type == ResourceType.CAMERA

    def test_get_entries_returns_copies(self):
        """Modifying returned entries should not affect internal state."""
        acl = ACLManager()
        acl.grant_user("a", ResourceType.TARGET, "t1", {Permission.READ})
        entries = acl.get_entries()
        entries[0].permissions.add(Permission.ADMIN)
        # Internal state should be unaffected
        assert not acl.check_access("a", ResourceType.TARGET, "t1", Permission.ADMIN)

    def test_get_user_permissions(self):
        acl = ACLManager()
        acl.assign_role("alice", "operator")
        acl.grant_user("alice", ResourceType.CAMERA, "cam_1", {Permission.WRITE})
        perms = acl.get_user_permissions("alice", ResourceType.CAMERA, "cam_1")
        assert perms == {Permission.READ, Permission.WRITE, Permission.OPERATE}


# ---------------------------------------------------------------------------
# ACLManager — persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "acl.json"
        acl = ACLManager()
        acl.assign_role("alice", "operator")
        acl.grant_user("alice", ResourceType.CAMERA, "cam_1", {Permission.WRITE})
        acl.grant_role("analyst", ResourceType.INVESTIGATION, "*", {Permission.READ})
        acl.define_role("zone_ops", ResourceType.ZONE, {Permission.OPERATE})
        acl.save(path)

        acl2 = ACLManager()
        acl2.load(path)
        assert acl2.get_user_roles("alice") == {"operator"}
        assert acl2.check_access("alice", ResourceType.CAMERA, "cam_1", Permission.WRITE)

        # Verify custom role survived roundtrip
        acl2.assign_role("bob", "zone_ops")
        assert acl2.check_access("bob", ResourceType.ZONE, "z1", Permission.OPERATE)

    def test_save_produces_valid_json(self, tmp_path):
        path = tmp_path / "acl.json"
        acl = ACLManager()
        acl.grant_user("x", ResourceType.TARGET, "t1", {Permission.READ})
        acl.save(path)
        data = json.loads(path.read_text())
        assert "user_roles" in data
        assert "entries" in data
        assert "custom_roles" in data

    def test_load_replaces_state(self, tmp_path):
        path = tmp_path / "acl.json"
        acl = ACLManager()
        acl.grant_user("old", ResourceType.TARGET, "t1", {Permission.READ})
        acl.save(path)

        acl2 = ACLManager()
        acl2.grant_user("new", ResourceType.CAMERA, "c1", {Permission.WRITE})
        acl2.load(path)
        # Old data should be present, new data should be gone
        assert acl2.check_access("old", ResourceType.TARGET, "t1", Permission.READ)
        assert not acl2.check_access("new", ResourceType.CAMERA, "c1", Permission.WRITE)


# ---------------------------------------------------------------------------
# ACLManager — clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_removes_everything(self):
        acl = ACLManager()
        acl.assign_role("alice", "admin")
        acl.grant_user("alice", ResourceType.TARGET, "t1", {Permission.WRITE})
        acl.define_role("custom", ResourceType.ZONE, {Permission.READ})
        acl.clear()
        assert acl.get_user_roles("alice") == set()
        assert not acl.check_access("alice", ResourceType.TARGET, "t1", Permission.WRITE)
        assert acl.get_entries() == []


# ---------------------------------------------------------------------------
# Module-level check_permission
# ---------------------------------------------------------------------------

class TestCheckPermission:
    def test_module_level_function(self):
        acl = ACLManager()
        acl.assign_role("alice", "viewer")
        assert check_permission(acl, "alice", ResourceType.TARGET, "t1", Permission.READ)
        assert not check_permission(acl, "alice", ResourceType.TARGET, "t1", Permission.WRITE)


# ---------------------------------------------------------------------------
# Import from auth package
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_from_auth_package(self):
        from tritium_lib.auth import (
            ACLEntry,
            ACLManager,
            BUILTIN_ROLES,
            Permission,
            ResourceType,
            check_permission,
        )
        assert Permission.READ.value == "read"
        assert callable(check_permission)
        assert isinstance(BUILTIN_ROLES, dict)
