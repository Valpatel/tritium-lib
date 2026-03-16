# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the User model, roles, permissions, and sessions."""

import pytest
from tritium_lib.models.user import (
    Permission,
    ROLE_PERMISSIONS,
    User,
    UserRole,
    UserSession,
)


class TestUserRole:
    def test_all_roles_defined(self):
        assert len(UserRole) == 5
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.COMMANDER.value == "commander"
        assert UserRole.ANALYST.value == "analyst"
        assert UserRole.OPERATOR.value == "operator"
        assert UserRole.OBSERVER.value == "observer"


class TestPermission:
    def test_permission_values(self):
        assert Permission.TARGETS_VIEW.value == "targets.view"
        assert Permission.SYSTEM_USERS.value == "system.users"
        assert Permission.AMY_COMMAND.value == "amy.command"

    def test_all_roles_have_permissions(self):
        for role in UserRole:
            assert role in ROLE_PERMISSIONS
            assert len(ROLE_PERMISSIONS[role]) > 0

    def test_admin_has_all_permissions(self):
        admin_perms = ROLE_PERMISSIONS[UserRole.ADMIN]
        assert admin_perms == set(Permission)

    def test_observer_read_only(self):
        observer_perms = ROLE_PERMISSIONS[UserRole.OBSERVER]
        for p in observer_perms:
            assert "view" in p.value, f"Observer should only have view permissions, got {p.value}"

    def test_commander_has_engage(self):
        perms = ROLE_PERMISSIONS[UserRole.COMMANDER]
        assert Permission.TARGETS_ENGAGE in perms

    def test_observer_no_engage(self):
        perms = ROLE_PERMISSIONS[UserRole.OBSERVER]
        assert Permission.TARGETS_ENGAGE not in perms


class TestUser:
    def test_create_default_user(self):
        user = User(username="test_op")
        assert user.username == "test_op"
        assert user.role == UserRole.OBSERVER
        assert user.user_id  # auto-generated
        assert user.color == "#00f0ff"

    def test_has_permission_role_default(self):
        user = User(username="cmd", role=UserRole.COMMANDER)
        assert user.has_permission(Permission.TARGETS_ENGAGE)
        assert user.has_permission("targets.engage")
        assert not user.has_permission(Permission.SYSTEM_USERS)

    def test_has_permission_explicit_override(self):
        user = User(
            username="custom",
            role=UserRole.OBSERVER,
            permissions={"targets.view", "targets.engage"},
        )
        assert user.has_permission(Permission.TARGETS_ENGAGE)
        assert not user.has_permission(Permission.FLEET_COMMAND)

    def test_get_effective_permissions(self):
        user = User(username="analyst", role=UserRole.ANALYST)
        perms = user.get_effective_permissions()
        assert "targets.view" in perms
        assert "intel.investigate" in perms
        assert len(perms) > 5

    def test_to_dict(self):
        user = User(username="mat", display_name="Matthew", role=UserRole.ADMIN)
        d = user.to_dict()
        assert d["username"] == "mat"
        assert d["display_name"] == "Matthew"
        assert d["role"] == "admin"
        assert "permissions" in d
        assert isinstance(d["permissions"], list)

    def test_from_dict_roundtrip(self):
        user = User(username="roundtrip", display_name="RT", role=UserRole.COMMANDER)
        d = user.to_dict()
        user2 = User.from_dict(d)
        assert user2.username == user.username
        assert user2.role == user.role
        assert user2.display_name == user.display_name


class TestUserSession:
    def test_create_session(self):
        session = UserSession(
            user_id="u1",
            username="operator1",
            display_name="Op One",
            role=UserRole.OPERATOR,
        )
        assert session.session_id
        assert session.username == "operator1"
        assert session.role == UserRole.OPERATOR

    def test_touch_updates_activity(self):
        session = UserSession(username="test")
        old_ts = session.last_activity
        session.touch()
        assert session.last_activity >= old_ts

    def test_cursor_position(self):
        session = UserSession(username="test")
        assert session.cursor_lat is None
        session.cursor_lat = 40.7128
        session.cursor_lng = -74.0060
        assert session.cursor_lat == pytest.approx(40.7128)

    def test_to_dict(self):
        session = UserSession(
            user_id="u2",
            username="analyst1",
            role=UserRole.ANALYST,
            color="#00f0ff",
        )
        d = session.to_dict()
        assert d["username"] == "analyst1"
        assert d["role"] == "analyst"
        assert d["color"] == "#00f0ff"
        assert "session_id" in d
        assert "connected_at" in d

    def test_from_dict_roundtrip(self):
        session = UserSession(
            user_id="u3",
            username="obs1",
            display_name="Observer 1",
            role=UserRole.OBSERVER,
        )
        d = session.to_dict()
        session2 = UserSession.from_dict(d)
        assert session2.username == session.username
        assert session2.role == session.role
