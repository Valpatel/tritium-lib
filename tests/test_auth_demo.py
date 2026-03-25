# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the auth demo — login flow, token refresh, RBAC, API key CRUD."""

import time

import pytest
from fastapi.testclient import TestClient

from tritium_lib.auth.demos.auth_demo import (
    ApiKeyRecord,
    DemoState,
    Role,
    User,
    _hash_password,
    _init_demo_state,
    authenticate_user,
    create_app,
    create_tokens,
    has_permission,
    SECRET_KEY,
)
from tritium_lib.auth.jwt import (
    create_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    TokenType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    """Fresh demo state with seed users."""
    return _init_demo_state()


@pytest.fixture
def client(state):
    """FastAPI test client wired to a fresh state."""
    app = create_app(state)
    return TestClient(app)


def _login(client, username="admin", password="admin123"):
    """Helper: login and return response JSON."""
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    return resp


def _auth_header(token: str) -> dict:
    """Helper: build Bearer auth header."""
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Test: Role-based access control helpers
# ---------------------------------------------------------------------------

class TestRBAC:
    def test_admin_has_admin_permission(self):
        assert has_permission("admin", Role.ADMIN) is True

    def test_admin_has_operator_permission(self):
        assert has_permission("admin", Role.OPERATOR) is True

    def test_admin_has_viewer_permission(self):
        assert has_permission("admin", Role.VIEWER) is True

    def test_operator_has_operator_permission(self):
        assert has_permission("operator", Role.OPERATOR) is True

    def test_operator_has_viewer_permission(self):
        assert has_permission("operator", Role.VIEWER) is True

    def test_operator_lacks_admin_permission(self):
        assert has_permission("operator", Role.ADMIN) is False

    def test_viewer_has_viewer_permission(self):
        assert has_permission("viewer", Role.VIEWER) is True

    def test_viewer_lacks_operator_permission(self):
        assert has_permission("viewer", Role.OPERATOR) is False

    def test_viewer_lacks_admin_permission(self):
        assert has_permission("viewer", Role.ADMIN) is False

    def test_invalid_role_denied(self):
        assert has_permission("hacker", Role.VIEWER) is False

    def test_empty_role_denied(self):
        assert has_permission("", Role.VIEWER) is False


# ---------------------------------------------------------------------------
# Test: Demo state initialization
# ---------------------------------------------------------------------------

class TestDemoState:
    def test_init_creates_three_users(self, state):
        assert len(state.users) == 3

    def test_admin_user_exists(self, state):
        user = state.users.get("user-admin")
        assert user is not None
        assert user.username == "admin"
        assert user.role == Role.ADMIN

    def test_operator_user_exists(self, state):
        user = state.users.get("user-operator")
        assert user is not None
        assert user.role == Role.OPERATOR

    def test_viewer_user_exists(self, state):
        user = state.users.get("user-viewer")
        assert user is not None
        assert user.role == Role.VIEWER

    def test_stats_initialized_to_zero(self, state):
        for key, val in state.stats.items():
            assert val == 0, f"stat '{key}' should start at 0"


# ---------------------------------------------------------------------------
# Test: Authentication helper
# ---------------------------------------------------------------------------

class TestAuthenticate:
    def test_valid_admin_credentials(self, state):
        user = authenticate_user(state, "admin", "admin123")
        assert user is not None
        assert user.username == "admin"

    def test_valid_operator_credentials(self, state):
        user = authenticate_user(state, "operator", "oper123")
        assert user is not None
        assert user.role == Role.OPERATOR

    def test_wrong_password_returns_none(self, state):
        assert authenticate_user(state, "admin", "wrong") is None

    def test_nonexistent_user_returns_none(self, state):
        assert authenticate_user(state, "nobody", "pass") is None


# ---------------------------------------------------------------------------
# Test: Login endpoint
# ---------------------------------------------------------------------------

class TestLoginEndpoint:
    def test_successful_login(self, client):
        resp = _login(client)
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "admin"
        assert data["user"]["role"] == "admin"

    def test_login_sets_cookies(self, client):
        resp = _login(client)
        assert resp.status_code == 200
        assert "tritium_token" in resp.cookies
        assert "tritium_refresh" in resp.cookies

    def test_login_increments_stats(self, client, state):
        assert state.stats["logins"] == 0
        _login(client)
        assert state.stats["logins"] == 1

    def test_invalid_credentials_401(self, client):
        resp = _login(client, "admin", "wrongpass")
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    def test_invalid_login_increments_failures(self, client, state):
        _login(client, "admin", "wrongpass")
        assert state.stats["auth_failures"] == 1

    def test_login_tokens_are_decodable(self, client):
        resp = _login(client)
        data = resp.json()
        access = decode_token(SECRET_KEY, data["access_token"])
        refresh = decode_token(SECRET_KEY, data["refresh_token"])
        assert access is not None
        assert access["sub"] == "user-admin"
        assert access["role"] == "admin"
        assert refresh is not None
        assert refresh["type"] == "refresh"

    def test_operator_login(self, client):
        resp = _login(client, "operator", "oper123")
        assert resp.status_code == 200
        assert resp.json()["user"]["role"] == "operator"

    def test_viewer_login(self, client):
        resp = _login(client, "viewer", "view123")
        assert resp.status_code == 200
        assert resp.json()["user"]["role"] == "viewer"


# ---------------------------------------------------------------------------
# Test: Token refresh endpoint
# ---------------------------------------------------------------------------

class TestRefreshEndpoint:
    def test_refresh_with_cookie(self, client):
        login_resp = _login(client)
        # The test client carries cookies automatically
        resp = client.post("/api/auth/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_refresh_with_bearer_header(self, client):
        login_resp = _login(client)
        refresh_token = login_resp.json()["refresh_token"]
        # Clear cookies to force header-based auth
        client.cookies.clear()
        resp = client.post("/api/auth/refresh", headers=_auth_header(refresh_token))
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_refresh_increments_stats(self, client, state):
        _login(client)
        client.post("/api/auth/refresh")
        assert state.stats["refreshes"] == 1

    def test_refresh_without_token_401(self, client):
        resp = client.post("/api/auth/refresh")
        assert resp.status_code == 401

    def test_refresh_with_access_token_rejected(self, client):
        login_resp = _login(client)
        access_token = login_resp.json()["access_token"]
        client.cookies.clear()
        resp = client.post("/api/auth/refresh", headers=_auth_header(access_token))
        assert resp.status_code == 401
        assert "Not a refresh token" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Test: Who-am-I endpoint
# ---------------------------------------------------------------------------

class TestMeEndpoint:
    def test_me_returns_user_info(self, client):
        login_resp = _login(client)
        token = login_resp.json()["access_token"]
        resp = client.get("/api/auth/me", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "user-admin"
        assert data["role"] == "admin"

    def test_me_without_auth_401(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test: API key management (admin only)
# ---------------------------------------------------------------------------

class TestApiKeys:
    def _admin_token(self, client):
        return _login(client).json()["access_token"]

    def test_create_api_key(self, client, state):
        token = self._admin_token(client)
        resp = client.post(
            "/api/keys",
            json={"name": "test-key", "role": "operator"},
            headers=_auth_header(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-key"
        assert data["role"] == "operator"
        assert data["api_key"].startswith("tritium_")
        assert state.stats["api_keys_created"] == 1

    def test_list_api_keys(self, client, state):
        token = self._admin_token(client)
        # Create a key first
        client.post("/api/keys", json={"name": "k1", "role": "viewer"}, headers=_auth_header(token))
        resp = client.get("/api/keys", headers=_auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["keys"][0]["name"] == "k1"

    def test_revoke_api_key(self, client, state):
        token = self._admin_token(client)
        create_resp = client.post("/api/keys", json={"name": "doomed", "role": "viewer"}, headers=_auth_header(token))
        key_id = create_resp.json()["key_id"]
        resp = client.delete(f"/api/keys/{key_id}", headers=_auth_header(token))
        assert resp.status_code == 200
        assert state.stats["api_keys_revoked"] == 1

    def test_revoke_nonexistent_key_404(self, client):
        token = self._admin_token(client)
        resp = client.delete("/api/keys/nonexistent", headers=_auth_header(token))
        assert resp.status_code == 404

    def test_revoke_already_revoked_400(self, client):
        token = self._admin_token(client)
        create_resp = client.post("/api/keys", json={"name": "x", "role": "viewer"}, headers=_auth_header(token))
        key_id = create_resp.json()["key_id"]
        client.delete(f"/api/keys/{key_id}", headers=_auth_header(token))
        resp = client.delete(f"/api/keys/{key_id}", headers=_auth_header(token))
        assert resp.status_code == 400

    def test_operator_cannot_create_keys(self, client):
        token = _login(client, "operator", "oper123").json()["access_token"]
        resp = client.post("/api/keys", json={"name": "x", "role": "viewer"}, headers=_auth_header(token))
        assert resp.status_code == 403

    def test_viewer_cannot_list_keys(self, client):
        token = _login(client, "viewer", "view123").json()["access_token"]
        resp = client.get("/api/keys", headers=_auth_header(token))
        assert resp.status_code == 403

    def test_api_key_auth_works(self, client, state):
        """Create an API key, then use it to authenticate."""
        admin_token = self._admin_token(client)
        create_resp = client.post(
            "/api/keys",
            json={"name": "robot-key", "role": "operator"},
            headers=_auth_header(admin_token),
        )
        raw_key = create_resp.json()["api_key"]

        # Use the API key to access a protected endpoint
        client.cookies.clear()
        resp = client.get("/api/targets", headers={"Authorization": f"ApiKey {raw_key}"})
        assert resp.status_code == 200
        assert resp.json()["total"] == 3

    def test_revoked_api_key_rejected(self, client, state):
        """A revoked API key should no longer authenticate."""
        admin_token = self._admin_token(client)
        create_resp = client.post(
            "/api/keys",
            json={"name": "temp", "role": "viewer"},
            headers=_auth_header(admin_token),
        )
        raw_key = create_resp.json()["api_key"]
        key_id = create_resp.json()["key_id"]

        # Revoke it
        client.delete(f"/api/keys/{key_id}", headers=_auth_header(admin_token))

        # Try using the revoked key
        client.cookies.clear()
        resp = client.get("/api/stats", headers={"Authorization": f"ApiKey {raw_key}"})
        assert resp.status_code == 401

    def test_create_key_invalid_role_400(self, client):
        token = self._admin_token(client)
        resp = client.post("/api/keys", json={"name": "x", "role": "superuser"}, headers=_auth_header(token))
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Test: Protected endpoints (RBAC)
# ---------------------------------------------------------------------------

class TestProtectedEndpoints:
    def test_targets_accessible_by_viewer(self, client):
        token = _login(client, "viewer", "view123").json()["access_token"]
        resp = client.get("/api/targets", headers=_auth_header(token))
        assert resp.status_code == 200

    def test_targets_accessible_by_operator(self, client):
        token = _login(client, "operator", "oper123").json()["access_token"]
        resp = client.get("/api/targets", headers=_auth_header(token))
        assert resp.status_code == 200

    def test_targets_accessible_by_admin(self, client):
        token = _login(client, "admin", "admin123").json()["access_token"]
        resp = client.get("/api/targets", headers=_auth_header(token))
        assert resp.status_code == 200

    def test_create_target_requires_operator(self, client):
        token = _login(client, "viewer", "view123").json()["access_token"]
        resp = client.post("/api/targets", json={"id": "test"}, headers=_auth_header(token))
        assert resp.status_code == 403

    def test_create_target_allowed_for_operator(self, client):
        token = _login(client, "operator", "oper123").json()["access_token"]
        resp = client.post("/api/targets", json={"id": "test"}, headers=_auth_header(token))
        assert resp.status_code == 200

    def test_admin_users_requires_admin(self, client):
        token = _login(client, "operator", "oper123").json()["access_token"]
        resp = client.get("/api/admin/users", headers=_auth_header(token))
        assert resp.status_code == 403

    def test_admin_users_accessible_by_admin(self, client):
        token = _login(client, "admin", "admin123").json()["access_token"]
        resp = client.get("/api/admin/users", headers=_auth_header(token))
        assert resp.status_code == 200
        assert resp.json()["total"] == 3

    def test_stats_accessible_by_viewer(self, client):
        token = _login(client, "viewer", "view123").json()["access_token"]
        resp = client.get("/api/stats", headers=_auth_header(token))
        assert resp.status_code == 200

    def test_unauthenticated_targets_401(self, client):
        resp = client.get("/api/targets")
        assert resp.status_code == 401

    def test_unauthenticated_stats_401(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test: HTML pages
# ---------------------------------------------------------------------------

class TestHTMLPages:
    def test_login_page_renders(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Tritium Auth Demo" in resp.text
        assert "loginForm" in resp.text

    def test_dashboard_requires_auth(self, client):
        resp = client.get("/dashboard", follow_redirects=False)
        # Returns a 302 meta-refresh redirect
        assert resp.status_code == 302 or "url=/" in resp.text

    def test_dashboard_renders_when_authenticated(self, client):
        _login(client)  # Sets cookie
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "Auth Dashboard" in resp.text
        assert "admin" in resp.text


# ---------------------------------------------------------------------------
# Test: Token creation helper
# ---------------------------------------------------------------------------

class TestCreateTokens:
    def test_returns_both_tokens(self, state):
        user = state.users["user-admin"]
        tokens = create_tokens(user)
        assert "access_token" in tokens
        assert "refresh_token" in tokens

    def test_access_token_has_role_claim(self, state):
        user = state.users["user-operator"]
        tokens = create_tokens(user)
        claims = decode_token(SECRET_KEY, tokens["access_token"])
        assert claims["role"] == "operator"

    def test_refresh_token_is_refresh_type(self, state):
        user = state.users["user-viewer"]
        tokens = create_tokens(user)
        claims = decode_token(SECRET_KEY, tokens["refresh_token"])
        assert claims["type"] == "refresh"
