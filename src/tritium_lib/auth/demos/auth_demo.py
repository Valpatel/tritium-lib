# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Auth package demo — full login flow with JWT, refresh, API keys, RBAC.

Demonstrates the complete Tritium auth pipeline:
  1. Login with username + password -> JWT access + refresh tokens
  2. Token refresh (exchange refresh token for new access token)
  3. Protected endpoints requiring valid JWT
  4. API key CRUD (create, list, revoke)
  5. Role-based access control (admin, operator, viewer)
  6. HTML login page + cyberpunk dashboard

Run with:
    PYTHONPATH=src python3 src/tritium_lib/auth/demos/auth_demo.py

Endpoints:
    GET  /                   — HTML login page
    POST /api/auth/login     — Login (username + password -> tokens)
    POST /api/auth/refresh   — Refresh access token
    GET  /api/auth/me        — Current user info (requires JWT)
    GET  /dashboard          — HTML dashboard (requires JWT cookie)
    POST /api/keys           — Create API key (admin only)
    GET  /api/keys           — List API keys (admin only)
    DELETE /api/keys/{key_id} — Revoke API key (admin only)
    GET  /api/targets        — Protected resource (operator+)
    POST /api/targets        — Create target (operator+)
    GET  /api/admin/users    — List users (admin only)
    GET  /api/stats          — Demo statistics (viewer+)
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from tritium_lib.auth.jwt import (
    TokenType,
    create_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    validate_api_key,
)
from tritium_lib.web.templates import full_page
from tritium_lib.web.theme import TritiumTheme

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_PORT = 9097
SECRET_KEY = "tritium-auth-demo-secret-key-do-not-use-in-production"
ACCESS_TTL = 900      # 15 minutes
REFRESH_TTL = 86400   # 24 hours

theme = TritiumTheme()


# ---------------------------------------------------------------------------
# Role-based access control
# ---------------------------------------------------------------------------

class Role(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


# Permission hierarchy: admin > operator > viewer
_ROLE_LEVELS = {
    Role.ADMIN: 3,
    Role.OPERATOR: 2,
    Role.VIEWER: 1,
}


def has_permission(user_role: str, required_role: Role) -> bool:
    """Check if a user role meets the required permission level."""
    try:
        user = Role(user_role)
    except ValueError:
        return False
    return _ROLE_LEVELS.get(user, 0) >= _ROLE_LEVELS.get(required_role, 99)


# ---------------------------------------------------------------------------
# In-memory data stores
# ---------------------------------------------------------------------------

@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    role: Role
    display_name: str


@dataclass
class ApiKeyRecord:
    key_id: str
    name: str
    key_hash: str
    owner_id: str
    role: Role
    created_at: float
    revoked: bool = False


@dataclass
class DemoState:
    """Mutable demo state — users, keys, tokens, stats."""
    users: dict[str, User] = field(default_factory=dict)
    api_keys: dict[str, ApiKeyRecord] = field(default_factory=dict)
    refresh_tokens: dict[str, str] = field(default_factory=dict)  # jti -> user_id
    revoked_jtis: set[str] = field(default_factory=set)
    stats: dict[str, int] = field(default_factory=lambda: {
        "logins": 0,
        "refreshes": 0,
        "api_keys_created": 0,
        "api_keys_revoked": 0,
        "auth_failures": 0,
        "protected_requests": 0,
    })


def _hash_password(password: str) -> str:
    """Simple password hash for demo purposes."""
    return hashlib.sha256(password.encode()).hexdigest()


def _init_demo_state() -> DemoState:
    """Create initial demo state with seed users."""
    state = DemoState()
    seed_users = [
        ("admin", "admin123", Role.ADMIN, "System Administrator"),
        ("operator", "oper123", Role.OPERATOR, "Field Operator"),
        ("viewer", "view123", Role.VIEWER, "Read-Only Analyst"),
    ]
    for username, password, role, display in seed_users:
        uid = f"user-{username}"
        state.users[uid] = User(
            user_id=uid,
            username=username,
            password_hash=_hash_password(password),
            role=role,
            display_name=display,
        )
    return state


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def authenticate_user(state: DemoState, username: str, password: str) -> Optional[User]:
    """Validate credentials, return User or None."""
    for user in state.users.values():
        if user.username == username and user.password_hash == _hash_password(password):
            return user
    return None


def create_tokens(user: User) -> dict:
    """Create access + refresh token pair for a user."""
    access = create_token(
        SECRET_KEY,
        user.user_id,
        token_type=TokenType.ACCESS,
        ttl_seconds=ACCESS_TTL,
        extra_claims={"role": user.role.value, "username": user.username},
    )
    refresh = create_token(
        SECRET_KEY,
        user.user_id,
        token_type=TokenType.REFRESH,
        ttl_seconds=REFRESH_TTL,
    )
    return {"access_token": access, "refresh_token": refresh}


def extract_user_from_token(
    state: DemoState, token: str
) -> Optional[dict]:
    """Decode a JWT and return claims if valid and not revoked."""
    claims = decode_token(SECRET_KEY, token)
    if claims is None:
        return None
    if claims.get("jti") in state.revoked_jtis:
        return None
    return claims


def extract_user_from_api_key(
    state: DemoState, key: str
) -> Optional[dict]:
    """Validate an API key and return synthetic claims."""
    for record in state.api_keys.values():
        if record.revoked:
            continue
        if validate_api_key(key, record.key_hash):
            return {
                "sub": record.owner_id,
                "role": record.role.value,
                "type": "api_key",
                "key_id": record.key_id,
            }
    return None


def get_auth_claims(state: DemoState, request: Request) -> Optional[dict]:
    """Extract auth claims from request (Bearer token, cookie, or API key)."""
    # 1. Check Authorization header (Bearer token or API key)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return extract_user_from_token(state, token)
    if auth_header.startswith("ApiKey "):
        key = auth_header[7:]
        return extract_user_from_api_key(state, key)

    # 2. Check cookie
    token = request.cookies.get("tritium_token")
    if token:
        return extract_user_from_token(state, token)

    return None


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

def _login_page_html() -> str:
    """Render the login page."""
    body = """
    <h1>Tritium Auth Demo</h1>
    <div class="card" style="max-width:400px;margin:40px auto">
        <h2>Login</h2>
        <div id="error" class="msg err" style="display:none"></div>
        <div id="success" class="msg ok" style="display:none"></div>
        <form id="loginForm" onsubmit="return doLogin(event)">
            <label class="label">Username</label>
            <input type="text" id="username" name="username" placeholder="admin" autocomplete="username">
            <label class="label" style="margin-top:8px">Password</label>
            <input type="password" id="password" name="password" placeholder="admin123" autocomplete="current-password">
            <div style="margin-top:16px">
                <button type="submit">Login</button>
            </div>
        </form>
        <div style="margin-top:16px;border-top:1px solid #1a1a1a;padding-top:12px">
            <div class="label">Demo Credentials</div>
            <table>
                <thead><tr><th>User</th><th>Password</th><th>Role</th></tr></thead>
                <tbody>
                    <tr><td>admin</td><td>admin123</td><td><span class="badge online">admin</span></td></tr>
                    <tr><td>operator</td><td>oper123</td><td><span class="badge updating">operator</span></td></tr>
                    <tr><td>viewer</td><td>view123</td><td><span class="badge offline">viewer</span></td></tr>
                </tbody>
            </table>
        </div>
    </div>
    <script>
    async function doLogin(e) {
        e.preventDefault();
        const errEl = document.getElementById('error');
        const okEl = document.getElementById('success');
        errEl.style.display = 'none';
        okEl.style.display = 'none';
        const body = JSON.stringify({
            username: document.getElementById('username').value,
            password: document.getElementById('password').value
        });
        try {
            const resp = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: body
            });
            const data = await resp.json();
            if (resp.ok) {
                okEl.textContent = 'Login successful! Redirecting...';
                okEl.style.display = 'block';
                setTimeout(() => window.location.href = '/dashboard', 500);
            } else {
                errEl.textContent = data.detail || 'Login failed';
                errEl.style.display = 'block';
            }
        } catch(err) {
            errEl.textContent = 'Connection error: ' + err.message;
            errEl.style.display = 'block';
        }
    }
    </script>
    """
    return full_page("Login", body, theme)


def _dashboard_html(claims: dict, state: DemoState) -> str:
    """Render the authenticated dashboard."""
    role = claims.get("role", "unknown")
    username = claims.get("username", claims.get("sub", "unknown"))
    user_id = claims.get("sub", "unknown")

    role_badge = {
        "admin": '<span class="badge online">admin</span>',
        "operator": '<span class="badge updating">operator</span>',
        "viewer": '<span class="badge offline">viewer</span>',
    }.get(role, f'<span class="badge error">{role}</span>')

    # Stats card
    stats_rows = "".join(
        f"<tr><td>{k}</td><td style='color:{theme.ACCENT}'>{v}</td></tr>"
        for k, v in state.stats.items()
    )

    # API keys card (admin only)
    keys_section = ""
    if role == "admin":
        key_rows = ""
        for rec in state.api_keys.values():
            status = "revoked" if rec.revoked else "active"
            badge = '<span class="badge error">revoked</span>' if rec.revoked else '<span class="badge online">active</span>'
            revoke_btn = (
                f'<button class="danger" onclick="revokeKey(\'{rec.key_id}\')">Revoke</button>'
                if not rec.revoked else "-"
            )
            key_rows += (
                f"<tr>"
                f"<td>{rec.key_id[:8]}...</td>"
                f"<td>{rec.name}</td>"
                f"<td>{rec.role.value}</td>"
                f"<td>{badge}</td>"
                f"<td>{revoke_btn}</td>"
                f"</tr>"
            )
        keys_section = f"""
        <div class="card">
            <h2>API Keys</h2>
            <div style="margin-bottom:12px">
                <input type="text" id="keyName" placeholder="Key name" style="width:200px;display:inline-block">
                <select id="keyRole" style="width:150px;display:inline-block">
                    <option value="admin">admin</option>
                    <option value="operator" selected>operator</option>
                    <option value="viewer">viewer</option>
                </select>
                <button onclick="createKey()">Create Key</button>
            </div>
            <div id="newKeyDisplay" class="msg ok" style="display:none"></div>
            <table>
                <thead><tr><th>ID</th><th>Name</th><th>Role</th><th>Status</th><th>Actions</th></tr></thead>
                <tbody id="keysBody">{key_rows}</tbody>
            </table>
        </div>
        """

    # Users card (admin only)
    users_section = ""
    if role == "admin":
        user_rows = ""
        for u in state.users.values():
            r_badge = {
                "admin": '<span class="badge online">admin</span>',
                "operator": '<span class="badge updating">operator</span>',
                "viewer": '<span class="badge offline">viewer</span>',
            }.get(u.role.value, u.role.value)
            user_rows += f"<tr><td>{u.user_id}</td><td>{u.username}</td><td>{u.display_name}</td><td>{r_badge}</td></tr>"
        users_section = f"""
        <div class="card">
            <h2>Users</h2>
            <table>
                <thead><tr><th>ID</th><th>Username</th><th>Display Name</th><th>Role</th></tr></thead>
                <tbody>{user_rows}</tbody>
            </table>
        </div>
        """

    body = f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
        <h1>Tritium Auth Dashboard</h1>
        <div>
            <span style="color:{theme.TEXT_DIM}">Logged in as</span>
            <strong style="color:{theme.ACCENT}">{username}</strong>
            {role_badge}
            <button class="danger" onclick="doLogout()" style="margin-left:12px">Logout</button>
        </div>
    </div>

    <div class="grid grid-2">
        <div class="card">
            <h2>Session Info</h2>
            <table>
                <tr><td class="label">User ID</td><td>{user_id}</td></tr>
                <tr><td class="label">Username</td><td>{username}</td></tr>
                <tr><td class="label">Role</td><td>{role_badge}</td></tr>
                <tr><td class="label">Token Type</td><td>{claims.get('type', 'unknown')}</td></tr>
                <tr><td class="label">Token ID</td><td style="font-size:11px">{claims.get('jti', 'N/A')}</td></tr>
            </table>
        </div>
        <div class="card">
            <h2>Auth Statistics</h2>
            <table>
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>{stats_rows}</tbody>
            </table>
        </div>
    </div>

    <div class="card" style="margin-top:12px">
        <h2>Token Operations</h2>
        <button onclick="testRefresh()">Refresh Token</button>
        <button onclick="testProtected()">Test Protected Endpoint</button>
        <button onclick="testMe()">Who Am I?</button>
        <div id="tokenResult" class="msg ok" style="display:none;margin-top:12px;word-break:break-all"></div>
    </div>

    {keys_section}
    {users_section}

    <script>
    function getToken() {{
        return document.cookie.split(';').map(c => c.trim())
            .find(c => c.startsWith('tritium_token='))?.split('=')[1] || '';
    }}

    async function apiCall(method, url, body) {{
        const opts = {{
            method,
            headers: {{'Authorization': 'Bearer ' + getToken(), 'Content-Type': 'application/json'}}
        }};
        if (body) opts.body = JSON.stringify(body);
        const resp = await fetch(url, opts);
        return {{ok: resp.ok, status: resp.status, data: await resp.json()}};
    }}

    function showResult(msg, isError) {{
        const el = document.getElementById('tokenResult');
        el.className = isError ? 'msg err' : 'msg ok';
        el.textContent = typeof msg === 'object' ? JSON.stringify(msg, null, 2) : msg;
        el.style.display = 'block';
    }}

    async function testRefresh() {{
        const r = await apiCall('POST', '/api/auth/refresh');
        showResult(r.data, !r.ok);
    }}

    async function testProtected() {{
        const r = await apiCall('GET', '/api/targets');
        showResult(r.data, !r.ok);
    }}

    async function testMe() {{
        const r = await apiCall('GET', '/api/auth/me');
        showResult(r.data, !r.ok);
    }}

    async function createKey() {{
        const name = document.getElementById('keyName').value || 'unnamed';
        const role = document.getElementById('keyRole').value;
        const r = await apiCall('POST', '/api/keys', {{name, role}});
        if (r.ok) {{
            document.getElementById('newKeyDisplay').textContent =
                'New API Key (save this!): ' + r.data.api_key;
            document.getElementById('newKeyDisplay').style.display = 'block';
            setTimeout(() => location.reload(), 2000);
        }} else {{
            showResult(r.data, true);
        }}
    }}

    async function revokeKey(keyId) {{
        const r = await apiCall('DELETE', '/api/keys/' + keyId);
        showResult(r.data, !r.ok);
        if (r.ok) setTimeout(() => location.reload(), 1000);
    }}

    function doLogout() {{
        document.cookie = 'tritium_token=; Max-Age=0; Path=/';
        document.cookie = 'tritium_refresh=; Max-Age=0; Path=/';
        window.location.href = '/';
    }}
    </script>
    """
    return full_page("Dashboard", body, theme)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(state: DemoState | None = None) -> FastAPI:
    """Create the auth demo FastAPI app.

    Args:
        state: Optional pre-built DemoState (for testing). Uses defaults if None.
    """
    if state is None:
        state = _init_demo_state()

    app = FastAPI(title="Tritium Auth Demo", version="1.0.0")

    # ── Login page ────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def login_page():
        return _login_page_html()

    # ── Login endpoint ────────────────────────────────────────────────

    @app.post("/api/auth/login")
    async def login(request: Request):
        body = await request.json()
        username = body.get("username", "")
        password = body.get("password", "")

        user = authenticate_user(state, username, password)
        if user is None:
            state.stats["auth_failures"] += 1
            return JSONResponse(
                {"detail": "Invalid username or password"},
                status_code=401,
            )

        tokens = create_tokens(user)
        state.stats["logins"] += 1

        # Track refresh token JTI for later refresh validation
        refresh_claims = decode_token(SECRET_KEY, tokens["refresh_token"])
        if refresh_claims:
            state.refresh_tokens[refresh_claims["jti"]] = user.user_id

        response = JSONResponse({
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": "bearer",
            "expires_in": ACCESS_TTL,
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "display_name": user.display_name,
            },
        })
        # Set cookies for browser-based access
        response.set_cookie(
            "tritium_token",
            tokens["access_token"],
            max_age=ACCESS_TTL,
            httponly=False,  # Demo needs JS access
            samesite="lax",
            path="/",
        )
        response.set_cookie(
            "tritium_refresh",
            tokens["refresh_token"],
            max_age=REFRESH_TTL,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response

    # ── Token refresh ─────────────────────────────────────────────────

    @app.post("/api/auth/refresh")
    async def refresh_token(request: Request):
        # Get refresh token from cookie or Authorization header
        refresh = request.cookies.get("tritium_refresh")
        if not refresh:
            auth = request.headers.get("authorization", "")
            if auth.startswith("Bearer "):
                refresh = auth[7:]

        if not refresh:
            return JSONResponse({"detail": "No refresh token provided"}, status_code=401)

        claims = decode_token(SECRET_KEY, refresh)
        if claims is None:
            state.stats["auth_failures"] += 1
            return JSONResponse({"detail": "Invalid or expired refresh token"}, status_code=401)

        if claims.get("type") != "refresh":
            return JSONResponse({"detail": "Not a refresh token"}, status_code=401)

        jti = claims.get("jti")
        if jti in state.revoked_jtis:
            return JSONResponse({"detail": "Refresh token has been revoked"}, status_code=401)

        user_id = claims.get("sub")
        user = state.users.get(user_id)
        if user is None:
            return JSONResponse({"detail": "User not found"}, status_code=401)

        # Issue new access token
        new_access = create_token(
            SECRET_KEY,
            user.user_id,
            token_type=TokenType.ACCESS,
            ttl_seconds=ACCESS_TTL,
            extra_claims={"role": user.role.value, "username": user.username},
        )
        state.stats["refreshes"] += 1

        response = JSONResponse({
            "access_token": new_access,
            "token_type": "bearer",
            "expires_in": ACCESS_TTL,
        })
        response.set_cookie(
            "tritium_token",
            new_access,
            max_age=ACCESS_TTL,
            httponly=False,
            samesite="lax",
            path="/",
        )
        return response

    # ── Who am I ──────────────────────────────────────────────────────

    @app.get("/api/auth/me")
    async def me(request: Request):
        claims = get_auth_claims(state, request)
        if claims is None:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)

        state.stats["protected_requests"] += 1
        user = state.users.get(claims.get("sub"))
        return {
            "user_id": claims.get("sub"),
            "username": claims.get("username", user.username if user else "unknown"),
            "role": claims.get("role"),
            "token_type": claims.get("type"),
            "token_id": claims.get("jti", "N/A"),
        }

    # ── Dashboard ─────────────────────────────────────────────────────

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        claims = get_auth_claims(state, request)
        if claims is None:
            return HTMLResponse(
                '<meta http-equiv="refresh" content="0;url=/">',
                status_code=302,
            )
        state.stats["protected_requests"] += 1
        return _dashboard_html(claims, state)

    # ── API key management (admin only) ───────────────────────────────

    @app.post("/api/keys")
    async def create_key(request: Request):
        claims = get_auth_claims(state, request)
        if claims is None:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        if not has_permission(claims.get("role", ""), Role.ADMIN):
            return JSONResponse({"detail": "Admin role required"}, status_code=403)

        body = await request.json()
        name = body.get("name", "unnamed")
        try:
            role = Role(body.get("role", "viewer"))
        except ValueError:
            return JSONResponse({"detail": "Invalid role"}, status_code=400)

        raw_key = generate_api_key()
        key_id = f"key-{uuid.uuid4().hex[:8]}"
        record = ApiKeyRecord(
            key_id=key_id,
            name=name,
            key_hash=hash_api_key(raw_key),
            owner_id=claims["sub"],
            role=role,
            created_at=time.time(),
        )
        state.api_keys[key_id] = record
        state.stats["api_keys_created"] += 1

        return {
            "key_id": key_id,
            "name": name,
            "role": role.value,
            "api_key": raw_key,
            "message": "Save this key — it cannot be retrieved later.",
        }

    @app.get("/api/keys")
    async def list_keys(request: Request):
        claims = get_auth_claims(state, request)
        if claims is None:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        if not has_permission(claims.get("role", ""), Role.ADMIN):
            return JSONResponse({"detail": "Admin role required"}, status_code=403)

        keys = []
        for rec in state.api_keys.values():
            keys.append({
                "key_id": rec.key_id,
                "name": rec.name,
                "role": rec.role.value,
                "owner_id": rec.owner_id,
                "created_at": rec.created_at,
                "revoked": rec.revoked,
            })
        return {"keys": keys, "total": len(keys)}

    @app.delete("/api/keys/{key_id}")
    async def revoke_key(key_id: str, request: Request):
        claims = get_auth_claims(state, request)
        if claims is None:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        if not has_permission(claims.get("role", ""), Role.ADMIN):
            return JSONResponse({"detail": "Admin role required"}, status_code=403)

        record = state.api_keys.get(key_id)
        if record is None:
            return JSONResponse({"detail": "API key not found"}, status_code=404)
        if record.revoked:
            return JSONResponse({"detail": "API key already revoked"}, status_code=400)

        record.revoked = True
        state.stats["api_keys_revoked"] += 1
        return {"message": f"API key {key_id} revoked", "key_id": key_id}

    # ── Protected resources (role-gated) ──────────────────────────────

    @app.get("/api/targets")
    async def list_targets(request: Request):
        claims = get_auth_claims(state, request)
        if claims is None:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        if not has_permission(claims.get("role", ""), Role.VIEWER):
            return JSONResponse({"detail": "Viewer role or higher required"}, status_code=403)

        state.stats["protected_requests"] += 1
        return {
            "targets": [
                {"id": "ble_aa:bb:cc:dd:ee:ff", "type": "phone", "alliance": "unknown"},
                {"id": "det_person_001", "type": "person", "alliance": "neutral"},
                {"id": "mesh_node_42", "type": "lora_node", "alliance": "friendly"},
            ],
            "total": 3,
            "accessed_by": claims.get("sub"),
        }

    @app.post("/api/targets")
    async def create_target(request: Request):
        claims = get_auth_claims(state, request)
        if claims is None:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        if not has_permission(claims.get("role", ""), Role.OPERATOR):
            return JSONResponse({"detail": "Operator role or higher required"}, status_code=403)

        body = await request.json()
        state.stats["protected_requests"] += 1
        return {
            "message": "Target created",
            "target": body,
            "created_by": claims.get("sub"),
        }

    @app.get("/api/admin/users")
    async def list_users(request: Request):
        claims = get_auth_claims(state, request)
        if claims is None:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        if not has_permission(claims.get("role", ""), Role.ADMIN):
            return JSONResponse({"detail": "Admin role required"}, status_code=403)

        users = []
        for u in state.users.values():
            users.append({
                "user_id": u.user_id,
                "username": u.username,
                "role": u.role.value,
                "display_name": u.display_name,
            })
        return {"users": users, "total": len(users)}

    @app.get("/api/stats")
    async def get_stats(request: Request):
        claims = get_auth_claims(state, request)
        if claims is None:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        if not has_permission(claims.get("role", ""), Role.VIEWER):
            return JSONResponse({"detail": "Viewer role or higher required"}, status_code=403)

        return {
            "stats": state.stats,
            "users": len(state.users),
            "api_keys": len(state.api_keys),
            "active_api_keys": sum(1 for k in state.api_keys.values() if not k.revoked),
        }

    return app


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the auth demo server."""
    import uvicorn

    print(f"\n  Tritium Auth Demo")
    print(f"  ─────────────────────────────")
    print(f"  Login page:  http://localhost:{DEMO_PORT}/")
    print(f"  Dashboard:   http://localhost:{DEMO_PORT}/dashboard")
    print(f"  API docs:    http://localhost:{DEMO_PORT}/docs")
    print(f"")
    print(f"  Demo credentials:")
    print(f"    admin    / admin123  (full access)")
    print(f"    operator / oper123   (targets + stats)")
    print(f"    viewer   / view123   (read-only)")
    print()

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=DEMO_PORT, log_level="info")


if __name__ == "__main__":
    main()
