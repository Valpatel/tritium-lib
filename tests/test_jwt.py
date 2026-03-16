# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for JWT token creation and validation."""

import time

from tritium_lib.auth import TokenType, create_token, decode_token


SECRET = "test-secret-key-for-unit-tests-32b"


class TestCreateToken:
    def test_creates_string(self):
        token = create_token(SECRET, subject="user-001")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_access_token(self):
        token = create_token(SECRET, "user-001", token_type=TokenType.ACCESS)
        claims = decode_token(SECRET, token)
        assert claims is not None
        assert claims["sub"] == "user-001"
        assert claims["type"] == "access"

    def test_device_token(self):
        token = create_token(SECRET, "dev-001", token_type=TokenType.DEVICE)
        claims = decode_token(SECRET, token)
        assert claims["type"] == "device"

    def test_refresh_token(self):
        token = create_token(SECRET, "user-001", token_type=TokenType.REFRESH)
        claims = decode_token(SECRET, token)
        assert claims["type"] == "refresh"

    def test_extra_claims(self):
        token = create_token(SECRET, "user-001", extra_claims={"role": "admin"})
        claims = decode_token(SECRET, token)
        assert claims["role"] == "admin"

    def test_has_jti(self):
        token = create_token(SECRET, "user-001")
        claims = decode_token(SECRET, token)
        assert "jti" in claims
        assert len(claims["jti"]) == 12

    def test_has_timestamps(self):
        token = create_token(SECRET, "user-001")
        claims = decode_token(SECRET, token)
        assert "iat" in claims
        assert "exp" in claims
        assert claims["exp"] > claims["iat"]


class TestDecodeToken:
    def test_valid_token(self):
        token = create_token(SECRET, "user-001")
        claims = decode_token(SECRET, token)
        assert claims is not None
        assert claims["sub"] == "user-001"

    def test_wrong_secret(self):
        token = create_token(SECRET, "user-001")
        claims = decode_token("wrong-secret", token)
        assert claims is None

    def test_expired_token(self):
        token = create_token(SECRET, "user-001", ttl_seconds=0)
        # Token is already expired (exp == iat)
        time.sleep(1)
        claims = decode_token(SECRET, token)
        assert claims is None

    def test_invalid_token_string(self):
        claims = decode_token(SECRET, "not-a-jwt")
        assert claims is None

    def test_unique_jti(self):
        t1 = create_token(SECRET, "user-001")
        t2 = create_token(SECRET, "user-001")
        c1 = decode_token(SECRET, t1)
        c2 = decode_token(SECRET, t2)
        assert c1["jti"] != c2["jti"]
