"""Tests for tritium_lib.auth."""

import time

from tritium_lib.auth import (
    create_token,
    decode_token,
    TokenType,
    generate_api_key,
    hash_api_key,
    validate_api_key,
)


class TestJWT:
    SECRET = "test-secret-key-do-not-use-32bc"

    def test_create_and_decode(self):
        token = create_token(self.SECRET, "esp32-001", TokenType.DEVICE)
        claims = decode_token(self.SECRET, token)
        assert claims is not None
        assert claims["sub"] == "esp32-001"
        assert claims["type"] == "device"

    def test_access_token(self):
        token = create_token(self.SECRET, "user-123", TokenType.ACCESS)
        claims = decode_token(self.SECRET, token)
        assert claims["type"] == "access"

    def test_extra_claims(self):
        token = create_token(
            self.SECRET,
            "esp32-001",
            extra_claims={"device_id": "esp32-001", "board": "touch-lcd-35bc"},
        )
        claims = decode_token(self.SECRET, token)
        assert claims["device_id"] == "esp32-001"
        assert claims["board"] == "touch-lcd-35bc"

    def test_wrong_secret(self):
        token = create_token(self.SECRET, "esp32-001")
        claims = decode_token("wrong-secret", token)
        assert claims is None

    def test_expired_token(self):
        token = create_token(self.SECRET, "esp32-001", ttl_seconds=0)
        time.sleep(1)
        claims = decode_token(self.SECRET, token)
        assert claims is None

    def test_invalid_token(self):
        assert decode_token(self.SECRET, "not.a.token") is None


class TestAPIKey:
    def test_generate(self):
        key = generate_api_key()
        assert key.startswith("tritium_")
        assert len(key) == 48  # "tritium_" (8) + 40 hex chars

    def test_uniqueness(self):
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100

    def test_hash_and_validate(self):
        key = generate_api_key()
        h = hash_api_key(key)
        assert validate_api_key(key, h) is True

    def test_wrong_key(self):
        key = generate_api_key()
        h = hash_api_key(key)
        assert validate_api_key("tritium_wrong", h) is False

    def test_empty(self):
        assert validate_api_key("", "somehash") is False
        assert validate_api_key("somekey", "") is False
