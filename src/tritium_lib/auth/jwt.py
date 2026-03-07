# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""JWT token utilities — shared between tritium-sc and tritium-edge.

Both systems use the same token format so that a token issued by one
can be validated by the other (when using the same secret key).
"""

import time
import uuid
from enum import Enum
from typing import Optional

import jwt

ALGORITHM = "HS256"


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    DEVICE = "device"


def create_token(
    secret: str,
    subject: str,
    token_type: TokenType = TokenType.ACCESS,
    ttl_seconds: int = 900,
    extra_claims: dict = None,
) -> str:
    """Create a signed JWT token.

    Args:
        secret: HMAC signing key
        subject: Token subject (user_id or device_id)
        token_type: access, refresh, or device
        ttl_seconds: Time-to-live in seconds
        extra_claims: Additional claims to include
    """
    now = int(time.time())
    payload = {
        "sub": subject,
        "type": token_type.value,
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": uuid.uuid4().hex[:12],
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_token(secret: str, token: str) -> Optional[dict]:
    """Decode and validate a JWT token.

    Returns claims dict on success, None on failure.
    """
    try:
        return jwt.decode(token, secret, algorithms=[ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None
