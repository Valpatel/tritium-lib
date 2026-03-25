# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SHA-256 integrity verification for evidence items.

Computes deterministic hashes over evidence data payloads so that any
modification after collection can be detected.  The hash covers the
JSON-serialized data dict using sorted keys for determinism.

No file I/O — operates purely on in-memory data structures.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import Evidence, EvidenceStatus


def _canonical_json(data: dict[str, Any]) -> str:
    """Produce a canonical JSON string for hashing.

    Uses sorted keys, no whitespace, and ensures_ascii for determinism
    across platforms and Python versions.

    Args:
        data: Dictionary to serialize.

    Returns:
        Canonical JSON string.
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def compute_sha256(data: dict[str, Any]) -> str:
    """Compute SHA-256 hash of a data payload.

    Args:
        data: Evidence data dictionary.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    canonical = _canonical_json(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_evidence(evidence: Evidence) -> str:
    """Compute and store the SHA-256 hash on an Evidence item.

    Hashes the evidence's data payload and stores the result in
    ``evidence.sha256``.  Returns the computed hash.

    Args:
        evidence: Evidence item to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    h = compute_sha256(evidence.data)
    evidence.sha256 = h
    return h


def verify_integrity(evidence: Evidence) -> bool:
    """Verify that an evidence item's data has not been tampered with.

    Recomputes the SHA-256 hash of the data payload and compares it
    to the stored hash.  If they match, the evidence is marked as
    VERIFIED; otherwise it is marked as CHALLENGED.

    Args:
        evidence: Evidence item to verify.

    Returns:
        True if the hash matches, False otherwise.
    """
    if not evidence.sha256:
        return False
    current_hash = compute_sha256(evidence.data)
    if current_hash == evidence.sha256:
        evidence.mark_verified()
        return True
    else:
        evidence.mark_challenged()
        return False


def verify_hash(data: dict[str, Any], expected_hash: str) -> bool:
    """Verify a data payload against an expected hash without modifying anything.

    Args:
        data: Evidence data dictionary.
        expected_hash: Expected SHA-256 hex digest.

    Returns:
        True if the computed hash matches the expected hash.
    """
    return compute_sha256(data) == expected_hash


def hash_bytes(raw: bytes) -> str:
    """Compute SHA-256 hash of raw bytes.

    Useful for hashing binary evidence payloads (screenshots, audio).

    Args:
        raw: Raw bytes to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(raw).hexdigest()
