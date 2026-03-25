# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.evidence.integrity — SHA-256 integrity verification."""

import hashlib
import json

from tritium_lib.evidence.integrity import (
    compute_sha256,
    hash_bytes,
    hash_evidence,
    verify_hash,
    verify_integrity,
)
from tritium_lib.evidence.models import Evidence, EvidenceStatus, EvidenceType


class TestComputeSha256:
    def test_empty_dict(self):
        h = compute_sha256({})
        expected = hashlib.sha256(b"{}").hexdigest()
        assert h == expected

    def test_deterministic(self):
        data = {"b": 2, "a": 1}
        h1 = compute_sha256(data)
        h2 = compute_sha256(data)
        assert h1 == h2

    def test_key_order_independent(self):
        h1 = compute_sha256({"a": 1, "b": 2})
        h2 = compute_sha256({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_data(self):
        h1 = compute_sha256({"key": "value1"})
        h2 = compute_sha256({"key": "value2"})
        assert h1 != h2

    def test_returns_hex_string(self):
        h = compute_sha256({"x": 1})
        assert len(h) == 64
        int(h, 16)  # should not raise

    def test_nested_dict(self):
        data = {"outer": {"inner": "value"}}
        h = compute_sha256(data)
        assert len(h) == 64


class TestHashEvidence:
    def test_sets_sha256(self):
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            data={"mac": "AA:BB:CC", "rssi": -55},
        )
        h = hash_evidence(ev)
        assert ev.sha256 == h
        assert len(h) == 64

    def test_empty_data(self):
        ev = Evidence(
            evidence_type=EvidenceType.MANUAL_NOTE,
            data={},
        )
        h = hash_evidence(ev)
        assert h == compute_sha256({})


class TestVerifyIntegrity:
    def test_pass(self):
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            data={"sensor": "ble_01", "rssi": -60},
        )
        hash_evidence(ev)
        assert verify_integrity(ev) is True
        assert ev.status == EvidenceStatus.VERIFIED

    def test_fail_tampered(self):
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            data={"sensor": "ble_01", "rssi": -60},
        )
        hash_evidence(ev)
        ev.data["rssi"] = -30  # tamper
        assert verify_integrity(ev) is False
        assert ev.status == EvidenceStatus.CHALLENGED

    def test_fail_no_hash(self):
        ev = Evidence(
            evidence_type=EvidenceType.SIGNAL_CAPTURE,
            data={"x": 1},
        )
        assert verify_integrity(ev) is False


class TestVerifyHash:
    def test_match(self):
        data = {"key": "val"}
        h = compute_sha256(data)
        assert verify_hash(data, h) is True

    def test_no_match(self):
        data = {"key": "val"}
        assert verify_hash(data, "0" * 64) is False


class TestHashBytes:
    def test_basic(self):
        raw = b"hello world"
        h = hash_bytes(raw)
        assert h == hashlib.sha256(raw).hexdigest()

    def test_empty(self):
        h = hash_bytes(b"")
        assert h == hashlib.sha256(b"").hexdigest()
