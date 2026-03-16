# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for correlation evidence models."""

import pytest

from tritium_lib.models.correlation_evidence import (
    CorrelationEvidence,
    EvidenceType,
    build_handoff_evidence,
    build_spatial_evidence,
    build_visual_evidence,
    compute_composite_confidence,
    make_pair_id,
)


class TestMakePairId:
    def test_deterministic_order(self):
        assert make_pair_id("a", "b") == make_pair_id("b", "a")

    def test_format(self):
        pid = make_pair_id("ble_aabb", "det_person_1")
        assert "::" in pid
        parts = pid.split("::")
        assert len(parts) == 2

    def test_same_target(self):
        pid = make_pair_id("x", "x")
        assert pid == "x::x"


class TestCorrelationEvidence:
    def test_create(self):
        ev = CorrelationEvidence(
            pair_id="a::b",
            evidence_type=EvidenceType.SPATIAL_PROXIMITY,
            confidence=0.85,
        )
        assert ev.pair_id == "a::b"
        assert ev.evidence_type == EvidenceType.SPATIAL_PROXIMITY
        assert ev.confidence == 0.85
        assert ev.evidence_id  # auto-generated

    def test_all_evidence_types(self):
        for et in EvidenceType:
            ev = CorrelationEvidence(
                pair_id="x::y",
                evidence_type=et,
                confidence=0.5,
            )
            assert ev.evidence_type == et

    def test_evidence_data(self):
        ev = CorrelationEvidence(
            pair_id="a::b",
            evidence_type=EvidenceType.SIGNAL_PATTERN,
            evidence_data={"rssi_delta": 5, "pattern": "stable"},
            confidence=0.6,
        )
        assert ev.evidence_data["rssi_delta"] == 5


class TestCompositeConfidence:
    def test_empty(self):
        assert compute_composite_confidence([]) == 0.0

    def test_single(self):
        ev = CorrelationEvidence(
            pair_id="a::b",
            evidence_type=EvidenceType.SPATIAL_PROXIMITY,
            confidence=0.8,
        )
        assert compute_composite_confidence([ev]) == pytest.approx(0.8, abs=0.01)

    def test_multiple_increases(self):
        evs = [
            CorrelationEvidence(
                pair_id="a::b",
                evidence_type=EvidenceType.SPATIAL_PROXIMITY,
                confidence=0.5,
            ),
            CorrelationEvidence(
                pair_id="a::b",
                evidence_type=EvidenceType.VISUAL_SIMILARITY,
                confidence=0.5,
            ),
        ]
        result = compute_composite_confidence(evs)
        assert result > 0.5  # Multiple evidence should increase confidence
        assert result == pytest.approx(0.75, abs=0.01)

    def test_diminishing_returns(self):
        evs = [
            CorrelationEvidence(
                pair_id="a::b",
                evidence_type=EvidenceType.SPATIAL_PROXIMITY,
                confidence=0.9,
            ),
            CorrelationEvidence(
                pair_id="a::b",
                evidence_type=EvidenceType.VISUAL_SIMILARITY,
                confidence=0.9,
            ),
        ]
        result = compute_composite_confidence(evs)
        assert result > 0.9
        assert result < 1.0  # Not quite 1.0


class TestBuildSpatialEvidence:
    def test_close_targets(self):
        ev = build_spatial_evidence("a", "b", distance_m=2.0)
        assert ev.evidence_type == EvidenceType.SPATIAL_PROXIMITY
        assert ev.confidence > 0.7

    def test_far_targets(self):
        ev = build_spatial_evidence("a", "b", distance_m=9.0)
        assert ev.confidence < 0.2

    def test_zero_distance(self):
        ev = build_spatial_evidence("a", "b", distance_m=0.0)
        assert ev.confidence == 1.0


class TestBuildVisualEvidence:
    def test_basic(self):
        ev = build_visual_evidence(
            "t1", "t2", similarity=0.85,
            camera_a="cam1", camera_b="cam2",
        )
        assert ev.evidence_type == EvidenceType.VISUAL_SIMILARITY
        assert ev.confidence == 0.85
        assert ev.evidence_data["camera_a"] == "cam1"


class TestBuildHandoffEvidence:
    def test_short_gap(self):
        ev = build_handoff_evidence(
            "t1", "t2",
            from_sensor="cam1", to_sensor="cam2",
            gap_seconds=5.0,
        )
        assert ev.evidence_type == EvidenceType.HANDOFF_MATCH
        assert ev.confidence > 0.9

    def test_long_gap(self):
        ev = build_handoff_evidence(
            "t1", "t2",
            from_sensor="cam1", to_sensor="cam2",
            gap_seconds=100.0,
        )
        assert ev.confidence < 0.3
