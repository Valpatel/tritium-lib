# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.intelligence.position_estimator."""

import pytest

from tritium_lib.intelligence.position_estimator import (
    estimate_from_multiple_anchors,
    estimate_from_single_anchor,
    rssi_to_distance,
)
from tritium_lib.models.position_anchor import (
    DetectionEdge,
    FusedPositionEstimate,
    PositionAnchor,
)


class TestRssiToDistance:
    def test_typical_rssi(self):
        """RSSI -60 dBm at default settings gives reasonable distance."""
        dist = rssi_to_distance(-60.0)
        assert 1.0 < dist < 100.0

    def test_close_rssi(self):
        """Strong signal (close RSSI) gives short distance."""
        dist = rssi_to_distance(-40.0)
        assert dist <= 2.0

    def test_far_rssi(self):
        """Weak signal (far RSSI) gives large distance."""
        dist = rssi_to_distance(-90.0)
        assert dist > 10.0

    def test_minimum_clamp(self):
        """Distance is clamped to minimum 0.5m."""
        dist = rssi_to_distance(-30.0)  # Very strong signal
        assert dist >= 0.5

    def test_negative_path_loss(self):
        """Negative path loss exponent raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            rssi_to_distance(-60.0, path_loss_exp=-1.0)

    def test_zero_path_loss(self):
        """Zero path loss exponent raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            rssi_to_distance(-60.0, path_loss_exp=0.0)

    def test_custom_tx_power(self):
        """Custom tx_power changes the distance estimate."""
        d1 = rssi_to_distance(-60.0, tx_power=-30.0)
        d2 = rssi_to_distance(-60.0, tx_power=-50.0)
        assert d1 > d2  # Higher tx_power = greater assumed distance

    def test_custom_path_loss_exponent(self):
        """Higher path loss exponent gives shorter distance estimate."""
        d1 = rssi_to_distance(-70.0, path_loss_exp=2.0)
        d2 = rssi_to_distance(-70.0, path_loss_exp=4.0)
        assert d1 > d2


class TestEstimateFromSingleAnchor:
    def test_basic_single_anchor(self):
        anchor = PositionAnchor(anchor_id="node1", lat=40.0, lng=-74.0)
        detection = DetectionEdge(
            detector_id="node1", detected_id="phone_01",
            detection_type="ble", rssi=-60.0,
        )
        result = estimate_from_single_anchor(anchor, detection)
        assert isinstance(result, FusedPositionEstimate)
        assert result.target_id == "phone_01"
        assert result.lat == 40.0
        assert result.lng == -74.0
        assert result.method == "proximity"
        assert result.anchor_count == 1
        assert 0.0 < result.confidence <= 1.0

    def test_with_distance_estimate(self):
        """When distance_estimate_m is set, RSSI is ignored."""
        anchor = PositionAnchor(anchor_id="a", lat=10.0, lng=20.0)
        detection = DetectionEdge(
            detector_id="a", detected_id="target1",
            detection_type="wifi", rssi=-80.0,
            distance_estimate_m=5.0,
        )
        result = estimate_from_single_anchor(anchor, detection)
        assert result.accuracy_m == 5.0

    def test_no_signal_data(self):
        """Without RSSI or distance, default radius of 100m is used."""
        anchor = PositionAnchor(anchor_id="a", lat=10.0, lng=20.0)
        detection = DetectionEdge(
            detector_id="a", detected_id="target1",
            detection_type="camera",
        )
        result = estimate_from_single_anchor(anchor, detection)
        assert result.accuracy_m == 100.0

    def test_confidence_penalty_for_single_anchor(self):
        """Single anchor gets 0.5 penalty to confidence."""
        anchor = PositionAnchor(anchor_id="a", lat=10.0, lng=20.0, confidence=1.0)
        detection = DetectionEdge(
            detector_id="a", detected_id="t",
            detection_type="ble", rssi=-50.0, confidence=1.0,
        )
        result = estimate_from_single_anchor(anchor, detection)
        assert result.confidence <= 0.5


class TestEstimateFromMultipleAnchors:
    def test_empty_inputs(self):
        assert estimate_from_multiple_anchors([], []) is None

    def test_no_matching_pairs(self):
        """Detections that don't match any anchor return None."""
        anchors = [PositionAnchor(anchor_id="a1", lat=10.0, lng=20.0)]
        detections = [DetectionEdge(
            detector_id="unknown_node", detected_id="phone",
            detection_type="ble", rssi=-60.0,
        )]
        result = estimate_from_multiple_anchors(anchors, detections)
        assert result is None

    def test_single_pair_falls_back(self):
        """With only one matching pair, falls back to single anchor logic."""
        anchors = [PositionAnchor(anchor_id="a1", lat=10.0, lng=20.0)]
        detections = [DetectionEdge(
            detector_id="a1", detected_id="phone",
            detection_type="ble", rssi=-55.0,
        )]
        result = estimate_from_multiple_anchors(anchors, detections)
        assert result is not None
        assert result.method == "proximity"
        assert result.anchor_count == 1

    def test_two_anchors_centroid(self):
        """Two anchors produce a centroid estimate."""
        anchors = [
            PositionAnchor(anchor_id="a1", lat=40.0, lng=-74.0),
            PositionAnchor(anchor_id="a2", lat=40.001, lng=-74.001),
        ]
        detections = [
            DetectionEdge(
                detector_id="a1", detected_id="phone",
                detection_type="ble", rssi=-60.0,
            ),
            DetectionEdge(
                detector_id="a2", detected_id="phone",
                detection_type="ble", rssi=-60.0,
            ),
        ]
        result = estimate_from_multiple_anchors(anchors, detections)
        assert result is not None
        assert result.method == "centroid"
        assert result.anchor_count == 2
        # Position should be between the two anchors
        assert 40.0 <= result.lat <= 40.001
        assert -74.001 <= result.lng <= -74.0

    def test_closer_anchor_pulls_harder(self):
        """Anchor with stronger signal (closer) dominates the position."""
        anchors = [
            PositionAnchor(anchor_id="close", lat=40.0, lng=-74.0),
            PositionAnchor(anchor_id="far", lat=40.01, lng=-74.01),
        ]
        detections = [
            DetectionEdge(
                detector_id="close", detected_id="phone",
                detection_type="ble", rssi=-40.0,  # Very close
            ),
            DetectionEdge(
                detector_id="far", detected_id="phone",
                detection_type="ble", rssi=-80.0,  # Far
            ),
        ]
        result = estimate_from_multiple_anchors(anchors, detections)
        assert result is not None
        # Should be much closer to the "close" anchor
        assert abs(result.lat - 40.0) < abs(result.lat - 40.01)

    def test_four_anchors_higher_confidence(self):
        """4+ anchors gives higher confidence than 2."""
        anchors_2 = [
            PositionAnchor(anchor_id=f"a{i}", lat=40.0 + i * 0.001, lng=-74.0)
            for i in range(2)
        ]
        dets_2 = [
            DetectionEdge(
                detector_id=f"a{i}", detected_id="phone",
                detection_type="ble", rssi=-55.0,
            )
            for i in range(2)
        ]
        anchors_4 = [
            PositionAnchor(anchor_id=f"a{i}", lat=40.0 + i * 0.001, lng=-74.0)
            for i in range(4)
        ]
        dets_4 = [
            DetectionEdge(
                detector_id=f"a{i}", detected_id="phone",
                detection_type="ble", rssi=-55.0,
            )
            for i in range(4)
        ]
        r2 = estimate_from_multiple_anchors(anchors_2, dets_2)
        r4 = estimate_from_multiple_anchors(anchors_4, dets_4)
        assert r2 is not None and r4 is not None
        assert r4.confidence > r2.confidence
