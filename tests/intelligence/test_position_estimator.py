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


class TestSnrToUncertaintyRadius:
    """ISM ring sizing (rtl433-receiver-position design §2):
    strong (> -8 dB SNR) → 30 m; medium → 75 m; weak → 150 m.
    Table-driven, never a false pinpoint."""

    def test_strong_snr(self):
        from tritium_lib.intelligence.position_estimator import (
            snr_to_uncertainty_radius,
        )
        assert snr_to_uncertainty_radius(-5.0) == 30.0
        assert snr_to_uncertainty_radius(12.0) == 30.0

    def test_medium_snr(self):
        from tritium_lib.intelligence.position_estimator import (
            snr_to_uncertainty_radius,
        )
        assert snr_to_uncertainty_radius(-10.0) == 75.0

    def test_weak_snr(self):
        from tritium_lib.intelligence.position_estimator import (
            snr_to_uncertainty_radius,
        )
        assert snr_to_uncertainty_radius(-20.0) == 150.0

    def test_none_snr_is_weak(self):
        from tritium_lib.intelligence.position_estimator import (
            snr_to_uncertainty_radius,
        )
        assert snr_to_uncertainty_radius(None) == 150.0


class TestEstimateFromRssiObservations:
    """Unified BLE/ISM entry (rtl433 design §3): one observation →
    proximity ring; 2+ → weighted centroid; the same pipeline both
    modalities call so tri-receiver yards localize for free."""

    def _obs(self, aid, lat, lng, rssi=None, snr=None,
             dtype="ism", detected="ism_acurite_123"):
        anchor = PositionAnchor(anchor_id=aid, lat=lat, lng=lng)
        det = DetectionEdge(
            detector_id=aid, detected_id=detected,
            detection_type=dtype, rssi=rssi, snr=snr,
        )
        return anchor, det

    def test_empty_returns_none(self):
        from tritium_lib.intelligence.position_estimator import (
            estimate_from_rssi_observations,
        )
        assert estimate_from_rssi_observations([]) is None

    def test_single_receiver_proximity(self):
        from tritium_lib.intelligence.position_estimator import (
            estimate_from_rssi_observations,
        )
        est = estimate_from_rssi_observations([
            self._obs("rx1", 40.0, -74.0, snr=-5.0),
        ])
        assert est is not None
        assert est.lat == 40.0 and est.lng == -74.0
        assert est.method == "rf_proximity"
        assert est.accuracy_m == 30.0  # strong SNR ring, not a pinpoint
        assert est.anchor_count == 1

    def test_single_receiver_weak_snr_wide_ring(self):
        from tritium_lib.intelligence.position_estimator import (
            estimate_from_rssi_observations,
        )
        est = estimate_from_rssi_observations([
            self._obs("rx1", 40.0, -74.0, snr=-20.0),
        ])
        assert est.accuracy_m == 150.0

    def test_ble_single_falls_back_to_rssi_distance(self):
        from tritium_lib.intelligence.position_estimator import (
            estimate_from_rssi_observations,
        )
        # BLE carries rssi, no snr: ring from the path-loss model
        est = estimate_from_rssi_observations([
            self._obs("rx1", 40.0, -74.0, rssi=-60.0, dtype="ble",
                      detected="ble_aa:bb"),
        ])
        assert est.method == "rf_proximity"
        assert 1.0 < est.accuracy_m < 100.0

    def test_multi_receiver_centroid(self):
        from tritium_lib.intelligence.position_estimator import (
            estimate_from_rssi_observations,
        )
        # Equidistant receivers east and west — centroid lands between
        est = estimate_from_rssi_observations([
            self._obs("rx1", 40.0, -74.0010, rssi=-60.0, snr=-5.0),
            self._obs("rx2", 40.0, -73.9990, rssi=-60.0, snr=-5.0),
        ])
        assert est is not None
        assert est.method == "rf_multilateration"
        assert est.anchor_count == 2
        assert abs(est.lng - (-74.0)) < 0.0005  # between the receivers

    def test_stronger_receiver_pulls_centroid(self):
        from tritium_lib.intelligence.position_estimator import (
            estimate_from_rssi_observations,
        )
        est = estimate_from_rssi_observations([
            self._obs("near", 40.0, -74.0010, rssi=-50.0),
            self._obs("far",  40.0, -73.9990, rssi=-80.0),
        ])
        # closer (stronger) receiver pulls the estimate west
        assert est.lng < -74.0
