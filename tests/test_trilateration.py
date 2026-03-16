# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for BLE trilateration — rssi_to_distance, trilaterate_2d, estimate_position."""

import math

import pytest

from tritium_lib.models.trilateration import (
    AnchorPoint,
    PositionEstimate,
    RSSIFilter,
    estimate_position,
    rssi_to_distance,
    trilaterate_2d,
)


# ---------------------------------------------------------------------------
# rssi_to_distance
# ---------------------------------------------------------------------------

class TestRssiToDistance:
    """Log-distance path loss model tests."""

    def test_at_reference_distance(self):
        """RSSI == tx_power means 1 meter."""
        d = rssi_to_distance(-59.0, tx_power=-59.0)
        assert d == pytest.approx(1.0, abs=0.01)

    def test_strong_signal_close(self):
        """Stronger RSSI should give shorter distance."""
        d = rssi_to_distance(-40.0, tx_power=-59.0, path_loss_exponent=2.5)
        assert d < 1.0

    def test_weak_signal_far(self):
        """Weaker RSSI should give longer distance."""
        d = rssi_to_distance(-80.0, tx_power=-59.0, path_loss_exponent=2.5)
        assert d > 1.0

    def test_known_value_free_space(self):
        """Verify formula: n=2.0, tx=-59, rssi=-69 -> d = 10^(10/20) = ~3.16m."""
        d = rssi_to_distance(-69.0, tx_power=-59.0, path_loss_exponent=2.0)
        expected = 10 ** (10.0 / 20.0)  # 3.162...
        assert d == pytest.approx(expected, rel=0.01)

    def test_minimum_clamp(self):
        """Very strong RSSI should clamp to 0.1m, not go to zero."""
        d = rssi_to_distance(-20.0, tx_power=-59.0)
        assert d >= 0.1

    def test_very_weak_signal(self):
        """RSSI of -100 should give a large distance."""
        d = rssi_to_distance(-100.0, tx_power=-59.0, path_loss_exponent=2.5)
        assert d > 20.0

    def test_invalid_path_loss(self):
        """Zero or negative path loss exponent should raise."""
        with pytest.raises(ValueError):
            rssi_to_distance(-60.0, path_loss_exponent=0)
        with pytest.raises(ValueError):
            rssi_to_distance(-60.0, path_loss_exponent=-1.0)

    def test_monotonic_decrease(self):
        """Distance should increase monotonically as RSSI decreases."""
        prev = 0.0
        for rssi in range(-40, -100, -5):
            d = rssi_to_distance(float(rssi))
            assert d >= prev, f"Distance should increase: rssi={rssi}, d={d}, prev={prev}"
            prev = d


# ---------------------------------------------------------------------------
# trilaterate_2d
# ---------------------------------------------------------------------------

class TestTrilaterate2d:
    """Weighted centroid trilateration tests."""

    def test_three_perfect_anchors(self):
        """Equilateral triangle with equal distances should give centroid."""
        # Three anchors at (0,0), (10,0), (5,8.66) — equilateral, same distance
        anchors = [
            (0.0, 0.0, 5.0),
            (10.0, 0.0, 5.0),
            (5.0, 8.66, 5.0),
        ]
        result = trilaterate_2d(anchors)
        assert result is not None
        # Equal distances -> simple centroid at (5.0, 2.887)
        assert result[0] == pytest.approx(5.0, abs=0.1)
        assert result[1] == pytest.approx(8.66 / 3.0, abs=0.1)

    def test_two_anchors(self):
        """Two anchors should still return an estimate."""
        anchors = [
            (0.0, 0.0, 2.0),
            (10.0, 0.0, 8.0),
        ]
        result = trilaterate_2d(anchors)
        assert result is not None
        # Closer to first anchor (distance=2 vs 8, weight ratio 16:1)
        assert result[0] < 5.0  # Should be biased toward (0,0)

    def test_one_anchor_returns_none(self):
        """Single anchor is insufficient."""
        result = trilaterate_2d([(5.0, 5.0, 1.0)])
        assert result is None

    def test_empty_returns_none(self):
        """Empty list returns None."""
        result = trilaterate_2d([])
        assert result is None

    def test_close_anchor_dominates(self):
        """Anchor with very small distance should dominate the estimate."""
        anchors = [
            (0.0, 0.0, 0.5),   # Very close
            (10.0, 0.0, 20.0),  # Far away
            (5.0, 10.0, 15.0),  # Far away
        ]
        result = trilaterate_2d(anchors)
        assert result is not None
        # Result should be very close to (0, 0)
        assert result[0] == pytest.approx(0.0, abs=1.0)
        assert result[1] == pytest.approx(0.0, abs=1.0)

    def test_known_position(self):
        """Device at (3, 4) with known distances from anchors."""
        # Anchors at (0,0), (10,0), (5,10) with exact distances
        target = (3.0, 4.0)
        anchors = [
            (0.0, 0.0, math.dist((0, 0), target)),
            (10.0, 0.0, math.dist((10, 0), target)),
            (5.0, 10.0, math.dist((5, 10), target)),
        ]
        result = trilaterate_2d(anchors)
        assert result is not None
        # Weighted centroid won't be exact, but should be in the right area
        assert result[0] == pytest.approx(3.0, abs=2.0)
        assert result[1] == pytest.approx(4.0, abs=2.0)


# ---------------------------------------------------------------------------
# estimate_position (end-to-end)
# ---------------------------------------------------------------------------

class TestEstimatePosition:
    """End-to-end estimation from sightings + node positions."""

    @pytest.fixture
    def node_positions(self):
        """Three nodes in a triangle (lat/lon coords)."""
        return {
            "node-A": (33.7490, -84.3880),  # Atlanta-ish
            "node-B": (33.7500, -84.3870),
            "node-C": (33.7495, -84.3860),
        }

    def test_three_nodes(self, node_positions):
        """Three sightings should produce a valid estimate."""
        sightings = [
            {"node_id": "node-A", "ble_rssi": -55},
            {"node_id": "node-B", "ble_rssi": -65},
            {"node_id": "node-C", "ble_rssi": -70},
        ]
        result = estimate_position(sightings, node_positions)
        assert result is not None
        assert "lat" in result
        assert "lon" in result
        assert result["anchors_used"] == 3
        assert result["method"] == "weighted_centroid"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_two_nodes(self, node_positions):
        """Two sightings should still produce an estimate with lower confidence."""
        sightings = [
            {"node_id": "node-A", "ble_rssi": -55},
            {"node_id": "node-B", "ble_rssi": -65},
        ]
        result = estimate_position(sightings, node_positions)
        assert result is not None
        assert result["anchors_used"] == 2

        # Three-node estimate should have higher confidence
        sightings_3 = sightings + [{"node_id": "node-C", "ble_rssi": -70}]
        result_3 = estimate_position(sightings_3, node_positions)
        assert result_3 is not None
        assert result_3["confidence"] >= result["confidence"]

    def test_one_node_returns_none(self, node_positions):
        """Single sighting is insufficient."""
        sightings = [{"node_id": "node-A", "ble_rssi": -55}]
        result = estimate_position(sightings, node_positions)
        assert result is None

    def test_unknown_node_ignored(self, node_positions):
        """Sightings from unknown nodes are silently dropped."""
        sightings = [
            {"node_id": "node-A", "ble_rssi": -55},
            {"node_id": "node-UNKNOWN", "ble_rssi": -45},
            {"node_id": "node-B", "ble_rssi": -65},
        ]
        result = estimate_position(sightings, node_positions)
        assert result is not None
        assert result["anchors_used"] == 2

    def test_missing_fields_ignored(self, node_positions):
        """Sightings missing required fields are silently dropped."""
        sightings = [
            {"node_id": "node-A", "ble_rssi": -55},
            {"node_id": "node-B"},  # missing ble_rssi
            {"ble_rssi": -70},      # missing node_id
            {"node_id": "node-C", "ble_rssi": -60},
        ]
        result = estimate_position(sightings, node_positions)
        assert result is not None
        assert result["anchors_used"] == 2

    def test_noisy_rssi(self, node_positions):
        """Noisy RSSI values should still produce a reasonable estimate."""
        # Simulate noisy readings — all roughly the same distance
        sightings = [
            {"node_id": "node-A", "ble_rssi": -62},
            {"node_id": "node-B", "ble_rssi": -58},
            {"node_id": "node-C", "ble_rssi": -65},
        ]
        result = estimate_position(sightings, node_positions)
        assert result is not None
        # Position should be within the bounding box of the nodes
        lats = [p[0] for p in node_positions.values()]
        lons = [p[1] for p in node_positions.values()]
        assert min(lats) - 0.001 <= result["lat"] <= max(lats) + 0.001
        assert min(lons) - 0.001 <= result["lon"] <= max(lons) + 0.001

    def test_strong_signal_pulls_toward_node(self, node_positions):
        """A very strong RSSI from one node should pull estimate toward it."""
        sightings = [
            {"node_id": "node-A", "ble_rssi": -35},  # Very close to A
            {"node_id": "node-B", "ble_rssi": -85},
            {"node_id": "node-C", "ble_rssi": -85},
        ]
        result = estimate_position(sightings, node_positions)
        assert result is not None
        # Should be much closer to node-A than the centroid
        node_a = node_positions["node-A"]
        centroid_lat = sum(p[0] for p in node_positions.values()) / 3
        dist_to_a = abs(result["lat"] - node_a[0])
        dist_to_centroid = abs(result["lat"] - centroid_lat)
        assert dist_to_a < dist_to_centroid

    def test_empty_sightings(self, node_positions):
        """Empty sightings list returns None."""
        assert estimate_position([], node_positions) is None

    def test_custom_path_loss(self, node_positions):
        """Different path_loss_exponent changes the distance scaling and can shift position."""
        sightings = [
            {"node_id": "node-A", "ble_rssi": -55},
            {"node_id": "node-B", "ble_rssi": -65},
            {"node_id": "node-C", "ble_rssi": -70},
        ]
        result_default = estimate_position(sightings, node_positions)
        # Higher path loss exponent compresses the distance range, changing relative weights
        result_custom = estimate_position(
            sightings, node_positions, path_loss_exponent=4.0
        )
        assert result_default is not None
        assert result_custom is not None
        # Both should produce valid estimates in the node neighborhood
        lats = [p[0] for p in node_positions.values()]
        assert min(lats) - 0.01 <= result_custom["lat"] <= max(lats) + 0.01


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TestModels:
    """Model validation tests."""

    def test_anchor_point(self):
        a = AnchorPoint(node_id="n1", lat=33.75, lon=-84.39, rssi=-60)
        assert a.distance == 0.0
        assert a.node_id == "n1"

    def test_position_estimate(self):
        e = PositionEstimate(lat=33.75, lon=-84.39, confidence=0.85, anchors_used=3)
        assert e.method == "weighted_centroid"
        assert e.confidence == 0.85

    def test_confidence_clamped(self):
        """Confidence must be between 0 and 1."""
        with pytest.raises(Exception):
            PositionEstimate(lat=0, lon=0, confidence=1.5, anchors_used=1)
        with pytest.raises(Exception):
            PositionEstimate(lat=0, lon=0, confidence=-0.1, anchors_used=1)


# ---------------------------------------------------------------------------
# RSSIFilter (Kalman)
# ---------------------------------------------------------------------------

class TestRSSIFilter:
    """Kalman filter for RSSI smoothing."""

    def test_initial_state(self):
        f = RSSIFilter(initial_estimate=-70.0)
        assert f.estimate == -70.0

    def test_converges_to_stable_signal(self):
        """Repeated identical readings should converge to that value."""
        f = RSSIFilter(initial_estimate=-70.0)
        for _ in range(20):
            f.update(-55.0)
        assert abs(f.estimate - (-55.0)) < 0.5

    def test_smooths_noisy_signal(self):
        """Noisy readings around -60 should produce a stable estimate near -60."""
        import random
        random.seed(42)
        f = RSSIFilter(initial_estimate=-60.0, measurement_noise=5.0)
        readings = [-60 + random.gauss(0, 5) for _ in range(50)]
        for r in readings:
            f.update(r)
        # Should be close to -60 despite noise
        assert abs(f.estimate - (-60.0)) < 3.0

    def test_responds_to_movement(self):
        """Filter should track a step change in signal."""
        f = RSSIFilter(initial_estimate=-50.0, process_noise=1.0)
        # Stabilize at -50
        for _ in range(10):
            f.update(-50.0)
        # Step change to -70
        for _ in range(20):
            f.update(-70.0)
        # Should have tracked most of the way to -70
        assert f.estimate < -65.0

    def test_reset(self):
        f = RSSIFilter()
        f.update(-80.0)
        f.update(-80.0)
        f.reset(-50.0)
        assert f.estimate == -50.0

    def test_single_update(self):
        """Single update should move estimate toward measurement."""
        f = RSSIFilter(initial_estimate=-70.0, measurement_noise=3.0)
        result = f.update(-60.0)
        # Should be between -70 and -60
        assert -70.0 < result < -60.0

    def test_improves_distance_estimate(self):
        """Filtered RSSI should give more stable distance than raw."""
        import random
        random.seed(123)
        f = RSSIFilter(initial_estimate=-65.0)

        raw_distances = []
        filtered_distances = []

        for _ in range(30):
            raw = -65 + random.gauss(0, 6)  # Noisy RSSI
            raw_distances.append(rssi_to_distance(raw))
            smoothed = f.update(raw)
            filtered_distances.append(rssi_to_distance(smoothed))

        # Variance of filtered distances should be lower
        raw_var = sum((d - sum(raw_distances)/len(raw_distances))**2 for d in raw_distances) / len(raw_distances)
        filt_var = sum((d - sum(filtered_distances)/len(filtered_distances))**2 for d in filtered_distances) / len(filtered_distances)
        assert filt_var < raw_var
