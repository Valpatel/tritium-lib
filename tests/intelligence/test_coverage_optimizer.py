# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.intelligence.coverage_optimizer module."""
import math

import pytest

from tritium_lib.intelligence.coverage_optimizer import (
    CoverageCell,
    CoverageGap,
    CoverageMap,
    PlacedSensor,
    PlacementResult,
    RedundancyZone,
    SensorSpec,
    SENSOR_RANGE_PROFILES,
    _cell_detection_prob,
    _detection_probability,
    _euclidean,
    _is_in_fov,
    _kmeans_cluster,
    build_coverage_map,
    coverage_gaps,
    optimize_placement,
    redundancy_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_area():
    """A 100x100 meter area."""
    return (0.0, 0.0, 100.0, 100.0)


@pytest.fixture
def ble_spec():
    """BLE radio sensor spec."""
    return SensorSpec.from_type("ble_radio")


@pytest.fixture
def wifi_spec():
    """WiFi radio sensor spec."""
    return SensorSpec.from_type("wifi_radio")


@pytest.fixture
def camera_spec():
    """Camera sensor spec."""
    return SensorSpec.from_type("camera")


@pytest.fixture
def center_ble_sensor(ble_spec):
    """A BLE sensor placed at center of 100x100 area."""
    return PlacedSensor(
        sensor_id="ble_center",
        x=50.0,
        y=50.0,
        spec=ble_spec,
    )


# ---------------------------------------------------------------------------
# SensorSpec tests
# ---------------------------------------------------------------------------

class TestSensorSpec:

    def test_from_known_type_ble(self):
        spec = SensorSpec.from_type("ble_radio")
        assert spec.sensor_type == "ble_radio"
        assert spec.max_range_m == 30.0
        assert spec.fov_degrees == 360.0

    def test_from_known_type_wifi(self):
        spec = SensorSpec.from_type("wifi_radio")
        assert spec.sensor_type == "wifi_radio"
        assert spec.max_range_m == 80.0

    def test_from_known_type_camera(self):
        spec = SensorSpec.from_type("camera")
        assert spec.sensor_type == "camera"
        assert spec.fov_degrees == 90.0
        assert spec.max_range_m == 50.0

    def test_from_unknown_type_uses_defaults(self):
        spec = SensorSpec.from_type("unknown_sensor")
        assert spec.sensor_type == "unknown_sensor"
        assert spec.max_range_m == 50.0  # fallback

    def test_all_profiles_exist(self):
        for stype in SENSOR_RANGE_PROFILES:
            spec = SensorSpec.from_type(stype)
            assert spec.sensor_type == stype
            assert spec.max_range_m > 0


# ---------------------------------------------------------------------------
# Detection probability model tests
# ---------------------------------------------------------------------------

class TestDetectionProbability:

    def test_zero_distance_returns_one(self, ble_spec):
        assert _detection_probability(0.0, ble_spec) == 1.0

    def test_beyond_max_range_returns_zero(self, ble_spec):
        assert _detection_probability(ble_spec.max_range_m + 1.0, ble_spec) == 0.0

    def test_probability_decreases_with_distance(self, ble_spec):
        p_near = _detection_probability(5.0, ble_spec)
        p_far = _detection_probability(25.0, ble_spec)
        assert p_near > p_far

    def test_wifi_has_longer_range_than_ble(self):
        ble = SensorSpec.from_type("ble_radio")
        wifi = SensorSpec.from_type("wifi_radio")
        # At 50m, WiFi should still detect, BLE should not
        p_ble = _detection_probability(50.0, ble)
        p_wifi = _detection_probability(50.0, wifi)
        assert p_ble == 0.0  # beyond 30m range
        assert p_wifi > 0.0  # within 80m range

    def test_camera_cosine_falloff(self, camera_spec):
        p_close = _detection_probability(5.0, camera_spec)
        p_mid = _detection_probability(25.0, camera_spec)
        p_far = _detection_probability(49.0, camera_spec)
        assert p_close > p_mid > p_far > 0.0

    def test_probability_always_bounded_zero_one(self):
        for stype in SENSOR_RANGE_PROFILES:
            spec = SensorSpec.from_type(stype)
            for dist in [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
                p = _detection_probability(dist, spec)
                assert 0.0 <= p <= 1.0, f"{stype} at {dist}m: p={p}"


# ---------------------------------------------------------------------------
# FOV tests
# ---------------------------------------------------------------------------

class TestFieldOfView:

    def test_omnidirectional_always_in_fov(self):
        assert _is_in_fov(0, 0, 10, 10, SensorSpec(fov_degrees=360.0))
        assert _is_in_fov(0, 0, -10, -10, SensorSpec(fov_degrees=360.0))

    def test_directional_north_facing(self):
        spec = SensorSpec(fov_degrees=90.0, heading_degrees=0.0)
        # North (+Y) should be in FOV
        assert _is_in_fov(0, 0, 0, 10, spec)
        # South (-Y) should NOT be in FOV
        assert not _is_in_fov(0, 0, 0, -10, spec)
        # East (+X) is 90 deg from north — outside half-FOV of 45 deg
        assert not _is_in_fov(0, 0, 10, 0, spec)
        # NE at 30 degrees — within 45-degree half-FOV
        assert _is_in_fov(0, 0, 5, 8.66, spec)

    def test_directional_south_facing(self):
        spec = SensorSpec(fov_degrees=90.0, heading_degrees=180.0)
        # South (-Y) should be in FOV
        assert _is_in_fov(0, 0, 0, -10, spec)
        # North (+Y) should NOT be in FOV
        assert not _is_in_fov(0, 0, 0, 10, spec)

    def test_narrow_fov_excludes_sides(self):
        spec = SensorSpec(fov_degrees=30.0, heading_degrees=0.0)
        # Directly north: in FOV
        assert _is_in_fov(0, 0, 0, 10, spec)
        # 45 degrees east: out of 30-degree FOV
        assert not _is_in_fov(0, 0, 10, 10, spec)

    def test_same_position_always_in_fov(self):
        spec = SensorSpec(fov_degrees=10.0, heading_degrees=90.0)
        assert _is_in_fov(5, 5, 5, 5, spec)


# ---------------------------------------------------------------------------
# Coverage map tests
# ---------------------------------------------------------------------------

class TestBuildCoverageMap:

    def test_empty_sensors_zero_coverage(self, small_area):
        cmap = build_coverage_map([], small_area, resolution=10)
        assert cmap.overall_coverage == 0.0
        assert cmap.avg_detection_prob == 0.0
        assert cmap.max_detection_prob == 0.0
        assert len(cmap.cells) == 10
        assert len(cmap.cells[0]) == 10

    def test_single_sensor_provides_coverage(self, small_area, center_ble_sensor):
        cmap = build_coverage_map([center_ble_sensor], small_area, resolution=10)
        assert cmap.overall_coverage > 0.0
        assert cmap.max_detection_prob > 0.0
        # Center cell (55, 55) is ~7m from sensor at (50, 50) — should detect
        center_cell = cmap.cells[5][5]
        assert center_cell.detection_prob > 0.3
        assert center_cell.sensor_count == 1

    def test_more_sensors_increase_coverage(self, small_area, ble_spec):
        s1 = PlacedSensor(sensor_id="s1", x=25, y=25, spec=ble_spec)
        s2 = PlacedSensor(sensor_id="s2", x=75, y=75, spec=ble_spec)
        s3 = PlacedSensor(sensor_id="s3", x=25, y=75, spec=ble_spec)
        s4 = PlacedSensor(sensor_id="s4", x=75, y=25, spec=ble_spec)

        cmap1 = build_coverage_map([s1], small_area, resolution=10)
        cmap4 = build_coverage_map([s1, s2, s3, s4], small_area, resolution=10)
        assert cmap4.overall_coverage > cmap1.overall_coverage

    def test_sensor_count_tracked_per_cell(self, small_area, ble_spec):
        # Two sensors close together
        s1 = PlacedSensor(sensor_id="s1", x=50, y=50, spec=ble_spec)
        s2 = PlacedSensor(sensor_id="s2", x=52, y=50, spec=ble_spec)
        cmap = build_coverage_map([s1, s2], small_area, resolution=10)
        center = cmap.cells[5][5]
        assert center.sensor_count == 2
        assert "s1" in center.contributing_sensors
        assert "s2" in center.contributing_sensors

    def test_degenerate_area_returns_empty(self):
        cmap = build_coverage_map([], (0, 0, 0, 0), resolution=5)
        assert cmap.overall_coverage == 0.0
        assert cmap.cells == []

    def test_resolution_clamped(self, small_area):
        cmap = build_coverage_map([], small_area, resolution=1)
        assert cmap.resolution == 2  # minimum
        cmap2 = build_coverage_map([], small_area, resolution=500)
        assert cmap2.resolution == 200  # maximum


# ---------------------------------------------------------------------------
# CoverageMap serialization tests
# ---------------------------------------------------------------------------

class TestCoverageMapSerialization:

    def test_to_heatmap_structure(self, small_area, center_ble_sensor):
        cmap = build_coverage_map([center_ble_sensor], small_area, resolution=5)
        hm = cmap.to_heatmap()
        assert "grid" in hm
        assert "area" in hm
        assert "resolution" in hm
        assert "overall_coverage" in hm
        assert "sensor_count_grid" in hm
        assert len(hm["grid"]) == 5
        assert len(hm["grid"][0]) == 5
        assert len(hm["sensor_count_grid"]) == 5

    def test_to_dict_structure(self, small_area, center_ble_sensor):
        cmap = build_coverage_map([center_ble_sensor], small_area, resolution=5)
        d = cmap.to_dict()
        assert "area" in d
        assert "resolution" in d
        assert "overall_coverage" in d
        assert "cell_count" in d
        assert d["cell_count"] == 25

    def test_heatmap_values_bounded(self, small_area, center_ble_sensor):
        cmap = build_coverage_map([center_ble_sensor], small_area, resolution=10)
        hm = cmap.to_heatmap()
        for row in hm["grid"]:
            for val in row:
                assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Optimize placement tests
# ---------------------------------------------------------------------------

class TestOptimizePlacement:

    def test_basic_placement(self, small_area):
        result = optimize_placement(
            small_area, sensor_count=3, sensor_types=["ble_radio"], resolution=10,
        )
        assert len(result.sensors) == 3
        assert result.total_coverage > 0.0
        assert result.coverage_map is not None

    def test_sensors_spread_across_area(self, small_area):
        result = optimize_placement(
            small_area, sensor_count=4, sensor_types=["ble_radio"], resolution=10,
        )
        # Sensors should not all be in the same spot
        positions = [(s.x, s.y) for s in result.sensors]
        unique = set(positions)
        assert len(unique) == 4

    def test_more_sensors_improve_coverage(self, small_area):
        r1 = optimize_placement(
            small_area, sensor_count=1, sensor_types=["ble_radio"], resolution=10,
        )
        r4 = optimize_placement(
            small_area, sensor_count=4, sensor_types=["ble_radio"], resolution=10,
        )
        assert r4.total_coverage >= r1.total_coverage

    def test_zero_sensors_returns_empty(self, small_area):
        result = optimize_placement(
            small_area, sensor_count=0, sensor_types=["ble_radio"], resolution=10,
        )
        assert len(result.sensors) == 0

    def test_existing_sensors_respected(self, small_area, ble_spec):
        existing = [
            PlacedSensor(sensor_id="existing_1", x=25, y=25, spec=ble_spec),
        ]
        result = optimize_placement(
            small_area,
            sensor_count=1,
            sensor_types=["ble_radio"],
            resolution=10,
            existing_sensors=existing,
        )
        assert len(result.sensors) == 1
        new_sensor = result.sensors[0]
        # New sensor should NOT be placed near the existing one
        dist = math.sqrt((new_sensor.x - 25) ** 2 + (new_sensor.y - 25) ** 2)
        assert dist > 10.0  # should be placed in a gap, away from existing

    def test_mixed_sensor_types(self, small_area):
        result = optimize_placement(
            small_area,
            sensor_count=3,
            sensor_types=["ble_radio", "wifi_radio", "camera"],
            resolution=10,
        )
        assert len(result.sensors) == 3
        types = [s.spec.sensor_type for s in result.sensors]
        assert types[0] == "ble_radio"
        assert types[1] == "wifi_radio"
        assert types[2] == "camera"

    def test_result_serializable(self, small_area):
        result = optimize_placement(
            small_area, sensor_count=2, sensor_types=["ble_radio"], resolution=10,
        )
        d = result.to_dict()
        assert d["sensor_count"] == 2
        assert "sensors" in d
        assert "total_coverage" in d
        assert "coverage_map" in d

    def test_degenerate_area(self):
        result = optimize_placement(
            (0, 0, 0, 0), sensor_count=2, sensor_types=["ble_radio"],
        )
        assert len(result.sensors) == 0


# ---------------------------------------------------------------------------
# Coverage gaps tests
# ---------------------------------------------------------------------------

class TestCoverageGaps:

    def test_no_sensors_all_gaps(self, small_area):
        gaps = coverage_gaps([], small_area, resolution=10, gap_threshold=0.1)
        assert len(gaps) > 0
        # All cells should be gaps
        total_cells = sum(g.cell_count for g in gaps)
        assert total_cells == 100  # 10x10

    def test_full_coverage_no_gaps(self, small_area):
        # Place many WiFi sensors to cover entire area
        wifi = SensorSpec.from_type("wifi_radio")
        sensors = [
            PlacedSensor(sensor_id=f"w{i}{j}", x=25+50*j, y=25+50*i, spec=wifi)
            for i in range(2) for j in range(2)
        ]
        gaps = coverage_gaps(
            sensors, small_area, resolution=10, gap_threshold=0.05,
        )
        # Should have few or no gaps with 4 WiFi sensors covering 100m area
        total_gap_cells = sum(g.cell_count for g in gaps)
        assert total_gap_cells < 50  # most area should be covered

    def test_gap_severity_bounded(self, small_area):
        gaps = coverage_gaps([], small_area, resolution=10)
        for g in gaps:
            assert 0.0 <= g.severity <= 1.0

    def test_gaps_sorted_by_severity(self, small_area, ble_spec):
        sensors = [
            PlacedSensor(sensor_id="s1", x=10, y=10, spec=ble_spec),
        ]
        gaps = coverage_gaps(sensors, small_area, resolution=10)
        for i in range(len(gaps) - 1):
            assert gaps[i].severity >= gaps[i + 1].severity

    def test_gap_serialization(self, small_area):
        gaps = coverage_gaps([], small_area, resolution=5)
        for g in gaps:
            d = g.to_dict()
            assert "center_x" in d
            assert "center_y" in d
            assert "severity" in d
            assert "cell_count" in d


# ---------------------------------------------------------------------------
# Redundancy analysis tests
# ---------------------------------------------------------------------------

class TestRedundancyAnalysis:

    def test_no_sensors_no_redundancy(self, small_area):
        zones = redundancy_analysis([], small_area, resolution=10)
        assert len(zones) == 0

    def test_single_sensor_no_redundancy(self, small_area, center_ble_sensor):
        zones = redundancy_analysis(
            [center_ble_sensor], small_area, resolution=10, redundancy_threshold=2,
        )
        assert len(zones) == 0

    def test_overlapping_sensors_detected(self, small_area, ble_spec):
        # Place 4 sensors very close together — massive overlap
        sensors = [
            PlacedSensor(sensor_id=f"s{i}", x=50+i, y=50, spec=ble_spec)
            for i in range(4)
        ]
        zones = redundancy_analysis(
            sensors, small_area, resolution=10, redundancy_threshold=3,
        )
        assert len(zones) > 0
        assert zones[0].max_sensor_count >= 3

    def test_spread_sensors_low_redundancy(self, small_area, ble_spec):
        # Sensors at corners — minimal overlap for BLE (30m range)
        sensors = [
            PlacedSensor(sensor_id="s0", x=5, y=5, spec=ble_spec),
            PlacedSensor(sensor_id="s1", x=95, y=5, spec=ble_spec),
            PlacedSensor(sensor_id="s2", x=5, y=95, spec=ble_spec),
            PlacedSensor(sensor_id="s3", x=95, y=95, spec=ble_spec),
        ]
        zones = redundancy_analysis(
            sensors, small_area, resolution=10, redundancy_threshold=3,
        )
        # With 30m range in 100m area, corner sensors don't overlap much
        total_redundant = sum(z.cell_count for z in zones)
        assert total_redundant < 20

    def test_redundancy_serialization(self, small_area, ble_spec):
        sensors = [
            PlacedSensor(sensor_id=f"s{i}", x=50, y=50, spec=ble_spec)
            for i in range(5)
        ]
        zones = redundancy_analysis(
            sensors, small_area, resolution=10, redundancy_threshold=2,
        )
        for z in zones:
            d = z.to_dict()
            assert "center_x" in d
            assert "max_sensor_count" in d
            assert "avg_sensor_count" in d


# ---------------------------------------------------------------------------
# K-means clustering tests
# ---------------------------------------------------------------------------

class TestKmeansClustering:

    def test_empty_input(self):
        assert _kmeans_cluster([], 3) == []

    def test_k_zero(self):
        assert _kmeans_cluster([(0, 0, 0)], 0) == []

    def test_k_exceeds_points(self):
        pts = [(0, 0, 1), (10, 10, 2)]
        result = _kmeans_cluster(pts, 5)
        assert len(result) == 2  # one cluster per point

    def test_two_distinct_clusters(self):
        pts = [
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
            (100, 100, 2), (101, 100, 2), (100, 101, 2), (101, 101, 2),
        ]
        clusters = _kmeans_cluster(pts, 2)
        assert len(clusters) == 2
        # Each cluster should have 4 points
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [4, 4]

    def test_single_cluster(self):
        pts = [(5, 5, 1), (6, 5, 1), (5, 6, 1)]
        clusters = _kmeans_cluster(pts, 1)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3


# ---------------------------------------------------------------------------
# Euclidean distance test
# ---------------------------------------------------------------------------

class TestEuclidean:

    def test_zero_distance(self):
        assert _euclidean(0, 0, 0, 0) == 0.0

    def test_known_distance(self):
        assert abs(_euclidean(0, 0, 3, 4) - 5.0) < 1e-10

    def test_symmetry(self):
        assert abs(_euclidean(1, 2, 5, 8) - _euclidean(5, 8, 1, 2)) < 1e-10


# ---------------------------------------------------------------------------
# Integration / end-to-end test
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_optimize_then_check_gaps(self, small_area):
        """Optimize, then verify gaps are reduced."""
        # Initial state: no sensors, lots of gaps
        initial_gaps = coverage_gaps([], small_area, resolution=10)
        initial_gap_cells = sum(g.cell_count for g in initial_gaps)

        # Optimize placement
        result = optimize_placement(
            small_area, sensor_count=4, sensor_types=["ble_radio"], resolution=10,
        )

        # Check gaps after placement
        post_gaps = coverage_gaps(
            result.sensors, small_area, resolution=10, gap_threshold=0.1,
        )
        post_gap_cells = sum(g.cell_count for g in post_gaps)

        assert post_gap_cells < initial_gap_cells

    def test_heatmap_round_trip(self, small_area):
        """Build map, export heatmap, verify structure."""
        result = optimize_placement(
            small_area, sensor_count=2, sensor_types=["wifi_radio"], resolution=10,
        )
        heatmap = result.coverage_map.to_heatmap()
        assert heatmap["resolution"] == 10
        assert len(heatmap["grid"]) == 10
        assert heatmap["overall_coverage"] == result.total_coverage
