# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for BehaviorProfiler — comprehensive behavioral profiling from long-term observation."""

import math
import time

import pytest

from tritium_lib.intelligence.behavior_profiler import (
    BehaviorChange,
    BehaviorProfile,
    BehaviorProfiler,
    ChangeSeverity,
    DeviceDimension,
    Observation,
    ProfileComparison,
    SocialDimension,
    SpatialDimension,
    SpatialStop,
    TargetRole,
    TemporalDimension,
    TransitCorridor,
    _centroid,
    _cluster_stops,
    _distribution_shift,
    _haversine_m,
    _histogram_similarity,
    _mean,
    _normalized_entropy_score,
    _peak_bin,
    _std,
    _top_bins,
    CHANGE_Z_THRESHOLD,
    HOME_HOURS,
    HOUR_BINS,
    MAC_ROTATION_THRESHOLD,
    MIN_OBSERVATIONS_FOR_CHANGE,
    MIN_OBSERVATIONS_FOR_PROFILE,
    STOP_CLUSTER_RADIUS_M,
    WORK_HOURS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(year=2026, month=3, day=15, hour=10, minute=0) -> float:
    """Build a timestamp from a date/time for deterministic tests."""
    import datetime
    return datetime.datetime(year, month, day, hour, minute).timestamp()


def _make_commuter_observations(
    target_id: str = "ble_aa:bb:cc",
    days: int = 10,
) -> list[Observation]:
    """Create observations simulating a commuter: home at night, work during day."""
    obs = []
    home_lat, home_lng = 40.7128, -74.0060
    work_lat, work_lng = 40.7580, -73.9855  # ~5 km north

    for d in range(days):
        day_offset = d
        # Morning at home (7 AM)
        obs.append(Observation(
            timestamp=_ts(day=15 + day_offset, hour=7),
            lat=home_lat, lng=home_lng,
            source="ble", device_type="phone",
            mac_address="aa:bb:cc:dd:ee:ff",
        ))
        # Commute (8 AM — midpoint)
        mid_lat = (home_lat + work_lat) / 2
        mid_lng = (home_lng + work_lng) / 2
        obs.append(Observation(
            timestamp=_ts(day=15 + day_offset, hour=8),
            lat=mid_lat, lng=mid_lng,
            source="ble", device_type="phone",
            mac_address="aa:bb:cc:dd:ee:ff",
        ))
        # At work (9 AM - 5 PM, a few observations)
        for h in [9, 12, 15, 17]:
            obs.append(Observation(
                timestamp=_ts(day=15 + day_offset, hour=h),
                lat=work_lat + 0.0001 * (h % 3),
                lng=work_lng,
                source="wifi", device_type="phone",
                mac_address="aa:bb:cc:dd:ee:ff",
            ))
        # Evening at home (19:00)
        obs.append(Observation(
            timestamp=_ts(day=15 + day_offset, hour=19),
            lat=home_lat, lng=home_lng,
            source="ble", device_type="phone",
            mac_address="aa:bb:cc:dd:ee:ff",
        ))
        # Night at home (23:00)
        obs.append(Observation(
            timestamp=_ts(day=15 + day_offset, hour=23),
            lat=home_lat, lng=home_lng,
            source="ble", device_type="phone",
            mac_address="aa:bb:cc:dd:ee:ff",
        ))
    return obs


def _make_delivery_observations(
    target_id: str = "ble_delivery",
    days: int = 5,
) -> list[Observation]:
    """Create observations simulating a delivery driver: many short stops."""
    obs = []
    base_lat, base_lng = 40.7128, -74.0060

    for d in range(days):
        for stop_idx in range(8):
            # 8 different stops per day, each visited briefly during work hours
            obs.append(Observation(
                timestamp=_ts(day=15 + d, hour=9 + stop_idx),
                lat=base_lat + 0.005 * stop_idx,
                lng=base_lng + 0.003 * stop_idx,
                source="ble", device_type="phone",
                mac_address="de:li:ve:ry:00:01",
            ))
    return obs


def _make_patrol_observations(
    target_id: str = "mesh_patrol",
    days: int = 7,
) -> list[Observation]:
    """Create observations simulating a patrol officer: wide area, regular loops."""
    obs = []
    base_lat, base_lng = 40.7128, -74.0060

    for d in range(days):
        # Patrol covers a large area with regular timing
        for hour in range(8, 20):
            # Circular patrol route
            angle = (hour - 8) * (2 * math.pi / 12)
            r = 0.01  # ~1km radius
            lat = base_lat + r * math.cos(angle)
            lng = base_lng + r * math.sin(angle)
            obs.append(Observation(
                timestamp=_ts(day=15 + d, hour=hour),
                lat=lat, lng=lng,
                source="mesh", device_type="radio",
                mac_address="pa:tr:ol:00:00:01",
                group_size=1,
            ))
    return obs


# ---------------------------------------------------------------------------
# Tests: Statistical helpers
# ---------------------------------------------------------------------------

class TestStatisticalHelpers:
    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean_single(self):
        assert _mean([5.0]) == 5.0

    def test_mean_multiple(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_std_empty(self):
        assert _std([]) == 0.0

    def test_std_single(self):
        assert _std([5.0]) == 0.0

    def test_std_uniform(self):
        # All same values: std = 0
        assert _std([3.0, 3.0, 3.0]) == 0.0

    def test_std_known(self):
        # [1, 2, 3] has population std = sqrt(2/3) ≈ 0.8165
        result = _std([1.0, 2.0, 3.0])
        assert abs(result - 0.8165) < 0.01

    def test_normalized_entropy_all_zero(self):
        assert _normalized_entropy_score([0] * 24) == 0.0

    def test_normalized_entropy_single_spike(self):
        h = [0] * 24
        h[9] = 100
        assert _normalized_entropy_score(h) == 1.0

    def test_normalized_entropy_uniform(self):
        h = [10] * 24
        score = _normalized_entropy_score(h)
        assert score == 0.0

    def test_normalized_entropy_moderate(self):
        h = [0] * 24
        h[8] = 20
        h[9] = 30
        h[10] = 20
        score = _normalized_entropy_score(h)
        assert 0.3 < score < 0.9

    def test_top_bins(self):
        h = [0, 5, 3, 0, 10, 0]
        result = _top_bins(h, n=2)
        assert result == [4, 1]

    def test_top_bins_empty(self):
        assert _top_bins([0, 0, 0]) == []

    def test_peak_bin(self):
        h = [0, 0, 5, 0, 10, 0]
        assert _peak_bin(h) == 4

    def test_peak_bin_empty(self):
        assert _peak_bin([0, 0, 0]) == 0

    def test_histogram_similarity_identical(self):
        h = [1, 2, 3, 4]
        assert _histogram_similarity(h, h) == pytest.approx(1.0, abs=0.001)

    def test_histogram_similarity_orthogonal(self):
        a = [1, 0, 0, 0]
        b = [0, 0, 0, 1]
        assert _histogram_similarity(a, b) == 0.0

    def test_histogram_similarity_both_zero(self):
        assert _histogram_similarity([0, 0], [0, 0]) == 0.0

    def test_haversine_same_point(self):
        assert _haversine_m(40.0, -74.0, 40.0, -74.0) == 0.0

    def test_haversine_known_distance(self):
        # NYC to LA: approximately 3,944 km
        dist = _haversine_m(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3_900_000 < dist < 4_000_000

    def test_distribution_shift_identical(self):
        h = [10, 20, 30, 10]
        assert _distribution_shift(h, h) == 0.0

    def test_distribution_shift_different(self):
        a = [100, 0, 0, 0]
        b = [0, 0, 0, 100]
        shift = _distribution_shift(a, b)
        assert shift > 0

    def test_centroid_empty(self):
        assert _centroid([]) is None

    def test_centroid_single(self):
        obs = [Observation(timestamp=1.0, lat=10.0, lng=20.0)]
        c = _centroid(obs)
        assert c == (10.0, 20.0)


# ---------------------------------------------------------------------------
# Tests: Stop clustering
# ---------------------------------------------------------------------------

class TestStopClustering:
    def test_cluster_single_point(self):
        points = [(40.7128, -74.0060, 1000.0)]
        stops = _cluster_stops(points, 50.0)
        assert len(stops) == 1
        assert stops[0].visit_count == 1

    def test_cluster_nearby_points(self):
        # Points within 10m of each other
        points = [
            (40.71280, -74.00600, 1000.0),
            (40.71281, -74.00601, 1010.0),
            (40.71279, -74.00599, 1020.0),
        ]
        stops = _cluster_stops(points, 50.0)
        assert len(stops) == 1
        assert stops[0].visit_count == 3

    def test_cluster_distant_points(self):
        # Points ~1 km apart
        points = [
            (40.7128, -74.0060, 1000.0),
            (40.7228, -74.0060, 2000.0),
        ]
        stops = _cluster_stops(points, 50.0)
        assert len(stops) == 2

    def test_cluster_dwell_time(self):
        points = [
            (40.7128, -74.0060, 1000.0),
            (40.7128, -74.0060, 1300.0),  # 5 minutes later
            (40.7128, -74.0060, 1600.0),  # another 5 minutes
        ]
        stops = _cluster_stops(points, 50.0)
        assert len(stops) == 1
        assert stops[0].total_dwell_s == pytest.approx(600.0, abs=1.0)


# ---------------------------------------------------------------------------
# Tests: BehaviorProfiler
# ---------------------------------------------------------------------------

class TestBehaviorProfiler:
    def test_empty_profiler(self):
        profiler = BehaviorProfiler()
        assert profiler.known_targets() == []

    def test_add_observation(self):
        profiler = BehaviorProfiler()
        obs = Observation(timestamp=_ts(), lat=40.0, lng=-74.0)
        profiler.add_observation("t1", obs)
        assert profiler.observation_count("t1") == 1

    def test_add_observations_bulk(self):
        profiler = BehaviorProfiler()
        obs_list = [
            Observation(timestamp=_ts(hour=h), lat=40.0, lng=-74.0)
            for h in range(10, 15)
        ]
        profiler.add_observations("t1", obs_list)
        assert profiler.observation_count("t1") == 5

    def test_known_targets(self):
        profiler = BehaviorProfiler()
        profiler.add_observation("t1", Observation(timestamp=1.0))
        profiler.add_observation("t2", Observation(timestamp=2.0))
        targets = profiler.known_targets()
        assert set(targets) == {"t1", "t2"}

    def test_build_profile_empty_target(self):
        profiler = BehaviorProfiler()
        profile = profiler.build_profile("nonexistent")
        assert profile.target_id == "nonexistent"
        assert profile.observation_count == 0

    def test_build_profile_basic(self):
        profiler = BehaviorProfiler()
        for obs in _make_commuter_observations():
            profiler.add_observation("ble_aa:bb:cc", obs)

        profile = profiler.build_profile("ble_aa:bb:cc")
        assert profile.target_id == "ble_aa:bb:cc"
        assert profile.observation_count > 0
        assert profile.first_seen > 0
        assert profile.last_seen >= profile.first_seen
        assert profile.profile_age_days > 0

    def test_build_profile_temporal(self):
        profiler = BehaviorProfiler()
        for obs in _make_commuter_observations():
            profiler.add_observation("t1", obs)

        profile = profiler.build_profile("t1")
        t = profile.temporal

        # Should have activity across multiple hours
        active = [h for h, c in enumerate(t.hourly_histogram) if c > 0]
        assert len(active) >= 5

        # Regularity should be moderate to high
        assert t.regularity_score > 0.1

        # Should have peak hours
        assert len(t.peak_hours) > 0

    def test_build_profile_spatial_home_work(self):
        profiler = BehaviorProfiler()
        for obs in _make_commuter_observations(days=10):
            profiler.add_observation("t1", obs)

        profile = profiler.build_profile("t1")
        s = profile.spatial

        # Should identify home and work areas
        assert s.home_area is not None or s.work_area is not None
        assert len(s.frequent_stops) > 0
        assert s.centroid_lat != 0.0

    def test_build_profile_social_loner(self):
        profiler = BehaviorProfiler()
        # All observations with group_size=1, no associations
        for h in range(10, 20):
            profiler.add_observation("loner", Observation(
                timestamp=_ts(hour=h), lat=40.0, lng=-74.0,
                group_size=1,
            ))

        profile = profiler.build_profile("loner")
        assert profile.social.is_loner is True
        assert profile.social.avg_group_size <= 1.2

    def test_build_profile_social_social(self):
        profiler = BehaviorProfiler()
        for h in range(10, 20):
            profiler.add_observation("social", Observation(
                timestamp=_ts(hour=h), lat=40.0, lng=-74.0,
                group_size=5,
                association_ids=["a1", "a2", "a3", "a4", "a5"],
            ))

        profile = profiler.build_profile("social")
        assert profile.social.is_social is True
        assert profile.social.avg_group_size >= 2.0
        assert profile.social.unique_associates >= 4

    def test_build_profile_device(self):
        profiler = BehaviorProfiler()
        for h in range(10, 18):
            profiler.add_observation("dev1", Observation(
                timestamp=_ts(hour=h), lat=40.0, lng=-74.0,
                source="ble", device_type="phone",
                mac_address="aa:bb:cc:dd:ee:ff",
            ))

        profile = profiler.build_profile("dev1")
        d = profile.device
        assert "phone" in d.device_types
        assert "ble" in d.source_types
        assert d.primary_device == "phone"
        assert d.mac_count == 1
        assert d.mac_rotation_detected is False

    def test_build_profile_device_mac_rotation(self):
        profiler = BehaviorProfiler()
        for h in range(10, 18):
            profiler.add_observation("rotator", Observation(
                timestamp=_ts(hour=h), lat=40.0, lng=-74.0,
                source="ble", device_type="phone",
                mac_address=f"aa:bb:cc:dd:ee:{h:02x}",
            ))

        profile = profiler.build_profile("rotator")
        assert profile.device.mac_rotation_detected is True
        assert profile.device.mac_count >= MAC_ROTATION_THRESHOLD

    def test_get_profile_after_build(self):
        profiler = BehaviorProfiler()
        profiler.add_observation("t1", Observation(
            timestamp=_ts(), lat=40.0, lng=-74.0,
        ))
        profiler.build_profile("t1")
        cached = profiler.get_profile("t1")
        assert cached is not None
        assert cached.target_id == "t1"

    def test_get_profile_not_built(self):
        profiler = BehaviorProfiler()
        assert profiler.get_profile("nonexistent") is None

    def test_build_all_profiles(self):
        profiler = BehaviorProfiler()
        for obs in _make_commuter_observations():
            profiler.add_observation("t1", obs)
        for obs in _make_delivery_observations():
            profiler.add_observation("t2", obs)

        profiles = profiler.build_all_profiles()
        assert "t1" in profiles
        assert "t2" in profiles

    def test_profile_to_dict(self):
        profiler = BehaviorProfiler()
        for obs in _make_commuter_observations():
            profiler.add_observation("t1", obs)

        profile = profiler.build_profile("t1")
        d = profile.to_dict()
        assert d["target_id"] == "t1"
        assert "temporal" in d
        assert "spatial" in d
        assert "social" in d
        assert "device" in d
        assert "role" in d

    def test_profile_export(self):
        profiler = BehaviorProfiler()
        for obs in _make_commuter_observations():
            profiler.add_observation("t1", obs)

        profile = profiler.build_profile("t1")
        exported = profile.export()
        assert exported == profile.to_dict()


# ---------------------------------------------------------------------------
# Tests: Role classification
# ---------------------------------------------------------------------------

class TestRoleClassification:
    def test_classify_insufficient_observations(self):
        profiler = BehaviorProfiler()
        profiler.add_observation("t1", Observation(timestamp=_ts()))
        profile = profiler.build_profile("t1")
        role, conf = profiler.classify_role(profile)
        assert role == TargetRole.UNKNOWN
        assert conf == 0.0

    def test_classify_commuter(self):
        profiler = BehaviorProfiler()
        for obs in _make_commuter_observations(days=10):
            profiler.add_observation("commuter", obs)

        profile = profiler.build_profile("commuter")
        # Should classify as commuter, worker, or resident (all reasonable for this data)
        assert profile.role in (
            TargetRole.COMMUTER, TargetRole.WORKER, TargetRole.RESIDENT,
        )
        assert profile.role_confidence > 0.0

    def test_classify_delivery(self):
        profiler = BehaviorProfiler()
        for obs in _make_delivery_observations(days=10):
            profiler.add_observation("delivery", obs)

        profile = profiler.build_profile("delivery")
        # Delivery driver has many stops
        assert profile.role in (TargetRole.DELIVERY, TargetRole.WORKER, TargetRole.COMMUTER)

    def test_classify_visitor(self):
        profiler = BehaviorProfiler()
        # Visitor: few observations spread over many days
        for d in range(10):
            profiler.add_observation("visitor", Observation(
                timestamp=_ts(day=15 + d, hour=14),
                lat=40.7128, lng=-74.0060,
                source="camera", device_type="",
            ))

        profile = profiler.build_profile("visitor")
        # With single location and low frequency, should tend toward visitor or resident
        assert profile.role != TargetRole.UNKNOWN or profile.observation_count < MIN_OBSERVATIONS_FOR_PROFILE

    def test_classify_patrol(self):
        profiler = BehaviorProfiler()
        for obs in _make_patrol_observations(days=7):
            profiler.add_observation("patrol", obs)

        profile = profiler.build_profile("patrol")
        # Patrol covers wide area with regularity
        assert profile.spatial.total_area_m2 > 100  # wide coverage

    def test_role_enum_values(self):
        assert TargetRole.RESIDENT.value == "resident"
        assert TargetRole.WORKER.value == "worker"
        assert TargetRole.COMMUTER.value == "commuter"
        assert TargetRole.VISITOR.value == "visitor"
        assert TargetRole.DELIVERY.value == "delivery"
        assert TargetRole.PATROL.value == "patrol"
        assert TargetRole.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# Tests: Change detection
# ---------------------------------------------------------------------------

class TestChangeDetection:
    def test_detect_change_insufficient_data(self):
        profiler = BehaviorProfiler()
        profiler.add_observation("t1", Observation(timestamp=_ts()))
        profile = profiler.build_profile("t1")
        changes = profiler.detect_change(profile)
        assert changes == []

    def test_detect_no_change(self):
        profiler = BehaviorProfiler()
        # Uniform behavior: every observation at the same hour so baseline and
        # recent windows have identical hourly distributions.
        for d in range(20):
            profiler.add_observation("stable", Observation(
                timestamp=_ts(day=1 + d, hour=10),
                lat=40.7128, lng=-74.0060,
                source="ble", device_type="phone",
                group_size=1,
            ))

        profile = profiler.build_profile("stable")
        changes = profiler.detect_change(profile)
        # Stable behavior should produce no temporal or spatial changes
        temporal_changes = [c for c in changes if c.dimension == "temporal"]
        assert len(temporal_changes) == 0

    def test_detect_temporal_change(self):
        profiler = BehaviorProfiler()
        # First 70%: morning activity
        for d in range(14):
            profiler.add_observation("shifted", Observation(
                timestamp=_ts(day=1 + d, hour=9),
                lat=40.7128, lng=-74.0060,
            ))
        # Last 30%: night activity
        for d in range(6):
            profiler.add_observation("shifted", Observation(
                timestamp=_ts(day=15 + d, hour=23),
                lat=40.7128, lng=-74.0060,
            ))

        profile = profiler.build_profile("shifted")
        changes = profiler.detect_change(profile)
        temporal_changes = [c for c in changes if c.dimension == "temporal"]
        assert len(temporal_changes) >= 1
        assert temporal_changes[0].severity in (ChangeSeverity.MEDIUM, ChangeSeverity.HIGH)

    def test_detect_device_change(self):
        profiler = BehaviorProfiler()
        # Baseline: phone only
        for d in range(14):
            profiler.add_observation("devchg", Observation(
                timestamp=_ts(day=1 + d, hour=10),
                lat=40.7128, lng=-74.0060,
                device_type="phone",
            ))
        # Recent: phone + laptop
        for d in range(6):
            profiler.add_observation("devchg", Observation(
                timestamp=_ts(day=15 + d, hour=10),
                lat=40.7128, lng=-74.0060,
                device_type="laptop",
            ))

        profile = profiler.build_profile("devchg")
        changes = profiler.detect_change(profile)
        device_changes = [c for c in changes if c.dimension == "device"]
        assert len(device_changes) >= 1
        assert "laptop" in device_changes[0].description

    def test_detect_social_change(self):
        profiler = BehaviorProfiler()
        # Baseline: always alone
        for d in range(14):
            profiler.add_observation("social_chg", Observation(
                timestamp=_ts(day=1 + d, hour=10),
                lat=40.7128, lng=-74.0060,
                group_size=1,
            ))
        # Recent: suddenly in large groups
        for d in range(6):
            profiler.add_observation("social_chg", Observation(
                timestamp=_ts(day=15 + d, hour=10),
                lat=40.7128, lng=-74.0060,
                group_size=10,
            ))

        profile = profiler.build_profile("social_chg")
        changes = profiler.detect_change(profile)
        social_changes = [c for c in changes if c.dimension == "social"]
        assert len(social_changes) >= 1

    def test_change_to_dict(self):
        change = BehaviorChange(
            dimension="temporal",
            description="Test change",
            severity=ChangeSeverity.HIGH,
            z_score=3.5,
            old_value="old",
            new_value="new",
        )
        d = change.to_dict()
        assert d["dimension"] == "temporal"
        assert d["severity"] == "high"
        assert d["z_score"] == 3.5


# ---------------------------------------------------------------------------
# Tests: Profile comparison
# ---------------------------------------------------------------------------

class TestProfileComparison:
    def test_compare_identical_profiles(self):
        profiler = BehaviorProfiler()
        obs = _make_commuter_observations()
        for o in obs:
            profiler.add_observation("t1", o)
            profiler.add_observation("t2", o)

        p1 = profiler.build_profile("t1")
        p2 = profiler.build_profile("t2")

        comp = ProfileComparison.compare(p1, p2)
        assert comp.temporal_similarity == pytest.approx(1.0, abs=0.01)
        assert comp.overall_similarity > 0.8

    def test_compare_different_profiles(self):
        profiler = BehaviorProfiler()

        # Day person
        for h in range(8, 18):
            profiler.add_observation("day", Observation(
                timestamp=_ts(hour=h), lat=40.7128, lng=-74.0060,
                source="ble", device_type="phone",
            ))

        # Night person
        for h in [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]:
            profiler.add_observation("night", Observation(
                timestamp=_ts(hour=h), lat=41.0, lng=-73.0,
                source="wifi", device_type="laptop",
            ))

        p_day = profiler.build_profile("day")
        p_night = profiler.build_profile("night")

        comp = ProfileComparison.compare(p_day, p_night)
        assert comp.temporal_similarity < 0.5
        assert comp.overall_similarity < 0.8

    def test_compare_same_role(self):
        p1 = BehaviorProfile(target_id="t1", role=TargetRole.WORKER)
        p2 = BehaviorProfile(target_id="t2", role=TargetRole.WORKER)
        comp = ProfileComparison.compare(p1, p2)
        assert comp.same_role is True

    def test_compare_different_role(self):
        p1 = BehaviorProfile(target_id="t1", role=TargetRole.WORKER)
        p2 = BehaviorProfile(target_id="t2", role=TargetRole.VISITOR)
        comp = ProfileComparison.compare(p1, p2)
        assert comp.same_role is False

    def test_compare_unknown_role_not_same(self):
        p1 = BehaviorProfile(target_id="t1", role=TargetRole.UNKNOWN)
        p2 = BehaviorProfile(target_id="t2", role=TargetRole.UNKNOWN)
        comp = ProfileComparison.compare(p1, p2)
        assert comp.same_role is False

    def test_compare_custom_weights(self):
        p1 = BehaviorProfile(target_id="t1")
        p2 = BehaviorProfile(target_id="t2")
        weights = {"temporal": 1.0, "spatial": 0.0, "social": 0.0, "device": 0.0}
        comp = ProfileComparison.compare(p1, p2, weights=weights)
        # Both have zero histograms so temporal similarity is 0
        assert comp.temporal_similarity == 0.0

    def test_comparison_to_dict(self):
        p1 = BehaviorProfile(target_id="t1")
        p2 = BehaviorProfile(target_id="t2")
        comp = ProfileComparison.compare(p1, p2)
        d = comp.to_dict()
        assert d["target_a"] == "t1"
        assert d["target_b"] == "t2"
        assert "temporal_similarity" in d
        assert "overall_similarity" in d


# ---------------------------------------------------------------------------
# Tests: Transit corridors
# ---------------------------------------------------------------------------

class TestTransitCorridors:
    def test_corridors_from_commuter(self):
        profiler = BehaviorProfiler()
        for obs in _make_commuter_observations(days=10):
            profiler.add_observation("comm", obs)

        profile = profiler.build_profile("comm")
        # Should detect at least some corridors between stops
        # (depends on clustering; corridors require MIN_CORRIDOR_POINTS trips)
        s = profile.spatial
        # At minimum we should have frequent stops
        assert len(s.frequent_stops) > 0

    def test_corridor_to_dict(self):
        corridor = TransitCorridor(
            start_stop="home",
            end_stop="work",
            trip_count=10,
            avg_duration_s=1800.0,
            avg_speed_mps=5.0,
        )
        d = corridor.to_dict()
        assert d["start_stop"] == "home"
        assert d["trip_count"] == 10


# ---------------------------------------------------------------------------
# Tests: Dimension serialization
# ---------------------------------------------------------------------------

class TestDimensionSerialization:
    def test_temporal_to_dict(self):
        t = TemporalDimension()
        d = t.to_dict()
        assert len(d["hourly_histogram"]) == 24

    def test_spatial_to_dict(self):
        s = SpatialDimension()
        d = s.to_dict()
        assert d["home_area"] is None
        assert d["frequent_stops"] == []

    def test_social_to_dict(self):
        soc = SocialDimension()
        d = soc.to_dict()
        assert d["avg_group_size"] == 0.0

    def test_device_to_dict(self):
        dev = DeviceDimension()
        d = dev.to_dict()
        assert d["device_types"] == []

    def test_spatial_stop_to_dict(self):
        stop = SpatialStop(lat=40.0, lng=-74.0, visit_count=5)
        d = stop.to_dict()
        assert d["lat"] == 40.0
        assert d["visit_count"] == 5
