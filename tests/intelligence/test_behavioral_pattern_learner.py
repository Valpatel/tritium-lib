# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for BehavioralPatternLearner — learns normal behavioral patterns and alerts on deviations."""

import math
import time

import pytest

from tritium_lib.intelligence.behavioral_pattern_learner import (
    BehavioralPatternLearner,
    BehavioralProfile,
    DeviationResult,
    FrequentZone,
    LearnedRoute,
    LearnedSchedule,
    LearnedWaypoint,
    ScheduleObservation,
    _mean_std,
    MIN_ROUTE_POINTS,
    MIN_SCHEDULE_OBSERVATIONS,
    SCHEDULE_BINS,
    DOW_BINS,
)
from tritium_lib.tracking.target_history import TargetHistory
from tritium_lib.tracking.movement_patterns import MovementPatternAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trail(
    waypoints: list[tuple[float, float]],
    start_time: float = 1000.0,
    dt: float = 10.0,
) -> list[tuple[float, float, float]]:
    """Build a trail from waypoints with uniform time spacing."""
    trail = []
    t = start_time
    for x, y in waypoints:
        trail.append((x, y, t))
        t += dt
    return trail


def _make_commute_trail(
    home: tuple[float, float] = (0.0, 0.0),
    work: tuple[float, float] = (100.0, 50.0),
    steps: int = 20,
    start_time: float = 1000.0,
    dt: float = 5.0,
) -> list[tuple[float, float, float]]:
    """Build a linear commute trail from home to work."""
    trail = []
    t = start_time
    for i in range(steps):
        frac = i / max(steps - 1, 1)
        x = home[0] + (work[0] - home[0]) * frac
        y = home[1] + (work[1] - home[1]) * frac
        trail.append((x, y, t))
        t += dt
    return trail


def _make_schedule_timestamps(
    hours: list[float],
    base_epoch: float = 1711929600.0,  # 2024-04-01 00:00:00 UTC (Monday)
    days: int = 7,
) -> list[float]:
    """Create timestamp list for schedule learning: given hours repeated over days."""
    timestamps = []
    for day in range(days):
        for hour in hours:
            ts = base_epoch + day * 86400 + hour * 3600
            timestamps.append(ts)
    return timestamps


# ---------------------------------------------------------------------------
# Tests: Utility
# ---------------------------------------------------------------------------

class TestMeanStd:
    def test_empty(self):
        mean, std = _mean_std([])
        assert mean == 0.0
        assert std == 0.0

    def test_single(self):
        mean, std = _mean_std([5.0])
        assert mean == 5.0
        assert std == 0.0

    def test_pair(self):
        mean, std = _mean_std([0.0, 10.0])
        assert mean == 5.0
        assert std > 0

    def test_known_values(self):
        mean, std = _mean_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(mean - 5.0) < 0.01
        assert std > 0


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_init(self):
        learner = BehavioralPatternLearner()
        assert learner._history is None
        assert learner._analyzer is None
        stats = learner.get_stats()
        assert stats["total_targets"] == 0

    def test_init_with_history(self):
        history = TargetHistory()
        analyzer = MovementPatternAnalyzer(history)
        learner = BehavioralPatternLearner(history=history, analyzer=analyzer)
        assert learner._history is history
        assert learner._analyzer is analyzer

    def test_custom_thresholds(self):
        learner = BehavioralPatternLearner(
            route_deviation_threshold=5.0,
            schedule_deviation_threshold=4.0,
        )
        assert learner._route_threshold == 5.0
        assert learner._schedule_threshold == 4.0


# ---------------------------------------------------------------------------
# Tests: Route learning
# ---------------------------------------------------------------------------

class TestLearnRoute:
    def test_no_history_no_trail(self):
        learner = BehavioralPatternLearner()
        result = learner.learn_route("target_1")
        assert result is None

    def test_too_few_points(self):
        learner = BehavioralPatternLearner()
        trail = _make_trail([(0, 0), (1, 1), (2, 2)])
        result = learner.learn_route("target_1", trail=trail)
        assert result is None

    def test_learn_simple_route(self):
        trail = _make_commute_trail(steps=20)
        learner = BehavioralPatternLearner()
        route = learner.learn_route("target_1", trail=trail)
        assert route is not None
        assert isinstance(route, LearnedRoute)
        assert route.target_id == "target_1"
        assert len(route.waypoints) > 0
        assert route.total_observations == 20
        assert route.last_updated > 0

    def test_route_waypoints_follow_trail(self):
        trail = _make_commute_trail(home=(0, 0), work=(100, 0), steps=20)
        learner = BehavioralPatternLearner()
        route = learner.learn_route("t1", trail=trail)
        assert route is not None
        # First waypoint should be near origin
        assert abs(route.waypoints[0].x) < 15
        # Last waypoint should be near (100, 0)
        assert abs(route.waypoints[-1].x - 100) < 15

    def test_route_to_dict(self):
        trail = _make_commute_trail(steps=15)
        learner = BehavioralPatternLearner()
        route = learner.learn_route("t1", trail=trail)
        d = route.to_dict()
        assert d["target_id"] == "t1"
        assert "waypoint_count" in d
        assert "waypoints" in d
        assert isinstance(d["waypoints"], list)

    def test_learn_route_from_history(self):
        history = TargetHistory()
        for i in range(20):
            history.record("t1", (float(i * 5), float(i * 2)), timestamp=float(1000 + i * 10))
        learner = BehavioralPatternLearner(history=history)
        route = learner.learn_route("t1")
        assert route is not None
        assert route.total_observations >= MIN_ROUTE_POINTS

    def test_incremental_route_learning(self):
        learner = BehavioralPatternLearner()
        trail1 = _make_commute_trail(steps=15)
        r1 = learner.learn_route("t1", trail=trail1)
        assert r1.total_observations == 15

        trail2 = _make_commute_trail(steps=20, start_time=2000.0)
        r2 = learner.learn_route("t1", trail=trail2)
        assert r2.total_observations == 35  # merged

    def test_route_duration_stats(self):
        trail = _make_commute_trail(steps=20, dt=5.0)
        learner = BehavioralPatternLearner()
        route = learner.learn_route("t1", trail=trail)
        assert route.mean_duration_s > 0


# ---------------------------------------------------------------------------
# Tests: Schedule learning
# ---------------------------------------------------------------------------

class TestLearnSchedule:
    def test_no_history_no_timestamps(self):
        learner = BehavioralPatternLearner()
        result = learner.learn_schedule("target_1")
        assert result is None

    def test_too_few_observations(self):
        learner = BehavioralPatternLearner()
        result = learner.learn_schedule("t1", timestamps=[1000.0, 2000.0])
        assert result is None

    def test_learn_schedule_basic(self):
        timestamps = _make_schedule_timestamps([8.0, 12.0, 17.0], days=5)
        learner = BehavioralPatternLearner()
        schedule = learner.learn_schedule("t1", timestamps=timestamps)
        assert schedule is not None
        assert isinstance(schedule, LearnedSchedule)
        assert schedule.target_id == "t1"
        assert schedule.total_observations == len(timestamps)

    def test_schedule_histogram_peaks(self):
        timestamps = _make_schedule_timestamps([9.0, 17.0], days=7)
        learner = BehavioralPatternLearner()
        schedule = learner.learn_schedule("t1", timestamps=timestamps)
        assert schedule is not None
        # Hour 9 and 17 should be peak
        assert schedule.hourly_histogram[9] > 0
        assert schedule.hourly_histogram[17] > 0
        # Hour 3 should be zero
        assert schedule.hourly_histogram[3] == 0

    def test_schedule_to_dict(self):
        timestamps = _make_schedule_timestamps([8.0, 17.0], days=5)
        learner = BehavioralPatternLearner()
        schedule = learner.learn_schedule("t1", timestamps=timestamps)
        d = schedule.to_dict()
        assert "hourly_histogram" in d
        assert "peak_hours" in d
        assert "peak_days" in d
        assert "mean_arrival_hour" in d

    def test_schedule_arrival_departure(self):
        # Sessions: 8am-12pm, then 2pm-5pm (gap > 1h between)
        base = 1711929600.0  # Monday 00:00 UTC
        timestamps = []
        for day in range(5):
            day_start = base + day * 86400
            # Morning session: 8:00 to 12:00, every 15 min
            for m in range(0, 241, 15):
                timestamps.append(day_start + 8 * 3600 + m * 60)
            # Afternoon session: 14:00 to 17:00, every 15 min
            for m in range(0, 181, 15):
                timestamps.append(day_start + 14 * 3600 + m * 60)

        learner = BehavioralPatternLearner()
        schedule = learner.learn_schedule("t1", timestamps=timestamps)
        assert schedule is not None
        assert len(schedule.arrival_times) > 0
        assert len(schedule.departure_times) > 0

    def test_incremental_schedule_learning(self):
        ts1 = _make_schedule_timestamps([9.0, 12.0], days=5)  # 10 obs >= MIN_SCHEDULE_OBSERVATIONS
        ts2 = _make_schedule_timestamps([9.0, 12.0], days=5)
        learner = BehavioralPatternLearner()
        s1 = learner.learn_schedule("t1", timestamps=ts1)
        assert s1 is not None
        s2 = learner.learn_schedule("t1", timestamps=ts2)
        assert s2.total_observations == s1.total_observations + len(ts2)


# ---------------------------------------------------------------------------
# Tests: Zone learning
# ---------------------------------------------------------------------------

class TestLearnZones:
    def test_no_history_no_trail(self):
        learner = BehavioralPatternLearner()
        zones = learner.learn_zones("t1")
        assert zones == []

    def test_too_few_points(self):
        learner = BehavioralPatternLearner()
        zones = learner.learn_zones("t1", trail=[(0, 0, 100)])
        assert zones == []

    def test_learn_zones_from_dwell(self):
        # Simulate a target that dwells at two locations
        trail = []
        t = 1000.0
        # Dwell at (10, 10) for 120 seconds
        for i in range(25):
            trail.append((10.0 + i * 0.1, 10.0 + i * 0.1, t))
            t += 5.0
        # Move to (100, 100)
        for i in range(5):
            trail.append((10 + i * 20, 10 + i * 20, t))
            t += 2.0
        # Dwell at (100, 100) for 120 seconds
        for i in range(25):
            trail.append((100.0 + i * 0.1, 100.0 + i * 0.1, t))
            t += 5.0

        learner = BehavioralPatternLearner()
        zones = learner.learn_zones("t1", trail=trail, cluster_radius=15.0, min_dwell_s=60.0)
        assert len(zones) >= 1
        assert all(isinstance(z, FrequentZone) for z in zones)

    def test_zone_to_dict(self):
        zone = FrequentZone(center_x=10.0, center_y=20.0, radius=5.0, visit_count=10, total_dwell_s=300.0)
        d = zone.to_dict()
        assert d["center_x"] == 10.0
        assert d["center_y"] == 20.0
        assert d["visit_count"] == 10


# ---------------------------------------------------------------------------
# Tests: Deviation detection
# ---------------------------------------------------------------------------

class TestDeviationDetection:
    def _learner_with_route(self) -> BehavioralPatternLearner:
        """Create a learner with a learned straight-line route along x-axis."""
        learner = BehavioralPatternLearner(route_deviation_threshold=2.0)
        # Straight line from (0,0) to (100,0)
        trail = _make_commute_trail(home=(0, 0), work=(100, 0), steps=50)
        learner.learn_route("t1", trail=trail)
        return learner

    def test_no_deviation_on_route(self):
        learner = self._learner_with_route()
        # Position right on the route
        result = learner.detect_deviation("t1", (50.0, 0.0))
        # Should either not be a deviation or have very low severity
        if result.is_deviation:
            assert result.severity < 0.5

    def test_deviation_far_from_route(self):
        learner = self._learner_with_route()
        # Position very far from the route
        result = learner.detect_deviation("t1", (50.0, 500.0))
        assert result.is_deviation is True
        assert result.severity > 0
        assert result.deviation_type == "route"

    def test_no_learned_data_no_deviation(self):
        learner = BehavioralPatternLearner()
        result = learner.detect_deviation("unknown_target", (10.0, 10.0))
        assert result.is_deviation is False

    def test_deviation_result_to_dict(self):
        result = DeviationResult(
            is_deviation=True,
            deviation_type="route",
            severity=0.75,
            distance_from_expected=25.0,
            sigma=4.5,
        )
        d = result.to_dict()
        assert d["is_deviation"] is True
        assert d["deviation_type"] == "route"
        assert d["severity"] == 0.75

    def test_zone_deviation(self):
        learner = BehavioralPatternLearner()
        # Manually inject a zone
        learner._zones["t1"] = [
            FrequentZone(center_x=50.0, center_y=50.0, radius=10.0, visit_count=20, total_dwell_s=600.0)
        ]
        # Inside zone: no deviation
        result = learner.detect_deviation("t1", (55.0, 55.0))
        assert result.is_deviation is False

        # Way outside zone
        result = learner.detect_deviation("t1", (200.0, 200.0))
        assert result.is_deviation is True
        assert result.deviation_type == "zone"

    def test_deviation_returns_most_severe(self):
        learner = BehavioralPatternLearner()
        # Route and zone that will both flag deviation
        trail = _make_commute_trail(home=(0, 0), work=(100, 0), steps=50)
        learner.learn_route("t1", trail=trail)
        learner._zones["t1"] = [
            FrequentZone(center_x=50.0, center_y=0.0, radius=5.0, visit_count=20, total_dwell_s=600.0)
        ]
        result = learner.detect_deviation("t1", (50.0, 500.0))
        assert result.is_deviation is True
        # Should have enriched details
        assert "all_deviation_types" in result.details or result.deviation_type in ("route", "zone")


# ---------------------------------------------------------------------------
# Tests: Behavioral profile
# ---------------------------------------------------------------------------

class TestBehavioralProfile:
    def test_empty_profile(self):
        learner = BehavioralPatternLearner()
        profile = learner.get_profile("unknown")
        assert isinstance(profile, BehavioralProfile)
        assert profile.target_id == "unknown"
        assert profile.route is None
        assert profile.schedule is None
        assert profile.zones == []
        assert profile.regularity_score == 0.0

    def test_profile_with_route(self):
        trail = _make_commute_trail(steps=20)
        learner = BehavioralPatternLearner()
        learner.learn_route("t1", trail=trail)
        profile = learner.get_profile("t1")
        assert profile.route is not None
        assert profile.total_observations > 0

    def test_profile_to_dict(self):
        trail = _make_commute_trail(steps=20)
        learner = BehavioralPatternLearner()
        learner.learn_route("t1", trail=trail)
        profile = learner.get_profile("t1")
        d = profile.to_dict()
        assert d["target_id"] == "t1"
        assert d["has_route"] is True
        assert d["has_schedule"] is False
        assert "regularity_score" in d

    def test_regularity_score_route_only(self):
        # Very consistent route with tight waypoints -> high regularity
        trail = _make_commute_trail(home=(0, 0), work=(100, 0), steps=50)
        learner = BehavioralPatternLearner()
        learner.learn_route("t1", trail=trail)
        profile = learner.get_profile("t1")
        # Route with very small std should have high regularity
        assert profile.regularity_score > 0.0

    def test_full_profile(self):
        learner = BehavioralPatternLearner()
        trail = _make_commute_trail(steps=20)
        learner.learn_route("t1", trail=trail)
        timestamps = _make_schedule_timestamps([9.0, 17.0], days=5)
        learner.learn_schedule("t1", timestamps=timestamps)
        profile = learner.get_profile("t1")
        assert profile.route is not None
        assert profile.schedule is not None


# ---------------------------------------------------------------------------
# Tests: Bulk operations and state management
# ---------------------------------------------------------------------------

class TestBulkAndState:
    def test_clear_single_target(self):
        learner = BehavioralPatternLearner()
        trail = _make_commute_trail(steps=20)
        learner.learn_route("t1", trail=trail)
        learner.learn_route("t2", trail=trail)
        assert learner.get_stats()["targets_with_routes"] == 2
        learner.clear("t1")
        assert learner.get_stats()["targets_with_routes"] == 1

    def test_clear_all(self):
        learner = BehavioralPatternLearner()
        trail = _make_commute_trail(steps=20)
        learner.learn_route("t1", trail=trail)
        learner.learn_route("t2", trail=trail)
        learner.clear()
        assert learner.get_stats()["total_targets"] == 0

    def test_get_stats(self):
        learner = BehavioralPatternLearner()
        trail = _make_commute_trail(steps=20)
        learner.learn_route("t1", trail=trail)
        timestamps = _make_schedule_timestamps([9.0], days=5)
        learner.learn_schedule("t2", timestamps=timestamps)

        stats = learner.get_stats()
        assert stats["targets_with_routes"] == 1
        assert stats["targets_with_schedules"] == 1
        assert stats["total_targets"] == 2

    def test_get_all_profiles(self):
        learner = BehavioralPatternLearner()
        trail = _make_commute_trail(steps=20)
        learner.learn_route("t1", trail=trail)
        learner.learn_route("t2", trail=trail)
        profiles = learner.get_all_profiles()
        assert "t1" in profiles
        assert "t2" in profiles

    def test_export(self):
        learner = BehavioralPatternLearner()
        trail = _make_commute_trail(steps=20)
        learner.learn_route("t1", trail=trail)
        timestamps = _make_schedule_timestamps([9.0, 17.0], days=5)
        learner.learn_schedule("t1", timestamps=timestamps)

        data = learner.export()
        assert "routes" in data
        assert "schedules" in data
        assert "zones" in data
        assert "stats" in data
        assert "t1" in data["routes"]
        assert "t1" in data["schedules"]

    def test_learn_all(self):
        history = TargetHistory()
        for i in range(20):
            history.record("t1", (float(i * 5), 0.0), timestamp=float(1000 + i * 10))
            history.record("t2", (0.0, float(i * 5)), timestamp=float(1000 + i * 10))
        learner = BehavioralPatternLearner(history=history)
        profiles = learner.learn_all(["t1", "t2"])
        assert "t1" in profiles
        assert "t2" in profiles


# ---------------------------------------------------------------------------
# Tests: Schedule observation conversion
# ---------------------------------------------------------------------------

class TestScheduleObservation:
    def test_timestamp_to_observation(self):
        # 2024-04-01 08:30:00 UTC (Monday)
        ts = 1711960200.0
        obs = BehavioralPatternLearner._timestamp_to_observation(ts)
        assert isinstance(obs, ScheduleObservation)
        assert 8.0 <= obs.hour <= 9.0  # should be ~8.5
        assert obs.day_of_week == 0  # Monday


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_learn_route_exact_min_points(self):
        trail = _make_trail([(float(i), 0.0) for i in range(MIN_ROUTE_POINTS)])
        learner = BehavioralPatternLearner()
        route = learner.learn_route("t1", trail=trail)
        assert route is not None

    def test_learn_schedule_exact_min_observations(self):
        timestamps = _make_schedule_timestamps([10.0], days=MIN_SCHEDULE_OBSERVATIONS)
        learner = BehavioralPatternLearner()
        schedule = learner.learn_schedule("t1", timestamps=timestamps)
        assert schedule is not None

    def test_deviation_with_zero_std_waypoints(self):
        """Route with all identical points should not crash on deviation check."""
        trail = _make_trail([(50.0, 50.0)] * 20)
        learner = BehavioralPatternLearner()
        learner.learn_route("t1", trail=trail)
        result = learner.detect_deviation("t1", (50.0, 50.0))
        # Should not crash
        assert isinstance(result, DeviationResult)

    def test_schedule_with_same_timestamps(self):
        """All observations at the same time should not crash."""
        ts = [1711929600.0] * 10
        learner = BehavioralPatternLearner()
        schedule = learner.learn_schedule("t1", timestamps=ts)
        assert schedule is not None

    def test_empty_zones_no_deviation(self):
        learner = BehavioralPatternLearner()
        learner._zones["t1"] = []
        result = learner.detect_deviation("t1", (10.0, 10.0))
        assert result.is_deviation is False
