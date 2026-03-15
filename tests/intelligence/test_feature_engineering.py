# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for intelligence.feature_engineering module."""

import time

import pytest

from tritium_lib.intelligence.feature_engineering import (
    EXTENDED_FEATURE_NAMES,
    build_extended_features,
    co_movement_score,
    device_type_match,
    source_diversity,
    time_similarity,
    wifi_probe_temporal_correlation,
)


class TestDeviceTypeMatch:
    def test_phone_person_strong_match(self):
        score = device_type_match("phone", "person", "ble", "yolo")
        assert score == 1.0

    def test_watch_person_strong(self):
        score = device_type_match("watch", "person", "ble", "yolo")
        assert score == 0.95

    def test_same_type_cross_sensor(self):
        score = device_type_match("phone", "phone", "ble", "wifi")
        assert score == 0.6

    def test_same_type_same_sensor(self):
        score = device_type_match("phone", "phone", "ble", "ble")
        assert score == 0.3

    def test_unknown_types(self):
        score = device_type_match("unknown", "unknown")
        assert score == 0.1

    def test_empty_type(self):
        score = device_type_match("", "person")
        assert score == 0.0

    def test_no_compatibility(self):
        score = device_type_match("beacon", "person")
        assert score == 0.0

    def test_case_insensitive(self):
        score = device_type_match("Phone", "Person", "ble", "yolo")
        assert score == 1.0


class TestCoMovementScore:
    def test_empty_trails(self):
        assert co_movement_score([], []) == 0.0

    def test_short_trails(self):
        assert co_movement_score([(0, 0, 0)], [(0, 0, 0)]) == 0.0

    def test_co_located_movement(self):
        # Two targets moving together
        trail_a = [(0, 0, 0), (1, 0, 5), (2, 0, 10), (3, 0, 15), (4, 0, 20)]
        trail_b = [(0.5, 0, 0), (1.5, 0, 5), (2.5, 0, 10), (3.5, 0, 15), (4.5, 0, 20)]
        score = co_movement_score(trail_a, trail_b)
        assert score > 0.3

    def test_distant_targets(self):
        # Targets far apart
        trail_a = [(0, 0, 0), (1, 0, 5), (2, 0, 10)]
        trail_b = [(100, 100, 0), (101, 100, 5), (102, 100, 10)]
        score = co_movement_score(trail_a, trail_b, max_distance=5.0)
        assert score == 0.0

    def test_no_temporal_overlap(self):
        trail_a = [(0, 0, 0), (1, 0, 5)]
        trail_b = [(0, 0, 100), (1, 0, 105)]
        score = co_movement_score(trail_a, trail_b)
        assert score == 0.0


class TestTimeSimilarity:
    def test_same_time(self):
        now = time.time()
        score = time_similarity(now, now)
        assert score == 1.0

    def test_explicit_time_of_day(self):
        score = time_similarity(0, 0, time_of_day_a=8.0, time_of_day_b=8.0)
        assert score == 1.0

    def test_opposite_times(self):
        score = time_similarity(0, 0, time_of_day_a=0.0, time_of_day_b=12.0)
        assert score == 0.0

    def test_six_hours_apart(self):
        score = time_similarity(0, 0, time_of_day_a=6.0, time_of_day_b=12.0)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_wraps_around_midnight(self):
        score = time_similarity(0, 0, time_of_day_a=23.0, time_of_day_b=1.0)
        assert score > 0.8  # 2 hours apart


class TestSourceDiversity:
    def test_single_source(self):
        score = source_diversity(["ble"], ["ble"])
        assert score == 0.0

    def test_two_sources(self):
        score = source_diversity(["ble"], ["yolo"])
        assert score > 0.3

    def test_many_sources(self):
        score = source_diversity(["ble", "wifi"], ["yolo", "acoustic"])
        assert score > 0.7

    def test_cross_category_bonus(self):
        # RF + visual should be higher than RF + RF
        rf_rf = source_diversity(["ble"], ["wifi"])
        rf_vis = source_diversity(["ble"], ["yolo"])
        assert rf_vis > rf_rf

    def test_empty_sources(self):
        score = source_diversity([], [])
        assert score == 0.0


class TestWifiProbeTemporalCorrelation:
    def test_simultaneous(self):
        now = time.time()
        score = wifi_probe_temporal_correlation(now, now)
        assert score == 1.0

    def test_within_window(self):
        now = time.time()
        score = wifi_probe_temporal_correlation(now, now + 5.0, max_window_s=10.0)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_outside_window(self):
        now = time.time()
        score = wifi_probe_temporal_correlation(now, now + 15.0, max_window_s=10.0)
        assert score == 0.0

    def test_same_observer_bonus(self):
        now = time.time()
        without = wifi_probe_temporal_correlation(now, now + 2.0)
        with_obs = wifi_probe_temporal_correlation(now, now + 2.0, same_observer=True)
        assert with_obs > without

    def test_capped_at_one(self):
        now = time.time()
        score = wifi_probe_temporal_correlation(now, now, same_observer=True)
        assert score <= 1.0


class TestBuildExtendedFeatures:
    def test_all_zeros(self):
        features = build_extended_features()
        assert len(features) == 10
        assert all(v == 0.0 for v in features.values())

    def test_keys_match_names(self):
        features = build_extended_features()
        assert set(features.keys()) == set(EXTENDED_FEATURE_NAMES)

    def test_custom_values(self):
        features = build_extended_features(
            distance=3.5,
            co_movement_duration=0.8,
            source_diversity_score=0.6,
        )
        assert features["distance"] == 3.5
        assert features["co_movement_duration"] == 0.8
        assert features["source_diversity_score"] == 0.6


class TestExtendedFeatureNames:
    def test_count(self):
        assert len(EXTENDED_FEATURE_NAMES) == 10

    def test_includes_original_six(self):
        original = ["distance", "rssi_delta", "co_movement",
                     "device_type_match", "time_gap", "signal_pattern"]
        for name in original:
            assert name in EXTENDED_FEATURE_NAMES

    def test_includes_new_four(self):
        new = ["co_movement_duration", "time_of_day_similarity",
               "source_diversity_score", "wifi_probe_correlation"]
        for name in new:
            assert name in EXTENDED_FEATURE_NAMES
