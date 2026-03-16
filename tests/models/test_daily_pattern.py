# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for DailyPattern model."""

import pytest

from tritium_lib.models.daily_pattern import DailyPattern


class TestDailyPattern:
    """Test suite for DailyPattern model."""

    def test_default_construction(self):
        dp = DailyPattern(target_id="ble_AA:BB:CC:DD:EE:FF")
        assert dp.target_id == "ble_AA:BB:CC:DD:EE:FF"
        assert len(dp.hourly_counts) == 24
        assert all(c == 0 for c in dp.hourly_counts)
        assert dp.peak_hour == 0
        assert dp.regularity_score == 0.0

    def test_add_sighting(self):
        dp = DailyPattern(target_id="t1")
        dp.add_sighting(8)
        dp.add_sighting(8)
        dp.add_sighting(17)
        assert dp.hourly_counts[8] == 2
        assert dp.hourly_counts[17] == 1
        assert dp.total_sightings == 3

    def test_add_sighting_out_of_range(self):
        dp = DailyPattern(target_id="t1")
        dp.add_sighting(25)
        dp.add_sighting(-1)
        assert dp.total_sightings == 0

    def test_compute_peak_hour(self):
        dp = DailyPattern(target_id="t1")
        dp.hourly_counts[14] = 50
        dp.hourly_counts[8] = 30
        dp.compute_peak_hour()
        assert dp.peak_hour == 14

    def test_compute_peak_hour_empty(self):
        dp = DailyPattern(target_id="t1")
        dp.compute_peak_hour()
        assert dp.peak_hour == 0

    def test_compute_quiet_hours(self):
        dp = DailyPattern(target_id="t1")
        dp.hourly_counts[12] = 100
        dp.hourly_counts[13] = 80
        dp.compute_quiet_hours()
        # All hours except 12 and 13 should be quiet (< 5% of 100 = 5)
        assert 0 in dp.quiet_hours
        assert 12 not in dp.quiet_hours
        assert 13 not in dp.quiet_hours

    def test_compute_regularity_score_single_spike(self):
        dp = DailyPattern(target_id="t1")
        dp.hourly_counts[10] = 100
        score = dp.compute_regularity_score()
        assert score == 1.0  # All activity in one hour = perfectly regular

    def test_compute_regularity_score_uniform(self):
        dp = DailyPattern(target_id="t1")
        dp.hourly_counts = [10] * 24
        score = dp.compute_regularity_score()
        assert score == 0.0  # Uniform = no pattern

    def test_compute_regularity_score_moderate(self):
        dp = DailyPattern(target_id="t1")
        # Morning commuter pattern
        dp.hourly_counts[7] = 5
        dp.hourly_counts[8] = 20
        dp.hourly_counts[9] = 10
        dp.hourly_counts[17] = 15
        dp.hourly_counts[18] = 8
        score = dp.compute_regularity_score()
        assert 0.3 < score < 0.9

    def test_recompute(self):
        dp = DailyPattern(target_id="t1")
        dp.hourly_counts[9] = 50
        dp.hourly_counts[17] = 30
        dp.recompute()
        assert dp.total_sightings == 80
        assert dp.peak_hour == 9
        assert dp.regularity_score > 0.5
        assert 0 in dp.quiet_hours

    def test_active_hours(self):
        dp = DailyPattern(target_id="t1")
        dp.hourly_counts[8] = 5
        dp.hourly_counts[12] = 3
        dp.hourly_counts[17] = 7
        assert dp.active_hours == [8, 12, 17]

    def test_is_daytime_only(self):
        dp = DailyPattern(target_id="t1")
        dp.hourly_counts[8] = 5
        dp.hourly_counts[12] = 3
        dp.hourly_counts[17] = 7
        dp.total_sightings = 15
        assert dp.is_daytime_only

    def test_is_daytime_only_false(self):
        dp = DailyPattern(target_id="t1")
        dp.hourly_counts[8] = 5
        dp.hourly_counts[22] = 3
        dp.total_sightings = 8
        assert not dp.is_daytime_only

    def test_is_nighttime_only(self):
        dp = DailyPattern(target_id="t1")
        dp.hourly_counts[22] = 10
        dp.hourly_counts[2] = 5
        dp.total_sightings = 15
        assert dp.is_nighttime_only

    def test_is_nighttime_only_false(self):
        dp = DailyPattern(target_id="t1")
        dp.hourly_counts[10] = 5
        dp.hourly_counts[22] = 5
        dp.total_sightings = 10
        assert not dp.is_nighttime_only

    def test_serialization_roundtrip(self):
        dp = DailyPattern(target_id="t1", days_observed=5)
        dp.hourly_counts[8] = 20
        dp.hourly_counts[17] = 15
        dp.recompute()

        data = dp.model_dump()
        dp2 = DailyPattern(**data)
        assert dp2.target_id == "t1"
        assert dp2.hourly_counts[8] == 20
        assert dp2.peak_hour == 8
        assert dp2.regularity_score == dp.regularity_score

    def test_empty_pattern_regularity(self):
        dp = DailyPattern(target_id="t1")
        score = dp.compute_regularity_score()
        assert score == 0.0
