# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for terrain analysis and RF propagation models."""

import math

import pytest

from tritium_lib.models.terrain import (
    CoverageAnalysis,
    CoverageCell,
    ElevationPoint,
    ElevationProfile,
    SensorPlacement,
    TerrainType,
    WeatherConditions,
    estimate_signal_strength,
    free_space_path_loss_db,
    terrain_path_loss_db,
)


class TestElevation:
    def test_elevation_point(self):
        p = ElevationPoint(latitude=37.7, longitude=-122.4, elevation_m=15.0)
        assert p.elevation_m == 15.0
        assert p.source == "srtm"

    def test_elevation_profile(self):
        profile = ElevationProfile(
            start_lat=37.7, start_lng=-122.4,
            end_lat=37.8, end_lng=-122.3,
            distance_m=14000,
            has_line_of_sight=True,
        )
        assert profile.distance_m == 14000
        assert profile.has_line_of_sight is True

    def test_profile_obstruction(self):
        profile = ElevationProfile(
            start_lat=0, start_lng=0,
            end_lat=1, end_lng=1,
            has_line_of_sight=False,
            obstruction_points=[5, 12],
        )
        assert not profile.has_line_of_sight
        assert len(profile.obstruction_points) == 2


class TestCoverage:
    def test_coverage_cell(self):
        cell = CoverageCell(
            latitude=37.7, longitude=-122.4,
            signal_strength_dbm=-65.0,
            covered=True,
            distance_m=50.0,
        )
        assert cell.covered is True
        assert cell.signal_strength_dbm == -65.0

    def test_coverage_analysis(self):
        analysis = CoverageAnalysis(
            sensor_id="ble01",
            sensor_lat=37.7,
            sensor_lng=-122.4,
            frequency_mhz=2400,
            range_m=100,
            coverage_percent=78.5,
        )
        assert analysis.frequency_mhz == 2400
        assert analysis.coverage_percent == 78.5

    def test_sensor_placement(self):
        p = SensorPlacement(
            latitude=37.7, longitude=-122.4,
            height_m=3.0,
            score=0.85,
            coverage_area_m2=3140.0,
        )
        assert p.score == 0.85


class TestRFPropagation:
    def test_fspl_known_values(self):
        """FSPL at 2.4 GHz, 1m should be ~40 dB."""
        loss = free_space_path_loss_db(1.0, 2400)
        assert 39.0 < loss < 41.0

    def test_fspl_increases_with_distance(self):
        loss_10 = free_space_path_loss_db(10.0, 2400)
        loss_100 = free_space_path_loss_db(100.0, 2400)
        assert loss_100 > loss_10
        # 20 dB per decade of distance
        assert abs((loss_100 - loss_10) - 20.0) < 0.1

    def test_fspl_increases_with_frequency(self):
        loss_900 = free_space_path_loss_db(100.0, 900)
        loss_2400 = free_space_path_loss_db(100.0, 2400)
        assert loss_2400 > loss_900

    def test_fspl_zero_distance(self):
        assert free_space_path_loss_db(0.0, 2400) == 0.0

    def test_fspl_zero_frequency(self):
        assert free_space_path_loss_db(100.0, 0.0) == 0.0

    def test_terrain_loss_urban_gt_rural(self):
        """Urban terrain should have more loss than rural."""
        urban = terrain_path_loss_db(100.0, 2400, TerrainType.URBAN)
        rural = terrain_path_loss_db(100.0, 2400, TerrainType.RURAL)
        assert urban > rural

    def test_terrain_loss_water_lowest(self):
        """Water terrain should have the least additional loss."""
        water = terrain_path_loss_db(100.0, 2400, TerrainType.WATER)
        forest = terrain_path_loss_db(100.0, 2400, TerrainType.FOREST)
        assert water < forest

    def test_signal_strength_decreases(self):
        """Signal should decrease with distance."""
        near = estimate_signal_strength(0, 10)
        far = estimate_signal_strength(0, 100)
        assert near > far

    def test_signal_strength_at_zero(self):
        """At zero distance, signal equals TX power."""
        assert estimate_signal_strength(10.0, 0) == 10.0

    def test_all_terrain_types(self):
        """Every terrain type should produce a valid loss."""
        for t in TerrainType:
            loss = terrain_path_loss_db(50.0, 2400, t)
            assert loss > 0


class TestWeather:
    def test_weather_defaults(self):
        w = WeatherConditions()
        assert w.temperature_c == 20.0
        assert w.rain_rate_mm_h == 0.0

    def test_no_rain_attenuation(self):
        w = WeatherConditions(rain_rate_mm_h=0)
        assert w.rain_attenuation_db_km == 0.0

    def test_rain_attenuation_increases(self):
        w_light = WeatherConditions(rain_rate_mm_h=5)
        w_heavy = WeatherConditions(rain_rate_mm_h=50)
        assert w_heavy.rain_attenuation_db_km > w_light.rain_attenuation_db_km
