# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for seasons + variable daylight + seasonal foliage/temp/snow.

Covers the "environment as a first-class layer" slice:
  - TimeOfDay variable daylight, with a byte-identity guarantee for the
    default sunrise=6/sunset=18 window (goldens depend on it).
  - SeasonalCycle: season bands, daylight geometry, foliage, temperature,
    snow classification.
  - Environment seasonal wiring + snow accumulation.
  - VisionSystem optional detection-range weather hook (byte-identical no-op
    when no environment is attached).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tritium_lib.sim_engine.environment import (
    Environment,
    Season,
    SeasonalCycle,
    TimeOfDay,
    Weather,
    WeatherSimulator,
)
from tritium_lib.sim_engine.world.vision import VisionSystem
from tritium_lib.sim_engine.core.spatial import SpatialGrid


# ======================================================================
# 1. Default TimeOfDay byte-identity (the golden-safety proof)
# ======================================================================

class TestDefaultTimeOfDayByteIdentity:

    def test_light_level_matches_old_formula(self):
        """Default sunrise=6/sunset=18 must equal (sin((h-6)*pi/12)+1)/2."""
        h = 0.0
        while h <= 24.0 + 1e-12:
            t = TimeOfDay(hour=h)
            old = max(0.0, min(1.0, (math.sin((h % 24.0 - 6.0) * math.pi / 12.0) + 1.0) / 2.0))
            assert abs(t.light_level() - old) < 1e-9, f"hour={h}"
            h += 0.1

    def test_sun_angle_matches_old_formula(self):
        """Default sun_angle must equal max(0, 90*sin((h-6)*pi/12))."""
        h = 0.0
        while h <= 24.0 + 1e-12:
            t = TimeOfDay(hour=h)
            old = max(0.0, 90.0 * math.sin((h % 24.0 - 6.0) * math.pi / 12.0))
            assert abs(t.sun_angle() - old) < 1e-9, f"hour={h}"
            h += 0.1

    def test_sun_angle_never_negative(self):
        for i in range(240):
            assert TimeOfDay(hour=i * 0.1).sun_angle() >= 0.0

    def test_default_key_predicates(self):
        assert TimeOfDay(6.0).is_day()
        assert TimeOfDay(18.0).is_day()
        assert not TimeOfDay(0.0).is_day()
        assert TimeOfDay(0.0).is_night()
        assert TimeOfDay(22.0).is_night()
        assert TimeOfDay(4.0).is_night()
        assert not TimeOfDay(12.0).is_night()
        assert TimeOfDay(5.5).is_twilight()
        assert TimeOfDay(19.0).is_twilight()
        assert not TimeOfDay(12.0).is_twilight()

    def test_solar_noon_default(self):
        assert TimeOfDay().solar_noon == pytest.approx(12.0)


# ======================================================================
# 2. Variable daylight
# ======================================================================

class TestVariableDaylight:

    def test_custom_window_is_day(self):
        t = TimeOfDay(hour=13.0, sunrise=8.0, sunset=16.0)
        assert t.is_day()
        assert t.solar_noon == pytest.approx(12.0)

    def test_light_peaks_near_solar_noon(self):
        t = TimeOfDay(hour=12.0, sunrise=8.0, sunset=16.0)
        peak = t.light_level()
        for h in (8.5, 10.0, 14.0, 15.5):
            other = TimeOfDay(hour=h, sunrise=8.0, sunset=16.0).light_level()
            assert peak >= other
        assert peak == pytest.approx(1.0)

    def test_short_day_window_narrower_than_long_day(self):
        def day_hours(sunrise, sunset):
            count = 0
            h = 0.0
            while h < 24.0:
                if TimeOfDay(hour=h, sunrise=sunrise, sunset=sunset).is_day():
                    count += 1
                h += 0.25
            return count

        winter = day_hours(8.0, 16.0)   # short day
        summer = day_hours(5.0, 21.0)   # long day
        assert winter < summer

    def test_set_daylight_mutates_window(self):
        t = TimeOfDay()
        t.set_daylight(7.0, 17.0)
        assert t.sunrise == 7.0
        assert t.sunset == 17.0
        assert t.solar_noon == pytest.approx(12.0)


# ======================================================================
# 3. Season bands
# ======================================================================

class TestSeasonBands:

    def test_northern_bands(self):
        assert SeasonalCycle(15, 40).season() is Season.WINTER
        assert SeasonalCycle(100, 40).season() is Season.SPRING
        assert SeasonalCycle(200, 40).season() is Season.SUMMER
        assert SeasonalCycle(300, 40).season() is Season.AUTUMN

    def test_southern_hemisphere_flips(self):
        # Day 200 is northern summer -> southern winter.
        assert SeasonalCycle(200, -40).season() is Season.WINTER


# ======================================================================
# 4. Daylight geometry
# ======================================================================

class TestDaylightHours:

    def test_summer_longer_than_winter_temperate(self):
        summer = SeasonalCycle(172, 40).daylight_hours()
        winter = SeasonalCycle(355, 40).daylight_hours()
        assert summer > 12.0 > winter

    def test_equator_near_twelve_both_solstices(self):
        assert SeasonalCycle(172, 0).daylight_hours() == pytest.approx(12.0, abs=1.0)
        assert SeasonalCycle(355, 0).daylight_hours() == pytest.approx(12.0, abs=1.0)

    def test_polar_midnight_sun_and_night(self):
        assert SeasonalCycle(172, 80).daylight_hours() == pytest.approx(24.0)
        assert SeasonalCycle(355, 80).daylight_hours() == pytest.approx(0.0)


# ======================================================================
# 5. Sunrise seasonality
# ======================================================================

class TestSunrise:

    def test_sunrise_earlier_in_summer(self):
        summer = SeasonalCycle(172, 40).sunrise_hour()
        winter = SeasonalCycle(355, 40).sunrise_hour()
        assert summer < winter


# ======================================================================
# 6. Foliage
# ======================================================================

class TestFoliage:

    def test_temperate_seasonal_swing(self):
        summer = SeasonalCycle(172, 40).foliage_state()
        autumn = SeasonalCycle(300, 40).foliage_state()
        winter = SeasonalCycle(355, 40).foliage_state()
        assert summer > autumn > winter
        assert winter < 0.3
        assert summer > 0.9

    def test_tropics_evergreen_constant(self):
        vals = [SeasonalCycle(d, 5).foliage_state() for d in range(1, 366)]
        assert max(vals) - min(vals) < 0.2
        assert all(v > 0.75 for v in vals)


# ======================================================================
# 7. Temperature
# ======================================================================

class TestTemperature:

    def test_temperate_summer_warmer_than_winter(self):
        summer = SeasonalCycle(172, 40).temperature_baseline()
        winter = SeasonalCycle(355, 40).temperature_baseline()
        assert summer > winter

    def test_high_latitude_bigger_swing(self):
        def amplitude(lat):
            vals = [SeasonalCycle(d, lat).temperature_baseline() for d in range(1, 366)]
            return max(vals) - min(vals)

        assert amplitude(60) > amplitude(10)


# ======================================================================
# 8. Snow classification
# ======================================================================

class TestPrecipIsSnow:

    def test_winter_is_snow_summer_is_not(self):
        assert SeasonalCycle(355, 40).precip_is_snow() is True
        assert SeasonalCycle(172, 40).precip_is_snow() is False

    def test_explicit_temperature_override(self):
        c = SeasonalCycle(172, 40)
        assert c.precip_is_snow(temperature_c=-3.0) is True
        assert c.precip_is_snow(temperature_c=10.0) is False


# ======================================================================
# 9. Environment seasonal wiring
# ======================================================================

class TestEnvironmentSeasonal:

    def test_winter_environment_snapshot(self):
        env = Environment(seasonal=SeasonalCycle(355, 40))
        snap = env.snapshot()
        assert snap["season"] == "winter"
        assert "foliage_state" in snap
        assert snap["sunrise"] > 6.0          # short winter day
        assert snap["is_snow_season"] is True

    def test_seasonal_narrows_time_window(self):
        env = Environment(seasonal=SeasonalCycle(355, 40))
        assert env.time.sunrise > 6.0
        assert env.time.sunset < 18.0

    def test_default_environment_snapshot_unchanged(self):
        # Byte-identity: no seasonal cycle => no seasonal keys leak in.
        snap = Environment().snapshot()
        for k in ("season", "foliage_state", "snow_depth_cm",
                  "seasonal_mobility_modifier", "daylight_hours",
                  "sunrise", "sunset", "is_snow_season"):
            assert k not in snap
        assert env_foliage_default() == 1.0

    def test_season_none_without_seasonal(self):
        assert Environment().season() is None


def env_foliage_default() -> float:
    return Environment().foliage_state()


# ======================================================================
# 10. Snow accumulation
# ======================================================================

class TestSnowAccumulation:

    def test_snow_accumulates_and_slows_movement(self):
        env = Environment(weather=WeatherSimulator(seed=0))
        # Cold so accumulated snow never melts; deterministic seed.
        env.weather.state.temperature = -5.0
        for _ in range(12):
            env.weather.state.current = Weather.SNOW
            env.weather.state.intensity = 1.0
            env.update(3600.0)
        assert env.snow_depth_cm > 0.0
        assert env.seasonal_mobility_modifier() < env.movement_speed_modifier()

    def test_no_snow_no_penalty(self):
        env = Environment()
        assert env.snow_depth_cm == 0.0
        assert env.seasonal_mobility_modifier() == pytest.approx(
            env.movement_speed_modifier()
        )


# ======================================================================
# 11. VisionSystem detection-range weather hook
# ======================================================================

def _mk(tid, x, y, alliance):
    return SimpleNamespace(
        target_id=tid,
        position=(x, y),
        alliance=alliance,
        asset_type="rover",
        status="active",
        heading=0.0,
        identity=SimpleNamespace(bluetooth_mac=None, wifi_mac=None, cell_id=None),
    )


class TestVisionWeatherHook:

    def _scene(self):
        grid = SpatialGrid(cell_size=50.0)
        friendly = _mk("f1", 0.0, 0.0, "friendly")
        # 12m out: inside the default 15m omni radius, beyond the fogged radius.
        hostile = _mk("h1", 12.0, 0.0, "hostile")
        targets = {"f1": friendly, "h1": hostile}
        grid.rebuild(list(targets.values()))
        return grid, targets

    def test_no_environment_is_byte_identical(self):
        grid, targets = self._scene()
        vs = VisionSystem()  # environment=None => det_mod 1.0
        state = vs.tick(0.1, targets, grid)
        assert "h1" in state.friendly_visible

    def test_fog_environment_degrades_detection(self):
        grid, targets = self._scene()
        fog = Environment(weather=WeatherSimulator(initial=Weather.FOG))
        det_mod = fog.detection_range_modifier()
        # Sanity: fog really does shrink the radius below the target distance.
        assert 15.0 * det_mod < 12.0
        vs = VisionSystem(environment=fog)
        state = vs.tick(0.1, targets, grid)
        assert "h1" not in state.friendly_visible
