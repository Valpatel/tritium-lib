# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for weather visual effects system (weather_fx.py).

60+ tests covering all subsystems: Rain, Snow, Fog, Lightning, Wind,
DayNightCycle, and WeatherFXEngine.
"""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.weather_fx import (
    Bounds,
    DayNightCycle,
    FogSystem,
    LightningSystem,
    RainSystem,
    SnowSystem,
    WeatherFXEngine,
    WindSystem,
)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

class TestBounds:
    def test_defaults(self):
        b = Bounds()
        assert b.width == 100.0
        assert b.height == 50.0
        assert b.depth == 100.0

    def test_custom(self):
        b = Bounds(x_min=-10, x_max=10, y_min=0, y_max=20, z_min=-5, z_max=5)
        assert b.width == 20.0
        assert b.height == 20.0
        assert b.depth == 10.0


# ---------------------------------------------------------------------------
# RainSystem
# ---------------------------------------------------------------------------

class TestRainSystem:
    def test_basic_generation(self):
        rain = RainSystem(seed=42)
        result = rain.generate_rain(0.5, 0.0, 5.0)
        assert "drop_count" in result
        assert "positions" in result
        assert "velocities" in result
        assert "sizes" in result
        assert "splashes" in result
        assert result["drop_count"] > 0
        assert len(result["positions"]) == result["drop_count"]
        assert len(result["velocities"]) == result["drop_count"]
        assert len(result["sizes"]) == result["drop_count"]

    def test_zero_intensity(self):
        rain = RainSystem(seed=42)
        result = rain.generate_rain(0.0, 0.0, 0.0)
        assert result["drop_count"] >= 1  # minimum 1 drop

    def test_max_intensity(self):
        rain = RainSystem(seed=42)
        result = rain.generate_rain(1.0, 0.0, 0.0)
        assert result["drop_count"] > 10

    def test_intensity_scales_count(self):
        rain = RainSystem(seed=42)
        low = rain.generate_rain(0.1, 0.0, 0.0)
        rain2 = RainSystem(seed=42)
        high = rain2.generate_rain(0.9, 0.0, 0.0)
        assert high["drop_count"] > low["drop_count"]

    def test_positions_within_bounds(self):
        b = Bounds(x_min=-10, x_max=10, y_min=0, y_max=30, z_min=-10, z_max=10)
        rain = RainSystem(seed=42)
        result = rain.generate_rain(0.5, 0.0, 0.0, bounds=b)
        for pos in result["positions"]:
            assert b.x_min <= pos[0] <= b.x_max
            assert b.y_min <= pos[1] <= b.y_max
            assert b.z_min <= pos[2] <= b.z_max

    def test_velocities_fall_down(self):
        rain = RainSystem(seed=42)
        result = rain.generate_rain(0.5, 0.0, 0.0)
        for vel in result["velocities"]:
            assert vel[1] < 0  # falling down

    def test_wind_affects_velocity(self):
        rain = RainSystem(seed=42)
        result = rain.generate_rain(0.5, 0.0, 10.0)
        # With wind from +X direction, drops should have positive vx
        avg_vx = sum(v[0] for v in result["velocities"]) / len(result["velocities"])
        assert avg_vx > 0

    def test_splashes_at_ground(self):
        b = Bounds(y_min=0, y_max=50)
        rain = RainSystem(seed=42)
        result = rain.generate_rain(0.8, 0.0, 0.0, bounds=b)
        for splash in result["splashes"]:
            assert splash["position"][1] == b.y_min
            assert splash["radius"] > 0
            assert 0.0 <= splash["age"] <= 0.3

    def test_cap_at_10000(self):
        huge = Bounds(x_min=-500, x_max=500, y_min=0, y_max=500, z_min=-500, z_max=500)
        rain = RainSystem(seed=42)
        result = rain.generate_rain(1.0, 0.0, 0.0, bounds=huge)
        assert result["drop_count"] <= 10000

    def test_clamps_intensity(self):
        rain = RainSystem(seed=42)
        result = rain.generate_rain(5.0, 0.0, 0.0)  # over 1.0
        assert result["drop_count"] > 0  # should not crash

    def test_each_position_is_3d(self):
        rain = RainSystem(seed=42)
        result = rain.generate_rain(0.5, 0.0, 0.0)
        for pos in result["positions"]:
            assert len(pos) == 3


# ---------------------------------------------------------------------------
# SnowSystem
# ---------------------------------------------------------------------------

class TestSnowSystem:
    def test_basic_generation(self):
        snow = SnowSystem(seed=42)
        result = snow.generate_snow(0.5)
        assert result["flake_count"] > 0
        assert len(result["positions"]) == result["flake_count"]
        assert len(result["velocities"]) == result["flake_count"]
        assert len(result["sizes"]) == result["flake_count"]
        assert len(result["rotations"]) == result["flake_count"]
        assert "accumulation_cm" in result

    def test_falls_slower_than_rain(self):
        snow = SnowSystem(seed=42)
        result = snow.generate_snow(0.5)
        for vel in result["velocities"]:
            assert vel[1] < 0  # falling
            assert abs(vel[1]) < RainSystem.TERMINAL_VELOCITY

    def test_accumulation_increases(self):
        snow = SnowSystem(seed=42)
        r1 = snow.generate_snow(1.0)
        r2 = snow.generate_snow(1.0)
        assert r2["accumulation_cm"] > r1["accumulation_cm"]

    def test_zero_accumulation_at_zero_intensity(self):
        snow = SnowSystem(seed=42)
        result = snow.generate_snow(0.0)
        assert result["accumulation_cm"] == 0.0

    def test_wind_affects_drift(self):
        snow = SnowSystem(seed=42)
        result = snow.generate_snow(0.5, wind_dir=0.0, wind_speed=10.0)
        avg_vx = sum(v[0] for v in result["velocities"]) / len(result["velocities"])
        assert avg_vx > 0  # wind pushes flakes in +X

    def test_rotations_in_range(self):
        snow = SnowSystem(seed=42)
        result = snow.generate_snow(0.5)
        for r in result["rotations"]:
            assert 0.0 <= r <= math.pi * 2

    def test_cap_at_8000(self):
        huge = Bounds(x_min=-500, x_max=500, y_min=0, y_max=500, z_min=-500, z_max=500)
        snow = SnowSystem(seed=42)
        result = snow.generate_snow(1.0, bounds=huge)
        assert result["flake_count"] <= 8000

    def test_custom_bounds(self):
        b = Bounds(x_min=-5, x_max=5, y_min=0, y_max=10, z_min=-5, z_max=5)
        snow = SnowSystem(seed=42)
        result = snow.generate_snow(0.5, bounds=b)
        for pos in result["positions"]:
            assert b.x_min <= pos[0] <= b.x_max
            assert b.y_min <= pos[1] <= b.y_max
            assert b.z_min <= pos[2] <= b.z_max


# ---------------------------------------------------------------------------
# FogSystem
# ---------------------------------------------------------------------------

class TestFogSystem:
    def test_basic_generation(self):
        fog = FogSystem(seed=42)
        result = fog.generate_fog(0.5)
        assert "grid_size" in result
        assert "cell_size" in result
        assert "origin" in result
        assert "densities" in result
        assert "wind_offset" in result

    def test_grid_dimensions(self):
        b = Bounds(x_min=0, x_max=30, y_min=0, y_max=20, z_min=0, z_max=30)
        fog = FogSystem(seed=42)
        result = fog.generate_fog(0.5, bounds=b, cell_size=10.0)
        nx, ny, nz = result["grid_size"]
        assert nx == 3
        assert ny == 2
        assert nz == 3
        assert len(result["densities"]) == nx * ny * nz

    def test_densities_non_negative(self):
        fog = FogSystem(seed=42)
        result = fog.generate_fog(0.8)
        for d in result["densities"]:
            assert d >= 0.0

    def test_zero_density(self):
        fog = FogSystem(seed=42)
        result = fog.generate_fog(0.0)
        for d in result["densities"]:
            assert d == 0.0

    def test_wind_shifts_offset(self):
        fog = FogSystem(seed=42)
        r1 = fog.generate_fog(0.5, wind_dir=0.0, wind_speed=10.0)
        r2 = fog.generate_fog(0.5, wind_dir=0.0, wind_speed=10.0)
        assert r2["wind_offset"][0] > r1["wind_offset"][0]

    def test_denser_near_ground(self):
        b = Bounds(x_min=0, x_max=10, y_min=0, y_max=50, z_min=0, z_max=10)
        fog = FogSystem(seed=42)
        result = fog.generate_fog(0.8, bounds=b, cell_size=10.0)
        nx, ny, nz = result["grid_size"]
        # Compare bottom row average vs top row average
        bottom_avg = 0.0
        top_avg = 0.0
        count = nx * nz
        for iz in range(nz):
            for ix in range(nx):
                bottom_avg += result["densities"][iz * ny * nx + 0 * nx + ix]
                top_avg += result["densities"][iz * ny * nx + (ny - 1) * nx + ix]
        bottom_avg /= count
        top_avg /= count
        assert bottom_avg >= top_avg

    def test_origin_matches_bounds(self):
        b = Bounds(x_min=-20, x_max=20, y_min=5, y_max=25, z_min=-15, z_max=15)
        fog = FogSystem(seed=42)
        result = fog.generate_fog(0.5, bounds=b)
        assert result["origin"] == [b.x_min, b.y_min, b.z_min]

    def test_cell_size_stored(self):
        fog = FogSystem(seed=42)
        result = fog.generate_fog(0.5, cell_size=5.0)
        assert result["cell_size"] == 5.0


# ---------------------------------------------------------------------------
# LightningSystem
# ---------------------------------------------------------------------------

class TestLightningSystem:
    def test_basic_bolt(self):
        lt = LightningSystem(seed=42)
        result = lt.generate_bolt((0, 100, 0), (0, 0, 0))
        assert "segments" in result
        assert "branch_count" in result
        assert "brightness" in result
        assert len(result["segments"]) > 0
        assert 0.0 <= result["brightness"] <= 1.0

    def test_segments_are_pairs(self):
        lt = LightningSystem(seed=42)
        result = lt.generate_bolt((0, 100, 0), (0, 0, 0))
        for seg in result["segments"]:
            assert len(seg) == 2
            assert len(seg[0]) == 3
            assert len(seg[1]) == 3

    def test_no_branches(self):
        lt = LightningSystem(seed=42)
        result = lt.generate_bolt((0, 100, 0), (0, 0, 0), branches=0)
        assert result["branch_count"] == 0

    def test_many_branches(self):
        lt = LightningSystem(seed=42)
        result = lt.generate_bolt((0, 100, 0), (0, 0, 0), branches=10)
        # Should have some branch segments
        assert result["branch_count"] > 0

    def test_jitter_zero_is_straight(self):
        lt = LightningSystem(seed=42)
        result = lt.generate_bolt((0, 100, 0), (0, 0, 0), branches=0, jitter=0.0)
        # All segments should be roughly along the Y axis
        for seg in result["segments"]:
            assert abs(seg[0][0]) < 1.0  # x stays near 0
            assert abs(seg[0][2]) < 1.0  # z stays near 0

    def test_strike_basic(self):
        lt = LightningSystem(seed=42)
        result = lt.strike((50, 0, 50))
        assert "bolt" in result
        assert "flash_intensity" in result
        assert "thunder_delay_s" in result
        assert "rumble_duration_s" in result
        assert result["flash_intensity"] > 0

    def test_thunder_delay_increases_with_distance(self):
        lt = LightningSystem(seed=42)
        near = lt.strike((10, 0, 0), observer=(0, 0, 0))
        lt2 = LightningSystem(seed=42)
        far = lt2.strike((1000, 0, 0), observer=(0, 0, 0))
        assert far["thunder_delay_s"] > near["thunder_delay_s"]

    def test_thunder_delay_physics(self):
        lt = LightningSystem(seed=42)
        result = lt.strike((343, 0, 0), observer=(0, 0, 0))
        # 343m at 343 m/s = ~1 second
        assert 0.9 < result["thunder_delay_s"] < 1.1

    def test_cloud_height_used(self):
        lt = LightningSystem(seed=42)
        result = lt.strike((0, 0, 0), cloud_height=5000.0)
        bolt = result["bolt"]
        # First segment should start near cloud height
        # The bolt starts at (0, 5000, 0) and ends at (0, 0, 0)
        all_y = [seg[0][1] for seg in bolt["segments"]] + [seg[1][1] for seg in bolt["segments"]]
        assert max(all_y) > 1000  # some point should be high up

    def test_deterministic_with_seed(self):
        lt1 = LightningSystem(seed=99)
        r1 = lt1.generate_bolt((0, 100, 0), (0, 0, 0), branches=2)
        lt2 = LightningSystem(seed=99)
        r2 = lt2.generate_bolt((0, 100, 0), (0, 0, 0), branches=2)
        assert r1["segments"] == r2["segments"]


# ---------------------------------------------------------------------------
# WindSystem
# ---------------------------------------------------------------------------

class TestWindSystem:
    def test_basic_wind(self):
        wind = WindSystem(base_dir=0.0, base_speed=10.0, seed=42)
        vx, vz = wind.get_wind_at((0, 0))
        assert isinstance(vx, float)
        assert isinstance(vz, float)

    def test_base_direction(self):
        wind = WindSystem(base_dir=0.0, base_speed=10.0, gust_strength=0.0,
                          turbulence=0.0, seed=42)
        vx, vz = wind.get_wind_at((0, 0), time=0.0)
        # Wind in +X direction
        assert vx > 5.0
        assert abs(vz) < 2.0

    def test_no_turbulence_is_uniform(self):
        wind = WindSystem(base_dir=0.0, base_speed=10.0, gust_strength=0.0,
                          turbulence=0.0, seed=42)
        v1 = wind.get_wind_at((0, 0), time=0.0)
        v2 = wind.get_wind_at((100, 100), time=0.0)
        assert abs(v1[0] - v2[0]) < 0.001
        assert abs(v1[1] - v2[1]) < 0.001

    def test_turbulence_varies_by_position(self):
        wind = WindSystem(base_dir=0.0, base_speed=10.0, turbulence=0.5, seed=42)
        v1 = wind.get_wind_at((0, 0), time=1.0)
        v2 = wind.get_wind_at((100, 100), time=1.0)
        # Should differ due to turbulence
        diff = abs(v1[0] - v2[0]) + abs(v1[1] - v2[1])
        assert diff > 0.01

    def test_advance_changes_gusts(self):
        wind = WindSystem(base_dir=0.0, base_speed=10.0, gust_strength=0.5, seed=42)
        v1 = wind.get_wind_at((0, 0))
        wind.advance(10.0)
        v2 = wind.get_wind_at((0, 0))
        # Gust should create some difference over time
        assert v1 != v2

    def test_to_three_js(self):
        wind = WindSystem(seed=42)
        result = wind.to_three_js(resolution=4)
        assert result["resolution"] == 4
        assert len(result["vectors"]) == 16  # 4x4
        assert "origin" in result
        assert "cell_size" in result
        assert "time" in result

    def test_to_three_js_vectors_are_2d(self):
        wind = WindSystem(seed=42)
        result = wind.to_three_js(resolution=3)
        for v in result["vectors"]:
            assert len(v) == 2

    def test_3d_position_input(self):
        wind = WindSystem(base_dir=0.0, base_speed=5.0, seed=42)
        v = wind.get_wind_at((10.0, 5.0, 20.0))
        assert isinstance(v, tuple)
        assert len(v) == 2

    def test_custom_bounds(self):
        b = Bounds(x_min=-100, x_max=100, z_min=-100, z_max=100)
        wind = WindSystem(seed=42)
        result = wind.to_three_js(bounds=b, resolution=5)
        assert result["origin"] == [-100, -100]


# ---------------------------------------------------------------------------
# DayNightCycle
# ---------------------------------------------------------------------------

class TestDayNightCycle:
    def test_sky_gradient_returns_three_colors(self):
        dnc = DayNightCycle()
        result = dnc.get_sky_gradient(12.0)
        assert len(result) == 3
        for c in result:
            assert c.startswith("#")

    def test_sky_gradient_all_hours(self):
        dnc = DayNightCycle()
        for h in range(24):
            result = dnc.get_sky_gradient(float(h))
            assert len(result) == 3

    def test_sun_position_noon(self):
        dnc = DayNightCycle()
        x, y, z = dnc.get_sun_position(12.0)
        # At noon, sun should be high (y near 1)
        assert y > 0.8

    def test_sun_position_sunrise(self):
        dnc = DayNightCycle()
        x, y, z = dnc.get_sun_position(6.0)
        # At 6am, sun is at horizon (y near 0)
        assert abs(y) < 0.3

    def test_sun_position_midnight(self):
        dnc = DayNightCycle()
        x, y, z = dnc.get_sun_position(0.0)
        # Below horizon
        assert y < 0

    def test_sun_position_is_normalized(self):
        dnc = DayNightCycle()
        for h in [0, 6, 12, 18]:
            x, y, z = dnc.get_sun_position(float(h))
            length = math.sqrt(x * x + y * y + z * z)
            assert abs(length - 1.0) < 0.01

    def test_moon_phase_cycle(self):
        dnc = DayNightCycle()
        assert dnc.get_moon_phase(0.0) == pytest.approx(0.0, abs=0.01)
        assert dnc.get_moon_phase(14.765) == pytest.approx(0.5, abs=0.01)
        assert dnc.get_moon_phase(29.53) == pytest.approx(0.0, abs=0.01)

    def test_moon_phase_range(self):
        dnc = DayNightCycle()
        for d in range(100):
            p = dnc.get_moon_phase(float(d))
            assert 0.0 <= p < 1.0

    def test_moon_position(self):
        dnc = DayNightCycle()
        x, y, z = dnc.get_moon_position(0.0)
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)

    def test_stars_visible_at_night(self):
        dnc = DayNightCycle()
        stars = dnc.get_star_field(0.0)  # midnight
        assert len(stars) > 0

    def test_stars_invisible_at_noon(self):
        dnc = DayNightCycle()
        stars = dnc.get_star_field(12.0)
        assert len(stars) == 0

    def test_star_has_required_fields(self):
        dnc = DayNightCycle()
        stars = dnc.get_star_field(0.0)
        for s in stars:
            assert "name" in s
            assert "x" in s
            assert "y" in s
            assert "z" in s
            assert "magnitude" in s
            assert "brightness" in s

    def test_star_brightness_positive(self):
        dnc = DayNightCycle()
        stars = dnc.get_star_field(0.0)
        for s in stars:
            assert s["brightness"] > 0

    def test_star_min_magnitude_filter(self):
        dnc = DayNightCycle()
        bright = dnc.get_star_field(0.0, min_magnitude=0.5)
        all_stars = dnc.get_star_field(0.0, min_magnitude=2.0)
        assert len(all_stars) >= len(bright)

    def test_sky_gradient_wraps_at_24(self):
        dnc = DayNightCycle()
        a = dnc.get_sky_gradient(0.0)
        b = dnc.get_sky_gradient(24.0)
        assert a == b


# ---------------------------------------------------------------------------
# WeatherFXEngine
# ---------------------------------------------------------------------------

class TestWeatherFXEngine:
    def _weather_state(self, weather="clear", intensity=0.5, wind_speed=5.0,
                        wind_direction=90.0):
        return {
            "weather": weather,
            "intensity": intensity,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
        }

    def _time_state(self, hour=12.0, day=0.0):
        return {"hour": hour, "day": day}

    def test_tick_returns_frame(self):
        engine = WeatherFXEngine(seed=42)
        frame = engine.tick(0.1, self._weather_state(), self._time_state())
        assert "time" in frame
        assert "hour" in frame
        assert "sky" in frame
        assert "wind" in frame
        assert "rain" in frame
        assert "snow" in frame
        assert "fog" in frame
        assert "lightning" in frame

    def test_clear_weather_no_effects(self):
        engine = WeatherFXEngine(seed=42)
        frame = engine.tick(0.1, self._weather_state("clear"), self._time_state())
        assert frame["rain"] is None
        assert frame["snow"] is None
        assert frame["fog"] is None

    def test_rain_weather_has_rain(self):
        engine = WeatherFXEngine(seed=42)
        frame = engine.tick(0.1, self._weather_state("rain", 0.5), self._time_state())
        assert frame["rain"] is not None
        assert frame["rain"]["drop_count"] > 0

    def test_heavy_rain_has_rain(self):
        engine = WeatherFXEngine(seed=42)
        frame = engine.tick(0.1, self._weather_state("heavy_rain", 0.5), self._time_state())
        assert frame["rain"] is not None

    def test_snow_weather_has_snow(self):
        engine = WeatherFXEngine(seed=42)
        frame = engine.tick(0.1, self._weather_state("snow", 0.7), self._time_state())
        assert frame["snow"] is not None
        assert frame["snow"]["flake_count"] > 0

    def test_fog_weather_has_fog(self):
        engine = WeatherFXEngine(seed=42)
        frame = engine.tick(0.1, self._weather_state("fog", 0.6), self._time_state())
        assert frame["fog"] is not None
        assert len(frame["fog"]["densities"]) > 0

    def test_storm_can_have_lightning(self):
        engine = WeatherFXEngine(seed=42)
        # Run many ticks to get at least one lightning strike
        got_lightning = False
        for _ in range(200):
            frame = engine.tick(1.0, self._weather_state("storm", 1.0), self._time_state())
            if frame["lightning"] is not None:
                got_lightning = True
                break
        assert got_lightning

    def test_sky_always_present(self):
        engine = WeatherFXEngine(seed=42)
        for weather in ["clear", "rain", "snow", "fog", "storm"]:
            frame = engine.tick(0.1, self._weather_state(weather), self._time_state())
            assert "gradient" in frame["sky"]
            assert "sun" in frame["sky"]
            assert "moon" in frame["sky"]

    def test_wind_always_present(self):
        engine = WeatherFXEngine(seed=42)
        frame = engine.tick(0.1, self._weather_state(), self._time_state())
        assert "vectors" in frame["wind"]
        assert "resolution" in frame["wind"]

    def test_time_advances(self):
        engine = WeatherFXEngine(seed=42)
        f1 = engine.tick(1.0, self._weather_state(), self._time_state())
        f2 = engine.tick(1.0, self._weather_state(), self._time_state())
        assert f2["time"] > f1["time"]

    def test_custom_bounds(self):
        engine = WeatherFXEngine(seed=42)
        b = Bounds(x_min=-5, x_max=5, y_min=0, y_max=10, z_min=-5, z_max=5)
        frame = engine.tick(0.1, self._weather_state("rain", 0.5),
                           self._time_state(), bounds=b)
        assert frame["rain"] is not None

    def test_night_sky_has_stars(self):
        engine = WeatherFXEngine(seed=42)
        frame = engine.tick(0.1, self._weather_state(), self._time_state(hour=0.0))
        assert len(frame["sky"]["stars"]) > 0

    def test_day_sky_no_stars(self):
        engine = WeatherFXEngine(seed=42)
        frame = engine.tick(0.1, self._weather_state(), self._time_state(hour=12.0))
        assert len(frame["sky"]["stars"]) == 0

    def test_deterministic(self):
        e1 = WeatherFXEngine(seed=100)
        f1 = e1.tick(0.5, self._weather_state("rain"), self._time_state())
        e2 = WeatherFXEngine(seed=100)
        f2 = e2.tick(0.5, self._weather_state("rain"), self._time_state())
        assert f1["rain"]["drop_count"] == f2["rain"]["drop_count"]
        assert f1["rain"]["positions"] == f2["rain"]["positions"]
