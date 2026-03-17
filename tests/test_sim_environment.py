# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.environment — weather and time-of-day simulation."""

from __future__ import annotations

import json
import math

import pytest

from tritium_lib.sim_engine.environment import (
    Environment,
    TimeOfDay,
    Weather,
    WeatherEffects,
    WeatherSimulator,
    WeatherState,
    _wind_compass,
)


# ======================================================================
# TimeOfDay
# ======================================================================

class TestTimeOfDay:

    def test_default_noon(self):
        t = TimeOfDay()
        assert t.hour == 12.0

    def test_custom_hour(self):
        t = TimeOfDay(hour=6.5)
        assert t.hour == pytest.approx(6.5)

    def test_wraps_at_24(self):
        t = TimeOfDay(hour=23.0)
        t.advance(2.0)
        assert t.hour == pytest.approx(1.0)

    def test_wraps_negative_init(self):
        t = TimeOfDay(hour=25.0)
        assert t.hour == pytest.approx(1.0)

    def test_advance_normal(self):
        t = TimeOfDay(hour=10.0)
        t.advance(3.0)
        assert t.hour == pytest.approx(13.0)

    def test_advance_multiple_wraps(self):
        t = TimeOfDay(hour=0.0)
        t.advance(50.0)
        assert t.hour == pytest.approx(2.0)

    def test_is_day_at_noon(self):
        assert TimeOfDay(12.0).is_day()

    def test_is_day_at_6(self):
        assert TimeOfDay(6.0).is_day()

    def test_is_day_at_18(self):
        assert TimeOfDay(18.0).is_day()

    def test_not_day_at_midnight(self):
        assert not TimeOfDay(0.0).is_day()

    def test_is_night_at_midnight(self):
        assert TimeOfDay(0.0).is_night()

    def test_is_night_at_22(self):
        assert TimeOfDay(22.0).is_night()

    def test_is_night_at_4(self):
        assert TimeOfDay(4.0).is_night()

    def test_not_night_at_noon(self):
        assert not TimeOfDay(12.0).is_night()

    def test_twilight_dawn(self):
        assert TimeOfDay(5.5).is_twilight()

    def test_twilight_dusk(self):
        assert TimeOfDay(19.0).is_twilight()

    def test_not_twilight_at_noon(self):
        assert not TimeOfDay(12.0).is_twilight()

    def test_light_level_noon_brightest(self):
        ll = TimeOfDay(12.0).light_level()
        assert ll == pytest.approx(1.0)

    def test_light_level_midnight_darkest(self):
        ll = TimeOfDay(0.0).light_level()
        assert ll == pytest.approx(0.0)

    def test_light_level_6am_mid(self):
        ll = TimeOfDay(6.0).light_level()
        assert ll == pytest.approx(0.5)

    def test_light_level_always_in_range(self):
        for h in [i * 0.5 for i in range(48)]:
            ll = TimeOfDay(h).light_level()
            assert 0.0 <= ll <= 1.0, f"hour={h}, light={ll}"

    def test_visibility_modifier_day_is_1(self):
        assert TimeOfDay(12.0).visibility_modifier() == pytest.approx(1.0)

    def test_visibility_modifier_night_is_03(self):
        assert TimeOfDay(0.0).visibility_modifier() == pytest.approx(0.3)

    def test_sun_angle_noon(self):
        assert TimeOfDay(12.0).sun_angle() == pytest.approx(90.0)

    def test_sun_angle_horizon_at_6(self):
        assert TimeOfDay(6.0).sun_angle() == pytest.approx(0.0, abs=0.1)

    def test_sun_angle_never_negative(self):
        for h in [i * 0.5 for i in range(48)]:
            assert TimeOfDay(h).sun_angle() >= 0.0


# ======================================================================
# Weather / WeatherState
# ======================================================================

class TestWeatherEnum:

    def test_all_members(self):
        assert len(Weather) == 8

    def test_values_are_strings(self):
        for w in Weather:
            assert isinstance(w.value, str)


class TestWeatherState:

    def test_defaults(self):
        ws = WeatherState()
        assert ws.current == Weather.CLEAR
        assert ws.intensity == 0.0
        assert ws.wind_speed == 0.0
        assert ws.temperature == 20.0


# ======================================================================
# WeatherEffects
# ======================================================================

class TestWeatherEffects:

    def test_clear_visibility(self):
        ws = WeatherState(current=Weather.CLEAR, intensity=0.0)
        assert WeatherEffects.visibility_modifier(ws) == pytest.approx(1.0)

    def test_fog_visibility_low(self):
        ws = WeatherState(current=Weather.FOG, intensity=0.5)
        v = WeatherEffects.visibility_modifier(ws)
        assert v < 0.25

    def test_storm_visibility(self):
        ws = WeatherState(current=Weather.STORM, intensity=0.5)
        assert WeatherEffects.visibility_modifier(ws) < 0.4

    def test_clear_movement(self):
        ws = WeatherState(current=Weather.CLEAR)
        assert WeatherEffects.movement_modifier(ws) == pytest.approx(1.0)

    def test_snow_movement_slow(self):
        ws = WeatherState(current=Weather.SNOW, intensity=0.5)
        assert WeatherEffects.movement_modifier(ws) < 0.7

    def test_storm_movement_slow(self):
        ws = WeatherState(current=Weather.STORM, intensity=0.8)
        assert WeatherEffects.movement_modifier(ws) < 0.5

    def test_accuracy_no_wind(self):
        ws = WeatherState(current=Weather.CLEAR, wind_speed=0.0, intensity=0.0)
        assert WeatherEffects.accuracy_modifier(ws) == pytest.approx(1.0)

    def test_accuracy_wind_degrades(self):
        ws = WeatherState(current=Weather.CLEAR, wind_speed=10.0)
        acc = WeatherEffects.accuracy_modifier(ws)
        assert acc == pytest.approx(0.8)

    def test_accuracy_rain_and_wind(self):
        ws = WeatherState(current=Weather.RAIN, wind_speed=5.0, intensity=0.5)
        acc = WeatherEffects.accuracy_modifier(ws)
        assert acc < 0.85

    def test_sound_clear_is_1(self):
        ws = WeatherState(current=Weather.CLEAR, intensity=0.0)
        assert WeatherEffects.sound_modifier(ws) == pytest.approx(1.0)

    def test_sound_storm_low(self):
        ws = WeatherState(current=Weather.STORM, intensity=0.8)
        assert WeatherEffects.sound_modifier(ws) < 0.4

    def test_combined_visibility_day_clear(self):
        ws = WeatherState(current=Weather.CLEAR, intensity=0.0)
        t = TimeOfDay(12.0)
        cv = WeatherEffects.combined_visibility(ws, t)
        assert cv == pytest.approx(1.0)

    def test_combined_visibility_night_fog(self):
        ws = WeatherState(current=Weather.FOG, intensity=0.5)
        t = TimeOfDay(0.0)
        cv = WeatherEffects.combined_visibility(ws, t)
        assert cv < 0.1

    def test_all_modifiers_clamped_above_zero(self):
        ws = WeatherState(
            current=Weather.STORM, intensity=1.0, wind_speed=30.0
        )
        assert WeatherEffects.visibility_modifier(ws) >= 0.05
        assert WeatherEffects.movement_modifier(ws) >= 0.1
        assert WeatherEffects.accuracy_modifier(ws) >= 0.1
        assert WeatherEffects.sound_modifier(ws) >= 0.1


# ======================================================================
# WeatherSimulator
# ======================================================================

class TestWeatherSimulator:

    def test_initial_state(self):
        ws = WeatherSimulator(initial=Weather.RAIN, seed=42)
        assert ws.state.current == Weather.RAIN

    def test_update_zero_dt(self):
        ws = WeatherSimulator(seed=42)
        old = ws.state.current
        ws.update(0.0)
        assert ws.state.current == old

    def test_produces_varied_weather_48h(self):
        """Over 48 simulated hours, weather should change at least once."""
        ws = WeatherSimulator(initial=Weather.CLEAR, seed=1)
        seen = {ws.state.current}
        for _ in range(480):
            ws.update(0.1)
            seen.add(ws.state.current)
        assert len(seen) > 1, "Weather never changed in 48 hours"

    def test_wind_speed_stays_bounded(self):
        ws = WeatherSimulator(seed=7)
        for _ in range(1000):
            ws.update(0.1)
            assert 0.0 <= ws.state.wind_speed <= 30.0

    def test_humidity_stays_bounded(self):
        ws = WeatherSimulator(seed=7)
        for _ in range(1000):
            ws.update(0.1)
            assert 0.0 <= ws.state.humidity <= 1.0

    def test_intensity_stays_bounded(self):
        ws = WeatherSimulator(seed=7)
        for _ in range(1000):
            ws.update(0.1)
            assert 0.0 <= ws.state.intensity <= 1.0

    def test_storm_eventually_ends(self):
        ws = WeatherSimulator(initial=Weather.STORM, seed=42)
        for _ in range(200):
            ws.update(0.1)
        # Storm lasts 1-3 hours, so after 20h it should have ended
        assert ws.state.current != Weather.STORM

    def test_seed_determinism(self):
        a = WeatherSimulator(seed=99)
        b = WeatherSimulator(seed=99)
        for _ in range(100):
            a.update(0.1)
            b.update(0.1)
        assert a.state.current == b.state.current
        assert a.state.wind_speed == b.state.wind_speed


# ======================================================================
# Environment
# ======================================================================

class TestEnvironment:

    def test_default_construction(self):
        env = Environment()
        assert env.time.hour == 12.0

    def test_update_advances_time(self):
        env = Environment(time=TimeOfDay(10.0))
        env.update(3600.0)  # 1 hour
        assert env.time.hour == pytest.approx(11.0)

    def test_update_advances_weather(self):
        env = Environment(weather=WeatherSimulator(seed=1))
        env.update(7200.0)  # 2 hours — weather will have been updated

    def test_visibility_range(self):
        env = Environment()
        v = env.visibility()
        assert 0.0 < v <= 1.0

    def test_movement_speed_modifier(self):
        env = Environment()
        m = env.movement_speed_modifier()
        assert 0.0 < m <= 1.0

    def test_accuracy_modifier(self):
        env = Environment()
        a = env.accuracy_modifier()
        assert 0.0 < a <= 1.0

    def test_detection_range_modifier(self):
        env = Environment()
        d = env.detection_range_modifier()
        assert 0.0 < d <= 1.0

    def test_snapshot_is_json_serializable(self):
        env = Environment()
        snap = env.snapshot()
        text = json.dumps(snap)
        assert isinstance(text, str)

    def test_snapshot_keys(self):
        snap = Environment().snapshot()
        expected = {
            "hour", "is_day", "is_night", "light_level", "sun_angle",
            "weather", "intensity", "wind_speed", "wind_direction",
            "temperature", "humidity", "visibility", "movement_modifier",
            "accuracy_modifier", "detection_range_modifier",
        }
        assert set(snap.keys()) == expected

    def test_describe_returns_string(self):
        desc = Environment().describe()
        assert isinstance(desc, str)
        assert len(desc) > 5

    def test_describe_contains_temperature(self):
        desc = Environment().describe()
        assert "\u00b0C" in desc

    def test_describe_day_label(self):
        env = Environment(time=TimeOfDay(12.0))
        assert "day" in env.describe()

    def test_describe_night_label(self):
        env = Environment(time=TimeOfDay(0.0))
        assert "night" in env.describe()


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:

    def test_hour_24_wraps(self):
        t = TimeOfDay(24.0)
        assert t.hour == pytest.approx(0.0)

    def test_wind_zero(self):
        ws = WeatherState(wind_speed=0.0)
        assert WeatherEffects.accuracy_modifier(ws) == pytest.approx(1.0)

    def test_intensity_boundary_0(self):
        ws = WeatherState(current=Weather.FOG, intensity=0.0)
        v = WeatherEffects.visibility_modifier(ws)
        assert v == pytest.approx(0.2)

    def test_intensity_boundary_1(self):
        ws = WeatherState(current=Weather.FOG, intensity=1.0)
        v = WeatherEffects.visibility_modifier(ws)
        assert v < 0.2

    def test_wind_compass_north(self):
        assert _wind_compass(0.0) == "N"

    def test_wind_compass_east(self):
        assert _wind_compass(math.pi / 2) == "E"

    def test_wind_compass_south(self):
        assert _wind_compass(math.pi) == "S"

    def test_wind_compass_west(self):
        assert _wind_compass(3 * math.pi / 2) == "W"

    def test_heavy_rain_movement(self):
        ws = WeatherState(current=Weather.HEAVY_RAIN, intensity=0.5)
        m = WeatherEffects.movement_modifier(ws)
        assert m < 0.85

    def test_sandstorm_visibility_worst(self):
        ws = WeatherState(current=Weather.SANDSTORM, intensity=1.0)
        v = WeatherEffects.visibility_modifier(ws)
        assert v < 0.15
