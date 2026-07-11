# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SeasonalCycle.for_date, WeatherSimulator.reseed, and Environment lightning.

These back three tick features:
  - auto-derive season from a real date + AO latitude (site standup),
  - deterministic fixed-weather goldens (reseed), and
  - live night-thunderstorm lightning VFX (strike/snapshot).

Golden safety is the load-bearing invariant: a default ``Environment()``
snapshot must stay byte-identical (no ``lightning``/season keys).
"""

from __future__ import annotations

import datetime

from tritium_lib.sim_engine.environment import (
    Environment,
    SeasonalCycle,
    Weather,
    WeatherSimulator,
)


# --------------------------------------------------------------------------
# SeasonalCycle.for_date — real-date + latitude derivation
# --------------------------------------------------------------------------

def test_for_date_northern_winter_vs_summer():
    """A northern AO in January is bare winter; in July it is lush summer."""
    winter = SeasonalCycle.for_date(datetime.date(2026, 1, 15), 40.0)
    summer = SeasonalCycle.for_date(datetime.date(2026, 7, 15), 40.0)
    assert winter.season().value == "winter"
    assert summer.season().value == "summer"
    # Real daylight: winter days are shorter than summer days.
    assert winter.daylight_hours() < summer.daylight_hours()
    # Real foliage: bare in winter, lush in summer.
    assert winter.foliage_state() < 0.4
    assert summer.foliage_state() > 0.8


def test_for_date_southern_hemisphere_flips():
    """January below the equator is summer, not winter."""
    jan_south = SeasonalCycle.for_date(datetime.date(2026, 1, 15), -33.9)
    assert jan_south.season().value == "summer"


def test_for_date_matches_explicit_day_of_year():
    """for_date is exactly SeasonalCycle(day_of_year, lat) for that date."""
    when = datetime.date(2026, 3, 21)
    doy = when.timetuple().tm_yday
    a = SeasonalCycle.for_date(when, 51.5)
    b = SeasonalCycle(day_of_year=doy, latitude=51.5)
    assert a.snapshot() == b.snapshot()


def test_for_date_accepts_datetime_and_defaults_to_now():
    """A datetime works, and None falls back to today's day-of-year."""
    dt = datetime.datetime(2026, 6, 21, 14, 30, tzinfo=datetime.timezone.utc)
    cyc = SeasonalCycle.for_date(dt, 40.0)
    assert cyc.day_of_year == dt.timetuple().tm_yday
    today = SeasonalCycle.for_date(None, 40.0)
    assert today.day_of_year == datetime.datetime.now(
        datetime.timezone.utc).timetuple().tm_yday


def test_for_date_latitude_drives_daylight_extremes():
    """Higher latitude => more extreme winter/summer daylight swing."""
    equator = SeasonalCycle.for_date(datetime.date(2026, 12, 21), 0.0)
    high = SeasonalCycle.for_date(datetime.date(2026, 12, 21), 60.0)
    # Near the equator the shortest day is ~12h; at 60N it is far shorter.
    assert high.daylight_hours() < equator.daylight_hours()


# --------------------------------------------------------------------------
# WeatherSimulator.reseed — deterministic fixed weather
# --------------------------------------------------------------------------

def test_reseed_makes_evolution_bit_identical():
    a = WeatherSimulator(initial=Weather.FOG)
    b = WeatherSimulator(initial=Weather.FOG)
    a.reseed(2026)
    b.reseed(2026)
    for _ in range(200):
        a.update(0.01)
        b.update(0.01)
    assert a.state.intensity == b.state.intensity
    assert a.state.wind_speed == b.state.wind_speed
    assert a.state.wind_direction == b.state.wind_direction
    assert a.state.current == b.state.current


def test_reseed_different_seeds_diverge():
    a = WeatherSimulator(initial=Weather.RAIN)
    b = WeatherSimulator(initial=Weather.RAIN)
    a.reseed(1)
    b.reseed(2)
    for _ in range(200):
        a.update(0.01)
        b.update(0.01)
    # Two different seeds almost surely produce a different intensity walk.
    assert a.state.intensity != b.state.intensity


# --------------------------------------------------------------------------
# Environment lightning — VFX only, golden-safe
# --------------------------------------------------------------------------

def test_default_snapshot_has_no_lightning_or_season_keys():
    """The golden byte-identity guard: default Environment is unchanged."""
    snap = Environment().snapshot()
    assert "lightning" not in snap
    assert "season" not in snap
    assert "foliage_state" not in snap


def test_strike_latches_into_snapshot():
    env = Environment()
    strike = env.strike(center=(0.0, 0.0), radius=80.0)
    snap = env.snapshot()
    assert "lightning" in snap
    L = snap["lightning"]
    assert L["strike_id"] == 1
    assert len(L["segments"]) > 10           # branching bolt geometry
    assert 0.0 <= L["flash_intensity"] <= 1.0
    assert L["thunder_delay_s"] >= 0.0       # physically-correct delay
    # Each segment is [[x1,y1,z1],[x2,y2,z2]] with y = height above ground.
    seg = L["segments"][0]
    assert len(seg) == 2 and len(seg[0]) == 3
    assert strike["strike_id"] == 1


def test_strike_ages_out_after_lifetime():
    env = Environment()
    env.strike()
    assert "lightning" in env.snapshot()
    # Advance past the strike lifetime.
    for _ in range(int(env._STRIKE_LIFETIME_S / 0.1) + 3):
        env.update(0.1)
    assert env.snapshot().get("lightning") is None


def test_storm_schedules_strikes_but_clear_never_does():
    """Storms fire lightning; clear weather never latches a strike."""
    # Clear weather: no strikes ever, over a long window.
    clear = Environment(weather=WeatherSimulator(initial=Weather.CLEAR))
    clear.weather.reseed(7)
    for _ in range(3000):
        clear.update(0.1)
    assert clear.snapshot().get("lightning") is None

    # Storm at full intensity: at least one strike within a minute.
    storm = Environment(weather=WeatherSimulator(initial=Weather.STORM))
    storm.weather.state.intensity = 1.0
    storm._lightning_rng.seed(3)
    fired = False
    for _ in range(600):
        # Keep it a storm (reseed keeps intensity high enough for scheduling).
        storm.weather.state.current = Weather.STORM
        storm.weather.state.intensity = 1.0
        storm.update(0.1)
        if storm._active_strike is not None:
            fired = True
            break
    assert fired
