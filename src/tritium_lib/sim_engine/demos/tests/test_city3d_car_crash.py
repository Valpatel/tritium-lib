"""
Tests for city3d.html car crash/damage visualization.
Source-string tests that verify the HTML file contains required code patterns.

Created by Matthew Valancy
Copyright 2026 Valpatel Software LLC
Licensed under AGPL-3.0
"""
import os
import pytest

CITY3D_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "city3d.html"
)


@pytest.fixture(scope="module")
def source():
    """Load city3d.html combined with all city3d/*.js modules.

    The frontend is split across city3d.html and external JS modules in
    city3d/*.js.  Tests must scan both to find all code patterns.
    """
    import glob as _glob
    parts = []
    with open(CITY3D_PATH, "r") as f:
        parts.append(f.read())
    js_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "city3d")
    for js_path in sorted(_glob.glob(os.path.join(js_dir, "*.js"))):
        with open(js_path, "r") as f:
            parts.append(f.read())
    return "\n".join(parts)


class TestCrashSound:
    def test_play_crash_function_exists(self, source):
        assert "function playCrash(" in source

    def test_crash_lower_pitch_than_gunshot(self, source):
        assert "playbackRate" in source, "Crash sound should use lower playback rate"

    def test_crash_uses_white_noise(self, source):
        idx = source.index("function playCrash(")
        snippet = source[idx:idx + 400]
        assert "Math.random()" in snippet, "Crash uses white noise burst"


class TestVehicleCollisionEffects:
    def test_crash_counter_on_car(self, source):
        assert "car.crashes" in source, "Cars should track crash count"

    def test_debris_particles_on_collision(self, source):
        assert "spawnParticle(car.x, 1.5, car.z" in source, "Debris particles at collision"

    def test_speed_reduction_on_crash(self, source):
        assert "car.speed *= 0.8" in source, "20% permanent speed reduction"

    def test_damage_tint_toward_gray(self, source):
        assert "0x333333" in source, "Damage tint lerps toward gray"
        assert "car.bodyColor" in source and "lerp" in source

    def test_smoke_trail_after_three_crashes(self, source):
        assert "car.crashes >= 3" in source or "(car.crashes || 0) >= 3" in source
        assert "isSmoke" in source, "Smoke particles use isSmoke flag"

    def test_smoke_emits_every_two_seconds(self, source):
        assert "car.smokeTimer" in source, "Smoke uses timer"
        assert "smokeTimer > 2" in source, "Smoke emits every 2 seconds"


class TestCarPedestrianCollision:
    def test_pedestrian_distance_check(self, source):
        assert "car.x - ped.x" in source, "Check distance between car and pedestrian"

    def test_knock_down_pedestrian(self, source):
        assert "knockDownPerson(ped)" in source

    def test_kill_feed_entry(self, source):
        assert "Pedestrian struck by vehicle" in source

    def test_car_pauses_after_hitting_pedestrian(self, source):
        assert "car.pauseTimer = 1.0" in source

    def test_liability_cost(self, source):
        assert "policeBudget -= COST_INJURY" in source

    def test_only_moving_cars_hit_pedestrians(self, source):
        assert "car.speed > 2" in source, "Only fast-moving cars can hit pedestrians"
