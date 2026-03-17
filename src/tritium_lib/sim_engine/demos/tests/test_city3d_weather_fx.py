"""
Tests for city3d.html dynamic sky colors and lightning during storms.
Demonstrates weather_fx.py (DayNightCycle, LightningSystem).
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
    with open(CITY3D_PATH, "r") as f:
        return f.read()


# =========================================================================
# 1. DYNAMIC SKY — time-of-day color gradient
# =========================================================================

class TestDynamicSky:
    def test_get_sky_color_function(self, source):
        assert "getSkyColor" in source, "Missing getSkyColor function"

    def test_dawn_color_0x0a0a2e(self, source):
        assert "0x0a0a2e" in source, "Missing dawn dark blue color"

    def test_dawn_color_0xff8844(self, source):
        assert "0xff8844" in source, "Missing dawn orange color"

    def test_day_color_0x88aacc(self, source):
        assert "0x88aacc" in source, "Missing daytime sky blue color"

    def test_dusk_color_0xff6622(self, source):
        assert "0xff6622" in source, "Missing dusk orange color"

    def test_dusk_color_0x2a1a3e(self, source):
        assert "0x2a1a3e" in source, "Missing dusk purple color"

    def test_night_color_0x0a0a0f(self, source):
        assert "0x0a0a0f" in source, "Missing night dark void color"

    def test_sky_color_lerp(self, source):
        assert ".lerp(" in source, "Missing color interpolation (lerp)"

    def test_sky_applied_to_background(self, source):
        assert "scene.background.set(skyColor)" in source, \
            "Sky color must be applied to scene.background"

    def test_sky_applied_to_fog(self, source):
        assert "scene.fog.color.set(skyColor)" in source, \
            "Sky color must be applied to scene.fog.color"

    def test_sky_uses_sim_time(self, source):
        assert "getSkyColor(simTime)" in source, \
            "getSkyColor must be called with simTime"


# =========================================================================
# 2. LIGHTNING — bolt + flash + thunder during rain+riot
# =========================================================================

class TestLightning:
    def test_lightning_count_variable(self, source):
        assert "lightningCount" in source, "Missing lightningCount counter"

    def test_lightning_timer_variable(self, source):
        assert "lightningTimer" in source, "Missing lightningTimer for flash duration"

    def test_bolt_geometry(self, source):
        assert "boltGeo" in source, "Missing pre-allocated bolt geometry"

    def test_bolt_line_segments(self, source):
        assert "LineSegments" in source, "Missing LineSegments for bolt mesh"

    def test_bolt_mesh_visible_toggle(self, source):
        assert "boltMesh.visible = true" in source, "Bolt must be made visible on flash"
        assert "boltMesh.visible = false" in source, "Bolt must be hidden after flash"

    def test_exposure_flash(self, source):
        assert "toneMappingExposure = 4.0" in source, \
            "Lightning flash must set exposure to 4.0"

    def test_rain_and_riot_condition(self, source):
        assert "rainActive" in source and "riotPhase" in source, \
            "Lightning requires both rain and riot phase check"

    def test_bolt_8_vertices(self, source):
        # 8 vertices * 3 floats = 24
        assert "Float32Array(24)" in source, \
            "Bolt geometry should have 24 floats (8 vertices)"

    def test_bolt_needs_update(self, source):
        assert "needsUpdate = true" in source, \
            "Bolt position attribute must be flagged needsUpdate"

    def test_thunder_sound(self, source):
        # Thunder uses low-frequency synthesis
        assert "ensureAudio()" in source, "Thunder must use audio system"


# =========================================================================
# 3. DEBUG OVERLAY — weather section
# =========================================================================

class TestWeatherDebug:
    def test_weather_section_in_overlay(self, source):
        assert "WEATHER" in source, "Debug overlay must have WEATHER section"

    def test_lightning_count_in_overlay(self, source):
        assert "Lightning" in source and "lightningCount" in source, \
            "Debug overlay must show lightning strike count"

    def test_sky_color_in_overlay(self, source):
        assert "getSkyColor(simTime)" in source, \
            "Debug overlay should show current sky color"
