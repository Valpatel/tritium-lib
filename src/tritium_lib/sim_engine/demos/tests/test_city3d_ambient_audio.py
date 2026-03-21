"""
Tests for city3d.html ambient city soundscape.
Demonstrates audio/spatial.py and soundtrack.py integration.
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


# =========================================================================
# 1. CITY AMBIENCE — continuous low traffic hum
# =========================================================================

class TestCityAmbience:
    def test_ambient_state_variable(self, source):
        assert "ambientOn" in source, "Missing ambientOn toggle state"

    def test_ambient_gain_node(self, source):
        assert "ambientGain" in source, "Missing ambientGain node for volume control"

    def test_low_frequency_oscillator(self, source):
        assert "80" in source and "oscillator" in source.lower(), \
            "Missing 80Hz low-frequency oscillator for traffic hum"

    def test_white_noise_buffer(self, source):
        assert "createBuffer" in source and "ambientNoise" in source, \
            "Missing white noise buffer for ambient sound"

    def test_s_key_toggle(self, source):
        assert "KeyS" in source, "Missing S key binding for ambient sound toggle"

    def test_day_night_volume(self, source):
        assert "ambientGain" in source and "simTime" in source, \
            "Ambient volume should vary with simTime (day/night)"


# =========================================================================
# 2. CAR HORNS — occasional beeps near collisions
# =========================================================================

class TestCarHorns:
    def test_horn_function(self, source):
        assert "playHorn" in source, "Missing playHorn function"

    def test_horn_frequency_440(self, source):
        assert "440" in source, "Horn should use 440Hz sine wave"

    def test_horn_short_duration(self, source):
        assert "0.15" in source, "Horn beep should be 0.15 seconds"

    def test_horn_triggered_by_collision(self, source):
        assert "playHorn" in source and "stuckTimer" in source, \
            "Horn should trigger near stuck/colliding cars"


# =========================================================================
# 3. CROWD NOISE — filtered noise during riot
# =========================================================================

class TestCrowdNoise:
    def test_crowd_noise_node(self, source):
        assert "crowdGain" in source, "Missing crowdGain node for riot crowd noise"

    def test_crowd_bandpass_filter(self, source):
        assert "crowdFilter" in source or "bandpass" in source, \
            "Crowd noise needs bandpass filter at 200-400Hz range"

    def test_crowd_scales_with_riot(self, source):
        assert "crowdGain" in source and "RIOT" in source, \
            "Crowd noise volume should scale during RIOT phase"


# =========================================================================
# 4. RAIN SOUND — high-frequency filtered noise
# =========================================================================

class TestRainSound:
    def test_rain_sound_node(self, source):
        assert "rainGain" in source, "Missing rainGain node for rain audio"

    def test_rain_highpass_filter(self, source):
        assert "rainFilter" in source or "highpass" in source, \
            "Rain sound needs highpass filter"

    def test_rain_tied_to_rain_active(self, source):
        assert "rainGain" in source and "rainActive" in source, \
            "Rain sound should be tied to rainActive state"


# =========================================================================
# 5. HUD — S key shown in help bar
# =========================================================================

class TestHUD:
    def test_s_key_in_help(self, source):
        assert ">S<" in source and "Sound" in source, \
            "Help bar should show S key for Sound toggle"
