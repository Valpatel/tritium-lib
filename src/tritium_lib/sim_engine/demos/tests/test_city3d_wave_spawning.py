"""
Tests for city3d.html wave-based protestor spawning mechanics.
Source-string tests that verify the HTML file contains required code patterns.

Demonstrates spawner.py from sim_engine — wave-based unit spawning
instead of all-at-once instantiation.

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
# 1. WAVE STATE VARIABLES (demonstrates spawner.py wave config)
# =========================================================================

class TestWaveState:
    def test_wave_current_variable(self, source):
        assert "spawnWave" in source, "Missing spawnWave current wave tracker"

    def test_wave_timer_variable(self, source):
        assert "spawnWaveTimer" in source, "Missing spawnWaveTimer for wave timing"

    def test_wave_definitions(self, source):
        assert "SPAWN_WAVES" in source, "Missing SPAWN_WAVES definitions array"

    def test_three_waves_defined(self, source):
        # Should have 3 wave entries with counts
        assert "count: 8" in source, "Wave 1 and 2 should spawn 8 protestors"
        assert "count: 4" in source, "Wave 3 should spawn 4 reinforcements"

    def test_wave_delays(self, source):
        assert "delay: 0" in source, "Wave 1 should have delay 0"
        assert "delay: 30" in source, "Wave 2 should have delay 30s"
        assert "delay: 60" in source, "Wave 3 should have delay 60s"


# =========================================================================
# 2. WAVE SPAWNING FUNCTION (demonstrates spawner.py spawn logic)
# =========================================================================

class TestWaveSpawnFunction:
    def test_spawn_wave_function(self, source):
        assert "spawnProtestorWave" in source, "Missing spawnProtestorWave function"

    def test_wave_aggression_scaling(self, source):
        """Later waves should have higher aggression."""
        assert "aggression" in source, "Protestors need aggression property"

    def test_wave_from_edge(self, source):
        """Wave 3 reinforcements should come from city edge."""
        assert "fromEdge" in source, "Wave 3 should have fromEdge flag for city-edge spawning"


# =========================================================================
# 3. SPAWN POINT MARKERS (ground markers at city edges)
# =========================================================================

class TestSpawnPointMarkers:
    def test_spawn_point_definitions(self, source):
        assert "spawnPoints" in source, "Missing spawnPoints array for ground markers"

    def test_spawn_marker_geometry(self, source):
        """Spawn points should use ring/circle geometry on ground."""
        assert "spawnMarker" in source, "Missing spawnMarker mesh for ground indicators"

    def test_spawn_marker_green_color(self, source):
        assert "0x05ffa1" in source, "Spawn markers should use green (#05ffa1)"


# =========================================================================
# 4. HUD WAVE INDICATOR
# =========================================================================

class TestWaveHUD:
    def test_wave_hud_element(self, source):
        assert "wave-indicator" in source, "Missing wave-indicator HUD element"

    def test_wave_progress_display(self, source):
        assert "WAVE" in source, "HUD should display WAVE label"

    def test_wave_progress_bar(self, source):
        assert "wave-progress" in source, "Missing wave-progress bar element"


# =========================================================================
# 5. WAVE NARRATION (dispatch messages on wave arrival)
# =========================================================================

class TestWaveNarration:
    def test_wave_narration_template(self, source):
        assert "wave_reinforcement" in source, "Missing wave_reinforcement narration template"

    def test_wave_dispatch_message(self, source):
        assert "protestors arriving" in source or "reinforcements" in source.lower(), \
            "Missing dispatch message about arriving protestors"
