"""
Tests for city3d.html medical triage visualization during riot.
Demonstrates medical.py — triage markers, medic officers, treatment logic.
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
# 1. TRIAGE MARKERS — color-coded by health severity
# =========================================================================

class TestTriageMarkers:
    def test_triage_color_red_critical(self, source):
        """RED for critical health < 0.3"""
        assert "0xff2a6d" in source, "Should have red/magenta triage color for critical"

    def test_triage_color_yellow_moderate(self, source):
        """YELLOW for moderate health 0.3-0.6"""
        assert "0xfcee0a" in source, "Should have yellow triage color for moderate"

    def test_triage_color_green_minor(self, source):
        """GREEN for minor health > 0.6"""
        assert "0x05ffa1" in source, "Should have green triage color for minor"

    def test_triage_health_thresholds(self, source):
        """Health thresholds at 0.3 and 0.6"""
        assert "h > 0.6" in source or "health > 0.6" in source, \
            "Should check health > 0.6 for green triage"
        assert "h > 0.3" in source or "health > 0.3" in source, \
            "Should check health > 0.3 for yellow triage"

    def test_per_marker_material(self, source):
        """Each marker has its own material for individual coloring"""
        assert "material.color.setHex" in source, \
            "Should set triage color per marker via material.color.setHex"


# =========================================================================
# 2. MEDIC OFFICERS — white-bodied police spawned in Phase 3
# =========================================================================

class TestMedicOfficers:
    def test_medics_array(self, source):
        assert "medics = []" in source or "const medics" in source, \
            "Should have medics array"

    def test_spawn_medics_function(self, source):
        assert "function spawnMedics()" in source, \
            "Should have spawnMedics function"

    def test_medic_white_body(self, source):
        """Medics have white body color to distinguish from regular police"""
        assert "0xFFFFFF" in source, "Medic body should be white"

    def test_medic_is_medic_flag(self, source):
        assert "isMedic: true" in source, \
            "Should flag medic officers with isMedic: true"

    def test_medics_spawned_phase3(self, source):
        """Medics spawn when campaign enters Phase 3 (Aftermath)"""
        assert "spawnMedics()" in source, \
            "spawnMedics should be called in phase 3 transition"

    def test_medic_kill_feed_deploy(self, source):
        assert "MEDIC officer deployed" in source, \
            "Kill feed should announce medic deployment"


# =========================================================================
# 3. MEDIC TREATMENT — walk to injured, treat, remove triage marker
# =========================================================================

class TestMedicTreatment:
    def test_update_medics_function(self, source):
        assert "function updateMedics(dt)" in source, \
            "Should have updateMedics function"

    def test_update_medics_called(self, source):
        assert "updateMedics(dt)" in source, \
            "updateMedics should be called in main loop"

    def test_treat_timer(self, source):
        """Medic stands near injured for ~5 seconds"""
        assert "treatTimer" in source, "Should have treatTimer for treatment duration"

    def test_treat_target(self, source):
        assert "treatTarget" in source, "Should track which person medic is treating"

    def test_medic_heals_person(self, source):
        """Treatment restores health"""
        assert "health = Math.min" in source or "health += " in source or \
            "(m.treatTarget.health || 0) + 0.5" in source, \
            "Treatment should restore person health"

    def test_medic_removes_marker(self, source):
        """Treatment removes triage marker"""
        assert "hideInjuredMarker" in source, \
            "Treatment should hide the injured marker"

    def test_medic_kill_feed_treat(self, source):
        assert "Medic treated casualty" in source, \
            "Kill feed should announce medic treatment"

    def test_medics_cleared_on_reset(self, source):
        assert "medics.length = 0" in source, \
            "Medics array should be cleared on riot reset"


# =========================================================================
# 4. AI PANEL — medic state display
# =========================================================================

class TestMedicAIState:
    def test_medic_treating_state(self, source):
        assert "'TREATING'" in source, \
            "AI state should show TREATING when medic is treating"

    def test_medic_responding_state(self, source):
        assert "'RESPONDING'" in source, \
            "AI state should show RESPONDING when medic walking to injured"
