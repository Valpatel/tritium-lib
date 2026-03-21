"""
Tests for city3d.html electronic warfare / jamming visualization.
Demonstrates cyber.py, electronic_warfare.py -- comm jamming zones,
GPS spoofing indicators, and countermeasure narration.
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
# 1. EW STATE VARIABLES
# =========================================================================

class TestEWState:
    def test_ew_jam_active_flag(self, source):
        assert "ewJamActive" in source, "Should have ewJamActive flag"

    def test_ew_jam_timer(self, source):
        assert "ewJamTimer" in source, "Should have ewJamTimer for duration tracking"

    def test_ew_jam_duration_15s(self, source):
        assert "EW_JAM_DURATION" in source and "15" in source, \
            "Jamming should last 15 seconds"

    def test_ew_jam_check_interval_30s(self, source):
        assert "EW_JAM_CHECK_INTERVAL" in source and "30" in source, \
            "Should check for jamming every 30 seconds"

    def test_ew_jam_chance_10pct(self, source):
        assert "EW_JAM_CHANCE" in source and "0.1" in source, \
            "Should have 10% chance per check"


# =========================================================================
# 2. JAMMING RING MESH -- pre-allocated, depthWrite:false
# =========================================================================

class TestJammingRing:
    def test_ring_geometry(self, source):
        assert "ewRingGeo" in source, "Should pre-allocate ring geometry"

    def test_ring_geometry_type(self, source):
        assert "RingGeometry" in source, "Should use THREE.RingGeometry for ground ring"

    def test_ring_material_magenta(self, source):
        assert "0xff2a6d" in source, "Ring should be magenta colored"

    def test_ring_depth_write_false(self, source):
        assert "depthWrite: false" in source, "Ring should have depthWrite: false"

    def test_ring_mesh_object(self, source):
        assert "ewRingMesh" in source, "Should have ewRingMesh object"

    def test_ring_initially_hidden(self, source):
        assert "ewRingMesh.visible = false" in source, "Ring should start hidden"


# =========================================================================
# 3. COMMS JAMMED STATUS
# =========================================================================

class TestCommsJammed:
    def test_comms_jammed_state(self, source):
        assert "'JAMMED'" in source or '"JAMMED"' in source, \
            "Should have JAMMED comms status"

    def test_comms_jammed_color_red(self, source):
        assert "JAMMED" in source and "#ff2a6d" in source, \
            "JAMMED status should show in red/magenta"

    def test_formation_gap_increases(self, source):
        assert "ewJamActive ? 25" in source or "ewJamActive? 25" in source, \
            "Formation gap tolerance should increase to 25 during jamming"


# =========================================================================
# 4. NARRATION TEMPLATES
# =========================================================================

class TestEWNarration:
    def test_ew_jam_start_template(self, source):
        assert "ew_jam_start" in source, "Should have ew_jam_start narration template"

    def test_ew_counter_template(self, source):
        assert "ew_counter" in source, "Should have ew_counter narration template"

    def test_jam_narration_mentions_interference(self, source):
        assert "interference" in source.lower(), \
            "Jam narration should mention signal interference"

    def test_counter_narration_mentions_countermeasures(self, source):
        assert "counter-measures" in source.lower() or "countermeasures" in source.lower(), \
            "Counter narration should mention deploying counter-measures"


# =========================================================================
# 5. GPS SPOOFING DEBUG INDICATOR
# =========================================================================

class TestGPSSpoofing:
    def test_gps_spoofed_label(self, source):
        assert "GPS" in source and "SPOOFED" in source, \
            "Should show GPS: SPOOFED in debug overlay during jamming"

    def test_gps_spoofed_yellow(self, source):
        assert "SPOOFED" in source and "#fcee0a" in source, \
            "GPS SPOOFED should be yellow"


# =========================================================================
# 6. ROBOT HEADING JITTER
# =========================================================================

class TestRobotJitter:
    def test_robot_heading_jitter(self, source):
        assert "rcar.rotY" in source and "0.6" in source, \
            "Robots in jammed zone should get heading jitter +/-0.3 rad (range 0.6)"

    def test_jitter_checks_jam_zone(self, source):
        assert "ewJamRadius" in source, \
            "Jitter should check distance to jam zone center"


# =========================================================================
# 7. COUNTERMEASURE SHRINK BEHAVIOR
# =========================================================================

class TestCountermeasure:
    def test_ring_shrinks(self, source):
        assert "remaining < 5" in source or "remaining<5" in source, \
            "Ring should shrink in last 5 seconds"

    def test_scale_reduction(self, source):
        assert "ewRingMesh.scale" in source, \
            "Should scale the ring mesh down during countermeasures"

    def test_counter_narrated_at_5s(self, source):
        assert "ewCounterNarrated" in source, \
            "Should track whether counter narration has played"


# =========================================================================
# 8. UPDATE FUNCTION INTEGRATION
# =========================================================================

class TestUpdateIntegration:
    def test_update_function_exists(self, source):
        assert "function updateEWJamming" in source, \
            "Should have updateEWJamming function"

    def test_update_called_in_loop(self, source):
        assert "updateEWJamming(" in source, \
            "updateEWJamming should be called in animation loop"

    def test_riot_phase_check(self, source):
        assert "riotPhase === 'RIOT'" in source and "ewJamCheckTimer" in source, \
            "Jamming should only trigger during RIOT phase"
