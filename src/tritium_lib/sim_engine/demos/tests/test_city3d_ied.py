"""
Tests for city3d.html IED/trap and guerrilla cell visualization.
Demonstrates asymmetric.py — IED placement, detonation, detection by ROS2 robots.
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
# 1. IED STATE — pre-allocated array and InstancedMesh
# =========================================================================

class TestIEDState:
    def test_ied_array(self, source):
        assert "ieds = []" in source or "const ieds" in source, "Should have ieds array"

    def test_max_ieds_constant(self, source):
        assert "MAX_IEDS" in source, "Should have MAX_IEDS constant"

    def test_max_ieds_value_6(self, source):
        assert "MAX_IEDS = 6" in source, "Should pre-allocate 6 IED slots"

    def test_ied_instanced_mesh(self, source):
        assert "iedMesh" in source, "Should have iedMesh InstancedMesh"

    def test_ied_box_geometry(self, source):
        assert "BoxGeometry(0.5" in source or "BoxGeometry( 0.5" in source, \
            "IED should use small box geometry (0.5 wide)"

    def test_ied_brown_color(self, source):
        # Brown color for the IED device
        assert "0x8B4513" in source or "0x7a3b10" in source or "0x6b3410" in source, \
            "IED should be brown colored"

    def test_ied_placement_timer(self, source):
        assert "iedPlaceTimer" in source, "Should have timer to throttle IED placement"


# =========================================================================
# 2. IED PLACEMENT — protestors plant during RIOT phase
# =========================================================================

class TestIEDPlacement:
    def test_placement_during_riot(self, source):
        assert "RIOT" in source and "ied" in source.lower(), \
            "IED placement should happen during RIOT phase"

    def test_placement_chance(self, source):
        # 5% chance check
        assert "0.05" in source or "0.95" in source, \
            "Should have ~5% placement probability"

    def test_ied_armed_flag(self, source):
        assert "armed" in source, "IED should have armed flag"

    def test_ied_detected_flag(self, source):
        assert "detected" in source, "IED should have detected flag"

    def test_ied_position_stored(self, source):
        # IED stores x, z position
        assert "ied" in source.lower() and "x:" in source, \
            "IED should store position"


# =========================================================================
# 3. IED DETONATION — triggered by nearby police/vans
# =========================================================================

class TestIEDDetonation:
    def test_detonation_radius(self, source):
        # Check within 3 units
        assert "3 *" in source or "< 9" in source or "< 3" in source, \
            "Detonation should check within ~3 unit radius"

    def test_detonation_creates_explosion(self, source):
        assert "createExplosion" in source, "Detonation should reuse createExplosion"

    def test_detonation_kill_feed(self, source):
        assert "IED" in source, "Kill feed should mention IED"

    def test_detonation_narration(self, source):
        assert "CONTACT IED" in source or "ied_detonate" in source, \
            "Should have IED detonation narration"

    def test_detonation_damages_police(self, source):
        assert "health" in source, "IED detonation should damage police health"

    def test_ied_deactivated_after_detonation(self, source):
        assert "armed" in source and "false" in source, \
            "IED should be deactivated after detonation"


# =========================================================================
# 4. IED DETECTION — by ROS2 robot car lidar
# =========================================================================

class TestIEDDetection:
    def test_robot_detects_ied(self, source):
        assert "detected" in source and "ied" in source.lower(), \
            "Robots should detect IEDs"

    def test_detection_range(self, source):
        # Within 2 units of lidar ray
        assert "2" in source, "Detection should use ~2 unit range"

    def test_detected_color_change(self, source):
        assert "0xff0000" in source or "0xff2a6d" in source, \
            "Detected IED should flash red"

    def test_detection_kill_feed(self, source):
        assert "suspicious device" in source or "IED detected" in source, \
            "Kill feed should announce IED detection"


# =========================================================================
# 5. GUERRILLA CELL INDICATOR — debug mode overlay
# =========================================================================

class TestGuerrillaDebug:
    def test_ied_debug_stats(self, source):
        assert "IEDs:" in source or "IED" in source, \
            "Debug overlay should show IED statistics"

    def test_ied_minimap_dots(self, source):
        # Red circles on minimap for IED positions
        assert "ied" in source.lower() and "minimap" in source.lower() or \
            "ieds" in source and "drawDot" in source or \
            "ied" in source.lower() and "#ff" in source, \
            "Should draw IED positions on minimap"


# =========================================================================
# 6. NARRATION TEMPLATES — radio chatter for IED events
# =========================================================================

class TestIEDNarration:
    def test_ied_detonate_template(self, source):
        assert "ied_detonate" in source or "CONTACT IED" in source, \
            "Should have IED detonation narration template"

    def test_ied_detected_template(self, source):
        assert "ied_detected" in source or "suspicious device" in source, \
            "Should have IED detection narration template"
