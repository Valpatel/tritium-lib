"""
Tests for city3d.html building damage and collapse from fire proximity.
Source-string tests that verify the HTML file contains required code patterns.

Demonstrates destruction.py — when fires burn near buildings for 15+ seconds,
buildings progress through damage states: intact -> damaged -> heavily damaged -> collapsed.

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
# 1. BUILDING DAMAGE PROPERTY
# =========================================================================

class TestBuildingDamageProperty:
    def test_damage_field_on_building(self, source):
        """Each building should get a damage property initialized to 0"""
        assert "damage: 0" in source or "damage:0" in source

    def test_damage_in_bldg_data(self, source):
        """bldgData should include damage field"""
        # The building push line should have damage
        assert "damage:" in source


# =========================================================================
# 2. FIRE DAMAGES BUILDINGS
# =========================================================================

class TestFireDamagesBuildings:
    def test_fire_building_proximity_check(self, source):
        """Fire update loop should check building proximity"""
        # Should iterate buildings near fires
        assert "building" in source and "damage" in source and "fire" in source.lower()

    def test_damage_increment(self, source):
        """Buildings near fire should accumulate damage"""
        assert ".damage +=" in source or ".damage+=" in source

    def test_fire_damage_radius(self, source):
        """Fire should damage buildings within 8 units"""
        # Check for the distance threshold
        assert "< 8" in source or "<8" in source or "< 10" in source

    def test_damage_capped_at_one(self, source):
        """Damage should not exceed 1.0"""
        assert "Math.min" in source and "damage" in source


# =========================================================================
# 3. VISUAL DAMAGE STATES
# =========================================================================

class TestVisualDamageStates:
    def test_damaged_windows_dark(self, source):
        """At damage > 0.3, windows should go dark"""
        assert "winStart" in source and "winEnd" in source
        # Windows get darkened for damaged buildings
        assert "0x111111" in source or "0x222222" in source or "0x333333" in source

    def test_body_color_darkens(self, source):
        """Damaged buildings should have darkened body color"""
        assert "setColorAt" in source and "buildingBodyMesh" in source

    def test_height_reduction_heavy_damage(self, source):
        """At heavy damage (0.6+), building height should reduce"""
        assert "setMatrixAt" in source and "buildingBodyMesh" in source

    def test_roof_hidden_on_collapse(self, source):
        """At collapse (0.9+), roof should be hidden"""
        assert "buildingRoofMesh" in source and "setMatrixAt" in source


# =========================================================================
# 4. RUBBLE AND DEBRIS
# =========================================================================

class TestRubbleDebris:
    def test_debris_particles_on_collapse(self, source):
        """Collapsed buildings should spawn debris particles"""
        assert "spawnParticle" in source and "rubble" in source.lower()

    def test_rubble_array(self, source):
        """Should track rubble positions for pathfinding avoidance"""
        assert "rubbleZones" in source or "rubble" in source.lower()


# =========================================================================
# 5. RESET CLEANS DAMAGE
# =========================================================================

class TestResetCleansDamage:
    def test_damage_reset_on_scenario_reset(self, source):
        """Building damage should reset when scenario resets"""
        assert "damage = 0" in source or "damage=0" in source
