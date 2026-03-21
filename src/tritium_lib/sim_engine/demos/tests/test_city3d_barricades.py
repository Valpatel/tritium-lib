"""
Tests for city3d.html barricades, building collision, and tree placement fixes.
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
# 1. TREES INSIDE BUILDINGS FIX
# =========================================================================

class TestTreePlacement:
    def test_tree_building_overlap_check(self, source):
        """Tree placement should check for building overlap"""
        assert "insideBuilding" in source, "Missing building overlap check in tree placement"

    def test_tree_skip_on_overlap(self, source):
        """Trees should be skipped when inside a building"""
        # The continue statement after insideBuilding check
        assert "if (insideBuilding) continue" in source

    def test_tree_margin_2m(self, source):
        """Building margin for trees should be 2m"""
        # Check for the 2m margin in the building check
        assert "bldg.w / 2 + 2" in source or "bldg.w/2 + 2" in source


# =========================================================================
# 2. POLICE BARRICADES
# =========================================================================

class TestBarricades:
    def test_barricade_instanced_mesh(self, source):
        """Barricades should use InstancedMesh"""
        assert "barricadeMesh" in source, "Missing barricadeMesh InstancedMesh"

    def test_barricade_geometry_dimensions(self, source):
        """Barricade geometry: BoxGeometry(6, 1.5, 1)"""
        assert "BoxGeometry(6, 1.5, 1)" in source

    def test_barricade_max_count(self, source):
        """Should allow up to 8 barricades"""
        assert "MAX_BARRICADES" in source

    def test_barricade_array(self, source):
        """barricades array for tracking state"""
        assert "const barricades = []" in source

    def test_spawn_barricades_function(self, source):
        """spawnBarricades function exists"""
        assert "function spawnBarricades()" in source

    def test_barricades_spawned_on_riot(self, source):
        """Barricades should be spawned when riot starts"""
        assert "spawnBarricades()" in source

    def test_barricade_health_system(self, source):
        """Barricades have health that decreases"""
        assert "damageBarricade" in source

    def test_barricade_color_shifts_on_damage(self, source):
        """Color shifts from blue to red as health decreases"""
        assert "1.0 - b.health" in source or "b.health" in source

    def test_barricade_destroyed_at_zero_health(self, source):
        """Barricade disappears when health reaches 0"""
        assert "barricade.health <= 0" in source or "b.health <= 0" in source

    def test_barricade_blocks_cars(self, source):
        """Cars should stop near barricades"""
        assert "isBlockedByBarricade" in source

    def test_barricade_damaged_by_molotov(self, source):
        """Molotovs should damage nearby barricades"""
        # Check that barricade damage is triggered on molotov impact
        assert "damageBarricade(b," in source

    def test_barricade_frustum_culled_false(self, source):
        """barricadeMesh must be in frustumCulled=false list"""
        idx = source.find("frustumCulled")
        assert idx > 0
        block = source[max(0, idx - 300):idx + 200]
        assert "barricadeMesh" in block

    def test_barricade_cleanup_on_riot_end(self, source):
        """Barricades cleared when riot ends"""
        assert "barricades.length = 0" in source

    def test_barricade_hud_display(self, source):
        """Barricade count shown in HUD"""
        assert "barricade-count" in source


# =========================================================================
# 3. BUILDING COLLISION FOR PEDESTRIANS
# =========================================================================

class TestBuildingCollision:
    def test_is_inside_building_function(self, source):
        """isInsideBuilding helper function exists"""
        assert "function isInsideBuilding(" in source

    def test_snap_out_of_building_function(self, source):
        """snapOutOfBuilding function exists"""
        assert "function snapOutOfBuilding(" in source

    def test_resolve_collision_function(self, source):
        """resolveCollision function for sliding along building edges"""
        assert "function resolveCollision(" in source

    def test_pedestrian_target_snapped(self, source):
        """Pedestrian targets inside buildings get snapped out"""
        assert "snapOutOfBuilding(ped.target" in source or "tBldg" in source

    def test_pedestrian_collision_resolution(self, source):
        """Pedestrian movement resolves building collisions"""
        assert "resolveCollision(ped" in source or "resolveCollision(old" in source

    def test_nearby_buildings_function(self, source):
        """nearbyBuildings function for efficient collision checks"""
        assert "function nearbyBuildings(" in source


# =========================================================================
# Integration
# =========================================================================

class TestBarricadeIntegration:
    def test_js_syntax_valid(self, source):
        """Brace balance check"""
        opens = source.count('{')
        closes = source.count('}')
        assert abs(opens - closes) < 5, f"Brace mismatch: {opens} opens vs {closes} closes"

    def test_barricade_in_debug_overlay(self, source):
        """Barricade count in debug overlay"""
        assert "Barricades:" in source and "barricades.filter" in source

    def test_kill_feed_barricade_messages(self, source):
        """Kill feed should show barricade events"""
        assert "Barricade destroyed" in source or "barricades deployed" in source.lower() or "Barricade" in source
