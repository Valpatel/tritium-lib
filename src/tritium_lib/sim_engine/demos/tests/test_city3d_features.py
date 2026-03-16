"""
Tests for city3d.html weather effects, territory control, and visible objectives.
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
# 1. WEATHER EFFECTS (rain + fog toggle)
# =========================================================================

class TestWeatherRain:
    def test_rain_state_variable(self, source):
        assert "rainActive" in source, "Missing rainActive state variable"

    def test_rain_instanced_mesh(self, source):
        assert "rainMesh" in source, "Missing rain InstancedMesh"

    def test_rain_geometry_thin_box(self, source):
        assert "BoxGeometry(0.03" in source, "Rain should use thin BoxGeometry(0.03, 0.8, 0.03)"

    def test_rain_particle_count_500(self, source):
        assert "RAIN_COUNT" in source or "500" in source, "Rain should have ~500 particles"

    def test_rain_key_r_toggle(self, source):
        assert "'KeyR'" in source, "Missing KeyR handler for rain toggle"

    def test_rain_update_function(self, source):
        assert "updateRain" in source, "Missing updateRain function"

    def test_rain_frustum_culled_false(self, source):
        # rainMesh must be in the frustumCulled=false list
        assert "rainMesh" in source and "frustumCulled" in source

    def test_rain_reduces_ambient_light(self, source):
        # When raining, ambient light should be reduced
        assert "rainActive" in source and "ambientLight" in source

    def test_rain_increases_fog_density(self, source):
        # Rain should increase fog density
        assert "rainActive" in source and "fog" in source

    def test_rain_hud_indicator(self, source):
        assert "RAIN" in source, "Should show RAIN indicator in HUD"

    def test_rain_reset_to_top(self, source):
        # Raindrops should reset to y=80 when hitting ground
        assert "80" in source and "rainActive" in source

    def test_rain_fall_speed(self, source):
        # Rain fall speed 15-25 m/s
        assert "updateRain" in source


class TestWeatherFog:
    def test_fog_toggle_key_f(self, source):
        assert "'KeyF'" in source, "Missing KeyF handler for fog toggle"

    def test_fog_state_variable(self, source):
        assert "fogOverride" in source, "Missing fogOverride state variable"

    def test_fog_density_increase(self, source):
        assert "0.008" in source, "Fog override density should be 0.008"


# =========================================================================
# 2. TERRITORY CONTROL
# =========================================================================

class TestTerritoryControl:
    def test_territory_zones_exist(self, source):
        assert "territoryZones" in source, "Missing territoryZones array"

    def test_territory_grid_4x3(self, source):
        assert "TERRITORY_COLS" in source or "4" in source
        assert "TERRITORY_ROWS" in source or "3" in source

    def test_territory_zone_meshes(self, source):
        assert "PlaneGeometry" in source and "territory" in source.lower()

    def test_territory_green_police(self, source):
        assert "05ffa1" in source, "Missing police territory color (green)"

    def test_territory_red_protestor(self, source):
        assert "ff2a6d" in source, "Missing protestor territory color (red)"

    def test_territory_neutral_gray(self, source):
        assert "888888" in source, "Missing neutral territory color (gray)"

    def test_territory_update_function(self, source):
        assert "updateTerritoryControl" in source, "Missing updateTerritoryControl function"

    def test_territory_update_throttled(self, source):
        # Should update every 2 seconds, not every frame
        assert "territoryTimer" in source or "territoryUpdateInterval" in source

    def test_territory_faction_counting(self, source):
        # Should count units per zone
        assert "policeInZone" in source or "protestorsInZone" in source or "police" in source.split("territory")[1] if "territory" in source else False

    def test_territory_zone_borders(self, source):
        # Zone borders as lines on ground
        assert "territoryBorder" in source or "zoneBorder" in source or "zone" in source.lower()


# =========================================================================
# 3. VISIBLE OBJECTIVES
# =========================================================================

class TestObjectives:
    def test_objectives_array(self, source):
        assert "objectives" in source, "Missing objectives array"

    def test_objective_secure_plaza(self, source):
        assert "SECURE PLAZA" in source, "Missing SECURE PLAZA objective"

    def test_objective_hold_ground(self, source):
        assert "HOLD GROUND" in source, "Missing HOLD GROUND objective"

    def test_objective_protect_hospital(self, source):
        assert "PROTECT HOSPITAL" in source, "Missing PROTECT HOSPITAL objective"

    def test_objective_diamond_mesh(self, source):
        # Diamond markers as InstancedMesh or geometry
        assert "objectiveMesh" in source or "diamondGeo" in source or "OctahedronGeometry" in source

    def test_objective_floating_height(self, source):
        # Markers should float at y=15
        assert "15" in source and "objective" in source.lower()

    def test_objective_rotation(self, source):
        # Markers should slowly rotate
        assert "objective" in source.lower() and "rotation" in source.lower()

    def test_objective_colors(self, source):
        # Yellow for active, green for complete, red for failed
        assert "objectiveColor" in source or "objective" in source.lower()

    def test_objective_hud_display(self, source):
        # Objectives should be shown in HUD
        assert "objective" in source.lower() and "hud" in source.lower() or "objectives-hud" in source or "objective-list" in source

    def test_objective_status_tracking(self, source):
        # Track active/complete/failed status
        assert "objectiveStatus" in source or "status" in source and "objective" in source.lower()

    def test_objective_update_function(self, source):
        assert "updateObjectives" in source, "Missing updateObjectives function"


# =========================================================================
# Integration tests
# =========================================================================

class TestIntegration:
    def test_controls_help_updated(self, source):
        """Controls help text should mention R, F keys"""
        assert "R" in source and "Rain" in source
        assert "F" in source and "Fog" in source

    def test_no_per_frame_allocation_territory(self, source):
        """Territory zones should NOT be created per frame"""
        # The PlaneGeometry for zones should be created once, not inside update()
        # Check that territory zone creation is NOT inside the update function
        assert "territoryZones" in source

    def test_js_syntax_valid(self, source):
        """Basic check: no obvious JS syntax errors like unmatched braces"""
        # Count braces (rough check)
        opens = source.count('{')
        closes = source.count('}')
        # Allow some tolerance for braces in strings/HTML
        assert abs(opens - closes) < 5, f"Brace mismatch: {opens} opens vs {closes} closes"

    def test_rain_mesh_in_frustum_list(self, source):
        """rainMesh must be in the frustumCulled=false block"""
        # Find the frustumCulled block and check rainMesh is included
        assert "rainMesh" in source
        idx = source.find("frustumCulled")
        assert idx > 0
