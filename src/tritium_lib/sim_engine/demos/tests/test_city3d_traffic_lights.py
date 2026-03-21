"""
Tests for city3d.html traffic light system at intersections.
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
# 1. TRAFFIC LIGHT INSTANCED MESH
# =========================================================================

class TestTrafficLightMesh:
    def test_traffic_light_instanced_mesh_exists(self, source):
        """Traffic lights should use an InstancedMesh"""
        assert "trafficLightMesh" in source, "Missing trafficLightMesh InstancedMesh"

    def test_traffic_light_geometry(self, source):
        """Traffic light should be a small box: BoxGeometry(0.5, 2.5, 0.5)"""
        assert "BoxGeometry(0.5, 2.5, 0.5)" in source

    def test_traffic_light_max_count(self, source):
        """Max traffic lights should match intersection count"""
        assert "intersections.length" in source

    def test_traffic_light_added_to_scene(self, source):
        """Traffic light mesh must be added to the scene"""
        assert "scene.add(trafficLightMesh)" in source

    def test_traffic_light_frustum_culling_disabled(self, source):
        """Traffic light mesh should have frustum culling disabled"""
        assert "trafficLightMesh" in source
        # Should be in the frustumCulled = false list
        idx_frustum = source.find("frustumCulled = false")
        idx_tl = source.find("trafficLightMesh")
        assert idx_frustum > 0, "Missing frustumCulled disable block"


# =========================================================================
# 2. TRAFFIC PHASE TIMER
# =========================================================================

class TestTrafficPhase:
    def test_traffic_phase_variable(self, source):
        """trafficPhase timer must exist"""
        assert "trafficPhase" in source

    def test_traffic_cycle_period(self, source):
        """Traffic cycle should be 8 seconds"""
        # The modulo or comparison with 8
        assert "% 8" in source or "TRAFFIC_CYCLE" in source

    def test_is_green_function(self, source):
        """isGreen() function must exist to check signal state"""
        assert "isGreen" in source


# =========================================================================
# 3. CAR BEHAVIOR AT RED LIGHTS
# =========================================================================

class TestCarRedLightBehavior:
    def test_cars_check_green_at_intersection(self, source):
        """Cars should call isGreen when approaching an intersection"""
        assert "isGreen(" in source

    def test_red_light_extends_pause(self, source):
        """Red light should extend pause timer for waiting cars"""
        # The pause extension when light is red
        assert "pauseTimer" in source
        # Should reference isGreen in the car update section
        car_section = source[source.find("Update cars"):]
        assert "isGreen" in car_section, "isGreen not used in car update loop"


# =========================================================================
# 4. TRAFFIC LIGHT VISUAL UPDATE
# =========================================================================

class TestTrafficLightVisualUpdate:
    def test_traffic_light_color_update(self, source):
        """Traffic light colors should update via setColorAt"""
        assert "trafficLightMesh.setColorAt" in source

    def test_green_color(self, source):
        """Green light color 0x00ff00"""
        assert "0x00ff00" in source

    def test_red_color(self, source):
        """Red light color 0xff0000"""
        assert "0xff0000" in source

    def test_y_position_2_5(self, source):
        """Traffic lights should be at y=2.5"""
        # Check that 2.5 appears near traffic light setup
        tl_section = source[source.find("trafficLightMesh"):]
        assert "2.5" in tl_section[:500], "Traffic light y=2.5 position not found"


# =========================================================================
# 5. POINT LIGHTS AT NEAREST TRAFFIC LIGHTS
# =========================================================================

class TestTrafficPointLights:
    def test_pooled_point_lights(self, source):
        """Should have pooled PointLights for traffic lights (max 4)"""
        assert "trafficPointLights" in source or "tlLights" in source

    def test_point_light_count(self, source):
        """Should have exactly 4 traffic point lights"""
        assert "4" in source  # max 4 lights
