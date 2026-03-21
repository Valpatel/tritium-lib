"""
Tests for city3d.html crosswalk markings and pedestrian crossing behavior.
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
# 1. CROSSWALK INSTANCED MESH
# =========================================================================

class TestCrosswalkMesh:
    def test_crosswalk_instanced_mesh_exists(self, source):
        """Crosswalks should use a single InstancedMesh for all strips"""
        assert "cwMesh" in source, "Missing cwMesh InstancedMesh"
        assert "InstancedMesh(cwGeo, cwMat" in source

    def test_crosswalk_geometry_is_plane(self, source):
        """Crosswalk strips should be PlaneGeometry"""
        assert "PlaneGeometry(0.8, 4)" in source, "Crosswalk strips should be 0.8 wide x 4 long"

    def test_crosswalk_material_is_white(self, source):
        """Crosswalk material should be white MeshStandardMaterial"""
        assert "color: 0xffffff" in source
        assert "cwMat" in source

    def test_crosswalk_strips_per_crosswalk(self, source):
        """Each crosswalk should have 6 strips"""
        assert "STRIPS_PER_CW = 6" in source

    def test_four_crosswalks_per_intersection(self, source):
        """4 crosswalks per intersection (one per road approach)"""
        assert "CW_PER_INT = 4" in source

    def test_crosswalk_y_position(self, source):
        """Crosswalks should be at y=0.02 just above road surface"""
        assert "0.02" in source

    def test_crosswalk_added_to_scene(self, source):
        """Crosswalk mesh must be added to the scene"""
        assert "scene.add(cwMesh)" in source

    def test_crosswalk_instance_matrix_updated(self, source):
        """InstancedMesh matrix must be flagged for update"""
        assert "cwMesh.instanceMatrix.needsUpdate = true" in source

    def test_crosswalk_pre_allocated(self, source):
        """Crosswalk InstancedMesh should be pre-allocated for all intersections"""
        assert "intersections.length * CW_PER_INT * STRIPS_PER_CW" in source


# =========================================================================
# 2. PEDESTRIAN CROSSWALK ROUTING
# =========================================================================

class TestPedestrianCrosswalkRouting:
    def test_plan_ped_path_function_exists(self, source):
        """planPedPath function should exist for crosswalk waypoint routing"""
        assert "function planPedPath(from, to)" in source

    def test_pedestrians_have_waypoints(self, source):
        """Pedestrians should have a waypoints array for multi-step paths"""
        assert "waypoints:" in source

    def test_pedestrians_use_plan_ped_path(self, source):
        """Pedestrian init should use planPedPath for routing"""
        assert "planPedPath(pos, dest)" in source or "planPedPath(pos," in source

    def test_pedestrians_shift_waypoints(self, source):
        """When reaching a waypoint, pedestrians should shift to the next"""
        assert "ped.waypoints.shift()" in source

    def test_new_target_uses_crosswalk_path(self, source):
        """When picking a new target, pedestrians should plan via crosswalks"""
        assert "planPedPath({ x: ped.x, z: ped.z }" in source

    def test_plan_ped_path_uses_intersections(self, source):
        """planPedPath should route via intersections (crosswalk locations)"""
        assert "for (const i of intersections)" in source

    def test_taxi_dropoff_uses_crosswalk_path(self, source):
        """Taxi drop-off respawned pedestrians should also use crosswalk routing"""
        assert "planPedPath(pos, _td)" in source
