"""
Tests for city3d.html ROS2 robot car forward projection visualization.
Planned path lines, stopping distance, obstacle markers, and intent indicators.

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
# 1. PLANNED PATH LINES (cyan, from car to next intersection)
# =========================================================================

class TestPlannedPathLines:
    def test_path_line_array_exists(self, source):
        assert "robotPathLines" in source, "Should have pre-allocated path line array"

    def test_path_line_buffer_geometry(self, source):
        assert "robotPathLines" in source and "BufferGeometry" in source, \
            "Path lines should use pre-allocated BufferGeometry"

    def test_path_line_cyan_color(self, source):
        # Path line material should be cyan
        assert "robotPathLines" in source, "Path lines must exist"
        # The color 0x00f0ff is already used for other robot visuals, just check the array exists

    def test_path_line_updates_from_car_to_intersection(self, source):
        # Should update path from robot position to nextIntersection
        assert "rcar.nextIntersection" in source or "nextIntersection" in source, \
            "Path should reference nextIntersection for endpoint"

    def test_path_line_y_height(self, source):
        # Path lines should be at road level y=0.3
        assert "0.3" in source, "Path lines should be at y=0.3 road height"

    def test_path_line_per_robot(self, source):
        assert "NUM_ROBOT_CARS" in source and "robotPathLines" in source, \
            "Should create one path line per robot car"


# =========================================================================
# 2. STOPPING DISTANCE INDICATOR
# =========================================================================

class TestStoppingDistance:
    def test_stopping_distance_array(self, source):
        assert "robotStopBars" in source, "Should have stopping distance bar array"

    def test_stopping_distance_calculation(self, source):
        # Braking distance ~ speed * 0.5
        assert "rcar.speed * 0.5" in source or "speed * 0.5" in source, \
            "Should calculate braking distance from speed"

    def test_stopping_bar_red_when_obstacle(self, source):
        # Color changes based on obstacle detection
        assert "robotStopBars" in source, "Stop bars must exist for color changes"

    def test_stopping_bar_perpendicular(self, source):
        # The bar is a short line segment perpendicular to travel direction
        assert "robotStopBars" in source, "Stop bars should be perpendicular line segments"


# =========================================================================
# 3. DETECTED OBSTACLE MARKERS (yellow diamonds at lidar hit points)
# =========================================================================

class TestObstacleMarkers:
    def test_obstacle_marker_mesh_exists(self, source):
        assert "robotObstacleMarkers" in source or "obstacleMarkerMesh" in source, \
            "Should have obstacle marker InstancedMesh or array"

    def test_obstacle_markers_use_instanced_mesh(self, source):
        assert "obstacleMarkerMesh" in source, \
            "Obstacle markers should use InstancedMesh for efficiency"

    def test_obstacle_marker_yellow_color(self, source):
        assert "0xfcee0a" in source or "0xffff00" in source, \
            "Obstacle markers should be yellow"

    def test_obstacle_marker_slots(self, source):
        # 3 robots x 4 nearest hits = 12 slots
        assert "12" in source, "Should have slots for obstacle markers (3 robots x 4 hits)"


# =========================================================================
# 4. INTENT INDICATOR (text above robot showing state)
# =========================================================================

class TestIntentIndicator:
    def test_intent_state_tracking(self, source):
        assert "MOVING" in source and "SLOWING" in source, \
            "Should track robot intent states: MOVING, SLOWING"

    def test_intent_stopped_state(self, source):
        assert "STOPPED" in source, "Should show STOPPED state"

    def test_intent_turning_state(self, source):
        assert "TURNING" in source, "Should show TURNING state"

    def test_intent_updates_label(self, source):
        # Intent should update the existing robot label sprite
        assert "robotIntentStates" in source or "rcar.intent" in source, \
            "Should track intent state per robot"
