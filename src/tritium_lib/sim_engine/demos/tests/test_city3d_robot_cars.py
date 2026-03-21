"""
Tests for city3d.html ROS2 robot cars with lidar visualization.
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
# 1. ROBOT CAR STATE AND CONSTANTS
# =========================================================================

class TestRobotCarState:
    def test_num_robot_cars_constant(self, source):
        assert "NUM_ROBOT_CARS = 3" in source, "Should have 3 robot cars"

    def test_lidar_rays_constant(self, source):
        assert "LIDAR_RAYS = 12" in source, "Should have 12 lidar rays"

    def test_lidar_range_constant(self, source):
        assert "LIDAR_RANGE = 15" in source, "Lidar range should be 15m"

    def test_robot_trail_length(self, source):
        assert "ROBOT_TRAIL_LENGTH = 20" in source, "Trail should track 20 positions"

    def test_robot_cars_array(self, source):
        assert "robotCars = []" in source, "Should have robotCars array"

    def test_robot_lidar_angle(self, source):
        assert "robotLidarAngle" in source, "Should track lidar rotation angle"

    def test_robot_total_hits(self, source):
        assert "robotTotalHits" in source, "Should track total lidar hits"


# =========================================================================
# 2. ROBOT CAR CREATION — first 3 cars from pool
# =========================================================================

class TestRobotCarCreation:
    def test_is_robot_flag(self, source):
        assert "isRobot" in source, "Cars should have isRobot flag"

    def test_robot_cyan_body_color(self, source):
        assert "0x00f0ff" in source, "Robot body color should be cyan 0x00f0ff"

    def test_robot_dark_cabin_color(self, source):
        assert "0x008899" in source, "Robot cabin should be dark cyan 0x008899"

    def test_robot_speed_constant(self, source):
        # Robot speed is fixed at 6, not random
        assert "isRobot ? 6" in source, "Robot speed should be fixed at 6"

    def test_robot_pushed_to_array(self, source):
        assert "robotCars.push" in source, "Robot cars should be pushed to robotCars array"


# =========================================================================
# 3. LIDAR VISUALIZATION
# =========================================================================

class TestLidarVisualization:
    def test_lidar_cylinder_geometry(self, source):
        assert "CylinderGeometry(0.3" in source, "Lidar should have spinning cylinder on roof"

    def test_lidar_cylinder_emissive(self, source):
        assert "emissiveIntensity: 0.4" in source, "Lidar cylinder should glow"

    def test_lidar_line_segments(self, source):
        assert "LineSegments" in source, "Lidar rays should use LineSegments"

    def test_lidar_vertex_colors(self, source):
        assert "vertexColors: true" in source, "Lidar rays should use vertex colors"

    def test_lidar_buffer_geometry(self, source):
        assert "LIDAR_RAYS * 2 * 3" in source, "Should pre-allocate 12 rays x 2 verts x 3 floats"

    def test_lidar_ray_building_test(self, source):
        # Lidar should test against buildings (AABB intersection)
        assert "bMinX" in source or "bMaxX" in source, "Lidar should test ray-AABB vs buildings"

    def test_lidar_ray_car_test(self, source):
        # Lidar should test against other cars
        assert "perpSq" in source, "Lidar should test sphere intersection vs cars"

    def test_lidar_ray_pedestrian_test(self, source):
        # Should also detect pedestrians
        assert "ped.x - rx" in source, "Lidar should detect pedestrians"

    def test_lidar_hit_turns_red(self, source):
        # When hit, end vertex should be red
        assert "colArr[vi + 3] = 1.0" in source, "Hit endpoint should turn red"

    def test_lidar_frustum_culled_false(self, source):
        assert "frustumCulled = false" in source, "Lidar meshes should have frustumCulled=false"

    def test_lidar_rotation_updates(self, source):
        assert "robotLidarAngle += dt" in source, "Lidar angle should update each tick"

    def test_lidar_positions_needsupdate(self, source):
        assert "position.needsUpdate = true" in source, "Lidar positions must flag needsUpdate"


# =========================================================================
# 4. ROBOT TRAIL
# =========================================================================

class TestRobotTrail:
    def test_trail_segments_constant(self, source):
        assert "TRAIL_SEGMENTS" in source, "Should define trail segment count"

    def test_trail_history_array(self, source):
        assert "trail.history" in source, "Trail should maintain position history"

    def test_trail_line_material(self, source):
        # Trail should be cyan with low opacity
        assert "opacity: 0.3" in source, "Trail should have low opacity"

    def test_trail_records_positions(self, source):
        assert "trail.timer" in source, "Trail should sample positions on a timer"


# =========================================================================
# 5. WARNING LIGHT
# =========================================================================

class TestWarningLight:
    def test_warning_light_amber(self, source):
        assert "0xffaa00" in source, "Warning light should be amber"

    def test_warning_light_near_obstacle(self, source):
        assert "nearObstacle" in source, "Should detect nearby pedestrians"

    def test_warning_light_blink(self, source):
        assert "blinkTimer" in source, "Warning light should blink"

    def test_pedestrian_detect_radius(self, source):
        assert "DETECT_RADIUS" in source, "Should define detection radius for pedestrians"


# =========================================================================
# 6. ROBOT-SPECIFIC BEHAVIOR
# =========================================================================

class TestRobotBehavior:
    def test_robot_never_pause_long(self, source):
        # Robots override long pauses
        assert "pauseTimer > 0.5" in source, "Robot should limit pause time"

    def test_robot_slow_near_pedestrians(self, source):
        assert "rcar.speed = 2.5" in source, "Robot should slow to 2.5 near pedestrians"

    def test_robot_normal_speed(self, source):
        assert "rcar.speed = 6" in source, "Robot should return to speed 6 when clear"


# =========================================================================
# 7. ROS2 LABEL SPRITE
# =========================================================================

class TestRobotLabel:
    def test_label_sprite_created(self, source):
        assert "makeRobotLabelSprite" in source, "Should create label sprite function"

    def test_label_text_ros2(self, source):
        assert "'ROS2-'" in source, "Label should say ROS2-N"

    def test_label_canvas_texture(self, source):
        assert "CanvasTexture" in source, "Label should use canvas texture"

    def test_label_position_above_car(self, source):
        assert "4.5" in source, "Label should float at y=4.5 above car"


# =========================================================================
# 8. HUD AND DEBUG OVERLAY
# =========================================================================

class TestRobotHUD:
    def test_hud_robot_count_element(self, source):
        assert 'id="robot-count"' in source, "HUD should have robot-count element"

    def test_hud_ros2_robots_label(self, source):
        assert "ROS2 Robots:" in source, "HUD should show 'ROS2 Robots:' label"

    def test_hud_updates_robot_count(self, source):
        assert "robot-count" in source and "robotCars.filter" in source, \
            "HUD should update robot count dynamically"

    def test_debug_overlay_robot_section(self, source):
        assert "ROS2 ROBOTS" in source, "Debug overlay should have ROS2 ROBOTS section"

    def test_debug_shows_lidar_hits(self, source):
        assert "Lidar hits" in source, "Debug should show lidar hit count"

    def test_debug_shows_robot_positions(self, source):
        assert "rc.x.toFixed" in source, "Debug should show robot positions"
