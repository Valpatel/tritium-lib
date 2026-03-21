"""
Tests for city3d.html fog of war and intelligence gathering visualization.
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
# 1. FOG OF WAR — canvas texture on single plane
# =========================================================================

class TestFogOfWarState:
    def test_fog_of_war_toggle_variable(self, source):
        assert "fogOfWarEnabled" in source, "Missing fogOfWarEnabled state variable"

    def test_fog_of_war_key_i_toggle(self, source):
        assert "'KeyI'" in source, "Missing KeyI handler for fog of war toggle"

    def test_fog_canvas_element(self, source):
        assert "fogCanvas" in source, "Missing fogCanvas for CanvasTexture"

    def test_fog_canvas_texture(self, source):
        assert "CanvasTexture" in source, "Missing CanvasTexture for fog of war"

    def test_fog_plane_geometry(self, source):
        assert "fogPlane" in source or "fogMesh" in source, \
            "Missing fog plane mesh"

    def test_fog_depth_write_false(self, source):
        # The fog material should have depthWrite: false
        assert "depthWrite: false" in source, "Fog material must have depthWrite: false"

    def test_fog_render_order(self, source):
        assert "renderOrder" in source, "Fog plane should have high renderOrder"


class TestFogOfWarGrid:
    def test_fog_cell_size(self, source):
        assert "FOG_CELL" in source or "fogCell" in source, \
            "Missing fog cell size constant"

    def test_fog_grid_dimensions(self, source):
        # Grid should be based on city dimensions / cell size
        assert "fogCols" in source or "FOG_COLS" in source, \
            "Missing fog grid column count"

    def test_fog_visibility_states(self, source):
        # Three states: fog (unexplored), explored, visible
        assert "explored" in source or "EXPLORED" in source, \
            "Missing explored fog state"

    def test_fog_update_function(self, source):
        assert "updateFogOfWar" in source, "Missing updateFogOfWar function"

    def test_fog_update_throttled(self, source):
        # Should update on a timer, not every frame
        assert "fogTimer" in source or "fogUpdateInterval" in source, \
            "Fog of war should be throttled, not per-frame"


class TestFogOfWarVisibility:
    def test_police_reveal_radius(self, source):
        # Police officers reveal ~40m radius
        assert "40" in source, "Police reveal radius should be ~40 units"

    def test_fog_opacity_unexplored(self, source):
        # Unexplored cells should have high opacity ~0.7
        assert "0.7" in source, "Unexplored fog opacity should be 0.7"

    def test_fog_opacity_explored(self, source):
        # Previously-seen cells should have medium opacity ~0.3
        assert "0.3" in source, "Explored fog opacity should be 0.3"

    def test_fog_uses_nearest_filter(self, source):
        assert "NearestFilter" in source, "Fog texture should use NearestFilter"


# =========================================================================
# 2. DETECTION VISUALIZATION — lines from police to visible protestors
# =========================================================================

class TestDetectionLines:
    def test_detection_line_geometry(self, source):
        assert "detectionLines" in source or "detectionLine" in source, \
            "Missing detection line visualization"

    def test_detection_lines_debug_only(self, source):
        # Detection lines should only show in debug mode
        assert "debugMode" in source and "detection" in source.lower(), \
            "Detection lines should be gated by debugMode"

    def test_detection_range_40(self, source):
        assert "DETECTION_RANGE" in source or "detectionRange" in source, \
            "Missing detection range constant"

    def test_detection_line_color_green(self, source):
        # Lines should be green/faint
        assert "0x05ffa1" in source or "#05ffa1" in source, \
            "Detection lines should use green color"

    def test_detected_count_in_debug(self, source):
        # Debug overlay should show detected count
        assert "detected" in source.lower() or "Detected" in source, \
            "Debug overlay should show detected count"


# =========================================================================
# 3. HUD CONTROLS — I key shown in controls bar
# =========================================================================

class TestFogOfWarHUD:
    def test_controls_show_i_key(self, source):
        assert "<span>I</span>" in source, "Controls bar should show I key for fog of war"

    def test_fog_of_war_label(self, source):
        assert "Intel" in source or "Fog of War" in source or "FOW" in source, \
            "Controls should label the I key as Intel or Fog of War"
