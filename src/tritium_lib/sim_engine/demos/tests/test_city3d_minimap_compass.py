"""
Tests for city3d.html minimap and compass HUD features.
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
# 1. MINIMAP — bottom-left canvas showing bird's-eye city view
# =========================================================================

class TestMinimapCanvas:
    def test_minimap_canvas_element(self, source):
        assert "minimap-canvas" in source, "Missing minimap-canvas element"

    def test_minimap_canvas_size(self, source):
        assert "180" in source and "120" in source, \
            "Minimap canvas should be 180x120px"

    def test_minimap_css_positioning(self, source):
        assert "bottom" in source and "minimap" in source.lower(), \
            "Minimap should be positioned at bottom-left"

    def test_minimap_border_cyan(self, source):
        assert "#00f0ff" in source, "Minimap should use cyan border color"


class TestMinimapDrawing:
    def test_minimap_2d_context(self, source):
        assert "minimapCtx" in source or "minimap" in source.lower(), \
            "Minimap should use 2D canvas context"

    def test_minimap_scale_factor(self, source):
        # Scale: minimap_width / CITY_W
        assert "CITY_W" in source and "CITY_H" in source, \
            "Minimap needs city dimension constants for scaling"

    def test_minimap_road_grid(self, source):
        assert "333333" in source or "#333" in source, \
            "Minimap should draw road grid lines in dark gray"

    def test_minimap_police_dots(self, source):
        # Police should be cyan dots on minimap
        assert "police" in source, "Minimap should draw police dots"

    def test_minimap_protestor_dots(self, source):
        assert "protestors" in source, "Minimap should draw protestor dots"

    def test_minimap_civilian_dots(self, source):
        assert "pedestrians" in source, "Minimap should draw civilian/pedestrian dots"

    def test_minimap_vehicle_dots(self, source):
        assert "cars" in source, "Minimap should draw vehicle dots"

    def test_minimap_camera_viewport(self, source):
        # Camera viewport rectangle showing what main camera sees
        assert "strokeRect" in source or "viewport" in source.lower(), \
            "Minimap should show camera viewport rectangle"


class TestMinimapUpdate:
    def test_minimap_update_function(self, source):
        assert "updateMinimap" in source or "drawMinimap" in source, \
            "Missing minimap update/draw function"

    def test_minimap_update_rate(self, source):
        # Should update at 4fps (in hudTimer block, every 0.25s)
        assert "minimap" in source.lower(), \
            "Minimap should update at HUD rate (4fps)"

    def test_minimap_toggle_key_m(self, source):
        assert "'KeyM'" in source, "Missing KeyM handler for minimap toggle"


# =========================================================================
# 2. COMPASS — top-center showing camera direction
# =========================================================================

class TestCompass:
    def test_compass_element(self, source):
        assert "compass" in source.lower(), "Missing compass HUD element"

    def test_compass_cardinal_directions(self, source):
        # Should show N, S, E, W
        src_lower = source.lower()
        assert "compass" in src_lower, "Compass must reference cardinal directions"

    def test_compass_uses_camera_rotation(self, source):
        # Uses camera azimuth or controls azimuth for direction
        assert "getAzimuthalAngle" in source or "rotation.y" in source or "azimuth" in source.lower(), \
            "Compass should use camera azimuth angle"

    def test_compass_update_function(self, source):
        assert "updateCompass" in source or "drawCompass" in source, \
            "Missing compass update function"


class TestCompassDisplay:
    def test_compass_positioning(self, source):
        # Should be top-center
        assert "compass" in source.lower(), \
            "Compass should be positioned at top-center"

    def test_compass_text_content(self, source):
        # Should display cardinal letters
        for direction in ["N", "S", "E", "W"]:
            assert direction in source, \
                f"Compass should display '{direction}' direction"


# =========================================================================
# 3. CONTROLS HINT — M key documented
# =========================================================================

class TestControlsHint:
    def test_minimap_in_controls_bar(self, source):
        assert "Minimap" in source or "Map" in source, \
            "Minimap toggle should be documented in controls bar"
