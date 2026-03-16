# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the visual analysis module."""

import os
import pytest

from tritium_lib.testing.visual_analysis import (
    file_exists, file_size_kb,
    BoundingBox, DetectedElement, FrameAnalysis,
    RenderMode,
)

try:
    import cv2
    import numpy as np
    from tritium_lib.testing.visual_analysis import (
        compare_screenshots, is_mostly_black, detect_changes,
        save_diff_image, detect_panels, detect_markers, detect_text_regions,
        analyze_frame, load_image,
    )
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class TestDataStructures:

    def test_bounding_box(self):
        bb = BoundingBox(10, 20, 100, 50)
        assert bb.area == 5000
        assert bb.center == (60, 45)
        assert bb.contains(50, 30)
        assert not bb.contains(5, 5)

    def test_detected_element(self):
        e = DetectedElement("panel", BoundingBox(0, 0, 100, 50), color="#00f0ff")
        assert e.element_type == "panel"
        assert e.bbox.area == 5000

    def test_frame_analysis(self):
        fa = FrameAnalysis(path="/tmp/test.png")
        fa.elements.append(DetectedElement("panel", BoundingBox(0, 0, 100, 50)))
        fa.elements.append(DetectedElement("marker", BoundingBox(50, 50, 10, 10)))
        assert fa.count("panel") == 1
        assert fa.count("marker") == 1

    def test_render_mode_constants(self):
        assert RenderMode.SOLID_BLOCKS == "solid_blocks"
        assert RenderMode.SEMANTIC == "semantic"
        assert RenderMode.url_param("wireframe") == "?render_mode=wireframe"
        assert "render_mode" in RenderMode.api_body("depth")

    def test_semantic_colors(self):
        assert "panel" in RenderMode.SEMANTIC_COLORS
        assert "button" in RenderMode.SEMANTIC_COLORS


class TestFileAssertions:

    def test_file_exists_true(self, tmp_path):
        p = tmp_path / "test.png"
        p.write_bytes(b"x" * 2000)
        assert file_exists(str(p))

    def test_file_exists_false(self, tmp_path):
        assert not file_exists(str(tmp_path / "nope.png"))

    def test_file_size(self, tmp_path):
        p = tmp_path / "test.png"
        p.write_bytes(b"x" * 5120)
        assert file_size_kb(str(p)) == pytest.approx(5.0)


@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestComparators:

    def _solid(self, path, color, w=100, h=100):
        img = np.full((h, w, 3), color, dtype=np.uint8)
        cv2.imwrite(str(path), img)
        return str(path)

    def test_identical_no_diff(self, tmp_path):
        a = self._solid(tmp_path / "a.png", (0, 0, 0))
        b = self._solid(tmp_path / "b.png", (0, 0, 0))
        diff = compare_screenshots(a, b)
        assert diff["changed_pixels"] == 0

    def test_different_has_diff(self, tmp_path):
        a = self._solid(tmp_path / "a.png", (0, 0, 0))
        b = self._solid(tmp_path / "b.png", (255, 255, 255))
        diff = compare_screenshots(a, b)
        assert diff["change_percent"] > 50

    def test_is_black_true(self, tmp_path):
        assert is_mostly_black(self._solid(tmp_path / "b.png", (5, 5, 5)))

    def test_is_black_false(self, tmp_path):
        assert not is_mostly_black(self._solid(tmp_path / "w.png", (200, 200, 200)))

    def test_detect_changes(self, tmp_path):
        a = self._solid(tmp_path / "a.png", (0, 0, 0))
        b = self._solid(tmp_path / "b.png", (100, 100, 100))
        assert detect_changes(a, b)

    def test_no_changes(self, tmp_path):
        a = self._solid(tmp_path / "a.png", (50, 50, 50))
        b = self._solid(tmp_path / "b.png", (50, 50, 50))
        assert not detect_changes(a, b)

    def test_save_diff(self, tmp_path):
        a = self._solid(tmp_path / "a.png", (0, 0, 0))
        b = self._solid(tmp_path / "b.png", (255, 255, 255))
        out = str(tmp_path / "diff.png")
        result = save_diff_image(a, b, out)
        assert result is not None
        assert os.path.exists(out)


@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestDetectors:

    def test_detect_markers_on_dark_bg(self, tmp_path):
        """Bright colored dots on dark background should be detected."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Place some cyan dots
        cv2.circle(img, (50, 50), 8, (255, 240, 0), -1)
        cv2.circle(img, (150, 100), 6, (0, 255, 100), -1)
        p = str(tmp_path / "markers.png")
        cv2.imwrite(p, img)
        markers = detect_markers(p)
        assert len(markers) >= 2

    def test_detect_text_regions(self, tmp_path):
        """Bright text-like regions should be detected."""
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(img, "HELLO WORLD", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, "STATUS: OK", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 1)
        p = str(tmp_path / "text.png")
        cv2.imwrite(p, img)
        regions = detect_text_regions(p)
        assert len(regions) >= 1

    def test_analyze_frame(self, tmp_path):
        """Full frame analysis returns structured result."""
        img = np.full((400, 600, 3), (30, 30, 40), dtype=np.uint8)  # dark gray, not black
        cv2.rectangle(img, (10, 10), (200, 150), (255, 240, 0), 2)
        cv2.rectangle(img, (10, 10), (200, 150), (100, 80, 40), -1)  # filled panel
        cv2.circle(img, (300, 200), 8, (0, 255, 100), -1)
        p = str(tmp_path / "frame.png")
        cv2.imwrite(p, img)
        analysis = analyze_frame(p)
        assert not analysis.is_black_screen
        assert analysis.pixel_stats["mean_brightness"] > 0
        assert len(analysis.dominant_colors) > 0

    def test_black_screen_detection(self, tmp_path):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        p = str(tmp_path / "black.png")
        cv2.imwrite(p, img)
        analysis = analyze_frame(p)
        assert analysis.is_black_screen
