# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the visual analysis module.

These test the ANALYSIS functions, not the browser collection.
Uses synthetic images to verify pixel comparison, color counting, etc.
"""

import os
import tempfile
import pytest

from tritium_lib.testing.visual_analysis import (
    file_exists, file_size_kb,
    VisualDiff, ColorCount, VisionDescription,
)

# OpenCV tests are conditional
try:
    import cv2
    import numpy as np
    from tritium_lib.testing.visual_analysis import (
        compare_screenshots, is_mostly_black, count_colored_pixels,
        detect_changes, save_diff_image,
    )
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class TestFileAssertions:
    """Layer 3: simple file assertions (no deps)."""

    def test_file_exists_true(self, tmp_path):
        p = tmp_path / "test.png"
        p.write_bytes(b"x" * 2000)
        assert file_exists(str(p))

    def test_file_exists_false(self, tmp_path):
        assert not file_exists(str(tmp_path / "nope.png"))

    def test_file_exists_too_small(self, tmp_path):
        p = tmp_path / "tiny.png"
        p.write_bytes(b"x" * 100)
        assert not file_exists(str(p))

    def test_file_size(self, tmp_path):
        p = tmp_path / "test.png"
        p.write_bytes(b"x" * 5120)
        assert file_size_kb(str(p)) == pytest.approx(5.0)


class TestDataclasses:
    """Verify dataclass construction."""

    def test_visual_diff(self):
        d = VisualDiff(changed_pixels=100, total_pixels=1000, change_percent=10.0, regions_changed=3)
        assert d.change_percent == 10.0

    def test_color_count(self):
        c = ColorCount(total_pixels=1000, matching_pixels=50, percent=5.0)
        assert c.percent == 5.0

    def test_vision_description(self):
        v = VisionDescription(description="A map", model="llava:7b", success=True)
        assert v.success


@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestOpenCVAnalysis:
    """Layer 1: OpenCV pixel analysis."""

    def _make_solid_image(self, path, color_bgr, w=100, h=100):
        img = np.full((h, w, 3), color_bgr, dtype=np.uint8)
        cv2.imwrite(str(path), img)
        return str(path)

    def test_identical_images_no_diff(self, tmp_path):
        a = self._make_solid_image(tmp_path / "a.png", (0, 0, 0))
        b = self._make_solid_image(tmp_path / "b.png", (0, 0, 0))
        diff = compare_screenshots(a, b)
        assert diff.changed_pixels == 0
        assert diff.change_percent == 0

    def test_different_images_have_diff(self, tmp_path):
        a = self._make_solid_image(tmp_path / "a.png", (0, 0, 0))
        b = self._make_solid_image(tmp_path / "b.png", (255, 255, 255))
        diff = compare_screenshots(a, b)
        assert diff.changed_pixels > 0
        assert diff.change_percent > 50

    def test_is_mostly_black_true(self, tmp_path):
        p = self._make_solid_image(tmp_path / "black.png", (5, 5, 5))
        assert is_mostly_black(p)

    def test_is_mostly_black_false(self, tmp_path):
        p = self._make_solid_image(tmp_path / "white.png", (200, 200, 200))
        assert not is_mostly_black(p)

    def test_count_cyan_pixels(self, tmp_path):
        # Image with some cyan pixels
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[0:50, :] = (255, 240, 0)  # BGR cyan
        p = str(tmp_path / "cyan.png")
        cv2.imwrite(p, img)
        result = count_colored_pixels(p, (255, 240, 0), tolerance=20)
        assert result.percent > 40

    def test_detect_changes_true(self, tmp_path):
        a = self._make_solid_image(tmp_path / "a.png", (0, 0, 0))
        b = self._make_solid_image(tmp_path / "b.png", (100, 100, 100))
        assert detect_changes(a, b)

    def test_detect_changes_false(self, tmp_path):
        a = self._make_solid_image(tmp_path / "a.png", (50, 50, 50))
        b = self._make_solid_image(tmp_path / "b.png", (50, 50, 50))
        assert not detect_changes(a, b)

    def test_save_diff_image(self, tmp_path):
        a = self._make_solid_image(tmp_path / "a.png", (0, 0, 0))
        b = self._make_solid_image(tmp_path / "b.png", (255, 255, 255))
        out = str(tmp_path / "diff.png")
        result = save_diff_image(a, b, out)
        assert result is not None
        assert os.path.exists(out)

    def test_partial_change(self, tmp_path):
        # 100x100 image, top half changed
        img_a = np.zeros((100, 100, 3), dtype=np.uint8)
        img_b = np.zeros((100, 100, 3), dtype=np.uint8)
        img_b[0:50, :] = (200, 200, 200)
        a = str(tmp_path / "a.png")
        b = str(tmp_path / "b.png")
        cv2.imwrite(a, img_a)
        cv2.imwrite(b, img_b)
        diff = compare_screenshots(a, b)
        assert 40 < diff.change_percent < 60  # ~50% changed
