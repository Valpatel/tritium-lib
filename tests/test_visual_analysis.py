# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for visual_analysis module.

Uses synthetic images — no browser needed. Tests each capability independently.
"""

import os
import pytest

from tritium_lib.testing.visual_analysis import (
    Box, UIElement, OverlapIssue, ScreenZone,
    file_exists, file_size_kb, find_overlaps,
    make_screen_zones, validate_layout,
)

try:
    import cv2
    import numpy as np
    from tritium_lib.testing.visual_analysis import (
        load, is_black_screen, brightness_stats, dominant_colors,
        detect_bright_rectangles, detect_small_colored_dots, detect_text_blocks,
        analyze_frame_sequence,
    )
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# ---- Data structures ----

class TestBox:
    def test_properties(self):
        b = Box(10, 20, 100, 50)
        assert b.area == 5000
        assert b.cx == 60
        assert b.cy == 45
        assert b.right == 110
        assert b.bottom == 70

    def test_overlaps_true(self):
        a = Box(0, 0, 100, 100)
        b = Box(50, 50, 100, 100)
        assert a.overlaps(b)

    def test_overlaps_false(self):
        a = Box(0, 0, 50, 50)
        b = Box(100, 100, 50, 50)
        assert not a.overlaps(b)

    def test_overlap_area(self):
        a = Box(0, 0, 100, 100)
        b = Box(50, 50, 100, 100)
        assert a.overlap_area(b) == 2500  # 50x50

    def test_iou(self):
        a = Box(0, 0, 100, 100)
        b = Box(0, 0, 100, 100)  # identical
        assert a.iou(b) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = Box(0, 0, 10, 10)
        b = Box(100, 100, 10, 10)
        assert a.iou(b) == 0.0


class TestOverlapDetection:
    def test_no_overlaps(self):
        elements = [
            UIElement("panel", Box(0, 0, 100, 100)),
            UIElement("panel", Box(200, 200, 100, 100)),
        ]
        assert find_overlaps(elements) == []

    def test_detects_overlap(self):
        elements = [
            UIElement("panel", Box(0, 0, 100, 100)),
            UIElement("toast", Box(50, 50, 100, 100)),
        ]
        issues = find_overlaps(elements, min_overlap_pct=10)
        assert len(issues) >= 1
        assert issues[0].severity in ("warning", "error", "critical")

    def test_critical_overlap(self):
        # Nearly identical boxes
        elements = [
            UIElement("panel", Box(10, 10, 200, 200)),
            UIElement("panel", Box(15, 15, 190, 190)),
        ]
        issues = find_overlaps(elements, min_overlap_pct=10)
        assert len(issues) >= 1
        assert issues[0].severity == "critical"


class TestLayoutValidation:
    def test_zone_creation(self):
        zones = make_screen_zones(1920, 1080)
        assert len(zones) == 6
        names = [z.name for z in zones]
        assert "top-left" in names
        assert "bottom-right" in names

    def test_overcrowded_zone(self):
        zones = [ScreenZone("test", Box(0, 0, 500, 500), max_elements=2)]
        elements = [
            UIElement("panel", Box(10, 10, 100, 100)),
            UIElement("panel", Box(10, 120, 100, 100)),
            UIElement("panel", Box(10, 230, 100, 100)),
            UIElement("toast", Box(10, 340, 100, 100)),
        ]
        issues = validate_layout(elements, zones)
        assert len(issues) >= 1
        assert issues[0]["issue"] == "overcrowded"

    def test_normal_layout(self):
        zones = [ScreenZone("test", Box(0, 0, 500, 500), max_elements=5)]
        elements = [UIElement("panel", Box(10, 10, 100, 100))]
        assert validate_layout(elements, zones) == []


class TestFileAssertions:
    def test_file_exists(self, tmp_path):
        p = tmp_path / "test.png"
        p.write_bytes(b"x" * 2000)
        assert file_exists(str(p))

    def test_file_missing(self, tmp_path):
        assert not file_exists(str(tmp_path / "nope.png"))

    def test_file_too_small(self, tmp_path):
        p = tmp_path / "tiny.png"
        p.write_bytes(b"x" * 100)
        assert not file_exists(str(p))

    def test_file_size(self, tmp_path):
        p = tmp_path / "test.png"
        p.write_bytes(b"x" * 5120)
        assert file_size_kb(str(p)) == pytest.approx(5.0)


# ---- OpenCV-dependent tests ----

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestBlackScreen:
    def _solid(self, path, color, w=200, h=200):
        img = np.full((h, w, 3), color, dtype=np.uint8)
        cv2.imwrite(str(path), img)
        return str(path)

    def test_black_is_black(self, tmp_path):
        assert is_black_screen(self._solid(tmp_path / "b.png", (5, 5, 5)))

    def test_gray_not_black(self, tmp_path):
        assert not is_black_screen(self._solid(tmp_path / "g.png", (80, 80, 80)))

    def test_bright_not_black(self, tmp_path):
        assert not is_black_screen(self._solid(tmp_path / "w.png", (200, 200, 200)))


@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestBrightnessStats:
    def test_dark_image(self, tmp_path):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        p = str(tmp_path / "dark.png")
        cv2.imwrite(p, img)
        stats = brightness_stats(p)
        assert stats["mean"] < 5
        assert stats["dark_pct"] > 95

    def test_bright_image(self, tmp_path):
        img = np.full((100, 100, 3), (220, 220, 220), dtype=np.uint8)
        p = str(tmp_path / "bright.png")
        cv2.imwrite(p, img)
        stats = brightness_stats(p)
        assert stats["mean"] > 200
        assert stats["bright_pct"] > 90


@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestDominantColors:
    def test_single_color(self, tmp_path):
        img = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)  # blue in BGR
        p = str(tmp_path / "blue.png")
        cv2.imwrite(p, img)
        colors = dominant_colors(p, n=3)
        assert len(colors) >= 1
        # Should be mostly blue (#0000ff)
        assert colors[0].startswith("#00")


@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestDetectors:
    def test_detect_colored_dots(self, tmp_path):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 10, (0, 255, 0), -1)   # green
        cv2.circle(img, (150, 150), 8, (255, 0, 0), -1)   # blue
        cv2.circle(img, (250, 100), 12, (0, 0, 255), -1)  # red
        p = str(tmp_path / "dots.png")
        cv2.imwrite(p, img)
        dots = detect_small_colored_dots(p)
        assert len(dots) >= 3

    def test_detect_text_blocks(self, tmp_path):
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(img, "HELLO WORLD", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        p = str(tmp_path / "text.png")
        cv2.imwrite(p, img)
        blocks = detect_text_blocks(p)
        assert len(blocks) >= 1

    def test_detect_rectangles(self, tmp_path):
        img = np.full((400, 600, 3), (20, 20, 25), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (250, 180), (255, 240, 0), 2)  # cyan border panel
        cv2.rectangle(img, (300, 50), (580, 350), (109, 42, 255), 2)  # magenta border panel
        p = str(tmp_path / "panels.png")
        cv2.imwrite(p, img)
        rects = detect_bright_rectangles(p)
        assert len(rects) >= 2


@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestFrameSequence:
    def test_static_sequence(self, tmp_path):
        # Same image repeated = all static
        img = np.full((100, 100, 3), (50, 50, 50), dtype=np.uint8)
        paths = []
        for i in range(5):
            p = str(tmp_path / f"f{i}.png")
            cv2.imwrite(p, img)
            paths.append(p)
        result = analyze_frame_sequence(paths)
        assert result["motion"] == 0
        assert result["static"] == 4  # 4 comparisons for 5 frames

    def test_motion_sequence(self, tmp_path):
        # Different images = motion
        paths = []
        for i in range(5):
            img = np.full((100, 100, 3), (i * 50, i * 40, i * 30), dtype=np.uint8)
            p = str(tmp_path / f"f{i}.png")
            cv2.imwrite(p, img)
            paths.append(p)
        result = analyze_frame_sequence(paths)
        assert result["motion"] > 0

    def test_black_screen_anomaly(self, tmp_path):
        paths = []
        for i in range(3):
            color = (100, 100, 100) if i != 1 else (0, 0, 0)  # frame 1 is black
            img = np.full((100, 100, 3), color, dtype=np.uint8)
            p = str(tmp_path / f"f{i}.png")
            cv2.imwrite(p, img)
            paths.append(p)
        result = analyze_frame_sequence(paths)
        assert len(result["anomalies"]) >= 1
        issues = [a["issue"] for a in result["anomalies"]]
        assert "black_screen" in issues
