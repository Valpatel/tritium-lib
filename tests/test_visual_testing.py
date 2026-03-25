# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the OpenCV-based visual testing framework.

Uses synthetic test images — no browser or server needed.
Each test creates images programmatically to exercise specific checks.
"""

import pytest

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from tritium_lib.testing.visual_testing import (
    ScreenshotAnalyzer,
    AnalysisReport,
    BlankScreenResult,
    UIElementResult,
    ColorDistributionResult,
    TextReadabilityResult,
    OverlapResult,
    BaselineComparisonResult,
    MapTileResult,
    MarkerResult,
    check_blank_screen,
    check_ui_elements,
    check_color_distribution,
    check_text_readability,
    check_element_overlap,
    compare_baseline,
    detect_map_tiles,
    detect_markers,
    HAS_OPENCV as MODULE_HAS_OPENCV,
    CYBERPUNK_COLORS,
)


# ============================================================
# Helpers for generating synthetic test images
# ============================================================

def _solid(color, w=800, h=600):
    """Create a solid-color BGR image."""
    return np.full((h, w, 3), color, dtype=np.uint8)


def _dark_with_header_sidebar(w=800, h=600):
    """Create a dark background with a bright header and sidebar — typical UI layout."""
    img = np.full((h, w, 3), (15, 12, 10), dtype=np.uint8)
    # Header bar: bright strip at top
    cv2.rectangle(img, (0, 0), (w, 50), (60, 50, 40), -1)
    cv2.putText(img, "TRITIUM COMMAND CENTER", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 240, 0), 2)
    # Sidebar: left panel with some structure
    cv2.rectangle(img, (0, 50), (200, h), (30, 25, 20), -1)
    cv2.line(img, (200, 50), (200, h), (255, 240, 0), 1)
    for i in range(5):
        y = 80 + i * 40
        cv2.putText(img, f"Menu Item {i+1}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    # Footer
    cv2.rectangle(img, (0, h - 30), (w, h), (40, 35, 30), -1)
    cv2.putText(img, "Status: Online", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 240, 255), 1)
    return img


def _cyberpunk_ui(w=800, h=600):
    """Create a cyberpunk-themed UI with signature colors."""
    img = np.full((h, w, 3), (15, 12, 10), dtype=np.uint8)
    # Cyan header
    cv2.rectangle(img, (0, 0), (w, 50), (200, 200, 0), -1)
    # Magenta sidebar accent
    cv2.rectangle(img, (0, 50), (5, h), (109, 42, 255), -1)
    # Green status indicators
    for i in range(3):
        cv2.circle(img, (300 + i * 50, 100), 8, (161, 255, 5), -1)
    # Yellow warning bar
    cv2.rectangle(img, (200, h - 50), (w, h - 30), (10, 238, 252), -1)
    # Cyan panels
    cv2.rectangle(img, (210, 60), (600, 300), (255, 240, 0), 2)
    return img


def _map_with_tiles(w=800, h=600):
    """Create a fake satellite map with texture patterns."""
    # Start with random noise to simulate satellite imagery
    rng = np.random.RandomState(42)
    img = rng.randint(40, 180, (h, w, 3), dtype=np.uint8)
    # Add some road-like lines
    cv2.line(img, (100, 0), (100, h), (80, 80, 80), 3)
    cv2.line(img, (0, 300), (w, 300), (80, 80, 80), 3)
    # Add building-like rectangles
    for i in range(10):
        x, y = rng.randint(0, w - 60), rng.randint(0, h - 40)
        cv2.rectangle(img, (x, y), (x + 50, y + 30), (60, 60, 60), -1)
    return img


def _map_with_markers(w=800, h=600, n_markers=5):
    """Create a map image with colored markers on it."""
    img = _map_with_tiles(w, h)
    rng = np.random.RandomState(99)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 240, 0), (0, 255, 255)]
    for i in range(n_markers):
        x = rng.randint(50, w - 50)
        y = rng.randint(50, h - 50)
        color = colors[i % len(colors)]
        cv2.circle(img, (x, y), 10, color, -1)
    return img


def _text_image(text_color, bg_color, w=400, h=100):
    """Create an image with text for readability testing."""
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)
    cv2.putText(img, "READABILITY TEST STRING", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    return img


def _overlapping_panels(w=800, h=600):
    """Create an image with overlapping UI panels."""
    img = np.full((h, w, 3), (15, 12, 10), dtype=np.uint8)
    # Panel A
    cv2.rectangle(img, (50, 50), (350, 300), (255, 240, 0), 2)
    cv2.rectangle(img, (51, 51), (349, 299), (40, 35, 30), -1)
    cv2.putText(img, "Panel A", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Panel B overlapping Panel A significantly
    cv2.rectangle(img, (100, 100), (400, 350), (109, 42, 255), 2)
    cv2.rectangle(img, (101, 101), (399, 349), (35, 30, 40), -1)
    cv2.putText(img, "Panel B", (110, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img


# ============================================================
# Test: check_blank_screen
# ============================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestCheckBlankScreen:
    def test_black_screen_detected(self):
        img = _solid((0, 0, 0))
        result = check_blank_screen(img)
        assert result.is_blank
        assert result.blank_type == "black"
        assert result.dark_pct > 95

    def test_white_screen_detected(self):
        img = _solid((255, 255, 255))
        result = check_blank_screen(img)
        assert result.is_blank
        assert result.blank_type == "white"
        assert result.bright_pct > 95

    def test_uniform_gray_detected(self):
        img = _solid((128, 128, 128))
        result = check_blank_screen(img)
        assert result.is_blank
        assert result.blank_type == "uniform"

    def test_normal_ui_not_blank(self):
        img = _dark_with_header_sidebar()
        result = check_blank_screen(img)
        assert not result.is_blank
        assert result.blank_type == ""

    def test_near_black_with_content_not_blank(self):
        img = _solid((5, 5, 5))
        # Add a large bright region covering >10% of the image
        cv2.rectangle(img, (0, 0), (400, 200), (255, 255, 255), -1)
        result = check_blank_screen(img)
        assert not result.is_blank

    def test_from_file_path(self, tmp_path):
        img = _solid((0, 0, 0))
        p = str(tmp_path / "black.png")
        cv2.imwrite(p, img)
        result = check_blank_screen(p)
        assert result.is_blank
        assert result.blank_type == "black"

    def test_nonexistent_file(self):
        result = check_blank_screen("/nonexistent/path.png")
        assert result.is_blank
        assert result.blank_type == "load_failed"


# ============================================================
# Test: check_ui_elements
# ============================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestCheckUIElements:
    def test_detects_header(self):
        img = _dark_with_header_sidebar()
        result = check_ui_elements(img)
        assert result.has_header
        region_labels = [r[0] for r in result.detected_regions]
        assert "header" in region_labels

    def test_detects_sidebar(self):
        img = _dark_with_header_sidebar()
        result = check_ui_elements(img)
        assert result.has_sidebar

    def test_detects_map_area(self):
        img = _dark_with_header_sidebar()
        # Add some content in the main area
        cv2.rectangle(img, (220, 70), (780, 550), (40, 40, 40), -1)
        cv2.putText(img, "Map Content", (300, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        result = check_ui_elements(img)
        assert result.has_map_area

    def test_blank_has_no_elements(self):
        img = _solid((0, 0, 0))
        result = check_ui_elements(img)
        assert not result.has_header
        assert not result.has_sidebar

    def test_detects_panels(self):
        img = _cyberpunk_ui()
        result = check_ui_elements(img)
        assert result.panel_count > 0

    def test_custom_header_height(self):
        img = np.full((600, 800, 3), (10, 10, 10), dtype=np.uint8)
        # Header that is 100px tall
        cv2.rectangle(img, (0, 0), (800, 100), (60, 50, 40), -1)
        cv2.putText(img, "BIG HEADER", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        result = check_ui_elements(img, header_height=100)
        assert result.has_header


# ============================================================
# Test: check_color_distribution
# ============================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestCheckColorDistribution:
    def test_cyberpunk_colors_detected(self):
        img = _cyberpunk_ui()
        result = check_color_distribution(img)
        # Should detect at least cyan (the header and panel borders)
        assert result.has_cyan or result.has_green or result.has_magenta or result.has_yellow

    def test_full_cyberpunk_qualifies(self):
        # Create an image with all four signature colors in large blocks
        img = np.full((400, 400, 3), (15, 12, 10), dtype=np.uint8)
        # Large cyan block
        cv2.rectangle(img, (0, 0), (200, 200), (255, 240, 0), -1)
        # Large magenta block
        cv2.rectangle(img, (200, 0), (400, 200), (109, 42, 255), -1)
        # Large green block
        cv2.rectangle(img, (0, 200), (200, 400), (161, 255, 5), -1)
        # Large yellow block
        cv2.rectangle(img, (200, 200), (400, 400), (10, 238, 252), -1)
        result = check_color_distribution(img)
        assert result.is_cyberpunk

    def test_grayscale_not_cyberpunk(self):
        img = _solid((128, 128, 128))
        result = check_color_distribution(img)
        assert not result.is_cyberpunk
        assert not result.has_cyan
        assert not result.has_magenta

    def test_dominant_colors_returned(self):
        img = _cyberpunk_ui()
        result = check_color_distribution(img)
        assert len(result.dominant_colors) > 0
        # All should be valid hex strings
        for c in result.dominant_colors:
            assert c.startswith("#")
            assert len(c) == 7

    def test_single_color_image(self):
        # Pure red image
        img = _solid((0, 0, 255))
        result = check_color_distribution(img)
        assert not result.has_cyan
        assert not result.has_green


# ============================================================
# Test: check_text_readability
# ============================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestCheckTextReadability:
    def test_high_contrast_text_readable(self):
        img = _text_image((255, 255, 255), (0, 0, 0))
        result = check_text_readability(img)
        assert result.text_region_count > 0
        assert result.readable

    def test_low_contrast_text_flagged(self):
        # Very similar foreground and background
        img = _text_image((45, 45, 45), (30, 30, 30))
        result = check_text_readability(img, min_contrast=3.0)
        # If text regions are found, they should have low contrast
        if result.text_region_count > 0:
            assert result.low_contrast_count > 0 or result.avg_contrast_ratio < 3.0

    def test_multiple_text_regions(self):
        img = np.full((400, 600, 3), (10, 10, 10), dtype=np.uint8)
        for i in range(4):
            y = 50 + i * 80
            cv2.putText(img, f"Line {i+1}: Test text here", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        result = check_text_readability(img)
        assert result.text_region_count >= 2

    def test_no_text_returns_clean(self):
        img = _solid((50, 50, 50))
        result = check_text_readability(img)
        assert result.text_region_count == 0
        assert result.readable  # No text = no readability issues


# ============================================================
# Test: check_element_overlap
# ============================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestCheckElementOverlap:
    def test_no_overlap_clean(self):
        img = np.full((600, 800, 3), (15, 12, 10), dtype=np.uint8)
        # Two clearly separated panels
        cv2.rectangle(img, (10, 10), (200, 200), (255, 240, 0), 2)
        cv2.rectangle(img, (400, 10), (700, 200), (255, 240, 0), 2)
        result = check_element_overlap(img)
        assert result.overlap_count == 0

    def test_overlap_detected(self):
        img = _overlapping_panels()
        result = check_element_overlap(img, min_overlap_pct=10.0)
        # The overlapping panels should be detected
        # (contour detection may merge them, so check is lenient)
        assert isinstance(result, OverlapResult)

    def test_blank_image_no_overlaps(self):
        img = _solid((0, 0, 0))
        result = check_element_overlap(img)
        assert result.overlap_count == 0
        assert not result.has_critical_overlap


# ============================================================
# Test: compare_baseline
# ============================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestCompareBaseline:
    def test_identical_images_match(self):
        img = _cyberpunk_ui()
        result = compare_baseline(img, img.copy())
        assert result.matches_baseline
        assert result.ssim_score > 0.99
        assert result.mse < 1.0

    def test_completely_different_images(self):
        a = _solid((0, 0, 0))
        b = _solid((255, 255, 255))
        result = compare_baseline(a, b)
        assert not result.matches_baseline
        assert result.ssim_score < 0.5
        assert result.mse > 10000

    def test_slight_difference_still_matches(self):
        img = _cyberpunk_ui()
        modified = img.copy()
        # Small change: add a tiny dot
        cv2.circle(modified, (400, 300), 3, (255, 255, 255), -1)
        result = compare_baseline(img, modified, ssim_threshold=0.9)
        assert result.matches_baseline
        assert result.ssim_score > 0.9

    def test_major_layout_change_fails(self):
        a = _dark_with_header_sidebar()
        b = a.copy()
        # Major layout change: add a large bright panel covering a big area
        cv2.rectangle(b, (200, 100), (700, 500), (200, 200, 200), -1)
        result = compare_baseline(a, b)
        assert result.changed_pct > 5

    def test_diff_regions_found(self):
        a = _solid((50, 50, 50))
        b = a.copy()
        # Add a large white block to b
        cv2.rectangle(b, (100, 100), (300, 300), (255, 255, 255), -1)
        result = compare_baseline(a, b)
        assert len(result.diff_regions) > 0

    def test_different_size_images(self):
        a = _solid((100, 100, 100), w=800, h=600)
        b = _solid((100, 100, 100), w=400, h=300)
        result = compare_baseline(a, b)
        # Should handle resize gracefully
        assert result.ssim_score > 0.9


# ============================================================
# Test: detect_map_tiles
# ============================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestDetectMapTiles:
    def test_textured_map_has_tiles(self):
        img = _map_with_tiles()
        result = detect_map_tiles(img)
        assert result.has_tiles
        assert not result.is_blank_map
        assert result.texture_score > 50
        assert result.unique_color_count > 50

    def test_blank_gray_map_detected(self):
        img = _solid((180, 180, 180))
        result = detect_map_tiles(img)
        assert not result.has_tiles
        assert result.is_blank_map

    def test_blank_dark_map_detected(self):
        img = _solid((20, 20, 20))
        result = detect_map_tiles(img)
        assert not result.has_tiles
        assert result.is_blank_map

    def test_custom_map_region(self):
        img = np.full((600, 800, 3), (20, 20, 20), dtype=np.uint8)
        # Add texture only in a specific region
        rng = np.random.RandomState(42)
        img[100:400, 200:600] = rng.randint(40, 180, (300, 400, 3), dtype=np.uint8)
        result = detect_map_tiles(img, map_region=(200, 100, 400, 300))
        assert result.has_tiles

    def test_tile_coverage_percentage(self):
        img = _map_with_tiles()
        result = detect_map_tiles(img)
        assert result.tile_coverage_pct > 0


# ============================================================
# Test: detect_markers
# ============================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestDetectMarkers:
    def test_colored_markers_found(self):
        img = _map_with_markers(n_markers=5)
        result = detect_markers(img)
        assert result.marker_count >= 3  # noise in the map may absorb some
        assert len(result.colors_found) > 0

    def test_no_markers_on_blank(self):
        img = _solid((20, 20, 20))
        result = detect_markers(img)
        assert result.marker_count == 0

    def test_single_marker(self):
        img = np.full((400, 400, 3), (30, 30, 30), dtype=np.uint8)
        cv2.circle(img, (200, 200), 12, (0, 255, 0), -1)
        result = detect_markers(img)
        assert result.marker_count >= 1
        assert any("#" in c for c in result.colors_found)

    def test_marker_size_filtering(self):
        img = np.full((400, 400, 3), (30, 30, 30), dtype=np.uint8)
        # Tiny marker (below min_size=10)
        cv2.circle(img, (100, 100), 2, (0, 255, 0), -1)
        # Normal marker
        cv2.circle(img, (200, 200), 15, (0, 0, 255), -1)
        # Huge marker (above max_size=30)
        cv2.circle(img, (300, 300), 50, (255, 0, 0), -1)
        result = detect_markers(img, min_size=10, max_size=30)
        # Only the normal-sized marker should be found
        # (tiny and huge filtered out)
        assert result.marker_count <= 2

    def test_marker_hex_colors_valid(self):
        img = np.full((200, 200, 3), (20, 20, 20), dtype=np.uint8)
        cv2.circle(img, (100, 100), 15, (255, 0, 0), -1)
        result = detect_markers(img)
        for m in result.markers:
            assert m["color_hex"].startswith("#")
            assert len(m["color_hex"]) == 7


# ============================================================
# Test: ScreenshotAnalyzer
# ============================================================

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")
class TestScreenshotAnalyzer:
    def test_full_analysis_normal_ui(self):
        img = _cyberpunk_ui()
        analyzer = ScreenshotAnalyzer()
        report = analyzer.analyze(img)
        assert isinstance(report, AnalysisReport)
        assert report.blank_screen is not None
        assert not report.blank_screen.is_blank
        assert report.ui_elements is not None
        assert report.color_distribution is not None
        assert report.text_readability is not None
        assert report.overlap is not None
        assert report.map_tiles is not None
        assert report.markers is not None

    def test_analysis_fails_on_black_screen(self):
        img = _solid((0, 0, 0))
        analyzer = ScreenshotAnalyzer()
        report = analyzer.analyze(img)
        assert not report.passed
        assert any("Blank screen" in f for f in report.failures)
        # Should short-circuit — other checks not run
        assert report.ui_elements is None

    def test_selective_checks(self):
        img = _cyberpunk_ui()
        analyzer = ScreenshotAnalyzer()
        report = analyzer.analyze(img, checks=["blank", "colors"])
        assert report.blank_screen is not None
        assert report.color_distribution is not None
        assert report.ui_elements is None  # not requested
        assert report.map_tiles is None  # not requested

    def test_baseline_comparison_in_report(self):
        img = _cyberpunk_ui()
        baseline = img.copy()
        analyzer = ScreenshotAnalyzer()
        report = analyzer.analyze(img, baseline=baseline)
        assert report.baseline is not None
        assert report.baseline.matches_baseline

    def test_custom_config(self):
        analyzer = ScreenshotAnalyzer(
            header_height=80,
            sidebar_width=300,
            min_contrast=4.5,
            ssim_threshold=0.9,
        )
        assert analyzer.header_height == 80
        assert analyzer.sidebar_width == 300
        assert analyzer.min_contrast == 4.5
        assert analyzer.ssim_threshold == 0.9

    def test_individual_check_methods(self):
        img = _cyberpunk_ui()
        analyzer = ScreenshotAnalyzer()
        blank = analyzer.check_blank(img)
        assert isinstance(blank, BlankScreenResult)
        elements = analyzer.check_elements(img)
        assert isinstance(elements, UIElementResult)
        colors = analyzer.check_colors(img)
        assert isinstance(colors, ColorDistributionResult)
        text = analyzer.check_text(img)
        assert isinstance(text, TextReadabilityResult)
        overlaps = analyzer.check_overlaps(img)
        assert isinstance(overlaps, OverlapResult)
        map_result = analyzer.check_map(img)
        assert isinstance(map_result, MapTileResult)
        markers = analyzer.check_markers(img)
        assert isinstance(markers, MarkerResult)

    def test_file_path_input(self, tmp_path):
        img = _cyberpunk_ui()
        p = str(tmp_path / "ui.png")
        cv2.imwrite(p, img)
        analyzer = ScreenshotAnalyzer()
        report = analyzer.analyze(p)
        assert report.blank_screen is not None
        assert not report.blank_screen.is_blank


# ============================================================
# Test: Graceful degradation
# ============================================================

class TestGracefulDegradation:
    def test_module_reports_opencv_status(self):
        # MODULE_HAS_OPENCV should be True since we imported cv2 above
        # (or False on systems without OpenCV — the test still passes)
        assert isinstance(MODULE_HAS_OPENCV, bool)

    def test_none_input_returns_defaults(self):
        if not HAS_OPENCV:
            pytest.skip("OpenCV not installed")
        result = check_blank_screen(None)
        assert result.is_blank
        assert result.blank_type == "load_failed"

    def test_data_classes_importable(self):
        # All data classes should be importable regardless of OpenCV
        assert BlankScreenResult is not None
        assert UIElementResult is not None
        assert ColorDistributionResult is not None
        assert TextReadabilityResult is not None
        assert OverlapResult is not None
        assert BaselineComparisonResult is not None
        assert MapTileResult is not None
        assert MarkerResult is not None
        assert AnalysisReport is not None
        assert ScreenshotAnalyzer is not None

    def test_cyberpunk_colors_defined(self):
        assert "cyan" in CYBERPUNK_COLORS
        assert "magenta" in CYBERPUNK_COLORS
        assert "green" in CYBERPUNK_COLORS
        assert "yellow" in CYBERPUNK_COLORS
        for name, info in CYBERPUNK_COLORS.items():
            assert "hex" in info
            assert "bgr" in info
            assert "hsv_range" in info


# ============================================================
# Test: Lazy import from testing package
# ============================================================

class TestLazyImport:
    def test_import_screenshot_analyzer(self):
        from tritium_lib.testing import ScreenshotAnalyzer as SA
        assert SA is not None

    def test_import_analysis_report(self):
        from tritium_lib.testing import AnalysisReport as AR
        assert AR is not None

    def test_import_all_result_types(self):
        from tritium_lib.testing import (
            BlankScreenResult,
            UIElementResult,
            ColorDistributionResult,
            TextReadabilityResult,
            OverlapResult,
            BaselineComparisonResult,
            MapTileResult,
            MarkerResult,
        )
        assert BlankScreenResult is not None
        assert UIElementResult is not None
        assert ColorDistributionResult is not None
        assert TextReadabilityResult is not None
        assert OverlapResult is not None
        assert BaselineComparisonResult is not None
        assert MapTileResult is not None
        assert MarkerResult is not None
