"""Tests for excessive empty space and UI density detection.

Creates synthetic frames that simulate:
  a) A launcher with tiny icons in a vast dark void (excessive empty space)
  b) A well-filled launcher that uses available space (passes)
  c) A UI with overlapping text/elements crammed together (too dense)
  d) A clean UI with proper spacing (passes)
"""

import numpy as np
import pytest

from tritium_lib.testing.visual import VisualCheck, Severity


@pytest.fixture
def checker():
    return VisualCheck(width=800, height=480)


# ---------------------------------------------------------------------------
# Helper: Tritium-like frames with varying content coverage
# ---------------------------------------------------------------------------

def make_empty_launcher(w=800, h=480):
    """Simulate the current broken launcher: tiny icons centered in vast void.

    Status bar at top, no nav bar (launcher mode), five ~90x80 icons
    clustered in the center. ~95% of viewport is black.
    """
    frame = np.full((h, w, 3), [15, 10, 10], dtype=np.uint8)  # T_VOID

    # Status bar (24px)
    frame[0:24, :] = [20, 14, 14]  # T_SURFACE1
    frame[0:24, 10:120] = [220, 224, 192]  # title text
    frame[0:24, 620:760] = [180, 190, 180]  # clock/icons

    # Five tiny launcher icons centered vertically and horizontally
    # Matches real device: each ~90x70, single row, centered in 456px viewport
    icon_y = 220
    icon_h = 70
    icon_w = 80
    gap = 8
    total_w = 5 * icon_w + 4 * gap
    start_x = (w - total_w) // 2

    for i in range(5):
        x = start_x + i * (icon_w + gap)
        # Icon card background (very dark, barely above void)
        frame[icon_y:icon_y + icon_h, x:x + icon_w] = [24, 17, 17]  # T_SURFACE2
        # Small icon glyph (cyan, 30x25)
        frame[icon_y + 8:icon_y + 33, x + 25:x + 55] = [255, 240, 0]
        # Label text (small, dim)
        frame[icon_y + 45:icon_y + 55, x + 15:x + 65] = [200, 208, 220]

    return frame


def make_filled_launcher(w=800, h=480):
    """Simulate a well-designed launcher that fills available space.

    Larger icons in a 5x2 or 3x2 grid using most of the viewport.
    """
    frame = np.full((h, w, 3), [15, 10, 10], dtype=np.uint8)  # T_VOID

    # Status bar
    frame[0:24, :] = [20, 14, 14]
    frame[0:24, 10:120] = [220, 224, 192]
    frame[0:24, 620:760] = [180, 190, 180]

    # Large launcher grid: 5 columns x 2 rows, fills most of viewport
    icon_w = 140
    icon_h = 160
    gap = 12
    margin_x = 20
    margin_y = 40  # from status bar

    for row in range(2):
        for col in range(5):
            x = margin_x + col * (icon_w + gap)
            y = 24 + margin_y + row * (icon_h + gap)
            if x + icon_w > w or y + icon_h > h:
                continue
            # Card
            frame[y:y + icon_h, x:x + icon_w] = [26, 18, 18]
            # Icon
            frame[y + 15:y + 75, x + 30:x + 110] = [255, 240, 0]
            # Label
            frame[y + 90:y + 110, x + 10:x + 130] = [200, 208, 220]

    return frame


def make_overlapping_ui(w=800, h=480):
    """Simulate UI with elements overlapping each other.

    Dense text rendered on top of other text, creating high edge density
    with overlapping content.
    """
    frame = np.full((h, w, 3), [15, 10, 10], dtype=np.uint8)

    # Status bar
    frame[0:24, :] = [20, 14, 14]
    frame[0:24, 10:120] = [220, 224, 192]

    # Nav bar
    frame[432:480, :] = [46, 26, 26]
    frame[432:434, :] = [255, 240, 0]

    # Create a region with extremely dense overlapping elements
    # Simulate text-on-text: alternating bright/dark horizontal lines
    # plus vertical bars creating a cross-hatch pattern (high edge density)
    dense_region = frame[60:200, 50:400]
    for y in range(0, 140, 2):
        dense_region[y, :] = [200, 210, 220]  # bright row
    for x in range(0, 350, 3):
        dense_region[:, x] = [180, 200, 240]  # bright column
    # Add another layer of diagonal lines
    for d in range(0, min(140, 350)):
        if d < 140 and d < 350:
            dense_region[d, d] = [255, 255, 255]
            if d + 1 < 350:
                dense_region[d, d + 1] = [255, 255, 255]

    frame[60:200, 50:400] = dense_region

    return frame


def make_clean_settings(w=800, h=480):
    """Simulate a clean settings panel with proper spacing."""
    frame = np.full((h, w, 3), [15, 10, 10], dtype=np.uint8)

    # Status bar
    frame[0:24, :] = [20, 14, 14]
    frame[0:24, 10:120] = [220, 224, 192]

    # Nav bar
    frame[432:480, :] = [46, 26, 26]
    frame[432:434, :] = [255, 240, 0]
    for bx in [130, 400, 660]:
        frame[450:470, bx:bx + 40] = [255, 240, 0]

    # Tab bar
    frame[24:60, :] = [30, 22, 22]
    for tx in [20, 180, 340, 500, 660]:
        frame[30:55, tx:tx + 120] = [255, 240, 0]

    # Content panels with proper spacing
    for py in [70, 170, 270, 360]:
        ph = 80
        if py + ph > 432:
            break
        frame[py:py + ph, 20:780] = [26, 18, 18]  # panel bg
        frame[py + 5:py + 15, 30:150] = [200, 210, 220]  # title
        frame[py + 25:py + 35, 30:750] = [40, 30, 30]  # slider track
        frame[py + 25:py + 35, 30:400] = [255, 240, 0]  # slider fill
        frame[py + 50:py + 60, 30:200] = [160, 170, 180]  # label

    return frame


# ============================================================================
# Test: Excessive empty space detection
# ============================================================================

class TestEmptySpace:
    """Verify that the empty space detector catches under-utilized UIs."""

    def test_tiny_icons_in_void_detected(self, checker):
        """Current broken launcher: 5 tiny icons, ~95% black. Should ERROR."""
        frame = make_empty_launcher()
        issues = checker.check_empty_space(frame, has_nav_bar=False)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1, f"Expected ERROR for empty launcher, got: {issues}"
        assert "empty" in errors[0].message.lower()

    def test_filled_launcher_passes(self, checker):
        """Well-designed launcher with large icons filling space. Should pass."""
        frame = make_filled_launcher()
        issues = checker.check_empty_space(frame, has_nav_bar=False)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"False positive on filled launcher: {issues}"

    def test_settings_page_passes(self, checker):
        """Settings page with panels filling viewport. Should pass."""
        frame = make_clean_settings()
        issues = checker.check_empty_space(frame, has_nav_bar=True)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"False positive on settings page: {issues}"

    def test_completely_black_viewport(self, checker):
        """Totally black viewport (no content at all). Should ERROR."""
        frame = np.full((480, 800, 3), [15, 10, 10], dtype=np.uint8)
        # Just a status bar, nothing else
        frame[0:24, :] = [20, 14, 14]
        frame[0:24, 10:80] = [220, 224, 192]
        issues = checker.check_empty_space(frame, has_nav_bar=False)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1, f"Expected ERROR for black viewport: {issues}"


# ============================================================================
# Test: UI density / overlapping elements detection
# ============================================================================

class TestDensity:
    """Verify that the density detector catches overlapping/crammed UI."""

    def test_overlapping_elements_detected(self, checker):
        """Crosshatch pattern simulating overlapping text. Should ERROR."""
        frame = make_overlapping_ui()
        issues = checker.check_density(frame, has_nav_bar=True)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) >= 1, f"Expected ERROR for overlapping UI, got: {issues}"
        assert "dense" in errors[0].message.lower() or "overlap" in errors[0].message.lower()

    def test_clean_settings_passes(self, checker):
        """Clean settings page with proper spacing. Should pass (no ERROR)."""
        frame = make_clean_settings()
        issues = checker.check_density(frame, has_nav_bar=True)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"False positive on clean settings: {issues}"

    def test_filled_launcher_passes(self, checker):
        """Well-filled launcher grid. Should pass (no ERROR)."""
        frame = make_filled_launcher()
        issues = checker.check_density(frame, has_nav_bar=False)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"False positive on filled launcher: {issues}"

    def test_empty_screen_passes(self, checker):
        """Empty screen has no density issues (it has empty space issues instead)."""
        frame = make_empty_launcher()
        issues = checker.check_density(frame, has_nav_bar=False)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"False positive on empty launcher: {issues}"
