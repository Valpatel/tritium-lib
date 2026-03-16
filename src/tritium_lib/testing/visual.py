"""OpenCV-based visual checks for Tritium-OS shell UI.

Detects layout problems, rendering glitches, and accessibility issues
that should never ship — clipped buttons, blank screens, overflows,
invisible nav bars, text overlap, etc.

Thresholds are tuned for the Tritium cyberpunk theme: dark backgrounds
(~#0a0a0f), cyan accents (#00f0ff), sparse bright UI elements on
mostly-dark surfaces.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class Severity(Enum):
    ERROR = "error"      # Test fails
    WARNING = "warning"  # Logged but test passes


@dataclass
class LayoutIssue:
    check: str
    severity: Severity
    message: str
    region: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h)

    def __str__(self):
        loc = f" at ({self.region})" if self.region else ""
        return f"[{self.severity.value}] {self.check}: {self.message}{loc}"


@dataclass
class VisualReport:
    app_name: str
    issues: list[LayoutIssue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[LayoutIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[LayoutIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


class VisualCheck:
    """Stateless visual analysis of Tritium-OS screenshots.

    All checks accept a BGR uint8 numpy array (H, W, 3) and return issues.
    Thresholds are tuned for the Tritium cyberpunk theme (dark backgrounds,
    cyan/magenta accents).
    """

    # Screen geometry (can be overridden per-board)
    STATUS_BAR_H = 24
    NAV_BAR_H = 48
    MIN_BUTTON_SIZE = 30  # Minimum tappable area in pixels

    # Dark theme background range — pixels in this range are "background"
    # and should not trigger corruption/banding alerts
    BG_DARK_THRESHOLD = 20  # grayscale value below which a row is "dark bg"

    def __init__(self, width: int = 800, height: int = 480):
        self.width = width
        self.height = height

    def run_all(self, img: np.ndarray, app_name: str = "unknown",
                has_nav_bar: bool = True) -> VisualReport:
        """Run all visual checks on a screenshot."""
        report = VisualReport(app_name=app_name)
        report.issues.extend(self.check_blank_screen(img))
        report.issues.extend(self.check_corruption(img))
        report.issues.extend(self.check_status_bar(img))
        if has_nav_bar:
            report.issues.extend(self.check_nav_bar(img))
        report.issues.extend(self.check_content_overflow(img, has_nav_bar))
        report.issues.extend(self.check_button_accessibility(img, has_nav_bar))
        report.issues.extend(self.check_text_rendering(img))
        report.issues.extend(self.check_empty_space(img, has_nav_bar))
        report.issues.extend(self.check_density(img, has_nav_bar))

        # Multi-frame flicker check requires a frame sequence
        # (called separately via check_flicker)

        # Store metrics for diagnostics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        report.metrics["mean_brightness"] = float(np.mean(gray))
        report.metrics["edge_density"] = float(
            np.mean(cv2.Canny(gray, 50, 150) > 0)
        )
        report.metrics["unique_colors"] = int(
            len(np.unique(img.reshape(-1, 3), axis=0))
        )
        return report

    def check_blank_screen(self, img: np.ndarray) -> list[LayoutIssue]:
        """Detect fully blank or near-blank screens."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blank_ratio = float(np.mean(gray < 5))
        if blank_ratio > 0.98:
            return [LayoutIssue(
                "blank_screen", Severity.ERROR,
                f"Screen is {blank_ratio*100:.0f}% black — display may not be rendering"
            )]
        if blank_ratio > 0.90:
            return [LayoutIssue(
                "blank_screen", Severity.WARNING,
                f"Screen is {blank_ratio*100:.0f}% black — minimal content"
            )]
        return []

    def check_corruption(self, img: np.ndarray) -> list[LayoutIssue]:
        """Detect visual corruption: noise, color banding, tearing.

        Checks both full-screen and localized tearing patterns:
        - Full-screen: consecutive rows with extreme brightness jumps
        - Localized: horizontal discontinuities within non-background regions
          where adjacent rows show incompatible content (half old/half new state)
        """
        issues = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Full-screen row-mean tearing ---
        # Sharp brightness jumps that span MULTIPLE consecutive rows (real
        # tearing), not single-row UI borders.
        row_means = np.mean(gray, axis=1)
        row_diffs = np.abs(np.diff(row_means))

        high_diff_mask = row_diffs > 80
        consecutive_count = 0
        tearing_bands = []
        for row_idx in range(len(high_diff_mask)):
            if high_diff_mask[row_idx]:
                consecutive_count += 1
            else:
                if consecutive_count >= 3:
                    tearing_bands.append((row_idx - consecutive_count, consecutive_count))
                consecutive_count = 0

        for band_start, band_len in tearing_bands:
            issues.append(LayoutIssue(
                "tearing", Severity.WARNING,
                f"Possible tearing at rows {band_start}-{band_start+band_len} "
                f"({band_len} disrupted rows)",
                region=(0, band_start, self.width, band_len),
            ))

        # --- Localized tearing detection ---
        # Scan for horizontal discontinuities WITHIN widget regions. Tearing
        # from unsynchronized buffer swaps creates a horizontal line where
        # the top portion shows one state and the bottom shows another.
        #
        # We detect this by looking for abrupt per-ROW color shifts within
        # contiguous non-background regions. A clean widget has smooth row
        # transitions; a torn widget has a sharp jump in the middle.
        #
        # Divide the screen into vertical columns and analyze each.
        col_width = 80
        for cx in range(0, self.width - col_width, col_width // 2):
            col_strip = gray[:, cx : cx + col_width]
            col_means = np.mean(col_strip, axis=1)

            # Find non-background spans (contiguous rows > threshold)
            above_bg = col_means > self.BG_DARK_THRESHOLD
            span_start = None
            for row in range(self.height):
                if above_bg[row]:
                    if span_start is None:
                        span_start = row
                elif span_start is not None:
                    span_len = row - span_start
                    if span_len >= 8:  # Need enough rows to detect a tear
                        # Check for a sharp horizontal discontinuity in this span
                        span_means = col_means[span_start:row]
                        span_diffs = np.abs(np.diff(span_means))

                        # Look for a single large jump (> 30) flanked by stable rows.
                        # This distinguishes a tear (one abrupt shift in an otherwise
                        # smooth region) from a gradient or multi-element layout.
                        for j in range(1, len(span_diffs) - 1):
                            if span_diffs[j] > 30:
                                # Verify surrounding rows are stable (< 10)
                                before_stable = all(span_diffs[max(0, j-2):j] < 10)
                                after_stable = all(span_diffs[j+1:min(len(span_diffs), j+3)] < 10)
                                if before_stable and after_stable:
                                    tear_row = span_start + j + 1
                                    # Check the color distribution above vs below the tear
                                    above_rgb = img[span_start:tear_row, cx:cx+col_width]
                                    below_rgb = img[tear_row:row, cx:cx+col_width]
                                    above_mean = np.mean(above_rgb, axis=(0, 1))
                                    below_mean = np.mean(below_rgb, axis=(0, 1))
                                    color_jump = float(np.max(np.abs(above_mean - below_mean)))
                                    if color_jump > 25:
                                        issues.append(LayoutIssue(
                                            "localized_tearing", Severity.ERROR,
                                            f"Horizontal tear at row {tear_row} in "
                                            f"column {cx}-{cx+col_width} "
                                            f"(color jump={color_jump:.0f}, "
                                            f"span rows {span_start}-{row})",
                                            region=(cx, span_start, col_width, span_len),
                                        ))
                                    break  # One tear per span is enough
                    span_start = None

        # --- Solid color bands (corruption pattern) ---
        for ch in range(3):
            channel = img[:, :, ch]
            col_std = np.std(channel, axis=1)
            row_mean = np.mean(channel, axis=1)
            suspicious = (col_std < 1.0) & (row_mean > self.BG_DARK_THRESHOLD)
            uniform_bright_rows = int(np.sum(suspicious))
            if uniform_bright_rows > self.height * 0.3:
                issues.append(LayoutIssue(
                    "color_banding", Severity.WARNING,
                    f"Channel {ch}: {uniform_bright_rows} uniform bright rows — possible corruption"
                ))

        return issues

    def check_frame_tearing(
        self,
        frames: list[np.ndarray],
        reference_frame: Optional[np.ndarray] = None,
    ) -> list[LayoutIssue]:
        """Detect tearing across a rapid frame sequence.

        Takes frames captured in rapid succession (~50ms apart) and checks
        each one for signs of a torn buffer — where the top portion of the
        screen shows the previous frame's content and the bottom shows the
        new frame's content, with a horizontal tear boundary between them.

        This is the definitive tearing test: it compares each frame against
        its neighbors to find frames that are a spatial composite of two
        temporally adjacent states.

        Args:
            frames: 3+ BGR frames captured in rapid succession
            reference_frame: Optional stable reference frame to compare against

        Returns:
            List of LayoutIssue for any tearing detected.
        """
        issues = []
        if len(frames) < 3:
            return issues

        for i in range(1, len(frames) - 1):
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
            next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)

            # For each row, measure similarity to prev vs next frame
            row_diff_prev = np.mean(np.abs(curr_gray - prev_gray), axis=1)
            row_diff_next = np.mean(np.abs(curr_gray - next_gray), axis=1)

            # Classify each row: closer to prev or closer to next.
            # A row is "prev-like" if it's nearly identical to prev AND
            # meaningfully different from next (and vice versa). Use both
            # an absolute threshold and a relative ratio to handle cases
            # where per-row diffs are small but still significant.
            match_threshold = 2.0   # "nearly identical" to one frame
            diff_threshold = 4.0    # "meaningfully different" from the other
            prev_like = (row_diff_prev < match_threshold) & (row_diff_next > diff_threshold)
            next_like = (row_diff_next < match_threshold) & (row_diff_prev > diff_threshold)

            # A torn frame has a contiguous block of prev-like rows at the top
            # followed by next-like rows at the bottom (or vice versa)
            if not (np.any(prev_like) and np.any(next_like)):
                continue

            # Find the transition point
            prev_indices = np.where(prev_like)[0]
            next_indices = np.where(next_like)[0]

            if len(prev_indices) < 3 or len(next_indices) < 3:
                continue

            # Check if prev-like rows are contiguous at one end and next-like at the other
            prev_median = np.median(prev_indices)
            next_median = np.median(next_indices)

            # They should be spatially separated (one group above the tear, one below).
            # Use a low threshold: even a 5% gap (~24px on 480p) is meaningful since
            # tearing on monitor-style UIs creates tears between closely-spaced panels.
            if abs(prev_median - next_median) < self.height * 0.05:
                continue  # Interleaved changes, not a tear

            # Find the tear boundary: the row where we transition from one state to another
            if prev_median < next_median:
                # Top = prev state, bottom = next state
                tear_row = int(prev_indices[-1]) + 1
                top_rows = int(np.sum(prev_like))
                bottom_rows = int(np.sum(next_like))
            else:
                tear_row = int(next_indices[-1]) + 1
                top_rows = int(np.sum(next_like))
                bottom_rows = int(np.sum(prev_like))

            issues.append(LayoutIssue(
                "frame_tearing", Severity.ERROR,
                f"Torn frame {i}: tear at row {tear_row} — "
                f"top {top_rows} rows match frame {i-1}, "
                f"bottom {bottom_rows} rows match frame {i+1}",
                region=(0, tear_row, self.width, 1),
            ))

        return issues

    def check_status_bar(self, img: np.ndarray) -> list[LayoutIssue]:
        """Verify status bar is present and has content."""
        issues = []
        bar = img[: self.STATUS_BAR_H, :, :]
        gray_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2GRAY)

        # Status bar should have SOME content (app name, clock, icons).
        # Tritium theme uses dim text (~brightness 30-40) on dark background,
        # so use low thresholds. Check for any non-background pixels.
        bright_pixels = float(np.mean(gray_bar > 15))
        max_brightness = float(np.max(gray_bar))

        # Status bar is empty only if there are NO non-background pixels
        if max_brightness < 10:
            issues.append(LayoutIssue(
                "status_bar", Severity.ERROR,
                f"Status bar appears empty (max_brightness={max_brightness:.0f})",
                region=(0, 0, self.width, self.STATUS_BAR_H),
            ))
        return issues

    def check_nav_bar(self, img: np.ndarray) -> list[LayoutIssue]:
        """Verify nav bar is visible, has buttons, and isn't clipped."""
        issues = []
        nav_y = self.height - self.NAV_BAR_H
        nav = img[nav_y:, :, :]
        gray_nav = cv2.cvtColor(nav, cv2.COLOR_BGR2GRAY)

        # Nav bar should have visible elements. The Tritium nav bar has:
        # - A 2px cyan border at the top (brightness ~111)
        # - T_SURFACE3 background (brightness ~26)
        # - 3 cyan icon buttons (brightness ~171)
        # Mean brightness will be low (~30) due to dark theme — check max instead.
        mean_brightness = float(np.mean(gray_nav))
        max_brightness = float(np.max(gray_nav))

        if max_brightness < 20:
            issues.append(LayoutIssue(
                "nav_bar_invisible", Severity.ERROR,
                f"Nav bar region is completely dark (max={max_brightness:.0f}) — "
                "buttons won't be visible or tappable",
                region=(0, nav_y, self.width, self.NAV_BAR_H),
            ))
            return issues

        # Check for distinct bright spots (button icons).
        # Cyan icons on dark background will have pixels > 60 brightness.
        bright_mask = gray_nav > 60
        bright_ratio = float(np.mean(bright_mask))
        if bright_ratio < 0.003:
            issues.append(LayoutIssue(
                "nav_bar_no_buttons", Severity.ERROR,
                f"Nav bar has no bright elements (ratio={bright_ratio:.4f}) — "
                "button icons may be invisible",
                region=(0, nav_y, self.width, self.NAV_BAR_H),
            ))

        # Check for edge structure (borders, icons)
        nav_edges = cv2.Canny(gray_nav, 50, 150)
        edge_density = float(np.mean(nav_edges > 0))
        if edge_density < 0.001:
            issues.append(LayoutIssue(
                "nav_bar_flat", Severity.WARNING,
                f"Nav bar has very low edge density ({edge_density:.4f}) — "
                "buttons may be hard to distinguish",
                region=(0, nav_y, self.width, self.NAV_BAR_H),
            ))

        # Check nav bar has distinct border or visual separation.
        # Look for a bright border line (cyan) in the top few rows of nav bar
        # OR a brightness difference from the viewport above.
        nav_top_strip = gray_nav[:4, :]
        top_strip_max = float(np.max(nav_top_strip))
        viewport_bottom = img[nav_y - 10 : nav_y, :, :]
        vp_mean = float(np.mean(cv2.cvtColor(viewport_bottom, cv2.COLOR_BGR2GRAY)))
        nav_mean = float(np.mean(gray_nav[4:14, :]))
        has_border = top_strip_max > 80  # bright border line
        has_contrast = abs(nav_mean - vp_mean) >= 3

        if not has_border and not has_contrast:
            issues.append(LayoutIssue(
                "nav_bar_no_contrast", Severity.WARNING,
                f"Nav bar blends into viewport (no border, delta={abs(nav_mean - vp_mean):.1f})"
            ))

        return issues

    def check_content_overflow(self, img: np.ndarray,
                               has_nav_bar: bool) -> list[LayoutIssue]:
        """Detect content rendering in the nav bar zone (overflow/clipping)."""
        issues = []
        if not has_nav_bar:
            return issues

        nav_y = self.height - self.NAV_BAR_H
        # Check the 5px strip right above the nav bar — if it has dense content
        # that continues INTO the nav bar, content is overflowing.
        # Skip the top 4px of nav bar (border area) to avoid false positives.
        above_nav = img[nav_y - 5 : nav_y, :, :]
        inside_nav = img[nav_y + 4 : nav_y + 9, :, :]
        above_edges = float(
            np.mean(cv2.Canny(cv2.cvtColor(above_nav, cv2.COLOR_BGR2GRAY), 50, 150) > 0)
        )
        inside_edges = float(
            np.mean(cv2.Canny(cv2.cvtColor(inside_nav, cv2.COLOR_BGR2GRAY), 50, 150) > 0)
        )

        if above_edges > 0.01 and inside_edges > 0.01:
            ratio = inside_edges / above_edges
            if ratio > 0.5:
                issues.append(LayoutIssue(
                    "content_overflow", Severity.WARNING,
                    f"Content may be overflowing into nav bar "
                    f"(edge ratio above/in = {above_edges:.3f}/{inside_edges:.3f})",
                    region=(0, nav_y - 5, self.width, 10),
                ))

        return issues

    def check_button_accessibility(self, img: np.ndarray,
                                   has_nav_bar: bool) -> list[LayoutIssue]:
        """Verify that interactive elements (buttons) are large enough to tap."""
        issues = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if has_nav_bar:
            viewport_bottom_y = self.height - self.NAV_BAR_H - 5
        else:
            viewport_bottom_y = self.height - 5

        # Look for rectangular bright regions near the bottom (likely buttons)
        bottom_strip = gray[viewport_bottom_y - 60 : viewport_bottom_y, :]
        _, thresh = cv2.threshold(bottom_strip, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            actual_y = viewport_bottom_y - 60 + y

            if has_nav_bar and actual_y + h > (self.height - self.NAV_BAR_H):
                if w > 40 and h > 15:  # Button-sized
                    issues.append(LayoutIssue(
                        "button_clipped", Severity.ERROR,
                        f"Button-like element ({w}x{h}) at y={actual_y} "
                        f"extends into nav bar zone (y>{self.height - self.NAV_BAR_H})",
                        region=(x, actual_y, w, h),
                    ))

        return issues

    def check_text_rendering(self, img: np.ndarray) -> list[LayoutIssue]:
        """Detect potential text rendering issues (overlapping, garbled)."""
        issues = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Look for regions with extremely high edge density (garbled/overlapping text).
        # Normal text on dark bg has edge density 0.05-0.20. Truly garbled text
        # (overlapping renders, corrupted glyphs) pushes above 0.25.
        # Raise threshold from 0.15 to 0.25 to avoid flagging normal dense text.
        cell_h, cell_w = 40, 100
        garbled_cells = 0
        for y in range(self.STATUS_BAR_H, self.height - self.NAV_BAR_H, cell_h):
            for x in range(0, self.width, cell_w):
                cell = gray[y : y + cell_h, x : x + cell_w]
                if cell.size == 0:
                    continue
                edges = cv2.Canny(cell, 50, 150)
                density = float(np.mean(edges > 0))
                mean_val = float(np.mean(cell))

                # Very high edge density in a non-bright area = likely garbled
                if density > 0.25 and mean_val < 100:
                    garbled_cells += 1

        # Only flag if multiple cells are garbled (isolated high-density cells
        # are likely just dense but legible text labels)
        if garbled_cells >= 3:
            issues.append(LayoutIssue(
                "text_garbled", Severity.WARNING,
                f"{garbled_cells} cells with very high edge density — "
                "possible text overlap or rendering artifact"
            ))

        return issues

    def check_flicker(self, frames: list[np.ndarray],
                      app_name: str = "unknown") -> list[LayoutIssue]:
        """Detect UI flicker/glitching by comparing consecutive frames.

        Takes a list of BGR frames captured in rapid succession (~200-500ms apart)
        and looks for:
        - Large regions that toggle on/off between frames (flashing elements)
        - Text areas that shift position (jittering labels)
        - Regions that appear/disappear (rendering glitches)

        Expects at least 2 frames. More frames give better flicker detection.
        """
        issues = []
        if len(frames) < 2:
            return issues

        flicker_regions = []
        total_changed_ratios = []

        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i - 1])
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            changed_ratio = float(np.mean(gray_diff > 15))
            total_changed_ratios.append(changed_ratio)

            if changed_ratio < 0.001:
                continue

            _, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w * h > 500:
                    flicker_regions.append((x, y, w, h, changed_ratio))

        if not total_changed_ratios:
            return issues

        max_change = max(total_changed_ratios)
        avg_change = sum(total_changed_ratios) / len(total_changed_ratios)

        # Large-area flicker: >5% of screen changing between frames
        if max_change > 0.05:
            issues.append(LayoutIssue(
                "flicker_large", Severity.ERROR,
                f"Major screen flicker: {max_change:.1%} of pixels changed "
                f"between frames (avg={avg_change:.1%})"
            ))

        # Persistent flicker: consistent changes >1% across multiple frame pairs
        flicker_frame_count = sum(1 for r in total_changed_ratios if r > 0.01)
        if flicker_frame_count >= len(total_changed_ratios) * 0.5 and avg_change > 0.01:
            issues.append(LayoutIssue(
                "flicker_persistent", Severity.WARNING,
                f"Persistent flicker: {avg_change:.2%} avg pixel change across "
                f"{flicker_frame_count}/{len(total_changed_ratios)} frame pairs"
            ))

        # Elements that appear/disappear repeatedly
        if len(flicker_regions) > 0:
            large_flickers = [r for r in flicker_regions if r[2] * r[3] > 2000]
            if len(large_flickers) > len(frames):
                issues.append(LayoutIssue(
                    "flicker_elements", Severity.WARNING,
                    f"{len(large_flickers)} large flickering regions detected "
                    f"across {len(frames)} frames"
                ))

        return issues

    def check_event_tearing(
        self,
        pre_frame: np.ndarray,
        event_frames: list[np.ndarray],
        post_frame: np.ndarray,
        roi: tuple[int, int, int, int],
        event_name: str = "tap",
    ) -> list[LayoutIssue]:
        """Detect localized visual tearing around a UI event (e.g. button press).

        Captures the region of interest (ROI) before, during, and after an event
        and checks for:
        - Partial rendering: top half shows old state, bottom half shows new state
          within the same frame (horizontal tear line inside the ROI)
        - Ghosting: blended/intermediate pixel values that don't match either the
          pre or post state (incomplete buffer swap)
        - Boundary bleed: pixels outside the ROI changing when they shouldn't
          (overdraw from a partial flush)

        Args:
            pre_frame: Screenshot captured before the event (BGR uint8)
            event_frames: Screenshots captured during/immediately after the event
            post_frame: Screenshot captured after the UI has settled
            roi: (x, y, w, h) region of interest (the button/widget area)
            event_name: Human-readable name for the event (for error messages)

        Returns:
            List of LayoutIssue for any tearing detected.
        """
        issues = []
        if not event_frames:
            return issues

        rx, ry, rw, rh = roi

        # Crop ROI from pre/post reference frames
        pre_roi = pre_frame[ry : ry + rh, rx : rx + rw]
        post_roi = post_frame[ry : ry + rh, rx : rx + rw]

        # Check if the button actually changed state (pre != post)
        state_diff = cv2.absdiff(pre_roi, post_roi)
        state_change = float(np.mean(cv2.cvtColor(state_diff, cv2.COLOR_BGR2GRAY) > 10))

        # If pre and post are identical, no state transition occurred — skip
        if state_change < 0.01:
            return issues

        for fidx, frame in enumerate(event_frames):
            frame_roi = frame[ry : ry + rh, rx : rx + rw]

            # --- Check 1: Horizontal tear line (partial render) ---
            # Compare each row of the event frame to both pre and post.
            # A torn frame will have some rows matching pre and others matching post,
            # with a sharp transition between them.
            gray_frame = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_pre = cv2.cvtColor(pre_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_post = cv2.cvtColor(post_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Per-row similarity to pre vs post state
            row_diff_pre = np.mean(np.abs(gray_frame - gray_pre), axis=1)
            row_diff_post = np.mean(np.abs(gray_frame - gray_post), axis=1)

            # A row is "pre-like" if it's much closer to pre than post, and vice versa
            pre_threshold = 8.0
            pre_like = row_diff_pre < pre_threshold
            post_like = row_diff_post < pre_threshold

            # Look for a tear boundary: consecutive pre-like rows followed by
            # consecutive post-like rows (or vice versa)
            if rh >= 6:  # Need enough rows to detect a boundary
                transitions = 0
                last_state = None
                for row in range(rh):
                    if pre_like[row] and not post_like[row]:
                        current = "pre"
                    elif post_like[row] and not pre_like[row]:
                        current = "post"
                    else:
                        current = None

                    if current and last_state and current != last_state:
                        transitions += 1
                    if current:
                        last_state = current

                # A clean transition has 0 or 1 transitions (fully pre, fully post,
                # or clean flip). A tear has exactly 1 transition in the middle
                # with both states present in the same frame.
                has_pre_rows = int(np.sum(pre_like & ~post_like))
                has_post_rows = int(np.sum(post_like & ~pre_like))

                if transitions >= 1 and has_pre_rows >= 2 and has_post_rows >= 2:
                    # Find the approximate tear line
                    for row in range(1, rh):
                        if pre_like[row - 1] and post_like[row]:
                            tear_y = ry + row
                            issues.append(LayoutIssue(
                                "event_tearing", Severity.ERROR,
                                f"Partial render during {event_name}: tear line at y={tear_y} "
                                f"in frame {fidx} — top={has_pre_rows} rows old state, "
                                f"bottom={has_post_rows} rows new state",
                                region=roi,
                            ))
                            break
                        if post_like[row - 1] and pre_like[row]:
                            tear_y = ry + row
                            issues.append(LayoutIssue(
                                "event_tearing", Severity.ERROR,
                                f"Partial render during {event_name}: tear line at y={tear_y} "
                                f"in frame {fidx} — reverse tear detected",
                                region=roi,
                            ))
                            break

            # --- Check 2: Ghosting (intermediate pixel values) ---
            # Pixels that don't match either pre or post within tolerance
            diff_from_pre = np.abs(gray_frame - gray_pre)
            diff_from_post = np.abs(gray_frame - gray_post)
            ghost_threshold = 15.0
            ghost_mask = (diff_from_pre > ghost_threshold) & (diff_from_post > ghost_threshold)
            ghost_ratio = float(np.mean(ghost_mask))

            if ghost_ratio > 0.15:
                issues.append(LayoutIssue(
                    "event_ghosting", Severity.WARNING,
                    f"Ghosting during {event_name}: {ghost_ratio:.1%} of ROI pixels "
                    f"don't match pre or post state in frame {fidx} — "
                    "possible incomplete buffer swap",
                    region=roi,
                ))

            # --- Check 3: Boundary bleed (changes outside the ROI) ---
            # Expand ROI by a margin and check if surrounding pixels changed
            margin = 10
            bx = max(0, rx - margin)
            by = max(0, ry - margin)
            bx2 = min(self.width, rx + rw + margin)
            by2 = min(self.height, ry + rh + margin)

            # Create a mask that excludes the original ROI
            surround_pre = pre_frame[by:by2, bx:bx2].copy()
            surround_evt = frame[by:by2, bx:bx2].copy()

            # Zero out the inner ROI in both so we only compare the surround
            inner_x = rx - bx
            inner_y = ry - by
            surround_pre[inner_y : inner_y + rh, inner_x : inner_x + rw] = 0
            surround_evt[inner_y : inner_y + rh, inner_x : inner_x + rw] = 0

            surround_diff = cv2.absdiff(surround_pre, surround_evt)
            surround_gray = cv2.cvtColor(surround_diff, cv2.COLOR_BGR2GRAY)
            bleed_ratio = float(np.mean(surround_gray > 10))

            if bleed_ratio > 0.05:
                issues.append(LayoutIssue(
                    "event_boundary_bleed", Severity.WARNING,
                    f"Boundary bleed during {event_name}: {bleed_ratio:.1%} of pixels "
                    f"outside ROI changed in frame {fidx} — possible overdraw",
                    region=(bx, by, bx2 - bx, by2 - by),
                ))

        return issues

    def check_animation_stability(
        self,
        frames: list[np.ndarray],
        expected_regions: list[tuple[int, int, int, int]] | None = None,
    ) -> list[LayoutIssue]:
        """Verify that subtle animations don't cause tearing or large-area flicker.

        Designed for validating "alive" animations (breathing dots, blinking colons)
        which should produce small, localized changes between frames — never
        large-area disruptions.

        Args:
            frames: Multiple frames captured at ~200-500ms intervals
            expected_regions: Optional list of (x, y, w, h) where animation is
                expected. Changes outside these regions are flagged.

        Returns:
            List of LayoutIssue for any problematic animation behavior.
        """
        issues = []
        if len(frames) < 2:
            return issues

        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i - 1])
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # Threshold for "significant" pixel change
            changed_mask = gray_diff > 12
            changed_ratio = float(np.mean(changed_mask))

            # Alive animations should cause <2% total screen change
            if changed_ratio > 0.02:
                issues.append(LayoutIssue(
                    "animation_too_large", Severity.ERROR,
                    f"Animation changed {changed_ratio:.1%} of screen between "
                    f"frames {i-1}-{i} — exceeds 2% threshold for subtle animation",
                ))

            # Check for tearing within animated regions
            if changed_ratio > 0.001:
                _, thresh = cv2.threshold(gray_diff, 12, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    if w * h < 50:
                        continue

                    # Check for horizontal tear lines within the animated region
                    region_diff = gray_diff[y : y + h, x : x + w]
                    row_means = np.mean(region_diff, axis=1)
                    row_diffs = np.abs(np.diff(row_means))

                    consecutive = 0
                    for rd in row_diffs:
                        if rd > 40:
                            consecutive += 1
                        else:
                            if consecutive >= 2:
                                issues.append(LayoutIssue(
                                    "animation_tearing", Severity.ERROR,
                                    f"Tearing in animated region at ({x},{y} {w}x{h}) "
                                    f"between frames {i-1}-{i}",
                                    region=(x, y, w, h),
                                ))
                            consecutive = 0

                    # If expected_regions provided, flag changes outside them
                    if expected_regions:
                        inside = False
                        for erx, ery, erw, erh in expected_regions:
                            if (x + w > erx and x < erx + erw and
                                    y + h > ery and y < ery + erh):
                                inside = True
                                break
                        if not inside and w * h > 200:
                            issues.append(LayoutIssue(
                                "animation_unexpected", Severity.WARNING,
                                f"Unexpected animation at ({x},{y} {w}x{h}) "
                                f"outside expected regions between frames {i-1}-{i}",
                                region=(x, y, w, h),
                            ))

        return issues

    # ------------------------------------------------------------------
    # Layout density checks
    # ------------------------------------------------------------------

    def check_empty_space(self, img: np.ndarray,
                          has_nav_bar: bool = True) -> list[LayoutIssue]:
        """Detect excessive empty/black space in the content viewport.

        Divides the content area (between status bar and nav bar) into a
        grid of cells and measures what fraction are nearly black. On a
        well-designed UI, content should use at least 25% of the viewport
        area. A launcher with tiny icons centered in a vast dark void fails
        this check.

        The algorithm:
          1. Extract the viewport region (below status bar, above nav bar).
          2. Divide it into NxN cells.
          3. A cell is "empty" if its mean brightness is below the dark
             background threshold.
          4. If >75% of cells are empty, the UI is wasting space.
        """
        issues = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        vp_top = self.STATUS_BAR_H
        vp_bottom = (self.height - self.NAV_BAR_H) if has_nav_bar else self.height
        viewport = gray[vp_top:vp_bottom, :]
        vp_h, vp_w = viewport.shape

        if vp_h < 10 or vp_w < 10:
            return issues

        # Grid analysis — 8x8 cells across viewport
        cell_rows, cell_cols = 8, 8
        cell_h = vp_h // cell_rows
        cell_w = vp_w // cell_cols
        empty_cells = 0
        total_cells = cell_rows * cell_cols

        for r in range(cell_rows):
            for c in range(cell_cols):
                cell = viewport[r * cell_h:(r + 1) * cell_h,
                                c * cell_w:(c + 1) * cell_w]
                if np.mean(cell) < self.BG_DARK_THRESHOLD:
                    empty_cells += 1

        empty_ratio = empty_cells / total_cells

        # Also measure the bounding box of all non-dark content
        content_mask = viewport > self.BG_DARK_THRESHOLD
        content_rows = np.any(content_mask, axis=1)
        content_cols = np.any(content_mask, axis=0)

        if np.any(content_rows) and np.any(content_cols):
            row_min = int(np.argmax(content_rows))
            row_max = vp_h - int(np.argmax(content_rows[::-1])) - 1
            col_min = int(np.argmax(content_cols))
            col_max = vp_w - int(np.argmax(content_cols[::-1])) - 1
            content_bbox_area = (row_max - row_min) * (col_max - col_min)
            viewport_area = vp_h * vp_w
            fill_ratio = content_bbox_area / viewport_area if viewport_area > 0 else 0
        else:
            fill_ratio = 0.0

        # Use both metrics: grid cell emptiness and bounding box fill.
        # A high fill_ratio (>80%) means content spans most of the viewport
        # even if individual cells are dark (e.g. dark-themed card backgrounds).
        if fill_ratio >= 0.80:
            # Content fills the space — only flag if nearly ALL cells are empty
            # (which would mean a tiny bright element in one corner, not real fill)
            if empty_ratio > 0.95:
                issues.append(LayoutIssue(
                    "excessive_empty_space", Severity.WARNING,
                    f"{empty_ratio:.0%} of viewport cells are empty — "
                    f"bounding box fills {fill_ratio:.0%} but content is sparse.",
                    region=(0, vp_top, vp_w, vp_h),
                ))
        elif empty_ratio > 0.85:
            issues.append(LayoutIssue(
                "excessive_empty_space", Severity.ERROR,
                f"{empty_ratio:.0%} of viewport cells are empty — "
                f"UI elements use only {fill_ratio:.0%} of available space. "
                f"Content should scale to fill the screen.",
                region=(0, vp_top, vp_w, vp_h),
            ))
        elif empty_ratio > 0.75:
            issues.append(LayoutIssue(
                "excessive_empty_space", Severity.WARNING,
                f"{empty_ratio:.0%} of viewport cells are empty — "
                f"consider enlarging UI elements to use available space.",
                region=(0, vp_top, vp_w, vp_h),
            ))

        return issues

    def check_density(self, img: np.ndarray,
                      has_nav_bar: bool = True) -> list[LayoutIssue]:
        """Detect excessively dense UI with overlapping elements.

        Scans the content area for regions where edge density is
        abnormally high, indicating overlapping widgets, text on text,
        or elements crammed too close together.

        The algorithm:
          1. Extract the viewport and compute Canny edges.
          2. Divide into cells (smaller than empty-space cells for precision).
          3. A cell is "overcrowded" if its edge density exceeds a threshold
             AND its mean brightness suggests visible content (not just noise
             on dark background).
          4. Adjacent overcrowded cells form "dense clusters" — if a cluster
             spans a significant area, elements are likely overlapping.
        """
        issues = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        vp_top = self.STATUS_BAR_H
        vp_bottom = (self.height - self.NAV_BAR_H) if has_nav_bar else self.height
        viewport = gray[vp_top:vp_bottom, :]
        vp_h, vp_w = viewport.shape

        if vp_h < 20 or vp_w < 20:
            return issues

        edges = cv2.Canny(viewport, 50, 150)

        # 16x16 cell grid for fine-grained density mapping
        cell_h = max(vp_h // 16, 1)
        cell_w = max(vp_w // 16, 1)
        rows = vp_h // cell_h
        cols = vp_w // cell_w

        # Build a density map
        dense_map = np.zeros((rows, cols), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                cell_edges = edges[r * cell_h:(r + 1) * cell_h,
                                   c * cell_w:(c + 1) * cell_w]
                cell_gray = viewport[r * cell_h:(r + 1) * cell_h,
                                     c * cell_w:(c + 1) * cell_w]
                density = float(np.mean(cell_edges > 0))
                brightness = float(np.mean(cell_gray))

                # High edge density + visible content = potential overlap
                if density > 0.30 and brightness > 25:
                    dense_map[r, c] = 255

        # Find connected clusters of dense cells
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dense_map, connectivity=8
        )

        for label_idx in range(1, num_labels):  # skip background
            area = stats[label_idx, cv2.CC_STAT_AREA]
            cx = stats[label_idx, cv2.CC_STAT_LEFT]
            cy = stats[label_idx, cv2.CC_STAT_TOP]
            cw = stats[label_idx, cv2.CC_STAT_WIDTH]
            ch = stats[label_idx, cv2.CC_STAT_HEIGHT]

            # A cluster of 6+ overcrowded cells indicates real overlap
            if area >= 6:
                pixel_x = cx * cell_w
                pixel_y = vp_top + cy * cell_h
                pixel_w = cw * cell_w
                pixel_h = ch * cell_h
                issues.append(LayoutIssue(
                    "ui_too_dense", Severity.ERROR,
                    f"Dense UI cluster ({area} cells, {pixel_w}x{pixel_h}px) — "
                    f"possible overlapping elements or text",
                    region=(pixel_x, pixel_y, pixel_w, pixel_h),
                ))
            elif area >= 4:
                pixel_x = cx * cell_w
                pixel_y = vp_top + cy * cell_h
                pixel_w = cw * cell_w
                pixel_h = ch * cell_h
                issues.append(LayoutIssue(
                    "ui_too_dense", Severity.WARNING,
                    f"Moderately dense UI cluster ({area} cells) — "
                    f"elements may be too close together",
                    region=(pixel_x, pixel_y, pixel_w, pixel_h),
                ))

        return issues
