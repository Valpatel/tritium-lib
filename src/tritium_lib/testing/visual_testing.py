# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""OpenCV-based visual testing framework for Tritium web UI screenshots.

Reusable analysis toolkit for verifying browser-rendered UI:
- Blank/white screen detection (rendering failures)
- UI element presence (header, sidebar, map, panels)
- Cyberpunk color scheme validation
- Text readability and contrast checking
- Element overlap detection
- Baseline comparison (structural similarity)
- Map tile verification (satellite imagery loaded)
- Marker detection (colored pins/dots on map)

All functions accept either a file path (str/Path) or a BGR numpy array.
Graceful degradation: returns safe defaults when OpenCV is not installed.

Design: each function is standalone and returns structured results.
ScreenshotAnalyzer composes them into a full report.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
    np = None


# ============================================================
# Cyberpunk palette constants (BGR order for OpenCV)
# ============================================================

CYBERPUNK_COLORS = {
    "cyan":    {"hex": "#00f0ff", "bgr": (255, 240, 0),   "hsv_range": ((80, 100, 100), (100, 255, 255))},
    "magenta": {"hex": "#ff2a6d", "bgr": (109, 42, 255),  "hsv_range": ((160, 100, 100), (180, 255, 255))},
    "green":   {"hex": "#05ffa1", "bgr": (161, 255, 5),   "hsv_range": ((45, 100, 100), (75, 255, 255))},
    "yellow":  {"hex": "#fcee0a", "bgr": (10, 238, 252),  "hsv_range": ((25, 100, 100), (35, 255, 255))},
}

# Minimum pixel coverage (%) for a color to be considered "present"
COLOR_PRESENCE_THRESHOLD = 0.05


# ============================================================
# Data structures
# ============================================================

@dataclass
class BlankScreenResult:
    """Result of blank screen check."""
    is_blank: bool
    blank_type: str = ""  # "black", "white", "uniform", ""
    dark_pct: float = 0.0
    bright_pct: float = 0.0
    mean_brightness: float = 0.0
    std_brightness: float = 0.0


@dataclass
class UIElementResult:
    """Result of UI element detection."""
    has_header: bool = False
    has_sidebar: bool = False
    has_map_area: bool = False
    has_footer: bool = False
    panel_count: int = 0
    button_count: int = 0
    detected_regions: List[Tuple[str, int, int, int, int]] = field(default_factory=list)
    # Each region: (label, x, y, w, h)


@dataclass
class ColorDistributionResult:
    """Result of color distribution analysis."""
    has_cyan: bool = False
    has_magenta: bool = False
    has_green: bool = False
    has_yellow: bool = False
    cyan_pct: float = 0.0
    magenta_pct: float = 0.0
    green_pct: float = 0.0
    yellow_pct: float = 0.0
    dominant_colors: List[str] = field(default_factory=list)  # hex strings
    is_cyberpunk: bool = False  # at least 2 signature colors present


@dataclass
class TextReadabilityResult:
    """Result of text readability analysis."""
    text_region_count: int = 0
    low_contrast_count: int = 0
    avg_contrast_ratio: float = 0.0
    readable: bool = True
    regions: List[dict] = field(default_factory=list)
    # Each region: {"x", "y", "w", "h", "contrast", "readable"}


@dataclass
class OverlapResult:
    """Result of element overlap detection."""
    overlap_count: int = 0
    overlaps: List[dict] = field(default_factory=list)
    # Each overlap: {"a": (x,y,w,h), "b": (x,y,w,h), "overlap_pct": float}
    has_critical_overlap: bool = False


@dataclass
class BaselineComparisonResult:
    """Result of baseline image comparison."""
    ssim_score: float = 0.0  # 0.0 = completely different, 1.0 = identical
    mse: float = 0.0
    changed_pct: float = 0.0
    matches_baseline: bool = False
    diff_regions: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class MapTileResult:
    """Result of map tile detection."""
    has_tiles: bool = False
    tile_coverage_pct: float = 0.0
    is_blank_map: bool = True
    texture_score: float = 0.0  # higher = more textured (satellite imagery)
    unique_color_count: int = 0


@dataclass
class MarkerResult:
    """Result of marker detection on map."""
    marker_count: int = 0
    markers: List[dict] = field(default_factory=list)
    # Each marker: {"x", "y", "w", "h", "color_hex"}
    colors_found: List[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Complete analysis report from ScreenshotAnalyzer."""
    blank_screen: Optional[BlankScreenResult] = None
    ui_elements: Optional[UIElementResult] = None
    color_distribution: Optional[ColorDistributionResult] = None
    text_readability: Optional[TextReadabilityResult] = None
    overlap: Optional[OverlapResult] = None
    baseline: Optional[BaselineComparisonResult] = None
    map_tiles: Optional[MapTileResult] = None
    markers: Optional[MarkerResult] = None
    passed: bool = True
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================
# Image loading
# ============================================================

def _load_image(img: Union[str, Path, "np.ndarray", None]) -> Optional["np.ndarray"]:
    """Load image from path or pass through numpy array."""
    if not HAS_OPENCV:
        return None
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        return img
    path = str(img)
    loaded = cv2.imread(path)
    return loaded


def _bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    """Convert BGR tuple to hex string."""
    return f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"


# ============================================================
# Check functions
# ============================================================

def check_blank_screen(img: Union[str, Path, "np.ndarray"]) -> BlankScreenResult:
    """Detect all-black, all-white, or uniform-color screens.

    These indicate rendering failures where the browser loaded but
    nothing rendered, or the page crashed to a blank state.
    """
    result = BlankScreenResult(is_blank=False)
    loaded = _load_image(img)
    if loaded is None:
        result.is_blank = True
        result.blank_type = "load_failed"
        return result

    gray = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total = h * w

    result.mean_brightness = float(np.mean(gray))
    result.std_brightness = float(np.std(gray))
    result.dark_pct = float(np.count_nonzero(gray < 20) / total * 100)
    result.bright_pct = float(np.count_nonzero(gray > 235) / total * 100)

    # All black
    if result.dark_pct > 95:
        result.is_blank = True
        result.blank_type = "black"
        return result

    # All white
    if result.bright_pct > 95:
        result.is_blank = True
        result.blank_type = "white"
        return result

    # Uniform color (very low standard deviation)
    if result.std_brightness < 3.0:
        result.is_blank = True
        result.blank_type = "uniform"
        return result

    return result


def check_ui_elements(img: Union[str, Path, "np.ndarray"],
                      header_height: int = 60,
                      sidebar_width: int = 250,
                      footer_height: int = 40) -> UIElementResult:
    """Detect presence of expected UI panels (header, sidebar, map area).

    Uses edge detection and contour analysis to find rectangular UI regions.
    Header is expected in the top strip, sidebar on the left, map in the
    remaining area.
    """
    result = UIElementResult()
    loaded = _load_image(img)
    if loaded is None:
        return result

    h, w = loaded.shape[:2]
    gray = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)

    # --- Header detection ---
    # The header strip should have content (edges, text, icons)
    header_region = gray[:header_height, :]
    header_edges = cv2.Canny(header_region, 30, 100)
    header_edge_density = float(np.mean(header_edges > 0))
    if header_edge_density > 0.01:
        result.has_header = True
        result.detected_regions.append(("header", 0, 0, w, header_height))

    # --- Sidebar detection ---
    # Left strip should have distinct brightness from the main content area
    sidebar_region = gray[header_height:, :sidebar_width]
    main_region = gray[header_height:, sidebar_width:]
    sidebar_mean = float(np.mean(sidebar_region))
    main_mean = float(np.mean(main_region))
    sidebar_edges = cv2.Canny(sidebar_region, 30, 100)
    sidebar_edge_density = float(np.mean(sidebar_edges > 0))

    # Sidebar is present if it has edges AND differs from main area
    if sidebar_edge_density > 0.005 and abs(sidebar_mean - main_mean) > 5:
        result.has_sidebar = True
        result.detected_regions.append(("sidebar", 0, header_height, sidebar_width, h - header_height))

    # --- Map area detection ---
    # The main content area should have texture (satellite tiles, roads, etc.)
    map_x = sidebar_width if result.has_sidebar else 0
    map_region = gray[header_height:h - footer_height, map_x:]
    map_edges = cv2.Canny(map_region, 30, 100)
    map_edge_density = float(np.mean(map_edges > 0))
    if map_edge_density > 0.005:
        result.has_map_area = True
        result.detected_regions.append(("map", map_x, header_height,
                                        w - map_x, h - header_height - footer_height))

    # --- Footer detection ---
    footer_region = gray[h - footer_height:, :]
    footer_edges = cv2.Canny(footer_region, 30, 100)
    footer_edge_density = float(np.mean(footer_edges > 0))
    if footer_edge_density > 0.01:
        result.has_footer = True
        result.detected_regions.append(("footer", 0, h - footer_height, w, footer_height))

    # --- Panel detection (contour-based) ---
    edges = cv2.Canny(gray, 40, 120)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = h * w
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        area_pct = area / total_area * 100
        if 0.5 < area_pct < 40 and 0.2 < (cw / max(ch, 1)) < 6.0:
            result.panel_count += 1
            result.detected_regions.append(("panel", x, y, cw, ch))

    # --- Button detection (small bright rectangles) ---
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if 500 < area < 15000 and 0.5 < (cw / max(ch, 1)) < 5.0:
            region_mean = float(np.mean(gray[y:y + ch, x:x + cw]))
            if region_mean > 40:  # brighter than background
                result.button_count += 1

    return result


def check_color_distribution(img: Union[str, Path, "np.ndarray"]) -> ColorDistributionResult:
    """Verify cyberpunk color scheme is present.

    Checks for the signature Tritium colors: cyan (#00f0ff), magenta (#ff2a6d),
    green (#05ffa1), yellow (#fcee0a). At least 2 must be present to qualify
    as cyberpunk.
    """
    result = ColorDistributionResult()
    loaded = _load_image(img)
    if loaded is None:
        return result

    hsv = cv2.cvtColor(loaded, cv2.COLOR_BGR2HSV)
    h, w = loaded.shape[:2]
    total = h * w

    for name, info in CYBERPUNK_COLORS.items():
        low, high = info["hsv_range"]
        mask = cv2.inRange(hsv, np.array(low, dtype=np.uint8),
                           np.array(high, dtype=np.uint8))
        pct = float(np.count_nonzero(mask) / total * 100)
        setattr(result, f"{name}_pct", round(pct, 3))
        present = pct > COLOR_PRESENCE_THRESHOLD
        setattr(result, f"has_{name}", present)

    # Count how many signature colors are present
    present_count = sum([result.has_cyan, result.has_magenta,
                         result.has_green, result.has_yellow])
    result.is_cyberpunk = present_count >= 2

    # Dominant colors via k-means
    small = cv2.resize(loaded, (64, 64))
    pixels = small.reshape(-1, 3).astype(np.float32)
    k = min(5, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    order = np.argsort(-counts)
    result.dominant_colors = [_bgr_to_hex(tuple(int(c) for c in centers[i]))
                              for i in order[:5]]

    return result


def check_text_readability(img: Union[str, Path, "np.ndarray"],
                           min_contrast: float = 3.0) -> TextReadabilityResult:
    """Detect text regions and verify they have sufficient contrast.

    Uses morphological analysis to find horizontal text regions, then
    measures foreground/background contrast for each region.
    WCAG AA minimum contrast ratio is 4.5:1 for normal text, 3:1 for large text.
    We use 3.0 as the threshold since this is a tactical UI with large text.
    """
    result = TextReadabilityResult()
    loaded = _load_image(img)
    if loaded is None:
        return result

    gray = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find text-like regions: bright horizontal strips
    _, bright = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(bright, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        # Text is wide and short
        if cw < 20 or ch < 5 or ch > 60 or cw / max(ch, 1) < 1.5:
            continue

        result.text_region_count += 1
        region = gray[y:y + ch, x:x + cw]

        # Calculate contrast: luminance of foreground vs background
        # Use Otsu's threshold to separate text from background
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_pixels = region[binary > 127]
        bg_pixels = region[binary <= 127]

        if len(fg_pixels) == 0 or len(bg_pixels) == 0:
            continue

        fg_lum = float(np.mean(fg_pixels)) / 255.0
        bg_lum = float(np.mean(bg_pixels)) / 255.0

        # WCAG relative luminance contrast ratio
        l1 = max(fg_lum, bg_lum) + 0.05
        l2 = min(fg_lum, bg_lum) + 0.05
        contrast = l1 / l2

        readable = contrast >= min_contrast
        if not readable:
            result.low_contrast_count += 1

        result.regions.append({
            "x": x, "y": y, "w": cw, "h": ch,
            "contrast": round(contrast, 2),
            "readable": readable,
        })

    if result.regions:
        result.avg_contrast_ratio = round(
            sum(r["contrast"] for r in result.regions) / len(result.regions), 2
        )

    result.readable = result.low_contrast_count == 0
    return result


def check_element_overlap(img: Union[str, Path, "np.ndarray"],
                          min_overlap_pct: float = 25.0) -> OverlapResult:
    """Detect overlapping UI elements.

    Finds rectangular regions (panels, buttons, tooltips) and checks if
    any pair overlaps by more than min_overlap_pct of the smaller element.
    """
    result = OverlapResult()
    loaded = _load_image(img)
    if loaded is None:
        return result

    gray = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total = h * w

    # Find rectangular UI elements via edge detection
    edges = cv2.Canny(gray, 40, 120)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        area_pct = area / total * 100
        if 0.3 < area_pct < 50 and 0.15 < (cw / max(ch, 1)) < 7.0:
            rects.append((x, y, cw, ch))

    # Check all pairs for overlap
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            ax, ay, aw, ah = rects[i]
            bx, by, bw, bh = rects[j]

            # Calculate overlap
            dx = min(ax + aw, bx + bw) - max(ax, bx)
            dy = min(ay + ah, by + bh) - max(ay, by)
            if dx <= 0 or dy <= 0:
                continue

            overlap_area = dx * dy
            smaller_area = min(aw * ah, bw * bh)
            if smaller_area == 0:
                continue

            overlap_pct = overlap_area / smaller_area * 100
            if overlap_pct >= min_overlap_pct:
                result.overlap_count += 1
                result.overlaps.append({
                    "a": rects[i],
                    "b": rects[j],
                    "overlap_pct": round(overlap_pct, 1),
                })
                if overlap_pct > 80:
                    result.has_critical_overlap = True

    return result


def compare_baseline(img: Union[str, Path, "np.ndarray"],
                     baseline: Union[str, Path, "np.ndarray"],
                     ssim_threshold: float = 0.85) -> BaselineComparisonResult:
    """Compare screenshot against a baseline using structural similarity.

    Uses a simplified SSIM (Structural Similarity Index) computation
    and pixel-level difference analysis. Returns regions that differ
    significantly from the baseline.

    Args:
        img: Current screenshot
        baseline: Reference/baseline screenshot
        ssim_threshold: Minimum SSIM to consider a match (0.0 to 1.0)
    """
    result = BaselineComparisonResult()
    loaded = _load_image(img)
    base = _load_image(baseline)
    if loaded is None or base is None:
        return result

    # Resize to match if needed
    if loaded.shape != base.shape:
        base = cv2.resize(base, (loaded.shape[1], loaded.shape[0]))

    gray_a = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_b = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # MSE
    diff = gray_a - gray_b
    result.mse = float(np.mean(diff ** 2))

    # Simplified SSIM
    # Constants for numerical stability
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(gray_a ** 2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(gray_b ** 2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)

    ssim_map = numerator / denominator
    result.ssim_score = float(np.mean(ssim_map))

    # Changed pixel percentage
    abs_diff = np.abs(gray_a - gray_b)
    changed_mask = abs_diff > 25
    result.changed_pct = float(np.count_nonzero(changed_mask) / gray_a.size * 100)

    result.matches_baseline = result.ssim_score >= ssim_threshold

    # Find contiguous diff regions
    changed_uint8 = changed_mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(changed_uint8, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch > 100:  # ignore tiny pixel noise
            result.diff_regions.append((x, y, cw, ch))

    return result


def detect_map_tiles(img: Union[str, Path, "np.ndarray"],
                     map_region: Optional[Tuple[int, int, int, int]] = None) -> MapTileResult:
    """Verify satellite imagery loaded (map is not blank).

    Analyzes the map region for texture patterns that indicate loaded
    satellite/terrain tiles vs. blank gray/white unfilled tiles.

    Args:
        img: Screenshot image
        map_region: Optional (x, y, w, h) of the map area. If None,
                    analyzes the center 60% of the image.
    """
    result = MapTileResult()
    loaded = _load_image(img)
    if loaded is None:
        return result

    h, w = loaded.shape[:2]

    if map_region:
        mx, my, mw, mh = map_region
        region = loaded[my:my + mh, mx:mx + mw]
    else:
        # Center 60% of image
        cx, cy = w // 5, h // 5
        region = loaded[cy:h - cy, cx:w - cx]

    if region.size == 0:
        return result

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    rh, rw = gray.shape

    # Texture score: Laplacian variance (high = textured imagery)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    result.texture_score = float(np.var(laplacian))

    # Unique colors in the region (downsampled for speed)
    small = cv2.resize(region, (min(rw, 100), min(rh, 100)))
    unique = len(np.unique(small.reshape(-1, 3), axis=0))
    result.unique_color_count = unique

    # Tile coverage: non-uniform areas (edges present = tiles loaded)
    edges = cv2.Canny(gray, 20, 60)
    edge_density = float(np.mean(edges > 0))
    result.tile_coverage_pct = round(edge_density * 100, 2)

    # A blank map has low texture, few unique colors, and few edges
    result.has_tiles = (result.texture_score > 50 and
                        result.unique_color_count > 100 and
                        edge_density > 0.01)
    result.is_blank_map = not result.has_tiles

    return result


def detect_markers(img: Union[str, Path, "np.ndarray"],
                   min_size: int = 4,
                   max_size: int = 60) -> MarkerResult:
    """Count colored markers on the map.

    Detects small bright colored dots/pins that represent tracked targets
    or sensor nodes on the map. Uses HSV color space to find saturated,
    bright regions within the size range.

    Args:
        img: Screenshot image
        min_size: Minimum marker dimension in pixels
        max_size: Maximum marker dimension in pixels
    """
    result = MarkerResult()
    loaded = _load_image(img)
    if loaded is None:
        return result

    hsv = cv2.cvtColor(loaded, cv2.COLOR_BGR2HSV)

    # Find saturated + bright pixels (colored elements, not gray/black/white)
    mask = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([180, 255, 255]))

    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colors_seen = set()
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_size or ch < min_size or cw > max_size or ch > max_size:
            continue

        # Get mean color of this marker
        marker_region = loaded[y:y + ch, x:x + cw]
        if marker_region.size == 0:
            continue
        mean_color = tuple(int(c) for c in np.mean(marker_region.reshape(-1, 3), axis=0))
        hex_color = _bgr_to_hex(mean_color)

        result.markers.append({
            "x": x, "y": y, "w": cw, "h": ch,
            "color_hex": hex_color,
        })
        colors_seen.add(hex_color)

    result.marker_count = len(result.markers)
    result.colors_found = sorted(colors_seen)
    return result


# ============================================================
# ScreenshotAnalyzer — composes all checks
# ============================================================

class ScreenshotAnalyzer:
    """Loads a screenshot and runs multiple visual checks.

    Usage:
        analyzer = ScreenshotAnalyzer()
        report = analyzer.analyze("screenshot.png")
        assert report.passed, report.failures

        # Or run individual checks:
        blank = analyzer.check_blank("screenshot.png")
        assert not blank.is_blank

    Configuration:
        analyzer = ScreenshotAnalyzer(
            header_height=80,
            sidebar_width=300,
            min_contrast=4.5,
            ssim_threshold=0.9,
        )
    """

    def __init__(self,
                 header_height: int = 60,
                 sidebar_width: int = 250,
                 footer_height: int = 40,
                 min_contrast: float = 3.0,
                 ssim_threshold: float = 0.85,
                 marker_min_size: int = 4,
                 marker_max_size: int = 60,
                 overlap_threshold: float = 25.0):
        self.header_height = header_height
        self.sidebar_width = sidebar_width
        self.footer_height = footer_height
        self.min_contrast = min_contrast
        self.ssim_threshold = ssim_threshold
        self.marker_min_size = marker_min_size
        self.marker_max_size = marker_max_size
        self.overlap_threshold = overlap_threshold

    def check_blank(self, img: Union[str, Path, "np.ndarray"]) -> BlankScreenResult:
        """Check for blank/black/white screens."""
        return check_blank_screen(img)

    def check_elements(self, img: Union[str, Path, "np.ndarray"]) -> UIElementResult:
        """Check for expected UI elements."""
        return check_ui_elements(img, self.header_height,
                                 self.sidebar_width, self.footer_height)

    def check_colors(self, img: Union[str, Path, "np.ndarray"]) -> ColorDistributionResult:
        """Check cyberpunk color scheme."""
        return check_color_distribution(img)

    def check_text(self, img: Union[str, Path, "np.ndarray"]) -> TextReadabilityResult:
        """Check text readability."""
        return check_text_readability(img, self.min_contrast)

    def check_overlaps(self, img: Union[str, Path, "np.ndarray"]) -> OverlapResult:
        """Check for element overlaps."""
        return check_element_overlap(img, self.overlap_threshold)

    def check_baseline(self, img: Union[str, Path, "np.ndarray"],
                       baseline: Union[str, Path, "np.ndarray"]) -> BaselineComparisonResult:
        """Compare against baseline."""
        return compare_baseline(img, baseline, self.ssim_threshold)

    def check_map(self, img: Union[str, Path, "np.ndarray"],
                  map_region: Optional[Tuple[int, int, int, int]] = None) -> MapTileResult:
        """Check map tiles loaded."""
        return detect_map_tiles(img, map_region)

    def check_markers(self, img: Union[str, Path, "np.ndarray"]) -> MarkerResult:
        """Detect markers on the map."""
        return detect_markers(img, self.marker_min_size, self.marker_max_size)

    def analyze(self, img: Union[str, Path, "np.ndarray"],
                baseline: Optional[Union[str, Path, "np.ndarray"]] = None,
                map_region: Optional[Tuple[int, int, int, int]] = None,
                checks: Optional[List[str]] = None) -> AnalysisReport:
        """Run all (or selected) checks and produce a unified report.

        Args:
            img: Screenshot to analyze
            baseline: Optional baseline image for comparison
            map_region: Optional (x, y, w, h) for map tile checks
            checks: Optional list of check names to run. If None, runs all.
                    Valid: "blank", "elements", "colors", "text",
                           "overlap", "baseline", "map", "markers"
        """
        report = AnalysisReport()
        all_checks = checks is None

        # 1. Blank screen (always run — fundamental check)
        if all_checks or "blank" in checks:
            report.blank_screen = self.check_blank(img)
            if report.blank_screen.is_blank:
                report.passed = False
                report.failures.append(
                    f"Blank screen detected: {report.blank_screen.blank_type}"
                )
                # If screen is blank, other checks are meaningless
                if report.blank_screen.blank_type in ("black", "white", "load_failed"):
                    return report

        # 2. UI elements
        if all_checks or "elements" in checks:
            report.ui_elements = self.check_elements(img)
            if not report.ui_elements.has_header and not report.ui_elements.has_map_area:
                report.warnings.append("No header or map area detected")

        # 3. Color distribution
        if all_checks or "colors" in checks:
            report.color_distribution = self.check_colors(img)

        # 4. Text readability
        if all_checks or "text" in checks:
            report.text_readability = self.check_text(img)
            if not report.text_readability.readable:
                report.warnings.append(
                    f"{report.text_readability.low_contrast_count} low-contrast text regions"
                )

        # 5. Element overlap
        if all_checks or "overlap" in checks:
            report.overlap = self.check_overlaps(img)
            if report.overlap.has_critical_overlap:
                report.passed = False
                report.failures.append(
                    f"Critical UI element overlap detected ({report.overlap.overlap_count} overlaps)"
                )

        # 6. Baseline comparison
        if baseline is not None and (all_checks or "baseline" in checks):
            report.baseline = self.check_baseline(img, baseline)
            if not report.baseline.matches_baseline:
                report.warnings.append(
                    f"Screenshot differs from baseline (SSIM={report.baseline.ssim_score:.3f})"
                )

        # 7. Map tiles
        if all_checks or "map" in checks:
            report.map_tiles = self.check_map(img, map_region)

        # 8. Markers
        if all_checks or "markers" in checks:
            report.markers = self.check_markers(img)

        return report
