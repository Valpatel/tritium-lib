# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Visual analysis toolkit for UI testing.

Three concerns, kept separate:

1. RENDERING MODES — tell the UI to render in analysis-friendly styles
2. DETECTORS — find specific UI patterns in screenshots/video frames
3. COMPARATORS — compare screenshots to detect changes

All functions take file paths or numpy arrays as input. No browser interaction.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import json

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ============================================================
# Data structures
# ============================================================

@dataclass
class BoundingBox:
    """A rectangle in pixel coordinates."""
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self): return self.w * self.h
    @property
    def center(self): return (self.x + self.w // 2, self.y + self.h // 2)
    def contains(self, px, py): return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


@dataclass
class DetectedElement:
    """A UI element found by a detector."""
    element_type: str          # panel, button, marker, overlay, text, etc.
    bbox: BoundingBox
    confidence: float = 1.0    # 0-1
    label: str = ""            # detected text or classification
    color: str = ""            # dominant color hex
    properties: dict = field(default_factory=dict)


@dataclass
class FrameAnalysis:
    """Complete analysis of a single frame/screenshot."""
    path: str
    elements: List[DetectedElement] = field(default_factory=list)
    is_black_screen: bool = False
    dominant_colors: List[str] = field(default_factory=list)
    text_regions: List[DetectedElement] = field(default_factory=list)
    pixel_stats: dict = field(default_factory=dict)

    def elements_of_type(self, t): return [e for e in self.elements if e.element_type == t]
    def count(self, t): return len(self.elements_of_type(t))


@dataclass
class VideoDiff:
    """Result of analyzing a video sequence (multiple frames)."""
    frame_count: int
    fps_estimate: float
    motion_frames: int             # frames with significant change from previous
    static_frames: int
    anomaly_frames: List[int] = field(default_factory=list)  # frame indices with issues
    summary: str = ""


# ============================================================
# 1. RENDERING MODES — request specific render styles from UI
# ============================================================

class RenderMode:
    """Constants for analysis-friendly rendering modes.

    These are passed to the UI via URL params or API calls.
    The UI renders differently to make detection easier.
    """
    NORMAL = "normal"                # Default visual style
    SOLID_BLOCKS = "solid_blocks"    # Each UI element as a solid colored rectangle
    WIREFRAME = "wireframe"          # Just outlines, no fills
    HEATMAP = "heatmap"              # Activity/interaction heatmap
    DEPTH = "depth"                  # Z-index visualization (brighter = higher z)
    SEMANTIC = "semantic"            # Each element type gets a unique color
    HIGH_CONTRAST = "high_contrast"  # Black bg, bright elements only
    ID_MAP = "id_map"               # Each element has unique RGB = element ID

    # Color assignments for SEMANTIC mode (element_type → BGR)
    SEMANTIC_COLORS = {
        "panel":     (255, 240, 0),    # cyan
        "button":    (0, 255, 161),    # green
        "menu":      (10, 238, 252),   # yellow
        "map":       (30, 30, 30),     # dark gray
        "overlay":   (109, 42, 255),   # magenta
        "text":      (255, 255, 255),  # white
        "marker":    (0, 165, 255),    # orange
        "toast":     (100, 200, 255),  # light orange
    }

    @classmethod
    def url_param(cls, mode):
        """Generate URL parameter for requesting a render mode."""
        return f"?render_mode={mode}"

    @classmethod
    def api_body(cls, mode):
        """Generate API request body for render mode."""
        return {"render_mode": mode}


# ============================================================
# 2. DETECTORS — find UI patterns in images
# ============================================================

def load_image(path_or_array):
    """Load an image from path or pass through numpy array."""
    if not HAS_OPENCV:
        return None
    if isinstance(path_or_array, np.ndarray):
        return path_or_array
    return cv2.imread(str(path_or_array))


def detect_panels(img) -> List[DetectedElement]:
    """Detect rectangular panel regions in a screenshot.

    Panels are dark rectangles with bright borders (cyan/magenta).
    """
    if not HAS_OPENCV:
        return []
    if isinstance(img, str):
        img = cv2.imread(img)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect edges (panel borders are strong edges)
    edges = cv2.Canny(gray, 50, 150)
    # Dilate to close gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    panels = []
    h, w = img.shape[:2]
    min_area = (w * h) * 0.005  # Panels are at least 0.5% of screen
    max_area = (w * h) * 0.6    # Panels are at most 60% of screen

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < min_area or area > max_area:
            continue
        # Panels are roughly rectangular (aspect ratio 0.3 to 3.0)
        aspect = cw / ch if ch > 0 else 0
        if aspect < 0.2 or aspect > 5.0:
            continue

        # Sample the border color
        border_color = _dominant_edge_color(img, x, y, cw, ch)

        panels.append(DetectedElement(
            element_type="panel",
            bbox=BoundingBox(x, y, cw, ch),
            color=_bgr_to_hex(border_color),
        ))

    return panels


def detect_markers(img, min_size=5, max_size=40) -> List[DetectedElement]:
    """Detect small colored markers/dots on the map (targets, units, etc.)."""
    if not HAS_OPENCV:
        return []
    if isinstance(img, str):
        img = cv2.imread(img)
    if img is None:
        return []

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    markers = []
    # Look for bright, saturated small regions
    # Saturation > 100, Value > 100 = colored, bright
    mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < min_size or ch < min_size or cw > max_size or ch > max_size:
            continue
        color = _dominant_color(img, x, y, cw, ch)
        markers.append(DetectedElement(
            element_type="marker",
            bbox=BoundingBox(x, y, cw, ch),
            color=_bgr_to_hex(color),
        ))

    return markers


def detect_text_regions(img) -> List[DetectedElement]:
    """Detect regions likely containing text (bright, horizontal clusters)."""
    if not HAS_OPENCV:
        return []
    if isinstance(img, str):
        img = cv2.imread(img)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Text is bright on dark background in cyberpunk UI
    _, bright = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    # Horizontal dilation to connect characters into word regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(bright, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        # Text regions are wider than tall and small-ish
        if cw < 20 or ch < 5 or ch > 40 or cw / ch < 2:
            continue
        regions.append(DetectedElement(
            element_type="text",
            bbox=BoundingBox(x, y, cw, ch),
        ))

    return regions


def detect_overlays(img) -> List[DetectedElement]:
    """Detect semi-transparent overlay regions (fog of war, territory zones, etc.)."""
    if not HAS_OPENCV:
        return []
    if isinstance(img, str):
        img = cv2.imread(img)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Overlays are large, semi-transparent areas — medium brightness, low contrast
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    # Areas where the blur and original are similar = uniform regions (overlays)
    diff = cv2.absdiff(gray, blurred)
    _, low_detail = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY_INV)

    # Large connected regions of low detail
    kernel = np.ones((20, 20), np.uint8)
    closed = cv2.morphologyEx(low_detail, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlays = []
    h, w = img.shape[:2]
    min_area = (w * h) * 0.05  # At least 5% of screen

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw * ch < min_area:
            continue
        color = _dominant_color(img, x, y, cw, ch)
        overlays.append(DetectedElement(
            element_type="overlay",
            bbox=BoundingBox(x, y, cw, ch),
            color=_bgr_to_hex(color),
        ))

    return overlays


# ============================================================
# 3. COMPARATORS — compare frames to detect changes
# ============================================================

def compare_screenshots(path_a, path_b, threshold=30):
    """Compare two screenshots pixel-by-pixel."""
    if not HAS_OPENCV:
        return None
    img_a = load_image(path_a)
    img_b = load_image(path_b)
    if img_a is None or img_b is None:
        return None
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    diff = cv2.absdiff(img_a, img_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    changed = int(np.count_nonzero(thresh))
    total = thresh.shape[0] * thresh.shape[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return {
        "changed_pixels": changed,
        "total_pixels": total,
        "change_percent": round(changed / total * 100, 2) if total > 0 else 0,
        "regions_changed": len(contours),
    }


def is_mostly_black(path, threshold=90.0):
    """Check if screenshot is mostly black (black screen detection)."""
    if not HAS_OPENCV:
        return False
    img = load_image(path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = np.count_nonzero(gray < 20)
    total = gray.shape[0] * gray.shape[1]
    return (dark / total * 100) > threshold


def detect_changes(baseline, current, min_percent=0.5):
    """Did something visually change between two frames?"""
    diff = compare_screenshots(baseline, current)
    if diff is None:
        return True
    return diff["change_percent"] > min_percent


def save_diff_image(path_a, path_b, output_path, threshold=30):
    """Save a visualization of differences between two images."""
    if not HAS_OPENCV:
        return None
    img_a = load_image(path_a)
    img_b = load_image(path_b)
    if img_a is None or img_b is None:
        return None
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    diff = cv2.absdiff(img_a, img_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    result = img_b.copy()
    result[mask > 0] = [255, 42, 109]  # Magenta highlight
    cv2.imwrite(str(output_path), result)
    return str(output_path)


# ============================================================
# 4. FULL FRAME ANALYSIS — combine all detectors
# ============================================================

def analyze_frame(path) -> FrameAnalysis:
    """Run all detectors on a single screenshot."""
    analysis = FrameAnalysis(path=str(path))

    if not HAS_OPENCV:
        return analysis

    img = load_image(path)
    if img is None:
        return analysis

    # Black screen check
    analysis.is_black_screen = is_mostly_black(path)

    # Pixel statistics
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    analysis.pixel_stats = {
        "mean_brightness": float(np.mean(gray)),
        "std_brightness": float(np.std(gray)),
        "dark_percent": float(np.count_nonzero(gray < 30) / gray.size * 100),
        "bright_percent": float(np.count_nonzero(gray > 200) / gray.size * 100),
    }

    # Dominant colors (top 5 by pixel count)
    analysis.dominant_colors = _top_colors(img, n=5)

    # Detect elements
    analysis.elements.extend(detect_panels(img))
    analysis.elements.extend(detect_markers(img))
    analysis.text_regions = detect_text_regions(img)
    analysis.elements.extend(detect_overlays(img))

    return analysis


def analyze_video(frame_paths: List[str], fps: float = 10.0) -> VideoDiff:
    """Analyze a sequence of frames for motion, anomalies, and stability."""
    if not HAS_OPENCV or len(frame_paths) < 2:
        return VideoDiff(frame_count=len(frame_paths), fps_estimate=fps,
                        motion_frames=0, static_frames=len(frame_paths))

    motion = 0
    static = 0
    anomalies = []
    prev = None

    for i, path in enumerate(frame_paths):
        img = load_image(path)
        if img is None:
            anomalies.append(i)
            continue

        if prev is not None:
            diff = compare_screenshots(prev, img)
            if diff and diff["change_percent"] > 0.5:
                motion += 1
            else:
                static += 1

            # Anomaly: sudden large change (>30%) or black screen
            if diff and diff["change_percent"] > 30:
                anomalies.append(i)
            if is_mostly_black(img):
                anomalies.append(i)

        prev = img

    return VideoDiff(
        frame_count=len(frame_paths),
        fps_estimate=fps,
        motion_frames=motion,
        static_frames=static,
        anomaly_frames=anomalies,
        summary=f"{motion} motion frames, {static} static, {len(anomalies)} anomalies",
    )


# ============================================================
# 5. VISION MODEL — semantic understanding via Ollama
# ============================================================

def describe_screenshot(path, prompt=None, model="llava:7b", ollama_url="http://localhost:11434"):
    """Ask a vision model to describe a screenshot."""
    if not HAS_REQUESTS:
        return {"success": False, "error": "requests not installed"}
    if prompt is None:
        prompt = "Describe this UI screenshot. What panels are visible? What's on the map? Any issues?"

    import base64
    try:
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        resp = _requests.post(f"{ollama_url}/api/generate", json={
            "model": model, "prompt": prompt, "images": [img_b64], "stream": False,
        }, timeout=60)
        if resp.ok:
            return {"success": True, "description": resp.json().get("response", ""), "model": model}
        return {"success": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# Internal helpers
# ============================================================

def _bgr_to_hex(bgr):
    if bgr is None or len(bgr) < 3:
        return "#000000"
    return f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"


def _dominant_color(img, x, y, w, h):
    """Get the most common color in a region."""
    region = img[y:y+h, x:x+w]
    if region.size == 0:
        return (0, 0, 0)
    pixels = region.reshape(-1, 3)
    # Simple: take the mean
    return tuple(int(c) for c in np.mean(pixels, axis=0))


def _dominant_edge_color(img, x, y, w, h, border_width=3):
    """Get the dominant color along the edges of a rectangle."""
    pixels = []
    for edge in [
        img[y:y+border_width, x:x+w],           # top
        img[y+h-border_width:y+h, x:x+w],       # bottom
        img[y:y+h, x:x+border_width],            # left
        img[y:y+h, x+w-border_width:x+w],        # right
    ]:
        if edge.size > 0:
            pixels.append(edge.reshape(-1, 3))
    if not pixels:
        return (0, 0, 0)
    all_pixels = np.vstack(pixels)
    return tuple(int(c) for c in np.mean(all_pixels, axis=0))


def _top_colors(img, n=5):
    """Get top N dominant colors as hex strings."""
    if img is None:
        return []
    # Downsample for speed
    small = cv2.resize(img, (50, 50))
    pixels = small.reshape(-1, 3).astype(np.float32)
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = min(n, len(pixels))
    if k < 1:
        return []
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    # Sort by frequency
    counts = np.bincount(labels.flatten())
    order = np.argsort(-counts)
    return [_bgr_to_hex(centers[i]) for i in order[:n]]


# ============================================================
# Simple assertions (no deps)
# ============================================================

def file_exists(path):
    p = Path(path)
    return p.exists() and p.stat().st_size > 1000

def file_size_kb(path):
    p = Path(path)
    return p.stat().st_size / 1024 if p.exists() else 0
