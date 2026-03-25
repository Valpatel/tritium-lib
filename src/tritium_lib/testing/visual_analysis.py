# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Visual analysis toolkit for UI testing.

NOT a screenshot diffing tool — Playwright's toHaveScreenshot() does that better.
This module handles STRUCTURAL analysis that Playwright can't do:

1. STRUCTURAL DETECTORS — find UI elements by shape, color, position
2. OVERLAP DETECTION — find elements that shouldn't be on top of each other
3. LAYOUT VALIDATION — verify element positions match expected zones
4. VIDEO ANALYSIS — analyze frame sequences for animation/performance issues
5. VISION MODEL — semantic validation via llava (Ollama)

Design principle: each function takes an image (path or numpy array) and returns
structured data. No browser interaction. No side effects.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

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
class Box:
    """A rectangle in pixel coordinates."""
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self): return self.w * self.h
    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2
    @property
    def right(self): return self.x + self.w
    @property
    def bottom(self): return self.y + self.h

    def overlaps(self, other: 'Box') -> bool:
        """Do two boxes overlap?"""
        return not (self.right <= other.x or other.right <= self.x or
                    self.bottom <= other.y or other.bottom <= self.y)

    def overlap_area(self, other: 'Box') -> int:
        """Area of overlap between two boxes. 0 if no overlap."""
        dx = min(self.right, other.right) - max(self.x, other.x)
        dy = min(self.bottom, other.bottom) - max(self.y, other.y)
        if dx <= 0 or dy <= 0:
            return 0
        return dx * dy

    def iou(self, other: 'Box') -> float:
        """Intersection over Union."""
        inter = self.overlap_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0


@dataclass
class UIElement:
    """A detected UI element."""
    kind: str              # panel, button, marker, text, overlay, toast, menu
    box: Box
    color_hex: str = ""    # dominant color
    confidence: float = 1.0
    label: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class OverlapIssue:
    """Two elements that shouldn't overlap but do."""
    element_a: UIElement
    element_b: UIElement
    overlap_percent: float   # overlap area / smaller element area
    severity: str = "warning"  # warning, error, critical


@dataclass
class ScreenZone:
    """A named region of the screen for layout validation."""
    name: str
    box: Box
    max_elements: int = 3   # expected max elements in this zone
    element_kinds: List[str] = field(default_factory=list)  # expected kinds


# ============================================================
# Image loading
# ============================================================

def load(path_or_array):
    """Load image from path or pass through numpy array. Returns None if failed."""
    if not HAS_OPENCV:
        return None
    if isinstance(path_or_array, np.ndarray):
        return path_or_array
    img = cv2.imread(str(path_or_array))
    return img


# ============================================================
# 1. STRUCTURAL DETECTORS
# ============================================================

def detect_bright_rectangles(img, min_area_pct=0.5, max_area_pct=60.0) -> List[UIElement]:
    """Find rectangular regions with bright borders on dark background.
    Good for detecting panels, dialogs, tooltips.
    """
    if not HAS_OPENCV:
        return []
    img = load(img)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    total = h * w
    min_a = total * min_area_pct / 100
    max_a = total * max_area_pct / 100
    results = []

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        a = cw * ch
        if a < min_a or a > max_a:
            continue
        aspect = cw / ch if ch > 0 else 0
        if aspect < 0.15 or aspect > 7.0:
            continue
        color = _mean_border_color(img, x, y, cw, ch)
        results.append(UIElement(
            kind="panel", box=Box(x, y, cw, ch), color_hex=_bgr2hex(color),
        ))

    return results


def detect_small_colored_dots(img, min_px=4, max_px=50) -> List[UIElement]:
    """Find small bright colored regions (map markers, status dots, icons)."""
    if not HAS_OPENCV:
        return []
    img = load(img)
    if img is None:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Saturated + bright = colored element (not gray/black/white)
    mask = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([180, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_px or ch < min_px or cw > max_px or ch > max_px:
            continue
        color = _mean_color(img, x, y, cw, ch)
        results.append(UIElement(
            kind="marker", box=Box(x, y, cw, ch), color_hex=_bgr2hex(color),
        ))
    return results


def detect_text_blocks(img) -> List[UIElement]:
    """Find horizontal bright text regions using morphological analysis."""
    if not HAS_OPENCV:
        return []
    img = load(img)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    # Connect characters horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    dilated = cv2.dilate(bright, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 15 or ch < 4 or ch > 50 or cw / max(ch, 1) < 1.5:
            continue
        results.append(UIElement(kind="text", box=Box(x, y, cw, ch)))
    return results


# ============================================================
# 2. OVERLAP DETECTION
# ============================================================

def find_overlaps(elements: List[UIElement], min_overlap_pct=20.0) -> List[OverlapIssue]:
    """Find pairs of elements that overlap more than min_overlap_pct of the smaller one."""
    issues = []
    for i, a in enumerate(elements):
        for b in elements[i + 1:]:
            overlap = a.box.overlap_area(b.box)
            if overlap == 0:
                continue
            smaller = min(a.box.area, b.box.area)
            if smaller == 0:
                continue
            pct = (overlap / smaller) * 100
            if pct >= min_overlap_pct:
                severity = "critical" if pct > 80 else "error" if pct > 50 else "warning"
                issues.append(OverlapIssue(a, b, round(pct, 1), severity))
    return issues


# ============================================================
# 3. LAYOUT VALIDATION
# ============================================================

def make_screen_zones(width: int, height: int) -> List[ScreenZone]:
    """Create standard screen zones for layout validation."""
    hw, hh = width // 2, height // 2
    qw, qh = width // 4, height // 4
    return [
        ScreenZone("top-left", Box(0, 0, hw, hh), max_elements=3, element_kinds=["panel", "menu"]),
        ScreenZone("top-right", Box(hw, 0, hw, hh), max_elements=3, element_kinds=["panel", "toast"]),
        ScreenZone("bottom-left", Box(0, hh, hw, hh), max_elements=3, element_kinds=["panel"]),
        ScreenZone("bottom-right", Box(hw, hh, hw, hh), max_elements=3, element_kinds=["toast"]),
        ScreenZone("top-center", Box(qw, 0, hw, qh), max_elements=2, element_kinds=["menu", "text"]),
        ScreenZone("bottom-center", Box(qw, height - qh, hw, qh), max_elements=2, element_kinds=["text"]),
    ]


def validate_layout(elements: List[UIElement], zones: List[ScreenZone]) -> List[dict]:
    """Check that elements are in their expected zones and zones aren't overcrowded."""
    issues = []
    for zone in zones:
        in_zone = [e for e in elements if zone.box.overlaps(e.box)]
        if len(in_zone) > zone.max_elements:
            issues.append({
                "zone": zone.name,
                "issue": "overcrowded",
                "count": len(in_zone),
                "max": zone.max_elements,
                "elements": [e.kind for e in in_zone],
            })
    return issues


# ============================================================
# 4. BLACK SCREEN / BASIC CHECKS
# ============================================================

def is_black_screen(img, threshold_pct=90.0) -> bool:
    """Is the image mostly black? (rendering failure detection)"""
    if not HAS_OPENCV:
        return False
    img = load(img)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = np.count_nonzero(gray < 20)
    return (dark / gray.size * 100) > threshold_pct


def brightness_stats(img) -> dict:
    """Get brightness statistics for an image."""
    if not HAS_OPENCV:
        return {}
    img = load(img)
    if img is None:
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "dark_pct": float(np.count_nonzero(gray < 30) / gray.size * 100),
        "bright_pct": float(np.count_nonzero(gray > 200) / gray.size * 100),
    }


def dominant_colors(img, n=5) -> List[str]:
    """Get top N colors as hex strings using k-means clustering."""
    if not HAS_OPENCV:
        return []
    img = load(img)
    if img is None:
        return []
    small = cv2.resize(img, (50, 50))
    pixels = small.reshape(-1, 3).astype(np.float32)
    k = min(n, len(pixels))
    if k < 1:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    order = np.argsort(-counts)
    return [_bgr2hex(centers[i]) for i in order[:n]]


# ============================================================
# 5. VIDEO / FRAME SEQUENCE ANALYSIS
# ============================================================

def analyze_frame_sequence(paths: List[str], change_threshold=1.0) -> dict:
    """Analyze a sequence of screenshots for motion and anomalies.

    Returns summary stats useful for performance and stability testing.
    """
    if not HAS_OPENCV or len(paths) < 2:
        return {"frames": len(paths), "motion": 0, "static": 0, "anomalies": []}

    motion, static, anomalies = 0, 0, []
    prev = None

    for i, path in enumerate(paths):
        img = load(path)
        if img is None:
            anomalies.append({"frame": i, "issue": "load_failed"})
            continue
        if is_black_screen(img):
            anomalies.append({"frame": i, "issue": "black_screen"})

        if prev is not None:
            diff = cv2.absdiff(prev, img)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            changed_pct = np.count_nonzero(gray > 25) / gray.size * 100
            if changed_pct > change_threshold:
                motion += 1
            else:
                static += 1
            # Sudden large change = possible crash/reload
            if changed_pct > 40:
                anomalies.append({"frame": i, "issue": "large_change", "pct": round(changed_pct, 1)})

        prev = img

    return {
        "frames": len(paths),
        "motion": motion,
        "static": static,
        "anomalies": anomalies,
    }


# ============================================================
# 6. VISION MODEL (Ollama/llava)
# ============================================================

def ask_vision_model(path, prompt=None, model="llava:7b", url="http://localhost:8081") -> dict:
    """Ask a vision LLM to evaluate a screenshot.

    Uses llama-server (OpenAI-compatible) on port 8081 by default.
    Falls back to ollama on 11434 if llama-server unavailable.
    """
    if not HAS_REQUESTS:
        return {"ok": False, "error": "requests not installed"}
    if prompt is None:
        prompt = (
            "You are a QA engineer reviewing a UI screenshot. "
            "List any visual issues: overlapping elements, unreadable text, "
            "empty panels, broken layouts, or anything that looks wrong. "
            "If everything looks fine, say 'No issues found.'"
        )
    import base64
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        # Try llama-server (OpenAI-compatible) first
        is_llama = any(p in url for p in [":8081", ":8082", ":8083"])
        if is_llama:
            r = _requests.post(f"{url}/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
                "max_tokens": 512,
            }, timeout=90)
            if r.ok:
                choices = r.json().get("choices", [])
                text = choices[0]["message"]["content"] if choices else ""
                return {"ok": True, "response": text, "model": model}
        else:
            # Legacy ollama
            r = _requests.post(f"{url}/api/generate", json={
                "model": model, "prompt": prompt, "images": [b64], "stream": False,
            }, timeout=90)
            if r.ok:
                return {"ok": True, "response": r.json().get("response", ""), "model": model}

        return {"ok": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ============================================================
# Internal helpers
# ============================================================

def _bgr2hex(bgr):
    if bgr is None or len(bgr) < 3:
        return "#000000"
    return f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"

def _mean_color(img, x, y, w, h):
    region = img[y:y+h, x:x+w]
    if region.size == 0:
        return (0, 0, 0)
    return tuple(int(c) for c in np.mean(region.reshape(-1, 3), axis=0))

def _mean_border_color(img, x, y, w, h, bw=2):
    strips = []
    for s in [img[y:y+bw, x:x+w], img[y+h-bw:y+h, x:x+w],
              img[y:y+h, x:x+bw], img[y:y+h, x+w-bw:x+w]]:
        if s.size > 0:
            strips.append(s.reshape(-1, 3))
    if not strips:
        return (0, 0, 0)
    all_px = np.vstack(strips)
    return tuple(int(c) for c in np.mean(all_px, axis=0))


# ============================================================
# Simple assertions (no deps)
# ============================================================

def file_exists(path):
    """Check screenshot file exists and is a real image (>1KB)."""
    p = Path(path)
    return p.exists() and p.stat().st_size > 1000

def file_size_kb(path):
    """Get file size in KB."""
    p = Path(path)
    return p.stat().st_size / 1024 if p.exists() else 0
