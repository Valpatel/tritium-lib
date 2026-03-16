# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Visual analysis toolkit for UI test screenshots.

ANALYSIS ONLY — no browser interaction. Takes screenshot paths as input,
returns structured results. Supports three analysis layers:

1. OpenCV pixel analysis (fast, deterministic)
2. Llava vision model (via Ollama, semantic understanding)
3. Simple assertions (pixel count thresholds)

Usage:
    from tritium_lib.testing.visual_analysis import (
        compare_screenshots, detect_changes, is_mostly_black,
        count_colored_pixels, describe_screenshot, diff_report,
    )
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Try OpenCV — graceful fallback if not installed
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Try requests for Ollama — graceful fallback
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class VisualDiff:
    """Result of comparing two screenshots."""
    changed_pixels: int
    total_pixels: int
    change_percent: float
    regions_changed: int  # number of distinct changed regions
    diff_image_path: Optional[str] = None


@dataclass
class ColorCount:
    """Result of counting colored pixels."""
    total_pixels: int
    matching_pixels: int
    percent: float


@dataclass
class VisionDescription:
    """Result from llava vision model."""
    description: str
    model: str
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Layer 1: OpenCV pixel analysis (fast, deterministic)
# ---------------------------------------------------------------------------

def compare_screenshots(path_a: str, path_b: str, threshold: int = 30) -> Optional[VisualDiff]:
    """Compare two screenshots pixel-by-pixel. Returns VisualDiff or None if OpenCV missing."""
    if not HAS_OPENCV:
        return None
    img_a = cv2.imread(str(path_a))
    img_b = cv2.imread(str(path_b))
    if img_a is None or img_b is None:
        return None
    if img_a.shape != img_b.shape:
        # Resize to match
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    diff = cv2.absdiff(img_a, img_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    changed = int(np.count_nonzero(thresh))
    total = thresh.shape[0] * thresh.shape[1]

    # Count distinct regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return VisualDiff(
        changed_pixels=changed,
        total_pixels=total,
        change_percent=round(changed / total * 100, 2) if total > 0 else 0,
        regions_changed=len(contours),
    )


def is_mostly_black(path: str, threshold: float = 90.0) -> bool:
    """Check if a screenshot is mostly black (> threshold% dark pixels)."""
    if not HAS_OPENCV:
        return False
    img = cv2.imread(str(path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = np.count_nonzero(gray < 20)
    total = gray.shape[0] * gray.shape[1]
    return (dark / total * 100) > threshold


def count_colored_pixels(path: str, color_bgr: tuple, tolerance: int = 40) -> Optional[ColorCount]:
    """Count pixels within tolerance of a target BGR color."""
    if not HAS_OPENCV:
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    lower = np.array([max(0, c - tolerance) for c in color_bgr], dtype=np.uint8)
    upper = np.array([min(255, c + tolerance) for c in color_bgr], dtype=np.uint8)
    mask = cv2.inRange(img, lower, upper)
    matching = int(np.count_nonzero(mask))
    total = img.shape[0] * img.shape[1]
    return ColorCount(
        total_pixels=total,
        matching_pixels=matching,
        percent=round(matching / total * 100, 4) if total > 0 else 0,
    )


def detect_changes(baseline: str, current: str, min_change_percent: float = 0.5) -> bool:
    """Returns True if screenshots differ by more than min_change_percent."""
    diff = compare_screenshots(baseline, current)
    if diff is None:
        return True  # Assume changed if we can't compare
    return diff.change_percent > min_change_percent


def save_diff_image(path_a: str, path_b: str, output_path: str, threshold: int = 30) -> Optional[str]:
    """Save a visual diff image highlighting changed pixels in magenta."""
    if not HAS_OPENCV:
        return None
    img_a = cv2.imread(str(path_a))
    img_b = cv2.imread(str(path_b))
    if img_a is None or img_b is None:
        return None
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

    diff = cv2.absdiff(img_a, img_b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Overlay magenta on changed pixels
    result = img_b.copy()
    result[mask > 0] = [255, 42, 109]  # Magenta (BGR)

    cv2.imwrite(str(output_path), result)
    return str(output_path)


# ---------------------------------------------------------------------------
# Layer 2: Llava vision model (via Ollama, semantic understanding)
# ---------------------------------------------------------------------------

def describe_screenshot(
    path: str,
    prompt: str = "Describe what you see in this UI screenshot. Focus on: layout, visible elements, colors, and any issues.",
    model: str = "llava:7b",
    ollama_url: str = "http://localhost:11434",
) -> VisionDescription:
    """Ask a vision model to describe a screenshot."""
    if not HAS_REQUESTS:
        return VisionDescription(description="", model=model, success=False, error="requests not installed")

    import base64
    try:
        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
            },
            timeout=60,
        )
        if resp.ok:
            data = resp.json()
            return VisionDescription(
                description=data.get("response", ""),
                model=model,
                success=True,
            )
        return VisionDescription(description="", model=model, success=False, error=f"HTTP {resp.status_code}")
    except Exception as e:
        return VisionDescription(description="", model=model, success=False, error=str(e))


# ---------------------------------------------------------------------------
# Layer 3: Simple assertions (no deps required)
# ---------------------------------------------------------------------------

def file_exists(path: str) -> bool:
    """Check if a screenshot file exists and has content."""
    p = Path(path)
    return p.exists() and p.stat().st_size > 1000  # >1KB = real image


def file_size_kb(path: str) -> float:
    """Get file size in KB."""
    p = Path(path)
    return p.stat().st_size / 1024 if p.exists() else 0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def diff_report(results: list, title: str, output_path: str) -> str:
    """Generate a markdown report from diff results."""
    md = f"# {title}\n\n"
    for r in results:
        md += f"## {r.get('name', '?')}\n"
        if r.get("diff"):
            d = r["diff"]
            md += f"- Changed pixels: {d.changed_pixels} ({d.change_percent}%)\n"
            md += f"- Regions changed: {d.regions_changed}\n"
        if r.get("screenshot"):
            md += f"- Screenshot: `{r['screenshot']}`\n"
        if r.get("description"):
            md += f"- Description: {r['description']}\n"
        md += "\n"

    Path(output_path).write_text(md)
    return output_path
