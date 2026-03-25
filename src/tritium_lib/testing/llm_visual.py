# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""LLM-based visual analysis for UI testing via local llama-server.

Uses local llama-server instances (ports 8081-8083) with vision models
(moondream, llava, granite3.2-vision, qwen3-vl, etc.) for semantic
screenshot understanding beyond what OpenCV structural analysis can do.

Design:
- Tries each configured llama-server port in order
- Uses the OpenAI-compatible /v1/chat/completions API with image_url
- Falls back to /completion endpoint with image_data if chat fails
- Graceful degradation: returns structured result even when no vision
  model is available (ok=False with fallback OpenCV analysis)

Usage:
    from tritium_lib.testing.llm_visual import LLMVisualAnalyzer

    analyzer = LLMVisualAnalyzer()
    result = analyzer.analyze("screenshot.png", "Is the map loaded?")
    if result.ok:
        print(result.text)
    else:
        print(f"Vision unavailable: {result.error}")
        # result.opencv_fallback has structural analysis if OpenCV available
"""

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# ============================================================
# Configuration
# ============================================================

DEFAULT_PORTS = [8081, 8082, 8083]
DEFAULT_TIMEOUT = 90
DEFAULT_MAX_TOKENS = 500


# ============================================================
# Data structures
# ============================================================

@dataclass
class VisionResult:
    """Result from a vision model analysis."""
    ok: bool
    text: str = ""
    error: str = ""
    model: str = ""
    port: int = 0
    elapsed_seconds: float = 0.0
    tokens_used: int = 0
    opencv_fallback: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        d = {
            "ok": self.ok,
            "text": self.text,
            "error": self.error,
            "model": self.model,
            "port": self.port,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "tokens_used": self.tokens_used,
        }
        if self.opencv_fallback:
            d["opencv_fallback"] = self.opencv_fallback
        return d


@dataclass
class ScreenshotAnalysis:
    """Complete analysis of a screenshot combining vision + OpenCV."""
    path: str
    vision: Optional[VisionResult] = None
    brightness: Optional[Dict[str, float]] = None
    dominant_colors: Optional[List[str]] = None
    is_black_screen: bool = False
    marker_count: int = 0
    panel_count: int = 0
    text_block_count: int = 0

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "path": self.path,
            "vision": self.vision.to_dict() if self.vision else None,
            "brightness": self.brightness,
            "dominant_colors": self.dominant_colors,
            "is_black_screen": self.is_black_screen,
            "marker_count": self.marker_count,
            "panel_count": self.panel_count,
            "text_block_count": self.text_block_count,
        }


# ============================================================
# Prompt library for tactical UI analysis
# ============================================================

PROMPTS = {
    "general": (
        "You are a QA engineer reviewing a screenshot of a tactical "
        "surveillance system called Tritium. Describe what you see: "
        "UI panels, map tiles, target markers, HUD elements, text labels, "
        "colors used. Note any visual issues: broken layouts, empty areas, "
        "overlapping elements, unreadable text."
    ),
    "map_loaded": (
        "Look at this screenshot of a mapping application. "
        "Is a map visible with satellite or street tiles? "
        "Are there any markers or icons on the map? "
        "Is the map area blank/black or does it show geographic features? "
        "Answer concisely."
    ),
    "target_markers": (
        "Count the colored markers, dots, or icons visible in this "
        "tactical map screenshot. What colors are they? Are they "
        "clustered or spread out? Do any have labels or popups?"
    ),
    "battle_hud": (
        "Is there a battle or combat HUD visible in this screenshot? "
        "Look for: unit health bars, kill feeds, team indicators, "
        "weapon icons, ammo counts, minimap, score displays. "
        "Describe what combat-related UI elements you can see."
    ),
    "ui_issues": (
        "You are a QA engineer. Find visual bugs in this screenshot: "
        "overlapping panels, clipped text, empty containers that should "
        "have content, broken borders, invisible elements, misaligned "
        "components. If everything looks fine, say 'No issues found.'"
    ),
    "accessibility": (
        "Evaluate this UI screenshot for accessibility: "
        "Is text readable against its background? Are colors "
        "distinguishable? Are interactive elements clearly visible? "
        "Is there sufficient contrast?"
    ),
}


# ============================================================
# Server discovery
# ============================================================

def probe_server(port: int, timeout: float = 3.0) -> Dict[str, Any]:
    """Check if a llama-server is running and what capabilities it has.

    Returns dict with keys: alive, has_vision, model_id, n_params, error.
    """
    if not HAS_REQUESTS:
        return {"alive": False, "has_vision": False, "error": "requests not installed"}

    result = {
        "alive": False,
        "has_vision": False,
        "model_id": "",
        "n_params": 0,
        "error": "",
    }

    try:
        # Health check
        r = _requests.get(f"http://localhost:{port}/health", timeout=timeout)
        if r.status_code != 200:
            result["error"] = f"health returned {r.status_code}"
            return result
        health = r.json()
        if health.get("status") != "ok":
            result["error"] = f"status={health.get('status')}"
            return result
        result["alive"] = True

        # Model info
        r = _requests.get(f"http://localhost:{port}/v1/models", timeout=timeout)
        if r.ok:
            data = r.json()
            models = data.get("data", [])
            if models:
                m = models[0]
                result["model_id"] = m.get("id", "")
                meta = m.get("meta", {})
                result["n_params"] = meta.get("n_params", 0)

        # Vision capability test: send a tiny image and check for error
        # A 1x1 red PNG, base64 encoded
        tiny_png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
        )
        r = _requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{tiny_png_b64}"
                    }},
                    {"type": "text", "text": "ok"},
                ]}],
                "max_tokens": 5,
            },
            timeout=timeout + 5,
        )
        if r.ok:
            result["has_vision"] = True
        else:
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            err_msg = body.get("error", {}).get("message", "")
            if "image input is not supported" in err_msg or "mmproj" in err_msg:
                result["has_vision"] = False
            else:
                # Some other error — might still support vision
                result["has_vision"] = False
                result["error"] = err_msg

    except Exception as e:
        result["error"] = str(e)

    return result


def discover_vision_servers(
    ports: Optional[List[int]] = None,
    timeout: float = 3.0,
) -> List[Dict[str, Any]]:
    """Scan ports for llama-server instances with vision capability.

    Returns list of server info dicts, vision-capable ones first.
    """
    ports = ports or DEFAULT_PORTS
    servers = []
    for port in ports:
        info = probe_server(port, timeout)
        info["port"] = port
        servers.append(info)
    # Sort: vision-capable first, then by param count (larger = smarter)
    servers.sort(key=lambda s: (not s["has_vision"], -s["n_params"]))
    return servers


# ============================================================
# Core analysis functions
# ============================================================

def _encode_image(path: str) -> str:
    """Read image file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _try_chat_completions(
    port: int,
    image_b64: str,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[VisionResult]:
    """Try the OpenAI-compatible /v1/chat/completions endpoint."""
    url = f"http://localhost:{port}/v1/chat/completions"
    start = time.time()
    try:
        r = _requests.post(url, json={
            "model": "default",
            "messages": [
                {"role": "system", "content": "You are a visual analysis assistant. Describe images accurately and concisely."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    }},
                    {"type": "text", "text": prompt},
                ]},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }, timeout=timeout)
        elapsed = time.time() - start

        if not r.ok:
            return None

        data = r.json()
        choices = data.get("choices", [])
        if not choices:
            return None

        text = choices[0].get("message", {}).get("content", "")
        if not text.strip():
            return None

        usage = data.get("usage", {})
        model = data.get("model", "")

        return VisionResult(
            ok=True,
            text=text.strip(),
            model=model,
            port=port,
            elapsed_seconds=elapsed,
            tokens_used=usage.get("total_tokens", 0),
        )
    except Exception:
        return None


def _try_completion(
    port: int,
    image_b64: str,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[VisionResult]:
    """Try the llama.cpp /completion endpoint with image_data."""
    url = f"http://localhost:{port}/completion"
    start = time.time()
    try:
        r = _requests.post(url, json={
            "prompt": prompt,
            "image_data": [{"data": image_b64, "id": 0}],
            "n_predict": max_tokens,
            "temperature": 0.2,
        }, timeout=timeout)
        elapsed = time.time() - start

        if not r.ok:
            return None

        data = r.json()
        text = data.get("content", "")
        if not text.strip():
            return None

        model = data.get("model", "")
        tokens = data.get("tokens_predicted", 0) + data.get("tokens_evaluated", 0)

        return VisionResult(
            ok=True,
            text=text.strip(),
            model=model,
            port=port,
            elapsed_seconds=elapsed,
            tokens_used=tokens,
        )
    except Exception:
        return None


def _opencv_fallback(path: str) -> Dict[str, Any]:
    """Run OpenCV structural analysis as fallback when no vision model available."""
    if not HAS_OPENCV:
        return {"available": False, "reason": "opencv not installed"}

    # Import from sister module
    from tritium_lib.testing.visual_analysis import (
        is_black_screen,
        brightness_stats,
        dominant_colors,
        detect_small_colored_dots,
        detect_bright_rectangles,
        detect_text_blocks,
    )

    result = {"available": True}

    try:
        result["is_black_screen"] = is_black_screen(path)
        result["brightness"] = brightness_stats(path)
        result["dominant_colors"] = dominant_colors(path, n=5)
        markers = detect_small_colored_dots(path)
        result["marker_count"] = len(markers)
        result["marker_colors"] = [m.color_hex for m in markers[:20]]
        panels = detect_bright_rectangles(path)
        result["panel_count"] = len(panels)
        text_blocks = detect_text_blocks(path)
        result["text_block_count"] = len(text_blocks)
    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================
# Main analyzer class
# ============================================================

class LLMVisualAnalyzer:
    """Analyze screenshots using local llama-server vision models.

    Tries ports in order, uses OpenAI-compatible API, falls back to
    /completion endpoint, then to OpenCV-only structural analysis.

    Example:
        analyzer = LLMVisualAnalyzer()
        result = analyzer.analyze("/tmp/screenshot.png")
        print(result.text)
    """

    def __init__(
        self,
        ports: Optional[List[int]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.ports = ports or list(DEFAULT_PORTS)
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._vision_ports: Optional[List[int]] = None

    def discover(self) -> List[Dict[str, Any]]:
        """Discover available vision servers. Caches vision-capable ports."""
        servers = discover_vision_servers(self.ports, timeout=3.0)
        self._vision_ports = [
            s["port"] for s in servers if s["has_vision"]
        ]
        return servers

    @property
    def vision_ports(self) -> List[int]:
        """Ports known to have vision capability. Runs discovery if needed."""
        if self._vision_ports is None:
            self.discover()
        return self._vision_ports or []

    @property
    def has_vision(self) -> bool:
        """Is at least one vision model available?"""
        return len(self.vision_ports) > 0

    def analyze(
        self,
        image_path: str,
        question: Optional[str] = None,
        prompt_key: Optional[str] = None,
        include_opencv: bool = True,
    ) -> VisionResult:
        """Analyze a screenshot with a vision model.

        Args:
            image_path: Path to PNG/JPEG screenshot.
            question: Custom question to ask about the image.
            prompt_key: Key into PROMPTS dict (e.g. "map_loaded", "battle_hud").
            include_opencv: If True and vision fails, include OpenCV fallback.

        Returns:
            VisionResult with ok=True if vision model responded,
            or ok=False with opencv_fallback if no vision available.
        """
        if not HAS_REQUESTS:
            return VisionResult(ok=False, error="requests library not installed")

        path = Path(image_path)
        if not path.exists():
            return VisionResult(ok=False, error=f"file not found: {image_path}")
        if path.stat().st_size < 100:
            return VisionResult(ok=False, error=f"file too small: {path.stat().st_size} bytes")

        # Build prompt
        if question:
            prompt = question
        elif prompt_key and prompt_key in PROMPTS:
            prompt = PROMPTS[prompt_key]
        else:
            prompt = PROMPTS["general"]

        # Encode image
        image_b64 = _encode_image(str(path))

        # Try vision-capable ports first, then all ports
        ports_to_try = list(self.vision_ports)
        for p in self.ports:
            if p not in ports_to_try:
                ports_to_try.append(p)

        # Try each port with chat/completions first, then /completion
        for port in ports_to_try:
            result = _try_chat_completions(
                port, image_b64, prompt, self.max_tokens, self.timeout
            )
            if result and result.ok:
                return result

            result = _try_completion(
                port, image_b64, prompt, self.max_tokens, self.timeout
            )
            if result and result.ok:
                return result

        # All ports failed — return fallback
        fallback = _opencv_fallback(str(path)) if include_opencv else None
        return VisionResult(
            ok=False,
            error="no vision model available on any port",
            opencv_fallback=fallback,
        )

    def analyze_full(self, image_path: str, question: Optional[str] = None) -> ScreenshotAnalysis:
        """Full analysis combining vision model + OpenCV structural analysis.

        Always runs both (if available), giving the richest possible result.
        """
        analysis = ScreenshotAnalysis(path=image_path)

        # Vision model analysis
        analysis.vision = self.analyze(image_path, question=question, include_opencv=False)

        # OpenCV structural analysis
        if HAS_OPENCV:
            from tritium_lib.testing.visual_analysis import (
                is_black_screen,
                brightness_stats,
                dominant_colors,
                detect_small_colored_dots,
                detect_bright_rectangles,
                detect_text_blocks,
            )
            try:
                analysis.is_black_screen = bool(is_black_screen(image_path))
                analysis.brightness = brightness_stats(image_path)
                analysis.dominant_colors = dominant_colors(image_path, n=5)
                analysis.marker_count = len(detect_small_colored_dots(image_path))
                analysis.panel_count = len(detect_bright_rectangles(image_path))
                analysis.text_block_count = len(detect_text_blocks(image_path))
            except Exception:
                pass

        return analysis

    def batch_analyze(
        self,
        items: List[Dict[str, str]],
    ) -> List[VisionResult]:
        """Analyze multiple screenshots.

        Args:
            items: List of dicts with 'path' and optional 'question' keys.

        Returns:
            List of VisionResult in same order.
        """
        results = []
        for item in items:
            path = item.get("path", "")
            question = item.get("question")
            results.append(self.analyze(path, question=question))
        return results


# ============================================================
# Convenience functions
# ============================================================

def analyze_screenshot(
    image_path: str,
    question: str = "Describe what you see in this screenshot of a tactical surveillance system.",
    ports: Optional[List[int]] = None,
) -> VisionResult:
    """One-shot screenshot analysis. Creates analyzer, runs analysis, returns result."""
    analyzer = LLMVisualAnalyzer(ports=ports)
    return analyzer.analyze(image_path, question=question)


def check_map_loaded(image_path: str) -> VisionResult:
    """Check if a map is loaded with tiles visible."""
    return LLMVisualAnalyzer().analyze(image_path, prompt_key="map_loaded")


def count_target_markers(image_path: str) -> VisionResult:
    """Ask vision model to count and describe target markers."""
    return LLMVisualAnalyzer().analyze(image_path, prompt_key="target_markers")


def check_battle_hud(image_path: str) -> VisionResult:
    """Check if a battle HUD is visible."""
    return LLMVisualAnalyzer().analyze(image_path, prompt_key="battle_hud")


def check_ui_issues(image_path: str) -> VisionResult:
    """Ask vision model to find visual bugs."""
    return LLMVisualAnalyzer().analyze(image_path, prompt_key="ui_issues")


def is_vision_available(ports: Optional[List[int]] = None) -> bool:
    """Quick check: is any vision model running?"""
    analyzer = LLMVisualAnalyzer(ports=ports)
    return analyzer.has_vision
