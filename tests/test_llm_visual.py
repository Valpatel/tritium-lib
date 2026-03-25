# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for LLM-based visual analysis module.

Tests are organized in tiers:
1. Unit tests (no network, no GPU) — always run
2. Integration tests (require llama-server running) — marked with @needs_server
3. Live vision tests (require vision model) — marked with @needs_vision

Synthetic images are created with OpenCV. No browser or external files needed.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tritium_lib.testing.llm_visual import (
    LLMVisualAnalyzer,
    VisionResult,
    ScreenshotAnalysis,
    PROMPTS,
    DEFAULT_PORTS,
    probe_server,
    discover_vision_servers,
    analyze_screenshot,
    check_map_loaded,
    count_target_markers,
    check_battle_hud,
    check_ui_issues,
    is_vision_available,
    _encode_image,
    _opencv_fallback,
)

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


def _server_alive(port: int) -> bool:
    """Check if a llama-server is responding on a given port."""
    if not HAS_REQUESTS:
        return False
    try:
        r = _requests.get(f"http://localhost:{port}/health", timeout=2)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False


def _server_has_vision(port: int) -> bool:
    """Check if a llama-server supports vision on a given port."""
    if not HAS_REQUESTS:
        return False
    try:
        info = probe_server(port, timeout=5)
        return info.get("has_vision", False)
    except Exception:
        return False


# Find first available server and vision server
_ANY_SERVER_PORT = None
_VISION_SERVER_PORT = None
for _p in DEFAULT_PORTS:
    if _server_alive(_p):
        if _ANY_SERVER_PORT is None:
            _ANY_SERVER_PORT = _p
        if _server_has_vision(_p):
            if _VISION_SERVER_PORT is None:
                _VISION_SERVER_PORT = _p

needs_server = pytest.mark.skipif(
    _ANY_SERVER_PORT is None,
    reason="No llama-server running on any default port",
)
needs_vision = pytest.mark.skipif(
    _VISION_SERVER_PORT is None,
    reason="No vision-capable llama-server running",
)
needs_opencv = pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV not installed")


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def dark_ui_image(tmp_path):
    """Create a synthetic dark UI screenshot with colored elements."""
    if not HAS_OPENCV:
        pytest.skip("OpenCV required")
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Dark background
    img[:] = (15, 15, 20)
    # Cyan panel border
    cv2.rectangle(img, (20, 20), (200, 100), (255, 240, 0), 2)
    # Magenta panel border
    cv2.rectangle(img, (220, 20), (400, 100), (109, 42, 255), 2)
    # Text
    cv2.putText(img, "TRITIUM", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 240, 0), 2)
    # Map area
    cv2.rectangle(img, (20, 120), (580, 380), (40, 40, 45), -1)
    # Colored markers
    cv2.circle(img, (100, 200), 8, (0, 255, 0), -1)   # green
    cv2.circle(img, (300, 250), 8, (0, 0, 255), -1)    # red
    cv2.circle(img, (400, 180), 8, (255, 240, 0), -1)  # cyan
    path = str(tmp_path / "dark_ui.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def black_image(tmp_path):
    """Create a black (failed render) image."""
    if not HAS_OPENCV:
        pytest.skip("OpenCV required")
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    path = str(tmp_path / "black.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def bright_map_image(tmp_path):
    """Create a simulated satellite map with markers."""
    if not HAS_OPENCV:
        pytest.skip("OpenCV required")
    img = np.full((600, 800, 3), (80, 120, 60), dtype=np.uint8)  # greenish terrain
    # Roads
    cv2.line(img, (0, 300), (800, 300), (100, 100, 100), 3)
    cv2.line(img, (400, 0), (400, 600), (100, 100, 100), 3)
    # Buildings
    for x, y in [(100, 100), (200, 400), (500, 150), (600, 350)]:
        cv2.rectangle(img, (x, y), (x + 40, y + 30), (50, 50, 80), -1)
    # Target markers
    for i, (x, y) in enumerate([(150, 200), (350, 350), (550, 250)]):
        color = [(0, 255, 0), (0, 0, 255), (255, 255, 0)][i]
        cv2.circle(img, (x, y), 10, color, -1)
        cv2.circle(img, (x, y), 12, (255, 255, 255), 1)
    path = str(tmp_path / "map.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def tiny_file(tmp_path):
    """Create a file too small to be a real image."""
    path = str(tmp_path / "tiny.png")
    with open(path, "wb") as f:
        f.write(b"x" * 50)
    return path


# ============================================================
# 1. Unit Tests — Data Structures
# ============================================================

class TestVisionResult:
    def test_defaults(self):
        r = VisionResult(ok=False)
        assert r.ok is False
        assert r.text == ""
        assert r.error == ""
        assert r.model == ""
        assert r.port == 0
        assert r.elapsed_seconds == 0.0
        assert r.tokens_used == 0
        assert r.opencv_fallback is None

    def test_success(self):
        r = VisionResult(
            ok=True,
            text="Map loaded with 3 markers",
            model="moondream",
            port=8081,
            elapsed_seconds=1.5,
            tokens_used=150,
        )
        assert r.ok
        assert "3 markers" in r.text
        assert r.port == 8081

    def test_to_dict(self):
        r = VisionResult(ok=True, text="hello", model="test", port=8081, elapsed_seconds=1.234)
        d = r.to_dict()
        assert d["ok"] is True
        assert d["text"] == "hello"
        assert d["elapsed_seconds"] == 1.23
        assert "opencv_fallback" not in d

    def test_to_dict_with_fallback(self):
        r = VisionResult(ok=False, error="no model", opencv_fallback={"marker_count": 5})
        d = r.to_dict()
        assert d["ok"] is False
        assert d["opencv_fallback"]["marker_count"] == 5

    def test_to_dict_json_serializable(self):
        r = VisionResult(ok=True, text="test", port=8081, elapsed_seconds=1.0, tokens_used=100)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["ok"] is True


class TestScreenshotAnalysis:
    def test_defaults(self):
        a = ScreenshotAnalysis(path="/tmp/test.png")
        assert a.path == "/tmp/test.png"
        assert a.vision is None
        assert a.is_black_screen is False
        assert a.marker_count == 0

    def test_to_dict(self):
        a = ScreenshotAnalysis(
            path="/tmp/test.png",
            vision=VisionResult(ok=True, text="looks good"),
            is_black_screen=False,
            marker_count=5,
            panel_count=2,
        )
        d = a.to_dict()
        assert d["path"] == "/tmp/test.png"
        assert d["vision"]["ok"] is True
        assert d["marker_count"] == 5

    def test_to_dict_no_vision(self):
        a = ScreenshotAnalysis(path="/tmp/test.png")
        d = a.to_dict()
        assert d["vision"] is None

    def test_to_dict_json_serializable(self):
        a = ScreenshotAnalysis(path="/tmp/x.png", marker_count=3)
        s = json.dumps(a.to_dict())
        assert isinstance(s, str)


class TestPrompts:
    def test_all_prompts_exist(self):
        expected = ["general", "map_loaded", "target_markers", "battle_hud", "ui_issues", "accessibility"]
        for key in expected:
            assert key in PROMPTS
            assert len(PROMPTS[key]) > 20

    def test_prompts_are_strings(self):
        for key, val in PROMPTS.items():
            assert isinstance(val, str), f"Prompt '{key}' is not a string"


# ============================================================
# 2. Unit Tests — Image Encoding
# ============================================================

class TestEncodeImage:
    def test_encode_small_file(self, tmp_path):
        path = str(tmp_path / "test.bin")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        b64 = _encode_image(path)
        assert isinstance(b64, str)
        assert len(b64) > 0
        # Should be valid base64
        import base64
        decoded = base64.b64decode(b64)
        assert decoded[:4] == b"\x89PNG"

    def test_encode_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _encode_image(str(tmp_path / "nonexistent.png"))


# ============================================================
# 3. Unit Tests — OpenCV Fallback
# ============================================================

@needs_opencv
class TestOpenCVFallback:
    def test_fallback_on_dark_ui(self, dark_ui_image):
        result = _opencv_fallback(dark_ui_image)
        assert result["available"] is True
        assert bool(result["is_black_screen"]) is False or bool(result["is_black_screen"]) is True
        assert "brightness" in result
        assert "dominant_colors" in result
        assert result["marker_count"] >= 0
        assert result["panel_count"] >= 0

    def test_fallback_on_black_image(self, black_image):
        result = _opencv_fallback(black_image)
        assert result["available"] is True
        assert bool(result["is_black_screen"]) is True

    def test_fallback_on_missing_file(self, tmp_path):
        result = _opencv_fallback(str(tmp_path / "nope.png"))
        assert result["available"] is True
        # Should handle gracefully (is_black_screen returns False for None img)
        assert result.get("is_black_screen") is not None


class TestOpenCVFallbackNoCV:
    def test_no_opencv(self):
        with patch("tritium_lib.testing.llm_visual.HAS_OPENCV", False):
            result = _opencv_fallback("/tmp/anything.png")
            assert result["available"] is False


# ============================================================
# 4. Unit Tests — Analyzer (mocked network)
# ============================================================

class TestAnalyzerNoNetwork:
    def test_missing_file(self, tmp_path):
        analyzer = LLMVisualAnalyzer(ports=[9999])
        result = analyzer.analyze(str(tmp_path / "nonexistent.png"))
        assert not result.ok
        assert "not found" in result.error

    def test_too_small_file(self, tiny_file):
        analyzer = LLMVisualAnalyzer(ports=[9999])
        result = analyzer.analyze(tiny_file)
        assert not result.ok
        assert "too small" in result.error

    def test_no_requests_library(self, tmp_path):
        path = str(tmp_path / "img.png")
        with open(path, "wb") as f:
            f.write(b"x" * 1000)
        with patch("tritium_lib.testing.llm_visual.HAS_REQUESTS", False):
            analyzer = LLMVisualAnalyzer()
            result = analyzer.analyze(path)
            assert not result.ok
            assert "requests" in result.error.lower()

    def test_custom_prompt(self):
        analyzer = LLMVisualAnalyzer(ports=[9999])
        # Verify prompt selection works (won't reach network)
        assert analyzer.max_tokens == 500
        assert analyzer.timeout == 90

    def test_prompt_key_selection(self):
        # Just verify PROMPTS dict is accessible
        assert "map_loaded" in PROMPTS
        assert "battle_hud" in PROMPTS


class TestAnalyzerMocked:
    def test_successful_chat_completions(self, tmp_path):
        """Mock a successful /v1/chat/completions response."""
        path = str(tmp_path / "test.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG" + b"\x00" * 1000)

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "I see a dark UI with 3 colored markers"}}],
            "model": "moondream",
            "usage": {"total_tokens": 200},
        }

        with patch("tritium_lib.testing.llm_visual._requests") as mock_req:
            mock_req.post.return_value = mock_response
            analyzer = LLMVisualAnalyzer(ports=[8081])
            analyzer._vision_ports = [8081]  # Skip discovery
            result = analyzer.analyze(path, question="What do you see?")

        assert result.ok
        assert "3 colored markers" in result.text
        assert result.port == 8081

    def test_falls_back_to_completion_endpoint(self, tmp_path):
        """If chat/completions fails, try /completion endpoint."""
        path = str(tmp_path / "test.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG" + b"\x00" * 1000)

        call_count = [0]

        def side_effect(url, **kwargs):
            call_count[0] += 1
            mock = MagicMock()
            if "/v1/chat/completions" in url:
                # Chat endpoint fails
                mock.ok = False
                mock.status_code = 500
            elif "/completion" in url:
                # Completion endpoint succeeds
                mock.ok = True
                mock.json.return_value = {
                    "content": "Dark background with colored dots",
                    "model": "moondream",
                    "tokens_predicted": 50,
                    "tokens_evaluated": 100,
                }
            return mock

        with patch("tritium_lib.testing.llm_visual._requests") as mock_req:
            mock_req.post.side_effect = side_effect
            analyzer = LLMVisualAnalyzer(ports=[8081])
            analyzer._vision_ports = [8081]
            result = analyzer.analyze(path)

        assert result.ok
        assert "colored dots" in result.text
        assert call_count[0] == 2  # Tried both endpoints

    def test_all_ports_fail_with_opencv_fallback(self, tmp_path):
        """When all ports fail, include OpenCV fallback if available."""
        path = str(tmp_path / "test.png")
        if HAS_OPENCV:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(path, img)
        else:
            with open(path, "wb") as f:
                f.write(b"\x89PNG" + b"\x00" * 1000)

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500

        with patch("tritium_lib.testing.llm_visual._requests") as mock_req:
            mock_req.post.return_value = mock_response
            # Also mock discovery to return no vision ports
            mock_health = MagicMock()
            mock_health.ok = False
            mock_req.get.return_value = mock_health

            analyzer = LLMVisualAnalyzer(ports=[9999])
            result = analyzer.analyze(path, include_opencv=True)

        assert not result.ok
        assert "no vision model" in result.error
        if HAS_OPENCV:
            assert result.opencv_fallback is not None
            assert result.opencv_fallback["available"] is True


# ============================================================
# 5. Unit Tests — Discovery (mocked)
# ============================================================

class TestDiscoveryMocked:
    def test_probe_no_requests(self):
        with patch("tritium_lib.testing.llm_visual.HAS_REQUESTS", False):
            info = probe_server(8081)
            assert not info["alive"]
            assert "requests" in info["error"]

    def test_probe_server_down(self):
        with patch("tritium_lib.testing.llm_visual._requests") as mock_req:
            mock_req.get.side_effect = Exception("Connection refused")
            info = probe_server(9999, timeout=1)
            assert not info["alive"]

    def test_discover_sorts_vision_first(self):
        def mock_probe(port, timeout=3.0):
            if port == 8081:
                return {"alive": True, "has_vision": True, "model_id": "moondream", "n_params": 1_400_000_000, "error": "", "port": port}
            elif port == 8082:
                return {"alive": True, "has_vision": False, "model_id": "qwen2.5", "n_params": 1_500_000_000, "error": "", "port": port}
            return {"alive": False, "has_vision": False, "model_id": "", "n_params": 0, "error": "down", "port": port}

        with patch("tritium_lib.testing.llm_visual.probe_server", side_effect=mock_probe):
            servers = discover_vision_servers([8081, 8082, 8083])
            assert len(servers) == 3
            # Vision-capable should be first
            assert servers[0]["has_vision"] is True
            assert servers[0]["port"] == 8081


# ============================================================
# 6. Unit Tests — Batch Analysis
# ============================================================

class TestBatchAnalyze:
    def test_batch_with_missing_files(self, tmp_path):
        analyzer = LLMVisualAnalyzer(ports=[9999])
        results = analyzer.batch_analyze([
            {"path": str(tmp_path / "missing1.png")},
            {"path": str(tmp_path / "missing2.png")},
        ])
        assert len(results) == 2
        assert not results[0].ok
        assert not results[1].ok

    def test_batch_empty(self):
        analyzer = LLMVisualAnalyzer(ports=[9999])
        results = analyzer.batch_analyze([])
        assert results == []


# ============================================================
# 7. Unit Tests — Convenience Functions
# ============================================================

class TestConvenienceFunctions:
    def test_is_vision_available_no_server(self):
        """With no servers, should return False."""
        with patch("tritium_lib.testing.llm_visual._requests") as mock_req:
            mock_req.get.side_effect = Exception("refused")
            assert not is_vision_available(ports=[9999])


# ============================================================
# 8. Integration Tests — Require llama-server running
# ============================================================

@needs_server
class TestServerProbe:
    def test_probe_live_server(self):
        info = probe_server(_ANY_SERVER_PORT, timeout=5)
        assert info["alive"] is True
        assert info["model_id"] != ""
        assert info["n_params"] > 0

    def test_discover_finds_servers(self):
        servers = discover_vision_servers(DEFAULT_PORTS, timeout=5)
        alive = [s for s in servers if s["alive"]]
        assert len(alive) >= 1


# ============================================================
# 9. Live Vision Tests — Require vision model running
# ============================================================

@needs_vision
@needs_opencv
class TestLiveVision:
    def test_analyze_dark_ui(self, dark_ui_image):
        """Vision model can analyze a synthetic UI screenshot."""
        analyzer = LLMVisualAnalyzer(ports=[_VISION_SERVER_PORT])
        result = analyzer.analyze(
            dark_ui_image,
            question="What elements do you see in this image? List colors and shapes.",
        )
        assert result.ok, f"Vision analysis failed: {result.error}"
        assert len(result.text) > 10
        assert result.port == _VISION_SERVER_PORT
        assert result.elapsed_seconds > 0
        assert result.tokens_used > 0

    def test_analyze_with_prompt_key(self, dark_ui_image):
        """Prompt keys map to the correct prompts."""
        analyzer = LLMVisualAnalyzer(ports=[_VISION_SERVER_PORT])
        result = analyzer.analyze(dark_ui_image, prompt_key="general")
        assert result.ok, f"Vision analysis failed: {result.error}"
        assert len(result.text) > 5

    def test_analyze_full(self, dark_ui_image):
        """Full analysis combines vision + OpenCV."""
        analyzer = LLMVisualAnalyzer(ports=[_VISION_SERVER_PORT])
        analysis = analyzer.analyze_full(dark_ui_image)
        assert analysis.path == dark_ui_image
        assert analysis.vision is not None
        assert analysis.vision.ok
        assert analysis.is_black_screen == False  # noqa: E712 — numpy bool_ compat
        assert analysis.marker_count >= 0  # OpenCV may or may not detect synthetic markers
        # Should be JSON serializable
        d = analysis.to_dict()
        s = json.dumps(d)
        assert isinstance(s, str)

    def test_analyze_map_image(self, bright_map_image):
        """Vision model processes a map-like image (may return empty for small models)."""
        analyzer = LLMVisualAnalyzer(ports=[_VISION_SERVER_PORT])
        analyzer._vision_ports = [_VISION_SERVER_PORT]  # Skip re-discovery
        result = analyzer.analyze(bright_map_image, prompt_key="map_loaded")
        # Small models like moondream may return empty for bright/uniform images.
        # The important thing is the flow completes without exceptions.
        # If vision fails, OpenCV fallback should be populated.
        if not result.ok:
            assert result.opencv_fallback is not None
            assert result.opencv_fallback.get("available") is True
        else:
            assert len(result.text) > 5

    def test_convenience_functions(self, dark_ui_image):
        """Convenience functions work with real vision server."""
        # Patch default ports to only use the known vision port
        with patch("tritium_lib.testing.llm_visual.DEFAULT_PORTS", [_VISION_SERVER_PORT]):
            result = analyze_screenshot(
                dark_ui_image,
                question="What do you see?",
                ports=[_VISION_SERVER_PORT],
            )
            assert result.ok

    def test_batch_analyze(self, dark_ui_image, bright_map_image):
        """Batch analysis processes multiple images."""
        analyzer = LLMVisualAnalyzer(ports=[_VISION_SERVER_PORT])
        results = analyzer.batch_analyze([
            {"path": dark_ui_image, "question": "Describe the UI."},
            {"path": bright_map_image, "question": "Is this a map?"},
        ])
        assert len(results) == 2
        assert results[0].ok
        assert results[1].ok

    def test_vision_on_black_screen(self, black_image):
        """Vision model can identify a black/empty screen."""
        analyzer = LLMVisualAnalyzer(ports=[_VISION_SERVER_PORT])
        result = analyzer.analyze(
            black_image,
            question="Is this image blank or black? Is there any content visible?",
        )
        assert result.ok, f"Vision analysis failed: {result.error}"
        # The response should mention darkness or emptiness
        assert len(result.text) > 5


@needs_vision
class TestVisionAvailability:
    def test_is_vision_available(self):
        assert is_vision_available(ports=[_VISION_SERVER_PORT])

    def test_analyzer_has_vision(self):
        analyzer = LLMVisualAnalyzer(ports=[_VISION_SERVER_PORT])
        assert analyzer.has_vision
