"""ESP32 device API client for remote UI testing.

Wraps the Tritium-OS REST API for screenshot capture, touch injection,
app navigation, and diagnostics.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests


DISPLAY_W = 800
DISPLAY_H = 480


@dataclass
class TouchDebug:
    hw_available: bool = False
    driver: str = "none"
    read_cb_calls: int = 0
    hw_touch_count: int = 0
    inject_count: int = 0
    last_raw_x: int = -1
    last_raw_y: int = -1
    currently_pressed: bool = False


@dataclass
class AppInfo:
    index: int
    name: str
    description: str
    system: bool


class DeviceAPI:
    """Client for Tritium-OS device REST API."""

    def __init__(self, host: str, timeout: float = 10.0):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.width = DISPLAY_W
        self.height = DISPLAY_H

    def _get(self, path: str, **kwargs) -> requests.Response:
        return requests.get(
            f"{self.host}{path}",
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

    def _post(self, path: str, **kwargs) -> requests.Response:
        return requests.post(
            f"{self.host}{path}",
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

    def screenshot_raw(self) -> Optional[np.ndarray]:
        """Capture a raw RGB565 screenshot and convert to BGR uint8 (480, 800, 3)."""
        r = self._get("/api/remote/screenshot")
        if r.status_code != 200:
            return None
        npix = self.width * self.height
        if len(r.content) < npix * 2:
            return None
        rgb565 = np.frombuffer(r.content[: npix * 2], dtype=np.uint16)
        r_ch = ((rgb565 >> 11) & 0x1F).astype(np.int32) * 255 // 31
        g_ch = ((rgb565 >> 5) & 0x3F).astype(np.int32) * 255 // 63
        b_ch = (rgb565 & 0x1F).astype(np.int32) * 255 // 31
        return (
            np.stack([b_ch, g_ch, r_ch], axis=-1)
            .astype(np.uint8)
            .reshape(self.height, self.width, 3)
        )

    def apps(self) -> list[AppInfo]:
        """List registered shell apps."""
        r = self._get("/api/shell/apps")
        data = r.json()
        return [
            AppInfo(
                index=a["index"],
                name=a["name"],
                description=a["description"],
                system=a["system"],
            )
            for a in data.get("apps", [])
        ]

    def launch(self, index: int) -> bool:
        """Launch an app by index."""
        r = self._post("/api/shell/launch", json={"index": index})
        return r.json().get("ok", False)

    def home(self) -> None:
        """Navigate to launcher."""
        try:
            self._post("/api/shell/home", timeout=3)
        except Exception:
            pass
        time.sleep(0.5)

    def tap(self, x: int, y: int) -> bool:
        """Inject a tap at (x, y)."""
        r = self._post("/api/remote/tap", json={"x": x, "y": y})
        return r.status_code == 200

    def touch_debug(self) -> TouchDebug:
        """Read touch subsystem diagnostics."""
        r = self._get("/api/debug/touch", timeout=3)
        d = r.json()
        return TouchDebug(
            hw_available=d.get("hw_available", False),
            driver=d.get("driver", "none"),
            read_cb_calls=d.get("read_cb_calls", 0),
            hw_touch_count=d.get("hw_touch_count", 0),
            inject_count=d.get("inject_count", 0),
            last_raw_x=d.get("last_raw_x", -1),
            last_raw_y=d.get("last_raw_y", -1),
            currently_pressed=d.get("currently_pressed", False),
        )

    def frame_stats(self) -> Optional[dict]:
        """Get per-frame flush stats ring buffer for flicker detection.

        Returns dict with keys:
            target_fps (int): Target frame rate (60)
            dropped (int): Total frames exceeding 2x target period
            count (int): Number of frames in ring buffer
            frames (list[dict]): Recent frames with:
                us (int): Total frame flush duration in microseconds
                fl (int): Number of flush calls this frame
                j (int): Jitter vs previous frame in microseconds
        """
        try:
            r = self._get("/api/diag/frames", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def swipe(self, x1: int, y1: int, x2: int, y2: int,
              duration_ms: int = 300) -> bool:
        """Inject a swipe gesture from (x1,y1) to (x2,y2)."""
        try:
            r = self._post("/api/remote/swipe", json={
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "duration_ms": duration_ms,
            })
            return r.status_code == 200
        except Exception:
            return False

    def disable_screensaver(self) -> bool:
        """Disable the screensaver (dismiss + set timeout to 0)."""
        try:
            r = self._post("/api/shell/screensaver", json={"action": "disable"}, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def enable_screensaver(self) -> bool:
        """Re-enable the screensaver (reload settings from NVS)."""
        try:
            r = self._post("/api/shell/screensaver", json={"action": "enable"}, timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def is_reachable(self) -> bool:
        """Check if the device web server is responding."""
        try:
            r = self._get("/api/shell/apps", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
