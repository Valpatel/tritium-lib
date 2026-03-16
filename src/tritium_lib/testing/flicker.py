"""Device-side flicker detection using frame stats from hal_diag or serial.

Supports two data sources:
1. REST API: /api/diag/frames endpoint (requires WiFi)
2. Serial: Per-frame JSON lines from ui_init.cpp (no WiFi needed)

Usage (REST):
    from tritium_lib.testing.flicker import FlickerAnalyzer
    from tritium_lib.testing.device import DeviceAPI

    device = DeviceAPI("http://10.42.0.237")
    analyzer = FlickerAnalyzer()
    result = analyzer.analyze(device)

Usage (Serial):
    from tritium_lib.testing.flicker import FlickerAnalyzer, capture_serial_frames

    frames = capture_serial_frames("/dev/ttyACM0", duration=3.0)
    analyzer = FlickerAnalyzer()
    result = analyzer.analyze_stats({"frames": frames, "dropped": 0})
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .device import DeviceAPI


def capture_serial_frames(
    port: str,
    duration: float = 3.0,
    baudrate: int = 115200,
) -> list[dict]:
    """Capture per-frame stats from serial output.

    Parses lines matching: I (timestamp) ui_stats: F:{"us":...,"fl":...,"j":...}
    Returns list of frame dicts compatible with FlickerAnalyzer.analyze_stats().
    """
    import serial

    frames = []
    pattern = re.compile(r'F:\{.*?"us":\d+.*?\}')

    ser = serial.Serial(port, baudrate, timeout=0.5)
    ser.reset_input_buffer()
    deadline = time.time() + duration

    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        m = pattern.search(line)
        if m:
            try:
                frame = json.loads(m.group(0)[2:])  # Strip "F:" prefix
                if "us" in frame and "fl" in frame and "j" in frame:
                    frames.append(frame)
            except json.JSONDecodeError:
                pass

    ser.close()
    return frames


def parse_serial_log(log_text: str) -> list[dict]:
    """Parse frame stats from a saved serial log string.

    Useful for offline analysis of captured serial output.
    """
    frames = []
    pattern = re.compile(r'F:\{.*?"us":\d+.*?\}')
    for line in log_text.splitlines():
        m = pattern.search(line)
        if m:
            try:
                frame = json.loads(m.group(0)[2:])
                if "us" in frame and "fl" in frame and "j" in frame:
                    frames.append(frame)
            except json.JSONDecodeError:
                pass
    return frames


@dataclass
class FlickerIssue:
    """A single flicker detection finding."""
    severity: str  # "ERROR" or "WARNING"
    check: str     # Which check found it
    message: str


@dataclass
class FlickerResult:
    """Result of flicker analysis on device frame stats."""
    verdict: str = "PASS"
    frame_count: int = 0
    dropped_frames: int = 0
    avg_frame_us: float = 0.0
    max_frame_us: int = 0
    avg_flush_count: float = 0.0
    max_flush_count: int = 0
    avg_jitter_us: float = 0.0
    max_jitter_us: int = 0
    issues: list[FlickerIssue] = field(default_factory=list)


class FlickerAnalyzer:
    """Analyze device frame stats for scroll flicker.

    Thresholds are calibrated for 60 FPS LVGL rendering on ESP32-S3.
    """

    def __init__(
        self,
        # A frame taking > 33ms (2x 16.6ms target) is "dropped"
        dropped_frame_threshold_us: int = 33333,
        # If >10% of frames are dropped, that's a flicker problem
        dropped_ratio_threshold: float = 0.10,
        # High flush count means many partial updates visible during scroll
        max_acceptable_flush_count: int = 4,
        # Frame-to-frame jitter > 20ms is perceptible
        jitter_threshold_us: int = 20000,
        # If >20% of frames have high jitter, it's perceptible flicker
        jitter_ratio_threshold: float = 0.20,
    ):
        self.dropped_frame_threshold_us = dropped_frame_threshold_us
        self.dropped_ratio_threshold = dropped_ratio_threshold
        self.max_acceptable_flush_count = max_acceptable_flush_count
        self.jitter_threshold_us = jitter_threshold_us
        self.jitter_ratio_threshold = jitter_ratio_threshold

    def analyze(self, device: DeviceAPI) -> FlickerResult:
        """Query device frame stats and check for flicker indicators.

        Call this after triggering a scroll gesture on the device.
        The device ring buffer holds the last 64 frames.
        """
        stats = device.frame_stats()
        if not stats:
            result = FlickerResult(verdict="ERROR")
            result.issues.append(FlickerIssue(
                severity="ERROR",
                check="connectivity",
                message="Could not retrieve frame stats from device",
            ))
            return result

        return self.analyze_stats(stats)

    def analyze_stats(self, stats: dict) -> FlickerResult:
        """Analyze raw frame stats dict (for testing without device)."""
        result = FlickerResult()
        frames = stats.get("frames", [])
        result.frame_count = len(frames)
        result.dropped_frames = stats.get("dropped", 0)

        if not frames:
            result.issues.append(FlickerIssue(
                severity="WARNING",
                check="no_data",
                message="No frame data in ring buffer — is the display active?",
            ))
            return result

        # Compute statistics
        frame_times = [f["us"] for f in frames]
        flush_counts = [f["fl"] for f in frames]
        jitters = [f["j"] for f in frames]

        result.avg_frame_us = sum(frame_times) / len(frame_times)
        result.max_frame_us = max(frame_times)
        result.avg_flush_count = sum(flush_counts) / len(flush_counts)
        result.max_flush_count = max(flush_counts)
        result.avg_jitter_us = sum(jitters) / len(jitters)
        result.max_jitter_us = max(jitters)

        # Check 1: Dropped frames ratio
        dropped = sum(1 for t in frame_times if t > self.dropped_frame_threshold_us)
        dropped_ratio = dropped / len(frame_times)
        if dropped_ratio > self.dropped_ratio_threshold:
            result.issues.append(FlickerIssue(
                severity="ERROR",
                check="dropped_frames",
                message=(
                    f"{dropped}/{len(frame_times)} frames exceeded "
                    f"{self.dropped_frame_threshold_us}us "
                    f"({dropped_ratio:.0%} > {self.dropped_ratio_threshold:.0%} threshold)"
                ),
            ))

        # Check 2: High flush count (many partial updates = visible tearing during scroll)
        # Only flag if a significant portion of frames have high flush count.
        # Occasional spikes from WiFi interrupts etc are normal.
        high_flush = sum(1 for c in flush_counts if c > self.max_acceptable_flush_count)
        high_flush_ratio = high_flush / len(frames)
        if high_flush_ratio > 0.25:
            result.issues.append(FlickerIssue(
                severity="WARNING" if high_flush_ratio < 0.5 else "ERROR",
                check="flush_count",
                message=(
                    f"{high_flush}/{len(frames)} frames had >{self.max_acceptable_flush_count} "
                    f"flush calls ({high_flush_ratio:.0%}, "
                    f"max={result.max_flush_count}, avg={result.avg_flush_count:.1f})"
                ),
            ))

        # Check 3: Jitter (inconsistent frame timing = perceptible flicker)
        high_jitter = sum(1 for j in jitters if j > self.jitter_threshold_us)
        jitter_ratio = high_jitter / len(jitters)
        if jitter_ratio > self.jitter_ratio_threshold:
            result.issues.append(FlickerIssue(
                severity="ERROR",
                check="jitter",
                message=(
                    f"{high_jitter}/{len(jitters)} frames had >{self.jitter_threshold_us}us jitter "
                    f"({jitter_ratio:.0%} > {self.jitter_ratio_threshold:.0%} threshold, "
                    f"max={result.max_jitter_us}us)"
                ),
            ))

        # Check 4: Sustained high frame time (everything is slow, not just spikes)
        if result.avg_frame_us > self.dropped_frame_threshold_us:
            result.issues.append(FlickerIssue(
                severity="ERROR",
                check="sustained_slowness",
                message=(
                    f"Average frame time {result.avg_frame_us:.0f}us exceeds "
                    f"target {self.dropped_frame_threshold_us}us — "
                    f"display pipeline is consistently too slow"
                ),
            ))

        # Set verdict
        errors = [i for i in result.issues if i.severity == "ERROR"]
        if errors:
            result.verdict = "FAIL"

        return result
