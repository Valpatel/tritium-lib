"""Tests for device-side flicker detection analysis.

Validates that FlickerAnalyzer correctly identifies flicker patterns
from frame stats data without requiring a real device.
"""

import pytest
from tritium_lib.testing.flicker import (
    FlickerAnalyzer,
    FlickerResult,
    parse_serial_log,
)


def make_stats(frames, dropped=0):
    """Build a frame stats dict from a list of (us, flush_count, jitter_us) tuples."""
    return {
        "target_fps": 60,
        "dropped": dropped,
        "count": len(frames),
        "frames": [
            {"us": us, "fl": fl, "j": j}
            for us, fl, j in frames
        ],
    }


class TestFlickerAnalyzerCleanFrames:
    """Verify that healthy frame data passes all checks."""

    def test_steady_60fps_passes(self):
        # 64 frames at ~16.6ms, 2 flushes each, low jitter
        frames = [(16500, 2, 200)] * 64
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        assert result.verdict == "PASS"
        assert len(result.issues) == 0

    def test_slight_variation_passes(self):
        # Normal frame time variation (15-18ms)
        import random
        random.seed(42)
        frames = [(random.randint(15000, 18000), 2, random.randint(0, 3000)) for _ in range(64)]
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        assert result.verdict == "PASS"

    def test_empty_buffer_warns(self):
        result = FlickerAnalyzer().analyze_stats(make_stats([]))
        assert result.verdict == "PASS"  # Warning, not failure
        assert len(result.issues) == 1
        assert result.issues[0].check == "no_data"


class TestFlickerAnalyzerDroppedFrames:
    """Detect frames exceeding 2x target period."""

    def test_many_dropped_frames_fails(self):
        # 50% of frames are >33ms = dropped
        frames = [(16000, 2, 500)] * 32 + [(50000, 2, 34000)] * 32
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        assert result.verdict == "FAIL"
        errors = [i for i in result.issues if i.check == "dropped_frames"]
        assert len(errors) == 1
        assert "32/64" in errors[0].message

    def test_few_dropped_frames_passes(self):
        # 3% dropped (2 out of 64) — below 10% threshold
        frames = [(16000, 2, 500)] * 62 + [(40000, 2, 24000)] * 2
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        dropped_errors = [i for i in result.issues if i.check == "dropped_frames"]
        assert len(dropped_errors) == 0


class TestFlickerAnalyzerFlushCount:
    """Detect high flush count (many partial updates per frame)."""

    def test_high_flush_count_errors(self):
        # Every frame has 8 flushes (>4 threshold, 100% > 50%) = ERROR
        frames = [(16000, 8, 500)] * 64
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        flush_issues = [i for i in result.issues if i.check == "flush_count"]
        assert len(flush_issues) == 1
        assert flush_issues[0].severity == "ERROR"

    def test_moderate_flush_count_warns(self):
        # ~30% of frames have high flush count (>25%, <50%) = WARNING
        frames = [(16000, 2, 500)] * 44 + [(16000, 6, 500)] * 20
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        flush_issues = [i for i in result.issues if i.check == "flush_count"]
        assert len(flush_issues) == 1
        assert flush_issues[0].severity == "WARNING"

    def test_low_flush_count_passes(self):
        # All frames have <=4 flushes
        frames = [(16000, 3, 500)] * 64
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        flush_issues = [i for i in result.issues if i.check == "flush_count"]
        assert len(flush_issues) == 0

    def test_occasional_spike_passes(self):
        # 5 out of 64 frames have high flush count (<25%) = not flagged
        frames = [(16000, 2, 500)] * 59 + [(16000, 7, 500)] * 5
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        flush_issues = [i for i in result.issues if i.check == "flush_count"]
        assert len(flush_issues) == 0


class TestFlickerAnalyzerJitter:
    """Detect inconsistent frame timing."""

    def test_high_jitter_fails(self):
        # 50% of frames have >20ms jitter
        frames = [(16000, 2, 500)] * 32 + [(16000, 2, 25000)] * 32
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        assert result.verdict == "FAIL"
        jitter_issues = [i for i in result.issues if i.check == "jitter"]
        assert len(jitter_issues) == 1

    def test_low_jitter_passes(self):
        # All frames have <5ms jitter
        frames = [(16000, 2, 4000)] * 64
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        jitter_issues = [i for i in result.issues if i.check == "jitter"]
        assert len(jitter_issues) == 0


class TestFlickerAnalyzerSustainedSlowness:
    """Detect consistently slow rendering."""

    def test_all_frames_slow_fails(self):
        # Average frame time 40ms >> 33ms threshold
        frames = [(40000, 2, 2000)] * 64
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        assert result.verdict == "FAIL"
        slow_issues = [i for i in result.issues if i.check == "sustained_slowness"]
        assert len(slow_issues) == 1

    def test_normal_speed_passes(self):
        frames = [(16000, 2, 500)] * 64
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        slow_issues = [i for i in result.issues if i.check == "sustained_slowness"]
        assert len(slow_issues) == 0


class TestFlickerAnalyzerEdgeCases:
    """Ensure single outliers and edge conditions don't cause false positives."""

    def test_single_bad_frame_passes(self):
        # One terrible frame among 63 good ones = should pass all checks
        frames = [(16000, 2, 500)] * 63 + [(80000, 10, 60000)]
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        assert result.verdict == "PASS"

    def test_wifi_interrupt_spike_passes(self):
        # WiFi interrupts cause periodic ~5ms spikes, 3 per 64 frames
        frames = [(16000, 2, 500)] * 61 + [(21000, 2, 5000)] * 3
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        assert result.verdict == "PASS"

    def test_single_frame_not_false_positive(self):
        # Only 1 frame in buffer (just booted)
        frames = [(16000, 2, 0)]
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        assert result.verdict == "PASS"

    def test_all_zeros_passes(self):
        # Device idle — 0 flush, 0 jitter
        frames = [(0, 0, 0)] * 10
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        # Should not crash or produce false positives
        assert result.frame_count == 10


class TestFlickerAnalyzerScrollScenario:
    """Simulate realistic scroll flicker patterns."""

    def test_scroll_flicker_pattern_detected(self):
        """Simulate the actual bug: scroll causes many flushes + frame drops."""
        # During scroll: 6-8 flushes per frame, some frames >33ms, high jitter
        frames = []
        for i in range(64):
            if i % 3 == 0:
                # Dropped frame during scroll redraw
                frames.append((45000, 7, 28000))
            else:
                # Normal frame but still lots of flushes
                frames.append((18000, 6, 5000))

        result = FlickerAnalyzer().analyze_stats(make_stats(frames, dropped=22))
        assert result.verdict == "FAIL"
        checks_hit = {i.check for i in result.issues}
        assert "dropped_frames" in checks_hit or "flush_count" in checks_hit

    def test_smooth_scroll_after_fix(self):
        """Simulate scroll after DMA sync fix: fewer flushes, stable timing."""
        # After fix: DMA wait reduces throughput but eliminates partial renders
        # Larger buffers mean fewer flush calls (2-3 instead of 6-8)
        frames = [(17000, 2, 1500)] * 64
        result = FlickerAnalyzer().analyze_stats(make_stats(frames))
        assert result.verdict == "PASS"
        assert len(result.issues) == 0


class TestSerialLogParsing:
    """Verify serial log parsing for WiFi-free flicker detection."""

    def test_parse_basic_frame_lines(self):
        log = (
            'I (1234) ui_stats: F:{"us":16500,"fl":2,"j":200}\n'
            'I (1250) ui_stats: F:{"us":17000,"fl":3,"j":500}\n'
        )
        frames = parse_serial_log(log)
        assert len(frames) == 2
        assert frames[0] == {"us": 16500, "fl": 2, "j": 200}
        assert frames[1] == {"us": 17000, "fl": 3, "j": 500}

    def test_parse_ignores_non_frame_lines(self):
        log = (
            'I (100) wifi: connected\n'
            'I (200) ui_stats: F:{"us":16000,"fl":2,"j":100}\n'
            'I (300) heap: free=123456\n'
        )
        frames = parse_serial_log(log)
        assert len(frames) == 1

    def test_parse_handles_malformed_json(self):
        log = (
            'I (100) ui_stats: F:{"us":16000,"fl":2,"j":100}\n'
            'I (200) ui_stats: F:{malformed}\n'
            'I (300) ui_stats: F:{"us":17000,"fl":2,"j":200}\n'
        )
        frames = parse_serial_log(log)
        assert len(frames) == 2

    def test_parse_empty_log(self):
        assert parse_serial_log("") == []
        assert parse_serial_log("\n\n\n") == []

    def test_end_to_end_serial_to_analyzer(self):
        """Full pipeline: serial log -> parse -> analyze -> verdict."""
        # Simulate flicker: high flush counts and jitter
        lines = []
        for i in range(64):
            if i % 3 == 0:
                lines.append(f'I ({i*16}) ui_stats: F:{{"us":45000,"fl":7,"j":28000}}')
            else:
                lines.append(f'I ({i*16}) ui_stats: F:{{"us":18000,"fl":6,"j":5000}}')

        frames = parse_serial_log("\n".join(lines))
        assert len(frames) == 64

        result = FlickerAnalyzer().analyze_stats({"frames": frames, "dropped": 0})
        assert result.verdict == "FAIL"

    def test_serial_clean_scroll_passes(self):
        """Clean serial data should pass analysis."""
        lines = [
            f'I ({i*16}) ui_stats: F:{{"us":16500,"fl":2,"j":300}}'
            for i in range(64)
        ]
        frames = parse_serial_log("\n".join(lines))
        result = FlickerAnalyzer().analyze_stats({"frames": frames, "dropped": 0})
        assert result.verdict == "PASS"
