"""
Tests for city3d.html replay recording and playback features.
Source-string tests that verify the HTML file contains required code patterns.

Created by Matthew Valancy
Copyright 2026 Valpatel Software LLC
Licensed under AGPL-3.0
"""
import os
import pytest

CITY3D_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "city3d.html"
)


@pytest.fixture(scope="module")
def source():
    with open(CITY3D_PATH, "r") as f:
        return f.read()


# =========================================================================
# 1. RECORDING — capture sim state each tick into a JS array
# =========================================================================

class TestRecordingState:
    def test_replay_max_frames_constant(self, source):
        assert "REPLAY_MAX_FRAMES" in source, "Missing REPLAY_MAX_FRAMES constant"

    def test_replay_max_frames_value(self, source):
        assert "3000" in source, "REPLAY_MAX_FRAMES should be 3000 (5 min at 10fps)"

    def test_recording_flag(self, source):
        assert "replayRecording" in source, "Missing replayRecording state flag"

    def test_replay_frames_array(self, source):
        assert "replayFrames" in source, "Missing replayFrames array"

    def test_capture_function(self, source):
        assert "captureReplayFrame" in source, "Missing captureReplayFrame function"

    def test_capture_rate(self, source):
        assert "REPLAY_CAPTURE_STEP" in source, "Missing replay capture rate constant"


class TestRecordingContent:
    def test_captures_pedestrians(self, source):
        assert "pedestrians.filter" in source and "slot" in source, \
            "Recording should capture pedestrian positions"

    def test_captures_cars(self, source):
        assert "cars.filter" in source, \
            "Recording should capture car positions"

    def test_captures_protestors(self, source):
        assert "protestors.filter" in source, \
            "Recording should capture protestor positions"

    def test_captures_police(self, source):
        assert "police.filter" in source, \
            "Recording should capture police positions"

    def test_captures_taxis(self, source):
        assert "taxis.filter" in source, \
            "Recording should capture taxi positions"

    def test_captures_car_drivers(self, source):
        assert "carDrivers.filter" in source, \
            "Recording should capture car driver positions"

    def test_captures_position_fields(self, source):
        # Each entity should store x, z, rotY
        assert "x: p.x" in source or "x: c.x" in source, \
            "Recording should capture x position"

    def test_auto_stop_at_max(self, source):
        assert "toggleRecording" in source and "REPLAY_MAX_FRAMES" in source, \
            "Recording should auto-stop at max frames"


# =========================================================================
# 2. RECORDING UI — red REC button with pulsing indicator
# =========================================================================

class TestRecordingUI:
    def test_rec_button_element(self, source):
        assert 'id="rec-btn"' in source, "Missing REC button element"

    def test_rec_dot_indicator(self, source):
        assert "rec-dot" in source, "Missing pulsing red dot indicator"

    def test_rec_pulse_animation(self, source):
        assert "rec-pulse" in source, "Missing REC pulse CSS animation"

    def test_rec_timer_element(self, source):
        assert 'id="rec-timer"' in source, "Missing recording timer display"

    def test_recording_timer_function(self, source):
        assert "updateRecordingTimer" in source, \
            "Missing updateRecordingTimer function"

    def test_timer_formats_mmss(self, source):
        assert "padStart(2" in source, \
            "Timer should format as MM:SS with zero padding"

    def test_toggle_recording_function(self, source):
        assert "function toggleRecording" in source, \
            "Missing toggleRecording function"

    def test_recording_css_class(self, source):
        assert "recording" in source, \
            "REC button should toggle 'recording' CSS class"


# =========================================================================
# 3. PLAYBACK CONTROLS — timeline scrubber, play/pause, speed, frame counter
# =========================================================================

class TestPlaybackUI:
    def test_playback_bar_element(self, source):
        assert 'id="playback-bar"' in source, "Missing playback bar element"

    def test_play_button(self, source):
        assert 'id="pb-play"' in source, "Missing play/pause button"

    def test_scrubber_element(self, source):
        assert 'id="playback-scrubber"' in source, "Missing timeline scrubber"

    def test_progress_bar(self, source):
        assert 'id="playback-progress"' in source, "Missing playback progress bar"

    def test_frame_counter(self, source):
        assert 'id="playback-frame"' in source, "Missing frame counter display"

    def test_speed_buttons(self, source):
        assert "0.5x" in source and "2x" in source and "4x" in source, \
            "Missing speed control buttons"

    def test_stop_button(self, source):
        assert "stopPlayback" in source, "Missing stop button"


class TestPlaybackLogic:
    def test_playback_flag(self, source):
        assert "replayPlayback" in source, "Missing replayPlayback state flag"

    def test_playback_index(self, source):
        assert "replayIdx" in source, "Missing replayIdx playback position"

    def test_playback_speed(self, source):
        assert "replaySpeed" in source, "Missing replaySpeed variable"

    def test_apply_frame_function(self, source):
        assert "function applyReplayFrame" in source, \
            "Missing applyReplayFrame function"

    def test_toggle_playback_function(self, source):
        assert "function togglePlayback" in source, \
            "Missing togglePlayback function"

    def test_seek_function(self, source):
        assert "function seekPlayback" in source, \
            "Missing seekPlayback function"

    def test_speed_function(self, source):
        assert "function setPlaybackSpeed" in source, \
            "Missing setPlaybackSpeed function"

    def test_playback_hud_update(self, source):
        assert "updatePlaybackHUD" in source, \
            "Missing updatePlaybackHUD function"

    def test_auto_pause_at_end(self, source):
        # Playback should auto-pause when reaching the last frame
        assert "replayFrames.length - 1" in source, \
            "Playback should detect end of recording"


# =========================================================================
# 4. ANIMATE LOOP INTEGRATION — freeze sim during playback
# =========================================================================

class TestAnimateIntegration:
    def test_playback_skips_update(self, source):
        # During playback, the live sim update should be skipped
        assert "if (replayPlayback)" in source, \
            "Animate loop should check replayPlayback to skip live update"

    def test_recording_hooks_into_tick(self, source):
        assert "replayRecordAccum" in source, \
            "Recording should accumulate time for capture rate"

    def test_playback_advances_frames(self, source):
        assert "replayAccum" in source, \
            "Playback should use accumulator for frame advancement"


# =========================================================================
# 5. KEYBOARD SHORTCUTS — K for record, L for playback
# =========================================================================

class TestKeyboardShortcuts:
    def test_key_k_recording(self, source):
        assert "KeyK" in source, "Missing K key binding for recording"

    def test_key_l_playback(self, source):
        assert "KeyL" in source, "Missing L key binding for playback"

    def test_controls_show_k(self, source):
        assert "<span>K</span>" in source and "Rec" in source, \
            "Controls bar should show K key for recording"

    def test_controls_show_l(self, source):
        assert "<span>L</span>" in source and "Play" in source, \
            "Controls bar should show L key for playback"


# =========================================================================
# 6. WINDOW SCOPE — functions accessible from onclick handlers
# =========================================================================

class TestWindowScope:
    def test_toggle_recording_on_window(self, source):
        assert "window.toggleRecording" in source, \
            "toggleRecording must be on window for onclick"

    def test_toggle_playback_on_window(self, source):
        assert "window.togglePlayback" in source, \
            "togglePlayback must be on window for onclick"

    def test_stop_playback_on_window(self, source):
        assert "window.stopPlayback" in source, \
            "stopPlayback must be on window for onclick"

    def test_set_speed_on_window(self, source):
        assert "window.setPlaybackSpeed" in source, \
            "setPlaybackSpeed must be on window for onclick"

    def test_seek_on_window(self, source):
        assert "window.seekPlayback" in source, \
            "seekPlayback must be on window for onclick"


# =========================================================================
# 7. CSS STYLING — cyberpunk aesthetic for replay controls
# =========================================================================

class TestReplayCSS:
    def test_rec_button_red_color(self, source):
        assert "#ff2a6d" in source, "REC button should use magenta/red color"

    def test_playback_bar_background(self, source):
        assert "#0a0a0f" in source, "Playback bar should use void background"

    def test_playback_buttons_cyan(self, source):
        assert "#00f0ff" in source, "Playback buttons should use cyan color"

    def test_frame_counter_green(self, source):
        assert "#05ffa1" in source, "Frame counter should use green color"

    def test_scrubber_progress_color(self, source):
        assert "playback-progress" in source and "#00f0ff" in source, \
            "Scrubber progress should be cyan"
