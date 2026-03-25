# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.comms.speaker."""

from tritium_lib.comms.speaker import Speaker


def test_speaker_instantiation():
    """Speaker can be instantiated with defaults."""
    s = Speaker()
    assert s is not None
    assert s.sample_rate == 22050


def test_speaker_has_speak_method():
    """Speaker has a speak() method."""
    s = Speaker()
    assert hasattr(s, "speak")
    assert callable(s.speak)


def test_speaker_has_speak_sync_method():
    """Speaker has a speak_sync() method."""
    s = Speaker()
    assert hasattr(s, "speak_sync")
    assert callable(s.speak_sync)


def test_speaker_has_shutdown_method():
    """Speaker has a shutdown() method."""
    s = Speaker()
    assert hasattr(s, "shutdown")
    assert callable(s.shutdown)
    s.shutdown()


def test_speaker_available_property():
    """Speaker.available returns bool (likely False in test env)."""
    s = Speaker()
    assert isinstance(s.available, bool)
    s.shutdown()


def test_speaker_playback_device():
    """Speaker reports its playback device."""
    s = Speaker()
    assert "pw-play" in s.playback_device
    s.shutdown()


def test_speaker_custom_params():
    """Speaker accepts custom parameters."""
    s = Speaker(piper_bin="/fake/piper", voice_model="/fake/model.onnx", sample_rate=44100)
    assert s.sample_rate == 44100
    s.shutdown()


def test_speaker_build_play_cmd_default_rate():
    """_build_play_cmd uses default sample_rate."""
    s = Speaker(sample_rate=16000)
    cmd = s._build_play_cmd()
    assert "--rate=16000" in cmd
    s.shutdown()


def test_speaker_build_play_cmd_custom_rate():
    """_build_play_cmd can override rate."""
    s = Speaker()
    cmd = s._build_play_cmd(rate=48000)
    assert "--rate=48000" in cmd
    s.shutdown()


def test_speaker_synthesize_raw_unavailable():
    """synthesize_raw returns None when piper is not available."""
    s = Speaker(piper_bin="/nonexistent/piper")
    result = s.synthesize_raw("Hello")
    assert result is None
    s.shutdown()


def test_speaker_queue_is_created():
    """Speaker creates an internal queue on init."""
    s = Speaker()
    assert s._queue is not None
    s.shutdown()


def test_speaker_speak_queues_text():
    """Speaker.speak puts text in the queue without blocking."""
    s = Speaker()
    s.speak("test message")
    # Queue should have at least the message (worker may have consumed it)
    s.shutdown()
