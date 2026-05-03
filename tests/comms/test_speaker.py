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


def test_speaker_synthesize_and_play_does_not_leak_pipes():
    """Regression guard for 2026-05-03 SC server hangs.

    Speaker._synthesize_and_play used to leak three subprocess pipe
    file descriptors per utterance (piper.stdin, piper.stdout, and
    player.stderr) on both the happy path AND on TimeoutExpired.
    Under sustained voice activity the SC event loop ended up
    deadlocked in futex_wait somewhere downstream of the leaked
    subprocess waitpid state -- twice today (13h13m and 1h10m
    uptime).

    This test runs the function many times against an unavailable
    piper binary (so it short-circuits without actually spawning
    subprocesses), proving the cleanup path is safe to call
    repeatedly even when the function returns early.  The deeper
    pipe-close behavior is exercised by the live-server tests that
    monitor FD count over time.
    """
    import os

    s = Speaker(piper_bin="/nonexistent/piper")
    # Available is false, so _synthesize_and_play should print and
    # return without touching subprocess.  Repeat many times -- we
    # are checking that the early-return path doesn't leak FDs
    # via accidental Popen() before the availability check.
    fd_before = len(os.listdir(f"/proc/{os.getpid()}/fd"))
    for _ in range(200):
        s._synthesize_and_play("hello")
    fd_after = len(os.listdir(f"/proc/{os.getpid()}/fd"))
    s.shutdown()
    # 200 calls should add at most a handful of FDs (e.g. one
    # /dev/null open if any).  Anything > 20 indicates the early
    # return is opening pipes.
    assert fd_after - fd_before < 20, (
        f"FD count grew by {fd_after - fd_before} after 200 short-circuit "
        f"_synthesize_and_play calls -- suspect pipe leak"
    )


def test_speaker_synthesize_and_play_finally_block_present():
    """The finally-block that closes pipe FDs must remain in source.

    Static guard: read the source file and assert the finally block
    is present in _synthesize_and_play.  Cheap defense against a
    future refactor accidentally removing the cleanup that
    prevented the SC server hangs of 2026-05-03.
    """
    import inspect
    from tritium_lib.comms import speaker as _sp

    src = inspect.getsource(_sp._synthesize_and_play.__func__) \
        if hasattr(_sp, '_synthesize_and_play') \
        else inspect.getsource(_sp.Speaker._synthesize_and_play)
    assert "finally:" in src, (
        "Speaker._synthesize_and_play must keep its finally-block; "
        "removing it caused the 2026-05-03 SC server hangs"
    )
    assert "handle.close()" in src, (
        "Speaker._synthesize_and_play must close pipe handles in finally"
    )
