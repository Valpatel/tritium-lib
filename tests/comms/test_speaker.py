# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.comms.speaker."""

import os

from tritium_lib.comms.speaker import Speaker


def _fake_piper(tmp_path, body: str):
    """Write an executable stand-in piper script and a dummy voice model.

    The script sees ``--help`` from the availability probe ($1 == --help)
    and ``--model <path> --output-raw`` from synthesis ($1 == --model), so
    each test controls both behaviors independently.
    """
    script = tmp_path / "piper"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(0o755)
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"onnx")
    return str(script), str(model)


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


# ---------------------------------------------------------------------------
# Availability = runnability (2026-07-18: dangling piper entry point served
# repeated TTS 500s while available reported True and stderr was swallowed)
# ---------------------------------------------------------------------------

def test_available_false_when_binary_exists_but_cannot_run(tmp_path):
    """THE incident: files exist, every spawn dies with ModuleNotFoundError.

    The old file-existence check reported available=True here.  Runnability
    must report False and last_error must name the actual root cause.
    """
    piper, model = _fake_piper(
        tmp_path,
        'echo "Traceback (most recent call last):" >&2\n'
        'echo "ModuleNotFoundError: No module named \'piper\'" >&2\n'
        "exit 1\n",
    )
    s = Speaker(piper_bin=piper, voice_model=model)
    assert s.available is False
    assert "No module named" in s.last_error
    s.shutdown()


def test_available_probe_runs_once_and_caches(tmp_path):
    """The probe must spawn at most one process, no matter how often
    available is read — availability of an install does not flap, and a
    per-call spawn was explicitly ruled out."""
    counter = tmp_path / "count"
    piper, model = _fake_piper(
        tmp_path,
        f'echo run >> "{counter}"\n'
        "exit 0\n",
    )
    s = Speaker(piper_bin=piper, voice_model=model)
    for _ in range(5):
        assert s.available is True
    assert counter.read_text().count("run") == 1
    s.shutdown()


def test_available_recheck_reprobes_after_a_fix(tmp_path):
    """recheck() drops the cached verdict so a repaired install is seen."""
    piper, model = _fake_piper(tmp_path, "exit 1\n")
    s = Speaker(piper_bin=piper, voice_model=model)
    assert s.available is False
    # Fix the install in place, same path.
    with open(piper, "w") as f:
        f.write("#!/bin/sh\nexit 0\n")
    os.chmod(piper, 0o755)
    assert s.available is False  # cached verdict holds until recheck
    assert s.recheck() is True
    assert s.available is True
    s.shutdown()


def test_available_missing_files_sets_reason(tmp_path):
    s = Speaker(
        piper_bin=str(tmp_path / "nope" / "piper"),
        voice_model=str(tmp_path / "nope" / "voice.onnx"),
    )
    assert s.available is False
    assert "not found" in s.last_error
    assert "piper" in s.last_error
    s.shutdown()


# ---------------------------------------------------------------------------
# synthesize_raw propagates stderr instead of swallowing it
# ---------------------------------------------------------------------------

def test_synthesize_raw_captures_stderr_on_failure(tmp_path):
    """Probe passes, synthesis fails: the caller must be able to log WHY
    without re-running piper themselves (which is what SC's tts router had
    to do while the lib swallowed stderr)."""
    piper, model = _fake_piper(
        tmp_path,
        'if [ "$1" = "--help" ]; then exit 0; fi\n'
        'echo "[E] failed to load model: corrupt onnx" >&2\n'
        "exit 1\n",
    )
    s = Speaker(piper_bin=piper, voice_model=model)
    assert s.available is True
    assert s.synthesize_raw("hello") is None
    assert "exited 1" in s.last_error
    assert "failed to load model: corrupt onnx" in s.last_error
    s.shutdown()


def test_synthesize_raw_reports_silent_empty_output(tmp_path):
    """Exit 0 with no audio is also a failure and must say so."""
    piper, model = _fake_piper(
        tmp_path,
        'if [ "$1" = "--help" ]; then exit 0; fi\n'
        "cat > /dev/null\n"
        "exit 0\n",
    )
    s = Speaker(piper_bin=piper, voice_model=model)
    assert s.synthesize_raw("hello") is None
    assert "produced no audio" in s.last_error
    s.shutdown()


def test_synthesize_raw_success_path_unchanged_and_clears_error(tmp_path):
    """Happy path: same bytes out as before the diagnostics work, and a
    success wipes any stale failure so last_error tracks the latest call."""
    piper, model = _fake_piper(
        tmp_path,
        'if [ "$1" = "--help" ]; then exit 0; fi\n'
        "cat > /dev/null\n"
        "printf 'PCM16DATA'\n"
        "exit 0\n",
    )
    s = Speaker(piper_bin=piper, voice_model=model)
    s.last_error = "stale failure from an earlier call"
    assert s.synthesize_raw("hello") == b"PCM16DATA"
    assert s.last_error is None
    s.shutdown()


def test_synthesize_raw_unavailable_reason_survives(tmp_path):
    """When unavailable, synthesize_raw returns None and the probe's root
    cause is still readable — the None is no longer unexplained."""
    piper, model = _fake_piper(
        tmp_path,
        'echo "ModuleNotFoundError: No module named \'piper\'" >&2\n'
        "exit 1\n",
    )
    s = Speaker(piper_bin=piper, voice_model=model)
    assert s.synthesize_raw("hello") is None
    assert "No module named" in s.last_error
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
