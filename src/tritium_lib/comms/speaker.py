# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Piper TTS output for Amy.

Uses ``pw-play`` (PipeWire) for audio playback.  PipeWire routes to the
default audio sink automatically.

Failure honesty (2026-07-18): ``available`` reflects *runnability* (a
one-time cached probe of the piper binary), not mere file existence, and
every failed synthesis leaves the real reason — including piper's captured
stderr — in :attr:`Speaker.last_error`.  Both changed after an SC
investigation traced repeated TTS 500s to a dangling piper entry point
(``ModuleNotFoundError: No module named 'piper'``) that this class reported
as available while swallowing the stderr that named the root cause.
"""

from __future__ import annotations

import os
import subprocess
import threading
import queue


DEFAULT_PIPER_DIR = os.path.expanduser("~/models/piper")
DEFAULT_PIPER_BIN = os.path.join(DEFAULT_PIPER_DIR, "piper")
DEFAULT_VOICE_MODEL = os.path.join(DEFAULT_PIPER_DIR, "en_US-amy-medium.onnx")

# How long the one-shot runnability probe may take.  ``piper --help`` loads
# no model, so a healthy binary answers in well under a second; the margin
# covers a cold page cache on a loaded box.
_PROBE_TIMEOUT_S = 15


def _stderr_tail(raw: bytes, lines: int = 3) -> str:
    """Last few non-empty stderr lines, joined — the part that names the bug."""
    text = raw.decode(errors="replace").strip()
    return " | ".join(
        line.strip() for line in text.splitlines()[-lines:] if line.strip()
    )


class Speaker:
    """Text-to-speech using Piper with queued playback via pw-play.

    Diagnostics contract: :attr:`available` is True only when piper can
    actually RUN (cached one-time probe, never a spawn per call), and after
    any failed synthesis :attr:`last_error` holds the real reason, stderr
    included — so a caller mapping a failure to an HTTP error can log the
    root cause instead of a bare None.
    """

    def __init__(
        self,
        piper_bin: str = DEFAULT_PIPER_BIN,
        voice_model: str = DEFAULT_VOICE_MODEL,
        sample_rate: int = 22050,
    ):
        self.piper_bin = os.path.abspath(piper_bin)
        self.voice_model = os.path.abspath(voice_model)
        self.sample_rate = sample_rate

        #: Why the most recent probe or synthesis failed (stderr included),
        #: or None.  A successful synthesis clears it; a successful probe
        #: leaves it untouched.
        self.last_error: str | None = None
        self._probe_lock = threading.Lock()
        self._probe_result: bool | None = None

        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def available(self) -> bool:
        """True only when piper can actually run — not merely when files exist.

        The old check (files exist) reported available=True for a dangling
        entry-point script whose every spawn died with ModuleNotFoundError,
        so callers kept synthesizing into 500s with no visible cause.  Now
        the files must exist AND a one-time ``piper --help`` probe must
        exit 0.  The probe is cached for the lifetime of this Speaker —
        never re-spawned per call — because an install does not flap;
        construct a new Speaker (or call :meth:`recheck`) after fixing an
        install.  When this returns False, :attr:`last_error` says why.
        """
        if not os.path.isfile(self.piper_bin):
            self.last_error = f"piper binary not found: {self.piper_bin}"
            return False
        if not os.path.isfile(self.voice_model):
            self.last_error = f"voice model not found: {self.voice_model}"
            return False
        return self._probe_runnable()

    def recheck(self) -> bool:
        """Drop the cached probe verdict and re-test availability once."""
        with self._probe_lock:
            self._probe_result = None
        return self.available

    def _probe_runnable(self) -> bool:
        """One-shot cached check that the piper binary can execute at all.

        ``piper --help`` is cheap (no model load) and fails fast on the
        real-world breakages: a dangling entry point (ModuleNotFoundError),
        a wrong-arch binary (ENOEXEC), a missing interpreter, a lost exec
        bit.  Runnable means: it spawned and exited 0 within the timeout.
        On failure the exception or the stderr tail lands in
        :attr:`last_error` so the root cause is loggable, not guessable.
        """
        with self._probe_lock:
            if self._probe_result is None:
                try:
                    proc = subprocess.run(
                        [self.piper_bin, "--help"],
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        timeout=_PROBE_TIMEOUT_S,
                    )
                except Exception as e:
                    self.last_error = (
                        f"piper probe failed: {type(e).__name__}: {e}"
                    )
                    self._probe_result = False
                else:
                    if proc.returncode == 0:
                        self._probe_result = True
                    else:
                        tail = _stderr_tail(proc.stderr)
                        self.last_error = (
                            f"piper --help exited {proc.returncode}"
                            + (f": {tail}" if tail else " with no stderr")
                        )
                        self._probe_result = False
            return self._probe_result

    @property
    def playback_device(self) -> str:
        """Human-readable name of the playback device for boot log."""
        return "pw-play (default sink)"

    def speak(self, text: str) -> None:
        self._queue.put(text)

    def speak_sync(self, text: str) -> None:
        self._synthesize_and_play(text)

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                break
            self._synthesize_and_play(text)

    def _build_play_cmd(self, rate: int | None = None) -> list[str]:
        """Build pw-play command for raw S16 PCM from stdin."""
        r = str(rate or self.sample_rate)
        return ["pw-play", "--format=s16", f"--rate={r}", "--channels=1", "-"]

    def _synthesize_and_play(self, text: str) -> None:
        if not self.available:
            reason = f" ({self.last_error})" if self.last_error else ""
            print(f'  [TTS unavailable] "{text}"{reason}')
            return

        # 2026-05-03: every Popen(stdin=PIPE, stdout=PIPE, stderr=PIPE)
        # call leaves three BufferedReader/Writer file objects holding
        # subprocess pipe FDs.  The previous version closed stdin/stdout
        # on the happy path but never closed player.stderr and never
        # closed anything on TimeoutExpired -- which under sustained
        # voice activity (commander narration during a battle) leaked
        # one FD set per utterance.  After ~60 utterances the SC event
        # loop ends up in a futex_wait somewhere downstream and stops
        # accepting requests entirely (observed twice today: 13h13m and
        # 1h10m hang times).  Always close the pipe handles, even on
        # exception.
        piper = None
        player = None
        try:
            piper = subprocess.Popen(
                [self.piper_bin, "--model", self.voice_model, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            player = subprocess.Popen(
                self._build_play_cmd(),
                stdin=piper.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            # Hand piper's stdout pipe to player and drop our reference
            # so the OS reclaims the FD when piper exits.
            piper.stdout.close()
            piper.stdout = None
            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()
            piper.stdin = None
            player.wait(timeout=60)
            piper.wait(timeout=5)
            play_err = player.stderr.read().decode(errors="replace").strip()
            if player.returncode != 0 and play_err:
                print(f"  [playback error] {play_err}")
        except subprocess.TimeoutExpired:
            print("  [TTS timeout]")
            for p in (piper, player):
                if p is None:
                    continue
                try:
                    p.kill()
                    p.wait(timeout=2)
                except Exception:
                    pass
        except Exception as e:
            print(f"  [TTS error] {e}")
        finally:
            # Defensive close of every pipe FD we may have opened.
            # Using p.stdin / p.stdout / p.stderr because Popen owns
            # those file objects and closing them releases the kernel
            # FDs even if .wait() never ran.
            for p in (piper, player):
                if p is None:
                    continue
                for handle in (p.stdin, p.stdout, p.stderr):
                    if handle is None:
                        continue
                    try:
                        handle.close()
                    except Exception:
                        pass

    def synthesize_raw(self, text: str) -> bytes | None:
        """Synthesize ``text`` to raw S16 PCM bytes, or None on failure.

        The success path is unchanged: same command, same timeout, same
        bytes returned.  What changed (2026-07-18): failure no longer
        discards the evidence.  On None, :attr:`last_error` holds the real
        reason — the exception, or piper's exit code plus its stderr tail —
        so the caller can log the root cause instead of re-running piper
        just to see what broke.  A successful call clears it.
        """
        if not self.available:
            # last_error already set by the availability check/probe.
            return None
        try:
            proc = subprocess.run(
                [self.piper_bin, "--model", self.voice_model, "--output-raw"],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
        except Exception as e:
            self.last_error = f"piper failed: {type(e).__name__}: {e}"
            return None
        if proc.returncode != 0 or not proc.stdout:
            tail = _stderr_tail(proc.stderr)
            self.last_error = (
                f"piper exited {proc.returncode}"
                + ("" if proc.stdout else ", produced no audio")
                + (f": {tail}" if tail else "")
            )
            return None
        self.last_error = None
        return proc.stdout

    def play_raw(self, raw_audio: bytes, rate: int | None = None) -> None:
        try:
            subprocess.run(
                self._build_play_cmd(rate),
                input=raw_audio,
                timeout=60,
            )
        except Exception as e:
            print(f"  [playback error] {e}")
