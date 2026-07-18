# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Hearing a legged body's turn past the sound of its own walking.

A wheeled body's measured yaw rate is its turn.  A legged body's is its turn
plus the left-right rock of its own gait, and on a live Newton-stepped Go2 the
rock is nearly as large as the entire command: standard deviation 0.567 rad/s
against a commanded magnitude capped at 0.800.  Feeding that to
:class:`~tritium_lib.control.yaw_rate_loop.YawRateLoop` does not merely add
noise — it injects stride-synchronous asymmetry at exactly the gait rhythm,
which is the textbook way to shake a legged body apart.  It was measured
doing so: loop gain up, body tilt up, monotonically.

The fix is the oldest one in the book and it is chosen for a specific
mathematical property rather than for smoothness.  A boxcar (moving average)
over a window of exactly ``T`` seconds has a frequency response of
``|sinc(f T)|``, which is **exactly zero** at ``f = 1/T`` and at *every*
harmonic of it.  Set ``T`` to one stride period and the gait's fundamental and
all its overtones are annihilated, not merely attenuated, while DC — the net
turn the loop actually wants — passes at unity gain.  No other single-pole
filter does that; a first-order low-pass with the same corner leaves the
fundamental at -3 dB and every harmonic above it partly intact.

**The cost, stated up front because it bounds what any loop built on this can
do.**  A boxcar's group delay is exactly half its window.  The Go2 gait these
runs use trots at 0.975 Hz — a 1.026 s period — so the filter costs ~0.51 s of
delay.  A feedback loop cannot be faster than the delay in its own measurement
path, so this filter caps the achievable rate-loop bandwidth at a fraction of
1 Hz.  That is not a defect of the filter; it is the honest statement of a
real constraint: **you cannot close a yaw-rate loop faster than the gait it
rides on.**  A caller wanting a fast loop must raise the stride frequency, not
lower the window.

Note that a quadruped's stride frequency is set by the *gait*, which is not
necessarily the follower's commanded cruise — on this driver the two are
decoupled (a gait table generated at 0.6 m/s while pure pursuit cruises at
0.3).  That is exactly why :meth:`from_stride_hz` takes the frequency as an
argument instead of deriving it from a speed: the null belongs on the
frequency the body actually emits, and only the caller holding the gait knows
what that is.

Consequently this class is also useful *without* any loop at all, purely as
the honest way to measure a legged body's steering authority offline — the
ratio of filtered-measured to commanded yaw rate is a plant gain; the ratio of
the raw signals is mostly gait.

Stdlib only, so it imports on a bare Jetson alongside the rest of the brain.
"""

from __future__ import annotations

from collections import deque

__all__ = ["StrideFilter"]


class StrideFilter:
    """Boxcar average over a fixed *time* window, with an exact gait null.

    The window is defined in seconds rather than in samples on purpose: the
    physics-step callback that drives this is not guaranteed to tick at a
    constant rate, and a sample-count window would silently retune its own
    null frequency whenever the step rate moved.
    """

    def __init__(self, window_s: float) -> None:
        if window_s <= 0.0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        self.window_s = float(window_s)
        self._samples: deque[tuple[float, float]] = deque()
        self._sum = 0.0
        self._span_seen = 0.0
        self._t0: float | None = None

    @classmethod
    def from_stride_hz(cls, stride_hz: float) -> "StrideFilter":
        """Window one full stride period, putting the null on the gait itself.

        This is the constructor callers should reach for: it ties the null to
        the body's actual cadence, so a gait that speeds up retunes the filter
        instead of leaving it notching a frequency nothing emits any more.
        """
        if stride_hz <= 0.0:
            raise ValueError(f"stride_hz must be > 0, got {stride_hz}")
        return cls(window_s=1.0 / float(stride_hz))

    @property
    def group_delay_s(self) -> float:
        """Half the window — the lag this filter costs a loop downstream."""
        return 0.5 * self.window_s

    @property
    def ready(self) -> bool:
        """Whether a full window has been seen, so the null is actually in force.

        Before this is true the output is an average over a partial window,
        which does *not* cancel the gait — it is a different, arbitrary filter
        whose null sits wherever the elapsed span happens to put it.  An
        integrator fed that during spin-up accumulates precisely the
        stride-synchronous error this class exists to remove, so callers are
        expected to gate on this rather than trust the first second of output.
        """
        return self._span_seen >= self.window_s

    def reset(self) -> None:
        """Forget all history — call between runs, not between ticks."""
        self._samples.clear()
        self._sum = 0.0
        self._span_seen = 0.0
        self._t0 = None

    def update(self, t_s: float, value: float) -> float:
        """Add ``value`` observed at ``t_s`` and return the windowed average.

        Time running backwards resets rather than corrupts.  Reusing one
        filter across two runs is a plausible mistake, and averaging across
        the seam would blend the tail of one run into the head of the next —
        a wrong number that looks entirely reasonable.
        """
        t = float(t_s)
        if self._t0 is not None and t < self._samples[-1][0]:
            self.reset()

        if self._t0 is None:
            self._t0 = t

        self._samples.append((t, float(value)))
        self._sum += float(value)

        # Evict everything strictly older than the window.  The comparison is
        # against the newest timestamp, not against wall time, so a stalled
        # or replayed clock cannot empty the window out from under the caller.
        cutoff = t - self.window_s
        while len(self._samples) > 1 and self._samples[0][0] < cutoff:
            _, dropped = self._samples.popleft()
            self._sum -= dropped

        self._span_seen = t - self._t0
        return self._sum / len(self._samples)

    def __len__(self) -> int:
        """Samples currently inside the window."""
        return len(self._samples)
