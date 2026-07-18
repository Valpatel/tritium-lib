# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""The filter that lets a rate loop hear past the body's own gait.

The tests that matter here are the two nulls.  A legged body's measured yaw
rate is its net turn *plus* the left-right rock of its own trot, and tick 18
measured the rock at nearly the amplitude of the entire command signal.  A
boxcar averaged over exactly one stride period has an exact zero at that
period and at every harmonic of it — that is the property being bought, so it
is the property under test, not merely "output is smoother than input".
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.control import StrideFilter


def _drive(filt, signal, duration_s, rate_hz):
    """Feed ``signal(t)`` at ``rate_hz`` for ``duration_s``; return samples."""
    out = []
    n = int(duration_s * rate_hz)
    for i in range(n):
        t = i / rate_hz
        out.append((t, filt.update(t, signal(t))))
    return out


class TestConstruction:
    def test_window_must_be_positive(self):
        with pytest.raises(ValueError):
            StrideFilter(window_s=0.0)
        with pytest.raises(ValueError):
            StrideFilter(window_s=-1.0)

    def test_from_stride_hz_uses_one_full_period(self):
        # The domain-meaningful constructor: the null must land on the gait.
        filt = StrideFilter.from_stride_hz(0.52)
        assert filt.window_s == pytest.approx(1.0 / 0.52)

    def test_from_stride_hz_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            StrideFilter.from_stride_hz(0.0)

    def test_group_delay_is_half_the_window(self):
        # Not a detail: this is the cost the loop pays, and the number the
        # bandwidth argument is built on.  Pinned so it cannot drift silently.
        assert StrideFilter(window_s=2.0).group_delay_s == pytest.approx(1.0)


class TestReadiness:
    def test_not_ready_until_a_full_window_has_been_seen(self):
        # A partial-window average does NOT null the oscillation.  Feeding one
        # to an integrator injects the exact transient the filter exists to
        # remove, so callers need to be able to tell the difference.
        filt = StrideFilter(window_s=1.0)
        filt.update(0.0, 1.0)
        assert not filt.ready
        filt.update(0.5, 1.0)
        assert not filt.ready
        filt.update(1.0, 1.0)
        assert filt.ready

    def test_reset_clears_readiness_and_history(self):
        filt = StrideFilter(window_s=1.0)
        _drive(filt, lambda t: 5.0, 2.0, 50.0)
        assert filt.ready
        filt.reset()
        assert not filt.ready
        assert filt.update(9.0, 0.0) == pytest.approx(0.0)


class TestNulls:
    def test_exact_null_at_the_stride_fundamental(self):
        # The headline property.  A pure oscillation at the window frequency
        # must average to zero once the window is full.
        stride_hz = 0.52
        filt = StrideFilter.from_stride_hz(stride_hz)
        samples = _drive(
            filt, lambda t: math.sin(2 * math.pi * stride_hz * t), 12.0, 60.0
        )
        settled = [v for t, v in samples if t > filt.window_s]
        assert max(abs(v) for v in settled) < 0.02

    def test_null_at_the_second_harmonic(self):
        # Tick 18's 1.04 Hz oscillation is 2x the 0.52 Hz stride, so the
        # harmonic null is the one that actually earns its keep here.
        stride_hz = 0.52
        filt = StrideFilter.from_stride_hz(stride_hz)
        samples = _drive(
            filt, lambda t: math.sin(2 * math.pi * 2 * stride_hz * t), 12.0, 60.0
        )
        settled = [v for t, v in samples if t > filt.window_s]
        assert max(abs(v) for v in settled) < 0.02

    def test_recovers_the_turn_hiding_under_the_gait(self):
        # The real signal: a constant net turn buried under a stride rock
        # whose amplitude dwarfs it — tick 18's measured situation.
        stride_hz = 0.52
        true_turn = 0.10
        filt = StrideFilter.from_stride_hz(stride_hz)
        samples = _drive(
            filt,
            lambda t: true_turn + 0.567 * math.sin(2 * math.pi * stride_hz * t),
            12.0,
            60.0,
        )
        settled = [v for t, v in samples if t > filt.window_s]
        assert all(abs(v - true_turn) < 0.02 for v in settled)

    def test_unfiltered_signal_really_is_swamped(self):
        # Control for the test above: without the filter the same input is
        # dominated by the gait, so the assertion above is not vacuous.
        stride_hz = 0.52
        raw = [
            0.10 + 0.567 * math.sin(2 * math.pi * stride_hz * (i / 60.0))
            for i in range(720)
        ]
        assert max(abs(v - 0.10) for v in raw) > 0.5


class TestPassthrough:
    def test_dc_passes_unchanged(self):
        filt = StrideFilter(window_s=1.0)
        samples = _drive(filt, lambda t: 0.42, 3.0, 60.0)
        assert all(v == pytest.approx(0.42) for _, v in samples)

    def test_slow_signal_survives_with_expected_lag(self):
        # A turn that evolves slower than the window must still get through —
        # a filter that removed it too would make the loop blind, not clean.
        filt = StrideFilter(window_s=1.0)
        samples = _drive(filt, lambda t: 0.05 * t, 8.0, 60.0)
        t_end, v_end = samples[-1]
        # Boxcar on a ramp lags by exactly the group delay.
        assert v_end == pytest.approx(0.05 * (t_end - filt.group_delay_s), abs=0.01)


class TestSampling:
    def test_sample_rate_agnostic(self):
        # Same physical signal at two rates must give the same answer: the
        # window is defined in seconds, not in samples, because the driver's
        # step cadence is not guaranteed constant.
        stride_hz = 0.52

        def sig(t):
            return 0.2 + 0.5 * math.sin(2 * math.pi * stride_hz * t)

        a = _drive(StrideFilter.from_stride_hz(stride_hz), sig, 10.0, 60.0)[-1][1]
        b = _drive(StrideFilter.from_stride_hz(stride_hz), sig, 10.0, 240.0)[-1][1]
        assert a == pytest.approx(b, abs=0.02)

    def test_repeated_timestamp_does_not_divide_by_zero(self):
        # A paused or re-entrant step callback repeats a timestamp; that is a
        # normal event and must not poison the filter.
        filt = StrideFilter(window_s=1.0)
        filt.update(0.5, 1.0)
        assert filt.update(0.5, 3.0) == pytest.approx(2.0)

    def test_time_going_backwards_resets_rather_than_corrupts(self):
        # Restarting a run without constructing a new filter is a foot-gun
        # worth handling explicitly instead of silently averaging across it.
        filt = StrideFilter(window_s=1.0)
        _drive(filt, lambda t: 5.0, 2.0, 50.0)
        assert filt.update(0.0, 1.0) == pytest.approx(1.0)
        assert not filt.ready

    def test_window_never_grows_without_bound(self):
        # 60 Hz for a long run must not accumulate samples forever.
        filt = StrideFilter(window_s=1.0)
        _drive(filt, lambda t: 1.0, 60.0, 60.0)
        assert len(filt) <= 70
