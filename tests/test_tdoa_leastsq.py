# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for REAL acoustic TDoA multilateration (GCC-PHAT + least-squares).

The legacy compute_tdoa_position is an inverse-distance weighted CENTROID — it
can only ever place the source somewhere *between* the sensors, so it cannot
localize a source outside the sensor hull and its "residual" is fabricated
(max_dt * c * (1 - sync)), not a real fit error. These tests pin the honest
behavior of the new solver:

  * gcc_phat() recovers a known inter-channel delay from raw audio.
  * compute_tdoa_position_leastsq() solves the hyperbolic TDoA equations and
    recovers a source OUTSIDE the sensor hull (where the centroid cannot reach),
    with a small, REAL residual — and beats the centroid by a wide margin.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tritium_lib.models.acoustic_tdoa import (
    TDoAObservation,
    compute_tdoa_position,
    compute_tdoa_position_leastsq,
    SPEED_OF_SOUND_MPS,
)
from tritium_lib.signals.gcc_phat import gcc_phat


# --- local ENU<->lat/lon helpers for ground-truth construction (planar approx) ---
LAT0 = 40.0
LON0 = -74.0
_M_PER_DEG_LAT = 110540.0
_M_PER_DEG_LON = 111320.0 * math.cos(math.radians(LAT0))


def enu_to_ll(x_east_m: float, y_north_m: float) -> tuple[float, float]:
    return LAT0 + y_north_m / _M_PER_DEG_LAT, LON0 + x_east_m / _M_PER_DEG_LON


def ll_dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dy = (lat2 - lat1) * _M_PER_DEG_LAT
    dx = (lon2 - lon1) * _M_PER_DEG_LON
    return math.hypot(dx, dy)


class TestGccPhat:
    def test_recovers_known_integer_delay(self):
        fs = 8000.0
        rng = np.random.default_rng(42)
        base = rng.standard_normal(2048)
        delay_samples = 17
        # mic2 is mic1 delayed by `delay_samples` (zeros pre-pended, tail trimmed)
        mic1 = base
        mic2 = np.concatenate([np.zeros(delay_samples), base])[: base.shape[0]]
        tau, cc = gcc_phat(mic2, mic1, fs=fs, interp=16)
        # tau is the delay of mic2 relative to mic1 -> +delay_samples/fs
        assert abs(tau - delay_samples / fs) < (1.0 / fs)  # within one sample

    def test_zero_delay_is_zero(self):
        fs = 16000.0
        rng = np.random.default_rng(7)
        sig = rng.standard_normal(1024)
        tau, _ = gcc_phat(sig, sig.copy(), fs=fs, interp=8)
        assert abs(tau) < (1.0 / fs)

    def test_max_tau_clamps_search(self):
        fs = 8000.0
        rng = np.random.default_rng(1)
        sig = rng.standard_normal(1024)
        # max_tau small -> shift search window bounded; still returns a finite tau
        tau, _ = gcc_phat(sig, sig.copy(), fs=fs, max_tau=0.005, interp=4)
        assert abs(tau) <= 0.005 + 1.0 / fs


class TestLeastSquaresMultilateration:
    def _square_sensors(self, half_m: float = 100.0):
        """4 sensors at the corners of a 2*half_m square centered on (LAT0, LON0)."""
        corners = [(-half_m, -half_m), (half_m, -half_m),
                   (-half_m, half_m), (half_m, half_m)]
        return [enu_to_ll(x, y) for x, y in corners]

    def _make_obs(self, sensors_ll, src_ll, *, sync=1.0, base_ms=1_000_000.0,
                  noise_ms=0.0, rng=None):
        obs = []
        for i, (slat, slon) in enumerate(sensors_ll):
            d = ll_dist_m(slat, slon, src_ll[0], src_ll[1])
            t = base_ms + (d / SPEED_OF_SOUND_MPS) * 1000.0
            if noise_ms and rng is not None:
                t += rng.normal(0.0, noise_ms)
            obs.append(TDoAObservation(
                sensor_id=f"mic{i}", arrival_time_ms=t,
                lat=slat, lon=slon, ntp_sync_quality=sync, confidence=1.0,
            ))
        return obs

    def test_recovers_source_inside_hull(self):
        sensors = self._square_sensors(100.0)
        src = enu_to_ll(30.0, -40.0)  # inside the square
        obs = self._make_obs(sensors, src)
        r = compute_tdoa_position_leastsq(obs)
        assert r is not None
        err = ll_dist_m(r.lat, r.lon, src[0], src[1])
        assert err < 5.0, f"recovered {err:.1f} m from truth"
        assert r.method == "tdoa_leastsq"
        assert r.residual_error_m < 2.0  # REAL, small fit error on clean data

    def test_recovers_source_OUTSIDE_hull_where_centroid_fails(self):
        sensors = self._square_sensors(100.0)
        src = enu_to_ll(40.0, 320.0)  # well NORTH of the sensor square (outside)
        obs = self._make_obs(sensors, src)

        ls = compute_tdoa_position_leastsq(obs)
        cen = compute_tdoa_position(obs)  # legacy centroid
        assert ls is not None and cen is not None

        ls_err = ll_dist_m(ls.lat, ls.lon, src[0], src[1])
        cen_err = ll_dist_m(cen.lat, cen.lon, src[0], src[1])
        # The centroid is trapped inside the hull (~>200 m off); least-squares nails it.
        assert ls_err < 15.0, f"leastsq {ls_err:.1f} m"
        assert cen_err > 150.0, f"centroid only {cen_err:.1f} m (expected far)"
        assert ls_err < cen_err / 5.0

    def test_residual_grows_with_timing_noise(self):
        sensors = self._square_sensors(120.0)
        src = enu_to_ll(10.0, 20.0)
        clean = compute_tdoa_position_leastsq(self._make_obs(sensors, src))
        rng = np.random.default_rng(123)
        noisy = compute_tdoa_position_leastsq(
            self._make_obs(sensors, src, noise_ms=2.0, rng=rng)
        )
        assert clean is not None and noisy is not None
        # Real fit error: noisy data should not produce a ~0 residual like the cheat did.
        assert noisy.residual_error_m > clean.residual_error_m

    def test_fewer_than_3_returns_none(self):
        sensors = self._square_sensors()[:2]
        src = enu_to_ll(0.0, 0.0)
        assert compute_tdoa_position_leastsq(self._make_obs(sensors, src)) is None

    def test_confidence_in_range(self):
        sensors = self._square_sensors(100.0)
        src = enu_to_ll(20.0, 20.0)
        r = compute_tdoa_position_leastsq(self._make_obs(sensors, src))
        assert r is not None
        assert 0.0 <= r.confidence <= 1.0
        assert r.confidence > 0.5  # clean geometry + perfect sync -> confident
