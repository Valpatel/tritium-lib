# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""GCC-PHAT — Generalized Cross-Correlation with Phase Transform.

The standard estimator for the time-difference-of-arrival (TDoA) between two
microphone channels observing the same source (Knapp & Carter, 1976). The
phase transform whitens the cross-power spectrum so the correlation peak stays
sharp under reverberation and coloured noise, which a plain cross-correlation
does not.

This is the missing UPSTREAM half of the acoustic localization pipeline: it
turns raw multi-channel audio into the precise inter-channel delays that
:func:`tritium_lib.models.acoustic_tdoa.compute_tdoa_position_leastsq` then
inverts into a source position. Pure NumPy — no SciPy, no heavy DSP dep.

Algorithm (public domain; the classic Knapp-Carter formulation):
    R(f) = X(f) * conj(Y(f)) / |X(f) * conj(Y(f))|     (PHAT weighting)
    r(t) = IFFT(R(f))      (optionally up-sampled by `interp` for sub-sample tau)
    tau  = argmax|r(t)| / fs
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def gcc_phat(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: float = 1.0,
    max_tau: Optional[float] = None,
    interp: int = 16,
) -> Tuple[float, np.ndarray]:
    """Estimate the delay of ``sig`` relative to ``refsig`` via GCC-PHAT.

    Args:
        sig: Signal whose delay is measured (1-D samples).
        refsig: Reference signal (1-D samples).
        fs: Sample rate in Hz. With the default 1.0 the returned tau is in
            samples; pass the real rate for seconds.
        max_tau: Optional cap on |tau| in seconds — restricts the peak search
            to a physically-plausible window (e.g. mic spacing / speed_of_sound).
        interp: Up-sampling factor for sub-sample peak resolution (>=1).

    Returns:
        ``(tau, cc)`` — the estimated delay in seconds (positive when ``sig``
        lags ``refsig``) and the (interpolated) cross-correlation vector.

    Raises:
        ValueError: if either input is empty or not 1-D.
    """
    sig = np.asarray(sig, dtype=np.float64).ravel()
    refsig = np.asarray(refsig, dtype=np.float64).ravel()
    if sig.size == 0 or refsig.size == 0:
        raise ValueError("gcc_phat requires non-empty signals")
    interp = max(1, int(interp))

    n = sig.size + refsig.size
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    # PHAT weighting: divide by magnitude to keep only phase information.
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    cc = np.fft.irfft(R / denom, n=interp * n)

    max_shift = int(interp * n // 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    max_shift = max(1, max_shift)

    # Center the correlation so index 0 is zero lag.
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    shift = int(np.argmax(np.abs(cc))) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc
