# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""WiFi Channel State Information (CSI) processor for occupancy detection.

CSI captures the amplitude and phase of each WiFi subcarrier, providing a
rich picture of the indoor RF environment.  Human presence disturbs these
subcarriers, enabling occupancy detection, motion sensing, and even
gesture recognition — all without cameras.

This module implements:
  - Hampel filter for outlier removal on CSI amplitude streams
  - Sliding-window variance for motion/occupancy detection
  - Subcarrier grouping into spectral bands
  - Binary occupancy classifier (static threshold + adaptive baseline)

Algorithm reference: Two-tier architecture described in
``docs/technical-brief-ruview-csi-analysis.md``.

Pure Python — no numpy required.

Usage::

    from tritium_lib.signals import CSIProcessor

    proc = CSIProcessor(num_subcarriers=64)
    proc.add_frame([amp1, amp2, ..., amp64], timestamp=1000.0)
    proc.add_frame([amp1, amp2, ..., amp64], timestamp=1001.0)

    result = proc.detect_occupancy()
    print(result.occupied, result.confidence)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_NUM_SUBCARRIERS = 64
DEFAULT_WINDOW_SIZE = 50  # frames
DEFAULT_OCCUPANCY_THRESHOLD = 2.0  # variance ratio above baseline
HAMPEL_WINDOW = 5
HAMPEL_THRESHOLD = 3.0  # number of MADs for outlier detection


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class OccupancyResult:
    """Result of occupancy detection analysis."""
    occupied: bool
    confidence: float
    variance_ratio: float
    mean_variance: float
    baseline_variance: float
    active_subcarriers: int
    total_subcarriers: int

    def to_dict(self) -> dict:
        return {
            "occupied": self.occupied,
            "confidence": round(self.confidence, 3),
            "variance_ratio": round(self.variance_ratio, 3),
            "mean_variance": round(self.mean_variance, 4),
            "baseline_variance": round(self.baseline_variance, 4),
            "active_subcarriers": self.active_subcarriers,
            "total_subcarriers": self.total_subcarriers,
        }


@dataclass
class CSIStats:
    """Per-frame CSI statistics."""
    timestamp: float
    mean_amplitude: float
    std_amplitude: float
    max_amplitude: float
    min_amplitude: float
    spectral_spread: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "mean_amplitude": round(self.mean_amplitude, 4),
            "std_amplitude": round(self.std_amplitude, 4),
            "max_amplitude": round(self.max_amplitude, 4),
            "min_amplitude": round(self.min_amplitude, 4),
            "spectral_spread": round(self.spectral_spread, 4),
        }


@dataclass
class SubcarrierBand:
    """A group of adjacent subcarriers for coarse analysis."""
    name: str
    start_idx: int
    end_idx: int
    mean_variance: float = 0.0
    is_active: bool = False


# ---------------------------------------------------------------------------
# Hampel filter
# ---------------------------------------------------------------------------

def hampel_filter(
    values: list[float],
    window_size: int = HAMPEL_WINDOW,
    threshold: float = HAMPEL_THRESHOLD,
) -> list[float]:
    """Apply Hampel filter to remove outliers from a 1D signal.

    The Hampel filter replaces values that deviate more than ``threshold``
    median absolute deviations (MADs) from the local median with the
    local median itself.

    Args:
        values: Input signal.
        window_size: Half-window size for median computation.
        threshold: Number of MADs beyond which a value is an outlier.

    Returns:
        Filtered signal (same length as input).
    """
    n = len(values)
    if n < 3:
        return list(values)

    result = list(values)
    for i in range(n):
        lo = max(0, i - window_size)
        hi = min(n, i + window_size + 1)
        window = sorted(values[lo:hi])
        w_len = len(window)
        median = window[w_len // 2]

        # Median absolute deviation
        deviations = sorted(abs(v - median) for v in window)
        mad = deviations[w_len // 2] * 1.4826  # scale factor for normal distribution

        deviation = abs(values[i] - median)
        if mad > 1e-10:
            if deviation > threshold * mad:
                result[i] = median
        elif deviation > 1e-10:
            # MAD is zero (all neighbours identical) but this value differs
            result[i] = median

    return result


# ---------------------------------------------------------------------------
# CSIProcessor
# ---------------------------------------------------------------------------

class CSIProcessor:
    """WiFi CSI processor for occupancy and motion detection.

    Maintains a sliding window of CSI frames (each frame is a list of
    subcarrier amplitudes) and computes variance-based occupancy metrics.

    Args:
        num_subcarriers: Number of OFDM subcarriers per frame.
        window_size: Number of frames in the analysis window.
        occupancy_threshold: Variance-ratio threshold for occupancy.
        num_bands: Number of spectral bands to group subcarriers into.
        baseline_frames: Number of initial frames used for baseline.
    """

    def __init__(
        self,
        num_subcarriers: int = DEFAULT_NUM_SUBCARRIERS,
        window_size: int = DEFAULT_WINDOW_SIZE,
        occupancy_threshold: float = DEFAULT_OCCUPANCY_THRESHOLD,
        num_bands: int = 4,
        baseline_frames: int = 20,
    ) -> None:
        self._num_sc = num_subcarriers
        self._window_size = window_size
        self._occupancy_threshold = occupancy_threshold
        self._num_bands = num_bands
        self._baseline_frames = baseline_frames

        # Sliding window of frames: each entry is (timestamp, amplitudes)
        self._frames: deque[tuple[float, list[float]]] = deque(maxlen=window_size)

        # Baseline variance per subcarrier (computed from initial frames)
        self._baseline_var: list[float] | None = None
        self._baseline_count: int = 0

        # Accumulator for computing baseline incrementally
        self._baseline_sum: list[float] = [0.0] * num_subcarriers
        self._baseline_sq_sum: list[float] = [0.0] * num_subcarriers

        self._total_frames: int = 0

    # -- Frame ingestion ----------------------------------------------------

    def add_frame(
        self,
        amplitudes: list[float],
        timestamp: float = 0.0,
    ) -> CSIStats | None:
        """Add one CSI frame (subcarrier amplitudes).

        Args:
            amplitudes: List of amplitude values, one per subcarrier.
                Length must match ``num_subcarriers``.
            timestamp: Epoch seconds for this frame.

        Returns:
            Per-frame statistics, or None if the frame was rejected.
        """
        if len(amplitudes) != self._num_sc:
            return None

        # Apply Hampel filter per frame to remove subcarrier outliers
        cleaned = hampel_filter(amplitudes)

        self._frames.append((timestamp, cleaned))
        self._total_frames += 1

        # Update baseline accumulator
        if self._baseline_var is None and self._baseline_count < self._baseline_frames:
            for i in range(self._num_sc):
                self._baseline_sum[i] += cleaned[i]
                self._baseline_sq_sum[i] += cleaned[i] * cleaned[i]
            self._baseline_count += 1

            if self._baseline_count >= self._baseline_frames:
                self._compute_baseline()

        # Compute per-frame stats
        mean_amp = sum(cleaned) / self._num_sc
        var_amp = sum((a - mean_amp) ** 2 for a in cleaned) / self._num_sc
        std_amp = math.sqrt(var_amp)

        return CSIStats(
            timestamp=timestamp,
            mean_amplitude=mean_amp,
            std_amplitude=std_amp,
            max_amplitude=max(cleaned),
            min_amplitude=min(cleaned),
            spectral_spread=max(cleaned) - min(cleaned),
        )

    # -- Occupancy detection ------------------------------------------------

    def detect_occupancy(self) -> OccupancyResult | None:
        """Analyse recent CSI frames and determine occupancy.

        Computes the variance of each subcarrier over the current window
        and compares to the baseline (empty room) variance. A significant
        increase indicates human presence.

        Returns:
            OccupancyResult, or None if insufficient data.
        """
        if self._baseline_var is None:
            return None
        if len(self._frames) < 5:
            return None

        # Compute current variance per subcarrier
        current_var = self._compute_window_variance()

        # Variance ratio per subcarrier
        active_count = 0
        ratios: list[float] = []
        for i in range(self._num_sc):
            base = max(self._baseline_var[i], 1e-10)
            ratio = current_var[i] / base
            ratios.append(ratio)
            if ratio > self._occupancy_threshold:
                active_count += 1

        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        mean_var = sum(current_var) / len(current_var)
        mean_baseline = sum(self._baseline_var) / len(self._baseline_var)

        occupied = mean_ratio > self._occupancy_threshold
        # Confidence: how far above/below threshold
        if occupied:
            confidence = min(1.0, (mean_ratio - self._occupancy_threshold) /
                             self._occupancy_threshold)
        else:
            confidence = min(1.0, (self._occupancy_threshold - mean_ratio) /
                             self._occupancy_threshold)

        return OccupancyResult(
            occupied=occupied,
            confidence=max(0.0, min(1.0, confidence)),
            variance_ratio=mean_ratio,
            mean_variance=mean_var,
            baseline_variance=mean_baseline,
            active_subcarriers=active_count,
            total_subcarriers=self._num_sc,
        )

    # -- Band analysis ------------------------------------------------------

    def get_band_activity(self) -> list[SubcarrierBand]:
        """Return per-band variance analysis.

        Groups subcarriers into ``num_bands`` equal bands and reports
        which bands are active (elevated variance vs. baseline).
        """
        if self._baseline_var is None or len(self._frames) < 5:
            return []

        current_var = self._compute_window_variance()
        band_size = max(1, self._num_sc // self._num_bands)

        bands: list[SubcarrierBand] = []
        for b in range(self._num_bands):
            start = b * band_size
            end = min(start + band_size, self._num_sc)
            if start >= self._num_sc:
                break

            band_var = current_var[start:end]
            base_var = self._baseline_var[start:end]

            mean_bv = sum(band_var) / len(band_var) if band_var else 0.0
            mean_base = sum(base_var) / len(base_var) if base_var else 1e-10
            ratio = mean_bv / max(mean_base, 1e-10)

            bands.append(SubcarrierBand(
                name=f"band_{b}",
                start_idx=start,
                end_idx=end - 1,
                mean_variance=mean_bv,
                is_active=ratio > self._occupancy_threshold,
            ))

        return bands

    # -- Queries ------------------------------------------------------------

    def is_baseline_ready(self) -> bool:
        """Check whether the baseline has been computed."""
        return self._baseline_var is not None

    def get_frame_count(self) -> int:
        """Return total frames processed."""
        return self._total_frames

    def get_window_size(self) -> int:
        """Return current frames in the analysis window."""
        return len(self._frames)

    def reset_baseline(self) -> None:
        """Clear baseline — will recompute from next frames."""
        self._baseline_var = None
        self._baseline_count = 0
        self._baseline_sum = [0.0] * self._num_sc
        self._baseline_sq_sum = [0.0] * self._num_sc

    def get_status(self) -> dict:
        """Return processor status."""
        return {
            "num_subcarriers": self._num_sc,
            "window_size": self._window_size,
            "frames_in_window": len(self._frames),
            "total_frames": self._total_frames,
            "baseline_ready": self.is_baseline_ready(),
            "baseline_frames_used": self._baseline_count,
            "occupancy_threshold": self._occupancy_threshold,
            "num_bands": self._num_bands,
        }

    # -- Internal -----------------------------------------------------------

    def _compute_baseline(self) -> None:
        """Compute baseline variance from accumulated initial frames."""
        n = self._baseline_count
        if n < 2:
            return
        self._baseline_var = []
        for i in range(self._num_sc):
            mean = self._baseline_sum[i] / n
            mean_sq = self._baseline_sq_sum[i] / n
            var = max(0.0, mean_sq - mean * mean)
            # Ensure nonzero baseline
            self._baseline_var.append(max(var, 1e-6))

    def _compute_window_variance(self) -> list[float]:
        """Compute variance of each subcarrier over the current window."""
        n = len(self._frames)
        if n == 0:
            return [0.0] * self._num_sc

        sums = [0.0] * self._num_sc
        sq_sums = [0.0] * self._num_sc

        for _, amps in self._frames:
            for i in range(self._num_sc):
                sums[i] += amps[i]
                sq_sums[i] += amps[i] * amps[i]

        variances: list[float] = []
        for i in range(self._num_sc):
            mean = sums[i] / n
            mean_sq = sq_sums[i] / n
            var = max(0.0, mean_sq - mean * mean)
            variances.append(var)

        return variances
