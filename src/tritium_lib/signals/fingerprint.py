# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Signal fingerprinting — build unique signatures from BLE/WiFi characteristics.

A SignalFingerprint captures the RF behaviour of a device over time:
  - RSSI histogram (power profile at different ranges)
  - Advertising / beacon interval statistics
  - Channel usage pattern
  - Temporal activity pattern (when the device is active)

Two fingerprints can be compared for similarity, enabling device re-identification
even when the MAC address has been randomized.

Pure Python — no numpy required.

Usage::

    from tritium_lib.signals import SignalFingerprint

    fp = SignalFingerprint("ble_AA:BB:CC:DD:EE:FF")
    fp.add_observation(rssi=-65.0, channel=37, timestamp=1000.0)
    fp.add_observation(rssi=-63.0, channel=38, timestamp=1001.0)
    fp.add_observation(rssi=-66.0, channel=37, timestamp=1002.0)

    fp2 = SignalFingerprint("ble_11:22:33:44:55:66")
    fp2.add_observation(rssi=-64.0, channel=37, timestamp=1000.5)
    fp2.add_observation(rssi=-62.0, channel=38, timestamp=1001.5)

    similarity = fp.compare(fp2)  # 0.0 - 1.0
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RSSI histogram bins: from -100 dBm to -20 dBm, 5 dBm per bin
_RSSI_BIN_MIN = -100
_RSSI_BIN_MAX = -20
_RSSI_BIN_WIDTH = 5
_RSSI_NUM_BINS = (_RSSI_BIN_MAX - _RSSI_BIN_MIN) // _RSSI_BIN_WIDTH

# Time-of-day bins (24 hours)
_HOUR_BINS = 24

# BLE advertising channels
_BLE_ADV_CHANNELS = {37, 38, 39}


# ---------------------------------------------------------------------------
# SignalFingerprint
# ---------------------------------------------------------------------------

class SignalFingerprint:
    """Unique signal fingerprint built from RF observations.

    Captures statistical patterns of a device's RF behaviour that persist
    even when identifiers like MAC addresses rotate.

    Args:
        device_id: Identifier for the device being fingerprinted.
        tx_power_hint: Optional known or estimated TX power in dBm.
    """

    def __init__(
        self,
        device_id: str,
        tx_power_hint: float = -59.0,
    ) -> None:
        self.device_id = device_id
        self.tx_power_hint = tx_power_hint
        self.observation_count: int = 0
        self.first_seen: float = 0.0
        self.last_seen: float = 0.0

        # RSSI histogram (counts per 5-dBm bin)
        self._rssi_histogram: list[int] = [0] * _RSSI_NUM_BINS

        # Channel usage counts
        self._channel_counts: dict[int, int] = defaultdict(int)

        # Inter-observation intervals (for beacon interval estimation)
        self._intervals: list[float] = []
        self._last_timestamp: float = 0.0

        # Hour-of-day activity histogram (if timestamps are epoch seconds)
        self._hour_histogram: list[int] = [0] * _HOUR_BINS

        # Service UUIDs and name fragments (for fingerprint enrichment)
        self._service_uuids: set[str] = set()
        self._name_fragments: set[str] = set()

    # -- Observation ingestion ----------------------------------------------

    def add_observation(
        self,
        rssi: float,
        channel: int = 0,
        timestamp: float = 0.0,
        service_uuids: list[str] | None = None,
        device_name: str = "",
    ) -> None:
        """Record one RF observation.

        Args:
            rssi: RSSI in dBm.
            channel: RF channel number (37/38/39 for BLE advertising).
            timestamp: Epoch seconds (0 = unknown).
            service_uuids: BLE service UUIDs advertised.
            device_name: BLE device name or WiFi SSID fragment.
        """
        self.observation_count += 1

        if timestamp > 0:
            if self.first_seen == 0.0:
                self.first_seen = timestamp
            self.last_seen = timestamp

        # RSSI histogram
        bin_idx = self._rssi_to_bin(rssi)
        self._rssi_histogram[bin_idx] += 1

        # Channel usage
        if channel > 0:
            self._channel_counts[channel] += 1

        # Beacon interval estimation
        if timestamp > 0 and self._last_timestamp > 0:
            interval = timestamp - self._last_timestamp
            if 0.01 < interval < 30.0:  # reasonable beacon range
                self._intervals.append(interval)
                # Keep bounded
                if len(self._intervals) > 200:
                    self._intervals = self._intervals[-200:]
        if timestamp > 0:
            self._last_timestamp = timestamp

        # Hour-of-day activity
        if timestamp > 0:
            try:
                import time as _time
                hour = _time.gmtime(timestamp).tm_hour
                self._hour_histogram[hour] += 1
            except (OSError, OverflowError):
                pass

        # Enrichment
        if service_uuids:
            self._service_uuids.update(service_uuids)
        if device_name:
            # Store lowercased prefix as a fragment
            frag = device_name.strip().lower()[:20]
            if frag:
                self._name_fragments.add(frag)

    # -- Derived metrics ----------------------------------------------------

    @property
    def mean_rssi(self) -> float:
        """Weighted mean RSSI from histogram."""
        total = sum(self._rssi_histogram)
        if total == 0:
            return -70.0
        weighted = 0.0
        for i, count in enumerate(self._rssi_histogram):
            bin_center = _RSSI_BIN_MIN + (i + 0.5) * _RSSI_BIN_WIDTH
            weighted += bin_center * count
        return weighted / total

    @property
    def rssi_std_dev(self) -> float:
        """Standard deviation of RSSI from histogram."""
        total = sum(self._rssi_histogram)
        if total == 0:
            return 0.0
        mean = self.mean_rssi
        variance = 0.0
        for i, count in enumerate(self._rssi_histogram):
            bin_center = _RSSI_BIN_MIN + (i + 0.5) * _RSSI_BIN_WIDTH
            variance += count * (bin_center - mean) ** 2
        return math.sqrt(variance / total)

    @property
    def estimated_beacon_interval(self) -> float | None:
        """Median beacon interval in seconds, or None if insufficient data."""
        if len(self._intervals) < 3:
            return None
        sorted_intervals = sorted(self._intervals)
        mid = len(sorted_intervals) // 2
        if len(sorted_intervals) % 2 == 0:
            return (sorted_intervals[mid - 1] + sorted_intervals[mid]) / 2.0
        return sorted_intervals[mid]

    @property
    def dominant_channel(self) -> int | None:
        """Most frequently observed channel, or None."""
        if not self._channel_counts:
            return None
        return max(self._channel_counts, key=self._channel_counts.get)  # type: ignore[arg-type]

    @property
    def channel_distribution(self) -> dict[int, float]:
        """Normalised channel usage as fractions summing to ~1.0."""
        total = sum(self._channel_counts.values())
        if total == 0:
            return {}
        return {ch: cnt / total for ch, cnt in sorted(self._channel_counts.items())}

    # -- Comparison ---------------------------------------------------------

    def compare(self, other: SignalFingerprint) -> float:
        """Compare this fingerprint to another and return similarity [0.0, 1.0].

        Compares:
          1. RSSI histogram (cosine similarity)          weight 0.35
          2. Beacon interval similarity                  weight 0.20
          3. Channel distribution similarity             weight 0.15
          4. Hour-of-day activity pattern                weight 0.15
          5. Service UUID overlap (Jaccard)              weight 0.15
        """
        scores: list[tuple[float, float]] = []

        # 1. RSSI histogram cosine similarity
        rssi_sim = _cosine_similarity(self._rssi_histogram, other._rssi_histogram)
        scores.append((rssi_sim, 0.35))

        # 2. Beacon interval similarity
        bi1 = self.estimated_beacon_interval
        bi2 = other.estimated_beacon_interval
        if bi1 is not None and bi2 is not None and (bi1 + bi2) > 0:
            bi_sim = 1.0 - abs(bi1 - bi2) / max(bi1, bi2)
            bi_sim = max(0.0, bi_sim)
        else:
            bi_sim = 0.5  # neutral if unknown
        scores.append((bi_sim, 0.20))

        # 3. Channel distribution cosine similarity
        all_channels = set(self._channel_counts.keys()) | set(other._channel_counts.keys())
        if all_channels:
            v1 = [self._channel_counts.get(ch, 0) for ch in sorted(all_channels)]
            v2 = [other._channel_counts.get(ch, 0) for ch in sorted(all_channels)]
            ch_sim = _cosine_similarity(v1, v2)
        else:
            ch_sim = 0.5
        scores.append((ch_sim, 0.15))

        # 4. Hour-of-day activity pattern cosine similarity
        hour_sim = _cosine_similarity(self._hour_histogram, other._hour_histogram)
        scores.append((hour_sim, 0.15))

        # 5. Service UUID Jaccard similarity
        if self._service_uuids or other._service_uuids:
            intersection = self._service_uuids & other._service_uuids
            union = self._service_uuids | other._service_uuids
            uuid_sim = len(intersection) / len(union) if union else 0.0
        else:
            uuid_sim = 0.5  # neutral
        scores.append((uuid_sim, 0.15))

        # Weighted sum
        total_weight = sum(w for _, w in scores)
        similarity = sum(s * w for s, w in scores) / total_weight if total_weight > 0 else 0.0
        return round(max(0.0, min(1.0, similarity)), 4)

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        """Export fingerprint as a serializable dict."""
        return {
            "device_id": self.device_id,
            "observation_count": self.observation_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "mean_rssi": round(self.mean_rssi, 2),
            "rssi_std_dev": round(self.rssi_std_dev, 2),
            "estimated_beacon_interval": (
                round(self.estimated_beacon_interval, 4)
                if self.estimated_beacon_interval is not None
                else None
            ),
            "dominant_channel": self.dominant_channel,
            "channel_distribution": {
                str(k): round(v, 4) for k, v in self.channel_distribution.items()
            },
            "service_uuids": sorted(self._service_uuids),
            "name_fragments": sorted(self._name_fragments),
            "rssi_histogram": list(self._rssi_histogram),
            "hour_histogram": list(self._hour_histogram),
        }

    # -- Internal -----------------------------------------------------------

    @staticmethod
    def _rssi_to_bin(rssi: float) -> int:
        """Map RSSI to histogram bin index."""
        clamped = max(_RSSI_BIN_MIN, min(_RSSI_BIN_MAX - 1, rssi))
        idx = int((clamped - _RSSI_BIN_MIN) / _RSSI_BIN_WIDTH)
        return max(0, min(_RSSI_NUM_BINS - 1, idx))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[int | float], b: list[int | float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a < 1e-12 or mag_b < 1e-12:
        return 0.0
    return dot / (mag_a * mag_b)
