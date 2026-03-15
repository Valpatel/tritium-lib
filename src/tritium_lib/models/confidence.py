# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Confidence decay models for tracked targets.

Different sensor sources have different confidence decay characteristics:
- BLE decays fast (devices move, signal fluctuates)
- Camera/YOLO decays moderately (object may leave frame)
- Mesh/GPS decays slowest (GPS positions are reliable, devices are stationary)
- Simulation targets don't decay (engine manages their lifecycle)

Uses exponential decay: confidence(t) = initial * exp(-lambda * dt)
where lambda = ln(2) / half_life_seconds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SourceType(str, Enum):
    """Sensor source types with associated decay characteristics."""
    BLE = "ble"
    WIFI = "wifi"
    YOLO = "yolo"
    CAMERA = "camera"
    MESH = "mesh"
    RF_MOTION = "rf_motion"
    SIMULATION = "simulation"
    MANUAL = "manual"


# Default half-life in seconds per source type.
# After one half-life, confidence drops to 50% of initial.
DEFAULT_HALF_LIVES: dict[str, float] = {
    "ble": 30.0,         # BLE: 30s half-life (fast decay, devices move)
    "wifi": 45.0,        # WiFi: 45s half-life (slightly more stable than BLE)
    "yolo": 15.0,        # YOLO: 15s half-life (object left frame = fast decay)
    "camera": 15.0,      # Camera alias for YOLO
    "rf_motion": 10.0,   # RF motion: 10s half-life (transient detections)
    "mesh": 120.0,       # Mesh/GPS: 2min half-life (GPS is reliable)
    "simulation": 0.0,   # Simulation: no decay (engine manages lifecycle)
    "manual": 300.0,     # Manual: 5min half-life (operator-placed targets)
}


@dataclass
class ConfidenceModel:
    """Configurable exponential confidence decay per source type.

    Usage:
        model = ConfidenceModel()
        # Get decayed confidence for a BLE target last seen 60s ago
        conf = model.decay("ble", initial=0.85, elapsed_seconds=60.0)
        # -> ~0.2125 (two half-lives worth of decay)

        # Custom half-lives
        model = ConfidenceModel(half_lives={"ble": 15.0})
    """

    half_lives: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HALF_LIVES))
    # Minimum confidence floor (below this, target is considered stale)
    min_confidence: float = 0.05

    def decay(self, source: str, initial: float, elapsed_seconds: float) -> float:
        """Compute decayed confidence using exponential decay.

        Args:
            source: Sensor source type (ble, yolo, mesh, etc.)
            initial: Initial confidence value (0.0 to 1.0)
            elapsed_seconds: Seconds since last detection

        Returns:
            Decayed confidence value, clamped to [min_confidence, 1.0].
            Returns 0.0 if below min_confidence (target is stale).
        """
        if elapsed_seconds <= 0.0:
            return max(0.0, min(1.0, initial))

        half_life = self.half_lives.get(source, self.half_lives.get("manual", 300.0))

        # No decay for simulation targets
        if half_life <= 0.0:
            return max(0.0, min(1.0, initial))

        # lambda = ln(2) / half_life
        decay_lambda = math.log(2) / half_life
        decayed = initial * math.exp(-decay_lambda * elapsed_seconds)

        if decayed < self.min_confidence:
            return 0.0

        return min(1.0, decayed)

    def is_stale(self, source: str, elapsed_seconds: float,
                 initial: float = 1.0) -> bool:
        """Check if a target should be considered stale (confidence ~0).

        A target is stale when its decayed confidence drops below
        min_confidence threshold.
        """
        return self.decay(source, initial, elapsed_seconds) == 0.0

    def time_to_stale(self, source: str, initial: float = 1.0) -> float:
        """Compute seconds until a target becomes stale from initial confidence.

        Returns float('inf') for simulation targets (never stale via decay).
        """
        half_life = self.half_lives.get(source, self.half_lives.get("manual", 300.0))

        if half_life <= 0.0:
            return float("inf")

        if initial <= self.min_confidence:
            return 0.0

        # Solve: min_confidence = initial * exp(-lambda * t)
        # t = -ln(min_confidence / initial) / lambda
        decay_lambda = math.log(2) / half_life
        return -math.log(self.min_confidence / initial) / decay_lambda

    def get_half_life(self, source: str) -> float:
        """Get the half-life for a given source type."""
        return self.half_lives.get(source, self.half_lives.get("manual", 300.0))

    def set_half_life(self, source: str, seconds: float) -> None:
        """Set the half-life for a given source type."""
        self.half_lives[source] = seconds

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "half_lives": dict(self.half_lives),
            "min_confidence": self.min_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConfidenceModel":
        """Deserialize from dictionary."""
        return cls(
            half_lives=data.get("half_lives", dict(DEFAULT_HALF_LIVES)),
            min_confidence=data.get("min_confidence", 0.05),
        )
