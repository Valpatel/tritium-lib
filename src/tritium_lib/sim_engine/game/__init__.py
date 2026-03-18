# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Game systems — wave-based game mode, ambient spawning, crowd density,
stats tracking, difficulty scaling, and morale.

Extracted from tritium-sc/src/engine/simulation/ for reuse by addons,
runners, and other consumers.
"""

from .game_mode import GameMode, WaveConfig, WAVE_CONFIGS, InfiniteWaveMode, InstigatorDetector
from .ambient import (
    AmbientSpawner,
    _generate_street_grid,
    _hour_activity,
)
from .crowd_density import CrowdDensityTracker
from .stats import StatsTracker, UnitStats, WaveStats
from .difficulty import DifficultyScaler, WaveRecord
from .morale import (
    MoraleSystem,
    DEFAULT_MORALE,
    BROKEN_THRESHOLD,
    SUPPRESSED_THRESHOLD,
    EMBOLDENED_THRESHOLD,
)

__all__ = [
    # game_mode
    "GameMode",
    "WaveConfig",
    "WAVE_CONFIGS",
    "InfiniteWaveMode",
    "InstigatorDetector",
    # ambient
    "AmbientSpawner",
    # crowd_density
    "CrowdDensityTracker",
    # stats
    "StatsTracker",
    "UnitStats",
    "WaveStats",
    # difficulty
    "DifficultyScaler",
    "WaveRecord",
    # morale
    "MoraleSystem",
    "DEFAULT_MORALE",
    "BROKEN_THRESHOLD",
    "SUPPRESSED_THRESHOLD",
    "EMBOLDENED_THRESHOLD",
]
