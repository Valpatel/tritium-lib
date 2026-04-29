# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""After-Action Report (AAR) data models.

An After-Action Report is a structured post-game summary of a finished
combat scenario.  It captures who fought, who survived, who fell, the
collateral damage cost, the unit-of-the-match, and the temporal arc of
the engagement (kill graph and morale curve).

The AAR is built once at scenario end (game_over event) by aggregating
data from the StatsTracker, MoraleSystem, civilian subsystem, and the
engine event log.  It is intentionally separate from the live kill feed
which keeps streaming during play.

Wave 202.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class KillGraphEntry(BaseModel):
    """A single kill in the chronological kill graph."""

    tick: int = 0  # Sim tick number (10 Hz typical)
    time_offset: float = 0.0  # Seconds from scenario start
    killer_id: str = ""
    killer_name: str = ""
    killer_alliance: str = ""  # friendly, hostile, neutral
    victim_id: str = ""
    victim_name: str = ""
    victim_alliance: str = ""
    weapon: str = "unknown"  # ballistic, missile, energy, melee, timeout, ...
    position: tuple[float, float] | None = None


class MoraleSample(BaseModel):
    """A single sample of the morale curve."""

    time_offset: float = 0.0  # Seconds from scenario start
    friendly_avg: float = 1.0  # 0.0..1.0
    hostile_avg: float = 1.0  # 0.0..1.0


class FactionSummary(BaseModel):
    """Aggregate summary for one faction (friendly or hostile)."""

    alliance: str = "friendly"
    units_started: int = 0
    units_lost: int = 0
    units_survived: int = 0
    total_kills: int = 0  # Kills inflicted BY this faction
    total_damage_dealt: float = 0.0
    total_damage_taken: float = 0.0
    accuracy: float = 0.0  # 0.0..1.0


class CivilianCollateral(BaseModel):
    """Collateral damage to civilians and infrastructure."""

    civilians_killed: int = 0
    civilians_injured: int = 0
    infrastructure_damage: float = 0.0
    causes: dict[str, int] = Field(default_factory=dict)
    # cause string -> count, e.g. {"crossfire": 3, "explosion": 1, "panic": 2}


class MVPHighlight(BaseModel):
    """The most valuable friendly unit of the engagement."""

    target_id: str = ""
    name: str = ""
    asset_type: str = "unknown"
    kills: int = 0
    accuracy: float = 0.0
    damage_dealt: float = 0.0
    survived: bool = True


class AfterActionReport(BaseModel):
    """Structured post-game summary of a finished combat scenario.

    Generated once at game_over.  Persisted to data/aar/{scenario_id}.json
    when the operator pins the report.
    """

    scenario_id: str = ""  # File-safe identifier (slug or UUID)
    scenario_name: str = ""  # Human-readable name (e.g. "Drone Swarm")
    game_mode_type: str = "battle"
    result: str = "draw"  # victory, defeat, draw, aborted
    started_at: float = Field(default_factory=time.time)
    ended_at: float = Field(default_factory=time.time)
    duration_seconds: float = 0.0
    waves_completed: int = 0
    final_score: int = 0

    friendly: FactionSummary = Field(default_factory=FactionSummary)
    hostile: FactionSummary = Field(
        default_factory=lambda: FactionSummary(alliance="hostile")
    )
    civilian: CivilianCollateral = Field(default_factory=CivilianCollateral)

    mvp: MVPHighlight | None = None

    kill_graph: list[KillGraphEntry] = Field(default_factory=list)
    morale_curve: list[MoraleSample] = Field(default_factory=list)

    pinned: bool = False  # True if persisted by operator
    notes: str = ""


__all__ = [
    "AfterActionReport",
    "FactionSummary",
    "CivilianCollateral",
    "MVPHighlight",
    "KillGraphEntry",
    "MoraleSample",
]
