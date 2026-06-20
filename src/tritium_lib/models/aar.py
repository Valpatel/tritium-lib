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
    # Of units_lost, how many fell to the stalemate timeout (no shooter
    # earned the kill).  Lets the AAR reconcile enemy losses against
    # attributed kills instead of showing 'KILLS 0' beside 'LOST 73'.
    units_lost_to_timeout: int = 0
    # Units that left the field alive (status 'escaped').  Escapees are
    # NOT losses — without this split, escaped hostiles inflated
    # units_lost and the kills-vs-lost math never reconciled.
    units_escaped: int = 0
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


class DeEscalationSummary(BaseModel):
    """Civil-unrest de-escalation outcome — how ORDER was (or wasn't) restored.

    Surfaces the mode's central metric in the after-action record: the riot is
    won by identifying/neutralizing enough instigators that the crowd
    self-calms (de_escalation_score crosses de_escalation_target), NOT by
    grinding every attrition wave.  ``weighted_score`` is the operator's
    headline number (30% combat score + 70% de-escalation).
    """

    score: int = 0
    target: int = 0
    target_met: bool = False
    weighted_score: int = 0


class EscortSummary(BaseModel):
    """Escort-mode outcome — did the VIP get delivered, and how far it got.

    Escort is the moving-objective mode: a friendly non-combatant protectee
    travels A->B.  The mission is won by ARRIVAL and lost by LOSS, never by
    clearing waves, so the after-action record reports the delivery outcome
    plus how much of the route was covered (``route_progress``, 0..1) and the
    metres still to go (``distance_remaining``) — the honest convoy debrief.
    """

    protectee_id: str = ""
    protectee_name: str = ""
    delivered: bool = False  # reached destination (victory)
    lost: bool = False  # destroyed / lost en route (defeat)
    destination: tuple[float, float] | None = None
    distance_remaining: float = 0.0  # metres from VIP to destination at end
    route_progress: float = 0.0  # 0.0..1.0 fraction of A->B covered


class PatrolSummary(BaseModel):
    """Patrol-mode outcome — was the protected perimeter HELD or BREACHED.

    Patrol is the static perimeter-security mode: a single hostile inside the
    ``breach_radius`` of the protected point is an immediate defeat, distinct
    from battle's all-friendlies-eliminated.  The after-action record reports
    whether the zone held and the closest a hostile got to the centre.
    """

    protected_point: tuple[float, float] | None = None
    breach_radius: float = 0.0
    held: bool = False  # perimeter survived every wave without a breach (victory)
    breached: bool = False  # a hostile entered the secure zone (defeat)
    closest_approach: float = 0.0  # nearest hostile distance to the point (m)


class InfrastructureSummary(BaseModel):
    """Infrastructure-defence outcome — was the protected structure HELD or DESTROYED.

    Drone-swarm and defense are the two structure-integrity modes: a single
    integrity pool (a comms relay under air assault, or a fixed ground
    strongpoint under siege) decays as attackers reach it, and the mission is
    lost the moment it hits zero — distinct from battle's all-friendlies wipe.
    The after-action record reports the final integrity, the lowest it ever
    sank to, total damage absorbed, and a per-source breakdown
    (``bomber_detonation`` / ``attack_fire`` / ``ground_siege``) so the
    operator can see HOW the structure was worn down, not just whether it fell.
    """

    final_integrity: float = 0.0  # health at game end
    max_integrity: float = 0.0  # starting / max health
    integrity_percent: float = 0.0  # 0.0..100.0 of max remaining at end
    min_integrity: float = 0.0  # lowest integrity reached during the engagement
    total_damage: float = 0.0  # cumulative damage absorbed
    destroyed: bool = False  # integrity hit 0 — mission-critical loss
    damage_by_source: dict[str, float] = Field(default_factory=dict)
    # source_type string -> damage dealt, e.g.
    # {"bomber_detonation": 450.0, "attack_fire": 75.0, "ground_siege": 120.0}


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
    result_reason: str = ""  # why it ended: order_restored, civilian_harm_limit,
    # all_friendlies_eliminated, all_waves_cleared, ...
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
    de_escalation: DeEscalationSummary | None = None  # civil_unrest only
    escort: EscortSummary | None = None  # escort mode only
    patrol: PatrolSummary | None = None  # patrol mode only
    infrastructure: InfrastructureSummary | None = None  # drone_swarm/defense only

    mvp: MVPHighlight | None = None

    kill_graph: list[KillGraphEntry] = Field(default_factory=list)
    morale_curve: list[MoraleSample] = Field(default_factory=list)

    pinned: bool = False  # True if persisted by operator
    notes: str = ""


__all__ = [
    "AfterActionReport",
    "FactionSummary",
    "CivilianCollateral",
    "DeEscalationSummary",
    "EscortSummary",
    "PatrolSummary",
    "InfrastructureSummary",
    "MVPHighlight",
    "KillGraphEntry",
    "MoraleSample",
]
