# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Artillery and indirect fire system for the Tritium sim engine.

Provides mortar, howitzer, MLRS, and naval gun simulation with parabolic
shell trajectories, CEP-based scatter, fire missions (point, area, barrage,
smoke, illumination, danger_close), and forward observer fire adjustment.

Usage::

    from tritium_lib.sim_engine.artillery import (
        ArtilleryEngine, ArtilleryType, ARTILLERY_TEMPLATES,
        ForwardObserver,
    )

    engine = ArtilleryEngine()
    piece = ARTILLERY_TEMPLATES[ArtilleryType.MORTAR_81MM]("m1", "blue", (100, 200))
    engine.add_piece(piece)

    fo = ForwardObserver("fo1", (300, 400), "blue")
    mission = fo.call_fire(engine, (500, 600), "point", 3)

    events = engine.tick(1.0)   # advance simulation
    three = engine.to_three_js()  # send to frontend

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import enum
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ArtilleryType(enum.Enum):
    """Classification of artillery piece."""
    MORTAR_60MM = "mortar_60mm"
    MORTAR_81MM = "mortar_81mm"
    HOWITZER_105MM = "howitzer_105mm"
    HOWITZER_155MM = "howitzer_155mm"
    MLRS = "mlrs"
    NAVAL_GUN = "naval_gun"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ArtilleryPiece:
    """An emplaced artillery piece with ammo, cooldown, and crew."""

    piece_id: str
    artillery_type: ArtilleryType
    alliance: str
    position: Vec2
    heading: float  # radians
    min_range: float  # meters
    max_range: float  # meters
    damage: float
    blast_radius: float  # meters
    reload_time: float  # seconds between rounds
    ammo: int
    max_ammo: int
    accuracy_cep: float  # circular error probable in meters
    ready: bool = True
    cooldown: float = 0.0
    crew: int = 3


@dataclass
class FireMission:
    """A requested fire mission with round count and pacing."""

    mission_id: str
    piece_id: str
    target_pos: Vec2
    mission_type: str  # point, area, barrage, smoke, illumination, danger_close
    rounds: int
    interval: float  # seconds between rounds
    rounds_fired: int = 0
    active: bool = True
    _interval_acc: float = field(default=0.0, repr=False)


@dataclass
class Shell:
    """A shell currently in flight along a parabolic arc."""

    shell_id: str
    origin: Vec2
    target: Vec2
    impact_pos: Vec2  # target + CEP scatter
    altitude: float  # current altitude (parabolic)
    time_of_flight: float  # total seconds from fire to impact
    elapsed: float = 0.0
    damage: float = 0.0
    blast_radius: float = 0.0
    shell_type: str = "he"  # he, smoke, illumination, wp
    piece_id: str = ""
    mission_id: str = ""


# ---------------------------------------------------------------------------
# Shell type mapping from mission type
# ---------------------------------------------------------------------------

_MISSION_SHELL_TYPE: dict[str, str] = {
    "point": "he",
    "area": "he",
    "barrage": "he",
    "smoke": "smoke",
    "illumination": "illumination",
    "danger_close": "he",
    "wp": "wp",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_cep_scatter(
    target: Vec2,
    cep: float,
    rng: random.Random | None = None,
) -> Vec2:
    """Offset *target* by a random displacement drawn from CEP distribution.

    CEP is the radius within which 50% of rounds land.  We model this as a
    2D Gaussian where sigma = CEP / 1.1774 (so that the 50th-percentile
    radius equals CEP).
    """
    r = rng or random.Random()
    sigma = cep / 1.1774
    dx = r.gauss(0.0, sigma)
    dy = r.gauss(0.0, sigma)
    return (target[0] + dx, target[1] + dy)


def _time_of_flight(range_m: float) -> float:
    """Estimate time of flight in seconds based on range.

    Simple model: roughly 1 second per 200 m, with a minimum of 2 s.
    """
    return max(2.0, range_m / 200.0)


def _parabolic_altitude(elapsed: float, tof: float, max_alt: float) -> float:
    """Return altitude at *elapsed* along a parabolic arc.

    Peak altitude is at tof/2, altitude is 0 at both ends.
    """
    if tof <= 0:
        return 0.0
    t = elapsed / tof  # 0..1
    # parabola: 4 * max_alt * t * (1 - t)
    return max(0.0, 4.0 * max_alt * t * (1.0 - t))


def _max_altitude(range_m: float) -> float:
    """Estimate peak shell altitude from range.  Higher angle = higher arc."""
    return max(50.0, range_m * 0.3)


# ---------------------------------------------------------------------------
# ArtilleryEngine
# ---------------------------------------------------------------------------

class ArtilleryEngine:
    """Central manager for artillery pieces, fire missions, and in-flight shells."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.pieces: dict[str, ArtilleryPiece] = {}
        self.fire_missions: list[FireMission] = []
        self.shells_in_flight: list[Shell] = []
        self._rng = rng or random.Random()
        # Accumulated events from the most recent tick
        self._pending_events: list[dict] = []

    # -- Piece management ---------------------------------------------------

    def add_piece(self, piece: ArtilleryPiece) -> None:
        """Register an artillery piece."""
        self.pieces[piece.piece_id] = piece

    def remove_piece(self, piece_id: str) -> None:
        """Remove an artillery piece and cancel its active missions."""
        self.pieces.pop(piece_id, None)
        for m in self.fire_missions:
            if m.piece_id == piece_id:
                m.active = False

    # -- Fire missions ------------------------------------------------------

    def request_fire_mission(
        self,
        piece_id: str,
        target: Vec2,
        mission_type: str = "point",
        rounds: int = 1,
        interval: float = 2.0,
    ) -> FireMission:
        """Request a fire mission from a specific piece.

        Raises ValueError if the piece doesn't exist, is out of range,
        or has no ammo.
        """
        piece = self.pieces.get(piece_id)
        if piece is None:
            raise ValueError(f"Unknown artillery piece: {piece_id}")

        rng = distance(piece.position, target)
        if rng < piece.min_range:
            raise ValueError(
                f"Target too close: {rng:.0f}m < min_range {piece.min_range:.0f}m"
            )
        if rng > piece.max_range:
            raise ValueError(
                f"Target too far: {rng:.0f}m > max_range {piece.max_range:.0f}m"
            )
        if piece.ammo <= 0:
            raise ValueError(f"Piece {piece_id} has no ammo")

        mission = FireMission(
            mission_id=uuid.uuid4().hex[:12],
            piece_id=piece_id,
            target_pos=target,
            mission_type=mission_type,
            rounds=rounds,
            interval=interval,
        )
        self.fire_missions.append(mission)
        return mission

    def cancel_mission(self, mission_id: str) -> bool:
        """Cancel an active fire mission.  Returns True if found."""
        for m in self.fire_missions:
            if m.mission_id == mission_id and m.active:
                m.active = False
                return True
        return False

    # -- Simulation tick ----------------------------------------------------

    def tick(self, dt: float) -> list[dict]:
        """Advance the artillery simulation by *dt* seconds.

        Returns a list of event dicts:
          - {"event": "fire", "piece_id", "mission_id", "shell_id", "target", "impact_pos"}
          - {"event": "impact", "shell_id", "position", "damage", "blast_radius", "shell_type"}
          - {"event": "smoke", "position", "radius"}
          - {"event": "illumination", "position", "radius"}
          - {"event": "mission_complete", "mission_id"}
        """
        events: list[dict] = []

        # 1. Update piece cooldowns
        for piece in self.pieces.values():
            if not piece.ready and piece.cooldown > 0:
                piece.cooldown -= dt
                if piece.cooldown <= 0:
                    piece.cooldown = 0.0
                    piece.ready = True

        # 2. Process fire missions — fire rounds at interval
        for mission in self.fire_missions:
            if not mission.active:
                continue
            if mission.rounds_fired >= mission.rounds:
                mission.active = False
                events.append({
                    "event": "mission_complete",
                    "mission_id": mission.mission_id,
                })
                continue

            piece = self.pieces.get(mission.piece_id)
            if piece is None or piece.ammo <= 0:
                mission.active = False
                events.append({
                    "event": "mission_complete",
                    "mission_id": mission.mission_id,
                })
                continue

            # Accumulate interval time
            mission._interval_acc += dt

            # Fire if ready and interval elapsed (first round fires immediately)
            while (
                mission.active
                and piece.ready
                and piece.ammo > 0
                and mission.rounds_fired < mission.rounds
                and (mission.rounds_fired == 0 or mission._interval_acc >= mission.interval)
            ):
                # Reset interval accumulator
                if mission.rounds_fired > 0:
                    mission._interval_acc -= mission.interval

                # Compute CEP scatter
                cep = piece.accuracy_cep
                # Area and barrage missions have wider scatter
                if mission.mission_type == "area":
                    cep *= 2.0
                elif mission.mission_type == "barrage":
                    cep *= 3.0
                # Danger close uses best accuracy
                elif mission.mission_type == "danger_close":
                    cep *= 0.7

                impact_pos = _apply_cep_scatter(mission.target_pos, cep, self._rng)

                rng_m = distance(piece.position, mission.target_pos)
                tof = _time_of_flight(rng_m)
                max_alt = _max_altitude(rng_m)

                shell_type = _MISSION_SHELL_TYPE.get(mission.mission_type, "he")

                shell = Shell(
                    shell_id=uuid.uuid4().hex[:12],
                    origin=piece.position,
                    target=mission.target_pos,
                    impact_pos=impact_pos,
                    altitude=0.0,
                    time_of_flight=tof,
                    elapsed=0.0,
                    damage=piece.damage if shell_type == "he" else 0.0,
                    blast_radius=piece.blast_radius,
                    shell_type=shell_type,
                    piece_id=piece.piece_id,
                    mission_id=mission.mission_id,
                )
                self.shells_in_flight.append(shell)

                piece.ammo -= 1
                mission.rounds_fired += 1
                piece.ready = False
                piece.cooldown = piece.reload_time

                events.append({
                    "event": "fire",
                    "piece_id": piece.piece_id,
                    "mission_id": mission.mission_id,
                    "shell_id": shell.shell_id,
                    "target": mission.target_pos,
                    "impact_pos": impact_pos,
                })

                # Only one round per tick if piece needs reload
                break

            # Check completion after firing
            if mission.rounds_fired >= mission.rounds:
                mission.active = False
                events.append({
                    "event": "mission_complete",
                    "mission_id": mission.mission_id,
                })

        # 3. Update shells in flight
        still_flying: list[Shell] = []
        for shell in self.shells_in_flight:
            shell.elapsed += dt
            max_alt = _max_altitude(distance(shell.origin, shell.target))
            shell.altitude = _parabolic_altitude(shell.elapsed, shell.time_of_flight, max_alt)

            if shell.elapsed >= shell.time_of_flight:
                # Impact
                if shell.shell_type == "smoke":
                    events.append({
                        "event": "smoke",
                        "shell_id": shell.shell_id,
                        "position": shell.impact_pos,
                        "radius": shell.blast_radius,
                        "mission_id": shell.mission_id,
                    })
                elif shell.shell_type == "illumination":
                    events.append({
                        "event": "illumination",
                        "shell_id": shell.shell_id,
                        "position": shell.impact_pos,
                        "radius": shell.blast_radius * 3.0,
                        "mission_id": shell.mission_id,
                    })
                else:
                    events.append({
                        "event": "impact",
                        "shell_id": shell.shell_id,
                        "position": shell.impact_pos,
                        "damage": shell.damage,
                        "blast_radius": shell.blast_radius,
                        "shell_type": shell.shell_type,
                        "mission_id": shell.mission_id,
                    })
            else:
                still_flying.append(shell)

        self.shells_in_flight = still_flying

        # 4. Cleanup completed missions
        self.fire_missions = [m for m in self.fire_missions if m.active]

        self._pending_events = events
        return events

    # -- Damage resolution --------------------------------------------------

    def resolve_impacts(
        self,
        events: list[dict],
        targets: list[tuple[Vec2, str, float]],
    ) -> list[dict]:
        """Resolve impact damage against a list of targets.

        *targets* is a list of ``(position, target_id, armor)`` tuples.
        Returns a list of damage result dicts.
        """
        results: list[dict] = []
        for evt in events:
            if evt.get("event") != "impact":
                continue
            center = evt["position"]
            blast_r = evt["blast_radius"]
            base_dmg = evt["damage"]

            for pos, tid, armor in targets:
                dist = distance(center, pos)
                if dist > blast_r:
                    continue
                # Linear falloff within blast radius
                ratio = dist / blast_r if blast_r > 0 else 0.0
                raw = base_dmg * (1.0 - ratio)
                effective = max(0.0, raw * (1.0 - min(1.0, armor)))
                results.append({
                    "target_id": tid,
                    "damage": round(effective, 2),
                    "distance": round(dist, 2),
                    "shell_id": evt["shell_id"],
                })

        return results

    # -- Three.js export ----------------------------------------------------

    def to_three_js(self) -> dict:
        """Export current state for Three.js rendering.

        Returns::

            {
                "shells": [...],       # in-flight shells with arc position
                "impacts": [...],      # impact explosions this tick
                "smoke_areas": [...],  # active smoke zones
                "illumination_areas": [...],  # illuminated zones
                "pieces": [...],       # artillery piece positions
            }
        """
        shells = []
        for s in self.shells_in_flight:
            # Interpolate XY position along origin->impact_pos
            if s.time_of_flight > 0:
                t = s.elapsed / s.time_of_flight
            else:
                t = 1.0
            x = s.origin[0] + (s.impact_pos[0] - s.origin[0]) * t
            y = s.origin[1] + (s.impact_pos[1] - s.origin[1]) * t
            shells.append({
                "id": s.shell_id,
                "x": round(x, 2),
                "y": round(y, 2),
                "altitude": round(s.altitude, 2),
                "shell_type": s.shell_type,
                "progress": round(t, 3),
            })

        impacts = []
        smoke_areas = []
        illumination_areas = []
        for evt in self._pending_events:
            if evt.get("event") == "impact":
                impacts.append({
                    "x": evt["position"][0],
                    "y": evt["position"][1],
                    "damage": evt["damage"],
                    "blast_radius": evt["blast_radius"],
                    "shell_type": evt["shell_type"],
                })
            elif evt.get("event") == "smoke":
                smoke_areas.append({
                    "x": evt["position"][0],
                    "y": evt["position"][1],
                    "radius": evt["radius"],
                })
            elif evt.get("event") == "illumination":
                illumination_areas.append({
                    "x": evt["position"][0],
                    "y": evt["position"][1],
                    "radius": evt["radius"],
                })

        pieces = []
        for p in self.pieces.values():
            pieces.append({
                "id": p.piece_id,
                "type": p.artillery_type.value,
                "x": p.position[0],
                "y": p.position[1],
                "heading": p.heading,
                "alliance": p.alliance,
                "ammo": p.ammo,
                "max_ammo": p.max_ammo,
                "ready": p.ready,
            })

        return {
            "shells": shells,
            "impacts": impacts,
            "smoke_areas": smoke_areas,
            "illumination_areas": illumination_areas,
            "pieces": pieces,
        }


# ---------------------------------------------------------------------------
# ARTILLERY_TEMPLATES — factory functions for each type
# ---------------------------------------------------------------------------

_TEMPLATES: dict[ArtilleryType, dict] = {
    ArtilleryType.MORTAR_60MM: {
        "min_range": 70.0,
        "max_range": 3500.0,
        "damage": 80.0,
        "blast_radius": 15.0,
        "reload_time": 3.0,
        "max_ammo": 60,
        "accuracy_cep": 20.0,
        "crew": 2,
    },
    ArtilleryType.MORTAR_81MM: {
        "min_range": 80.0,
        "max_range": 5600.0,
        "damage": 120.0,
        "blast_radius": 25.0,
        "reload_time": 4.0,
        "max_ammo": 40,
        "accuracy_cep": 25.0,
        "crew": 3,
    },
    ArtilleryType.HOWITZER_105MM: {
        "min_range": 500.0,
        "max_range": 11500.0,
        "damage": 200.0,
        "blast_radius": 35.0,
        "reload_time": 6.0,
        "max_ammo": 30,
        "accuracy_cep": 35.0,
        "crew": 5,
    },
    ArtilleryType.HOWITZER_155MM: {
        "min_range": 2000.0,
        "max_range": 24000.0,
        "damage": 350.0,
        "blast_radius": 50.0,
        "reload_time": 8.0,
        "max_ammo": 20,
        "accuracy_cep": 40.0,
        "crew": 8,
    },
    ArtilleryType.MLRS: {
        "min_range": 10000.0,
        "max_range": 70000.0,
        "damage": 500.0,
        "blast_radius": 75.0,
        "reload_time": 30.0,
        "max_ammo": 12,
        "accuracy_cep": 100.0,
        "crew": 3,
    },
    ArtilleryType.NAVAL_GUN: {
        "min_range": 1000.0,
        "max_range": 37000.0,
        "damage": 400.0,
        "blast_radius": 60.0,
        "reload_time": 5.0,
        "max_ammo": 100,
        "accuracy_cep": 50.0,
        "crew": 6,
    },
}


def create_piece(
    artillery_type: ArtilleryType,
    piece_id: str,
    alliance: str,
    position: Vec2,
    heading: float = 0.0,
    ammo: int | None = None,
) -> ArtilleryPiece:
    """Create an ArtilleryPiece from a template type."""
    tmpl = _TEMPLATES[artillery_type]
    return ArtilleryPiece(
        piece_id=piece_id,
        artillery_type=artillery_type,
        alliance=alliance,
        position=position,
        heading=heading,
        min_range=tmpl["min_range"],
        max_range=tmpl["max_range"],
        damage=tmpl["damage"],
        blast_radius=tmpl["blast_radius"],
        reload_time=tmpl["reload_time"],
        ammo=ammo if ammo is not None else tmpl["max_ammo"],
        max_ammo=tmpl["max_ammo"],
        accuracy_cep=tmpl["accuracy_cep"],
        crew=tmpl["crew"],
    )


# Expose templates dict for direct inspection
ARTILLERY_TEMPLATES: dict[ArtilleryType, dict] = dict(_TEMPLATES)


# ---------------------------------------------------------------------------
# ForwardObserver
# ---------------------------------------------------------------------------

class ForwardObserver:
    """A forward observer who can call for and adjust indirect fire."""

    def __init__(
        self,
        observer_id: str,
        position: Vec2,
        alliance: str,
    ) -> None:
        self.observer_id = observer_id
        self.position = position
        self.alliance = alliance
        self._active_missions: dict[str, str] = {}  # mission_id -> piece_id

    def call_fire(
        self,
        engine: ArtilleryEngine,
        target: Vec2,
        mission_type: str = "point",
        rounds: int = 3,
        piece_id: str | None = None,
        interval: float = 2.0,
    ) -> FireMission:
        """Request fire from the engine.

        If *piece_id* is None, selects the best available piece for the
        target range from matching alliance.
        """
        if piece_id is None:
            piece_id = self._select_piece(engine, target)
            if piece_id is None:
                raise ValueError("No available artillery piece in range")

        mission = engine.request_fire_mission(
            piece_id, target, mission_type, rounds, interval,
        )
        self._active_missions[mission.mission_id] = piece_id
        return mission

    def adjust_fire(
        self,
        engine: ArtilleryEngine,
        mission_id: str,
        offset: Vec2,
    ) -> FireMission | None:
        """Adjust an active fire mission by an offset (walk fire onto target).

        Cancels the old mission and creates a new one at the adjusted position
        with the remaining rounds.
        """
        old_mission: FireMission | None = None
        for m in engine.fire_missions:
            if m.mission_id == mission_id and m.active:
                old_mission = m
                break

        if old_mission is None:
            return None

        remaining = old_mission.rounds - old_mission.rounds_fired
        if remaining <= 0:
            return None

        new_target = (
            old_mission.target_pos[0] + offset[0],
            old_mission.target_pos[1] + offset[1],
        )

        engine.cancel_mission(mission_id)

        new_mission = engine.request_fire_mission(
            old_mission.piece_id,
            new_target,
            old_mission.mission_type,
            remaining,
            old_mission.interval,
        )
        self._active_missions.pop(mission_id, None)
        self._active_missions[new_mission.mission_id] = old_mission.piece_id
        return new_mission

    def end_mission(self, engine: ArtilleryEngine, mission_id: str) -> bool:
        """End (cancel) a fire mission."""
        self._active_missions.pop(mission_id, None)
        return engine.cancel_mission(mission_id)

    def _select_piece(self, engine: ArtilleryEngine, target: Vec2) -> str | None:
        """Pick the best in-range, same-alliance piece with ammo."""
        best_id: str | None = None
        best_range_diff = float("inf")

        for piece in engine.pieces.values():
            if piece.alliance != self.alliance:
                continue
            if piece.ammo <= 0:
                continue

            rng = distance(piece.position, target)
            if rng < piece.min_range or rng > piece.max_range:
                continue

            # Prefer piece whose max_range is closest to actual range (best accuracy)
            diff = abs(rng - (piece.min_range + piece.max_range) / 2.0)
            if diff < best_range_diff:
                best_range_diff = diff
                best_id = piece.piece_id

        return best_id
