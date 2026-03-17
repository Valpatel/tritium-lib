# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Civilian population and infrastructure simulation module.

Simulates civilian population behaviour, infrastructure systems (power, water,
roads, telecom), and collateral damage tracking.  Civilians react to nearby
threats and explosions, flee or shelter, and accumulate fear.  Infrastructure
can be damaged or repaired, and its operational state cascades into the
population (e.g. hospital without power is degraded).  Population sentiment
tracks the "hearts and minds" dimension.

All spatial units are meters.  Positions are Vec2 = tuple[float, float].
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    normalize,
    _add,
    _sub,
    _scale,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CivilianState(IntEnum):
    """Lifecycle states for a simulated civilian."""
    NORMAL = 0
    SHELTERING = 1
    FLEEING = 2
    INJURED = 3
    DEAD = 4


class InfrastructureType(IntEnum):
    """Types of infrastructure that can exist in the simulation."""
    POWER_PLANT = 0
    WATER_TREATMENT = 1
    HOSPITAL = 2
    SCHOOL = 3
    MARKET = 4
    ROAD = 5
    BRIDGE = 6
    TELECOM_TOWER = 7


# ---------------------------------------------------------------------------
# Infrastructure templates
# ---------------------------------------------------------------------------

INFRASTRUCTURE_TEMPLATES: dict[InfrastructureType, dict[str, Any]] = {
    InfrastructureType.POWER_PLANT: {
        "health": 200.0,
        "radius": 500.0,
        "population_capacity": 5000,
        "repair_rate": 0.05,
    },
    InfrastructureType.WATER_TREATMENT: {
        "health": 150.0,
        "radius": 400.0,
        "population_capacity": 4000,
        "repair_rate": 0.08,
    },
    InfrastructureType.HOSPITAL: {
        "health": 120.0,
        "radius": 300.0,
        "population_capacity": 500,
        "repair_rate": 0.1,
    },
    InfrastructureType.SCHOOL: {
        "health": 80.0,
        "radius": 200.0,
        "population_capacity": 300,
        "repair_rate": 0.15,
    },
    InfrastructureType.MARKET: {
        "health": 60.0,
        "radius": 150.0,
        "population_capacity": 200,
        "repair_rate": 0.2,
    },
    InfrastructureType.ROAD: {
        "health": 100.0,
        "radius": 50.0,
        "population_capacity": 0,
        "repair_rate": 0.05,
    },
    InfrastructureType.BRIDGE: {
        "health": 150.0,
        "radius": 30.0,
        "population_capacity": 0,
        "repair_rate": 0.03,
    },
    InfrastructureType.TELECOM_TOWER: {
        "health": 80.0,
        "radius": 1000.0,
        "population_capacity": 10000,
        "repair_rate": 0.1,
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Civilian:
    """A single simulated civilian."""
    civilian_id: str
    position: Vec2
    state: CivilianState = CivilianState.NORMAL
    home_position: Vec2 = (0.0, 0.0)
    work_position: Vec2 | None = None
    destination: Vec2 | None = None
    speed: float = 3.0
    fear: float = 0.0
    health: float = 100.0


@dataclass
class Infrastructure:
    """A piece of simulated infrastructure."""
    infra_id: str
    infra_type: InfrastructureType
    position: Vec2
    radius: float
    health: float = 100.0
    max_health: float = 100.0
    is_operational: bool = True
    serves_population: int = 0
    repair_rate: float = 0.1


@dataclass
class CollateralDamage:
    """Record of a collateral damage event."""
    event_id: str
    position: Vec2
    timestamp: float
    civilian_casualties: int = 0
    infrastructure_damage: list[str] = field(default_factory=list)
    cause: str = "unknown"
    severity: float = 0.0
    hearts_minds_impact: float = 0.0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

# Tuning constants
_THREAT_AWARENESS_RADIUS = 200.0   # civilians react to threats within this range
_EXPLOSION_LETHAL_RADIUS = 15.0    # instant death
_EXPLOSION_INJURY_RADIUS = 40.0    # injury zone
_EXPLOSION_FEAR_RADIUS = 120.0     # fear propagation
_FEAR_DECAY_RATE = 0.05            # per second when no threats
_FEAR_FLEE_THRESHOLD = 0.6         # above this civilians flee
_FEAR_SHELTER_THRESHOLD = 0.3      # above this civilians shelter
_SENTIMENT_DECAY_RATE = 0.001      # slow drift toward neutral per tick
_COLLATERAL_SENTIMENT_PENALTY = 0.05  # per casualty
_INFRA_DAMAGE_SENTIMENT_PENALTY = 0.02  # per damaged infra


class CivilianSimulator:
    """Simulate civilian population, infrastructure, and collateral damage."""

    def __init__(self) -> None:
        self.civilians: list[Civilian] = []
        self.infrastructure: dict[str, Infrastructure] = {}
        self.collateral_events: list[CollateralDamage] = []
        self.population_sentiment: float = 0.5
        self._time: float = 0.0

    # ----- population spawning -----

    def spawn_population(
        self,
        center: Vec2,
        count: int,
        radius: float,
        with_infrastructure: bool = True,
    ) -> None:
        """Spawn *count* civilians around *center* within *radius*.

        If *with_infrastructure* is True, a default set of infrastructure
        is placed near the population centre.
        """
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            r = radius * math.sqrt(random.random())
            pos = (center[0] + r * math.cos(angle), center[1] + r * math.sin(angle))
            home = pos
            # ~60 % of civilians have a work position offset from home
            work: Vec2 | None = None
            if random.random() < 0.6:
                wa = random.uniform(0, 2 * math.pi)
                wr = random.uniform(50, radius)
                work = (center[0] + wr * math.cos(wa), center[1] + wr * math.sin(wa))
            civ = Civilian(
                civilian_id=f"civ_{uuid.uuid4().hex[:8]}",
                position=pos,
                home_position=home,
                work_position=work,
            )
            self.civilians.append(civ)

        if with_infrastructure:
            self._spawn_default_infrastructure(center, radius)

    def _spawn_default_infrastructure(self, center: Vec2, radius: float) -> None:
        """Place a basic infrastructure set around *center*."""
        placements: list[tuple[InfrastructureType, Vec2]] = [
            (InfrastructureType.POWER_PLANT, (center[0] + radius * 0.6, center[1])),
            (InfrastructureType.WATER_TREATMENT, (center[0] - radius * 0.5, center[1] + radius * 0.3)),
            (InfrastructureType.HOSPITAL, (center[0], center[1] + radius * 0.2)),
            (InfrastructureType.SCHOOL, (center[0] - radius * 0.3, center[1] - radius * 0.2)),
            (InfrastructureType.MARKET, (center[0] + radius * 0.2, center[1] - radius * 0.3)),
            (InfrastructureType.TELECOM_TOWER, (center[0], center[1] - radius * 0.5)),
        ]
        for itype, pos in placements:
            tpl = INFRASTRUCTURE_TEMPLATES[itype]
            infra = Infrastructure(
                infra_id=f"infra_{uuid.uuid4().hex[:8]}",
                infra_type=itype,
                position=pos,
                radius=tpl["radius"],
                health=tpl["health"],
                max_health=tpl["health"],
                serves_population=tpl["population_capacity"],
                repair_rate=tpl["repair_rate"],
            )
            self.infrastructure[infra.infra_id] = infra

    # ----- tick -----

    def tick(
        self,
        dt: float,
        threats: list[tuple[Vec2, float]] | None = None,
        explosions: list[tuple[Vec2, float]] | None = None,
    ) -> dict[str, Any]:
        """Advance the simulation by *dt* seconds.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        threats : list of (position, danger_radius)
            Active threat locations (combat, gunfire).
        explosions : list of (position, blast_radius)
            New explosion events this tick.

        Returns a summary dict with keys: casualties, injured, infrastructure_damaged,
        sentiment, fear_avg.
        """
        if threats is None:
            threats = []
        if explosions is None:
            explosions = []

        self._time += dt
        casualties = 0
        newly_injured = 0
        damaged_infra_ids: list[str] = []

        # --- process explosions ---
        for exp_pos, exp_radius in explosions:
            lethal_r = min(exp_radius, _EXPLOSION_LETHAL_RADIUS)
            injury_r = max(exp_radius, _EXPLOSION_INJURY_RADIUS)
            fear_r = max(exp_radius * 3, _EXPLOSION_FEAR_RADIUS)

            civ_killed = 0
            civ_injured_here = 0

            for civ in self.civilians:
                if civ.state == CivilianState.DEAD:
                    continue
                d = distance(civ.position, exp_pos)
                if d <= lethal_r:
                    civ.state = CivilianState.DEAD
                    civ.health = 0.0
                    civ_killed += 1
                elif d <= injury_r:
                    dmg = max(0.0, 80.0 * (1.0 - d / injury_r))
                    civ.health = max(0.0, civ.health - dmg)
                    if civ.health <= 0:
                        civ.state = CivilianState.DEAD
                        civ_killed += 1
                    elif civ.state != CivilianState.INJURED:
                        civ.state = CivilianState.INJURED
                        civ_injured_here += 1
                    civ.fear = min(1.0, civ.fear + 0.5)
                if d <= fear_r and civ.state not in (CivilianState.DEAD, CivilianState.INJURED):
                    fear_add = 0.8 * (1.0 - d / fear_r)
                    civ.fear = min(1.0, civ.fear + fear_add)

            # infrastructure damage from explosions
            exp_infra: list[str] = []
            for iid, infra in self.infrastructure.items():
                d = distance(infra.position, exp_pos)
                if d <= exp_radius * 1.5:
                    dmg = infra.max_health * 0.4 * (1.0 - d / (exp_radius * 1.5))
                    infra.health = max(0.0, infra.health - dmg)
                    if infra.health <= 0:
                        infra.is_operational = False
                    exp_infra.append(iid)
                    if iid not in damaged_infra_ids:
                        damaged_infra_ids.append(iid)

            casualties += civ_killed
            newly_injured += civ_injured_here

            # record collateral event
            severity = min(1.0, (civ_killed * 0.3 + civ_injured_here * 0.1 + len(exp_infra) * 0.15))
            hm_impact = -(civ_killed * _COLLATERAL_SENTIMENT_PENALTY
                          + civ_injured_here * _COLLATERAL_SENTIMENT_PENALTY * 0.5
                          + len(exp_infra) * _INFRA_DAMAGE_SENTIMENT_PENALTY)
            self.population_sentiment = max(0.0, self.population_sentiment + hm_impact)

            event = CollateralDamage(
                event_id=f"cd_{uuid.uuid4().hex[:8]}",
                position=exp_pos,
                timestamp=self._time,
                civilian_casualties=civ_killed + civ_injured_here,
                infrastructure_damage=exp_infra,
                cause="explosion",
                severity=severity,
                hearts_minds_impact=hm_impact,
            )
            self.collateral_events.append(event)

        # --- civilian reactions to threats ---
        for civ in self.civilians:
            if civ.state == CivilianState.DEAD:
                continue

            # accumulate fear from nearby threats
            for threat_pos, threat_radius in threats:
                d = distance(civ.position, threat_pos)
                awareness = max(threat_radius, _THREAT_AWARENESS_RADIUS)
                if d < awareness:
                    fear_add = 0.3 * dt * (1.0 - d / awareness)
                    civ.fear = min(1.0, civ.fear + fear_add)

            # state transitions based on fear
            if civ.state == CivilianState.INJURED:
                # injured stay injured, slowly lose health without hospital
                if not self._hospital_nearby(civ.position):
                    civ.health = max(0.0, civ.health - 0.5 * dt)
                    if civ.health <= 0:
                        civ.state = CivilianState.DEAD
                        casualties += 1
                else:
                    civ.health = min(100.0, civ.health + 2.0 * dt)
                    if civ.health >= 50.0:
                        civ.state = CivilianState.SHELTERING
                        civ.fear = max(civ.fear, _FEAR_SHELTER_THRESHOLD)
                continue

            if civ.fear >= _FEAR_FLEE_THRESHOLD:
                civ.state = CivilianState.FLEEING
                # flee away from nearest threat
                nearest_threat = self._nearest_threat(civ.position, threats)
                if nearest_threat is not None:
                    away = normalize(_sub(civ.position, nearest_threat))
                    move = _scale(away, civ.speed * dt)
                    civ.position = _add(civ.position, move)
                else:
                    # flee toward home
                    civ.destination = civ.home_position
            elif civ.fear >= _FEAR_SHELTER_THRESHOLD:
                civ.state = CivilianState.SHELTERING
                # move toward home
                civ.destination = civ.home_position
            else:
                civ.state = CivilianState.NORMAL

            # movement toward destination
            if civ.destination is not None and civ.state != CivilianState.DEAD:
                d = distance(civ.position, civ.destination)
                if d > 1.0:
                    direction = normalize(_sub(civ.destination, civ.position))
                    step = min(civ.speed * dt, d)
                    civ.position = _add(civ.position, _scale(direction, step))
                else:
                    civ.destination = None

            # fear decay when safe
            if not threats and not explosions:
                civ.fear = max(0.0, civ.fear - _FEAR_DECAY_RATE * dt)

        # --- infrastructure cascading effects ---
        self._update_infrastructure_cascades()

        # --- sentiment drift toward neutral ---
        if not explosions and not threats:
            if self.population_sentiment < 0.5:
                self.population_sentiment = min(0.5, self.population_sentiment + _SENTIMENT_DECAY_RATE * dt)
            elif self.population_sentiment > 0.5:
                self.population_sentiment = max(0.5, self.population_sentiment - _SENTIMENT_DECAY_RATE * dt)

        # clamp sentiment
        self.population_sentiment = max(0.0, min(1.0, self.population_sentiment))

        alive = [c for c in self.civilians if c.state != CivilianState.DEAD]
        fear_avg = sum(c.fear for c in alive) / max(1, len(alive))

        return {
            "casualties": casualties,
            "injured": newly_injured,
            "infrastructure_damaged": damaged_infra_ids,
            "sentiment": round(self.population_sentiment, 4),
            "fear_avg": round(fear_avg, 4),
        }

    def _hospital_nearby(self, pos: Vec2) -> bool:
        """Return True if an operational hospital is within range of *pos*."""
        for infra in self.infrastructure.values():
            if (
                infra.infra_type == InfrastructureType.HOSPITAL
                and infra.is_operational
                and distance(pos, infra.position) <= infra.radius
            ):
                # check if power is available — hospital effectiveness degrades without power
                if self._has_power(infra.position):
                    return True
                # degraded hospital: 30 % chance of effective treatment
                return random.random() < 0.3
        return False

    def _has_power(self, pos: Vec2) -> bool:
        """Return True if an operational power plant covers *pos*."""
        for infra in self.infrastructure.values():
            if (
                infra.infra_type == InfrastructureType.POWER_PLANT
                and infra.is_operational
                and distance(pos, infra.position) <= infra.radius
            ):
                return True
        return False

    def _nearest_threat(self, pos: Vec2, threats: list[tuple[Vec2, float]]) -> Vec2 | None:
        """Return the position of the nearest threat, or None."""
        best: Vec2 | None = None
        best_d = float("inf")
        for tp, _ in threats:
            d = distance(pos, tp)
            if d < best_d:
                best_d = d
                best = tp
        return best

    def _update_infrastructure_cascades(self) -> None:
        """Apply cascading effects: no power degrades hospitals, etc."""
        for infra in self.infrastructure.values():
            if infra.health <= 0:
                infra.is_operational = False
            elif infra.health < infra.max_health * 0.2:
                # below 20 % health — becomes non-operational
                infra.is_operational = False
            else:
                infra.is_operational = True

    # ----- humanitarian aid -----

    def provide_aid(self, position: Vec2, radius: float, amount: float) -> float:
        """Provide humanitarian aid, improving sentiment and healing injured.

        Returns the total sentiment improvement applied.
        """
        helped = 0
        for civ in self.civilians:
            if civ.state == CivilianState.DEAD:
                continue
            if distance(civ.position, position) <= radius:
                helped += 1
                civ.fear = max(0.0, civ.fear - amount * 0.1)
                if civ.state == CivilianState.INJURED:
                    civ.health = min(100.0, civ.health + amount * 2.0)
                    if civ.health >= 50.0:
                        civ.state = CivilianState.SHELTERING

        if helped == 0:
            return 0.0

        improvement = min(0.1, amount * 0.01 * helped / max(1, len(self.civilians)))
        self.population_sentiment = min(1.0, self.population_sentiment + improvement)
        return improvement

    # ----- infrastructure repair -----

    def repair_infrastructure(self, infra_id: str, engineers: int, dt: float) -> float:
        """Repair infrastructure by *engineers* over *dt* seconds.

        Returns the amount of health restored.
        """
        infra = self.infrastructure.get(infra_id)
        if infra is None:
            return 0.0
        if infra.health >= infra.max_health:
            return 0.0

        repair_amount = infra.repair_rate * engineers * dt
        old_health = infra.health
        infra.health = min(infra.max_health, infra.health + repair_amount)

        # re-evaluate operational status
        if infra.health >= infra.max_health * 0.2:
            infra.is_operational = True

        restored = infra.health - old_health

        # repairing infrastructure improves sentiment slightly
        self.population_sentiment = min(
            1.0, self.population_sentiment + restored * 0.0005
        )
        return restored

    # ----- reporting -----

    def get_population_report(self) -> dict[str, Any]:
        """Return a summary of population status."""
        alive = 0
        injured = 0
        dead = 0
        sheltering = 0
        fleeing = 0
        normal = 0

        for civ in self.civilians:
            if civ.state == CivilianState.DEAD:
                dead += 1
            elif civ.state == CivilianState.INJURED:
                injured += 1
                alive += 1
            else:
                alive += 1
                if civ.state == CivilianState.SHELTERING:
                    sheltering += 1
                elif civ.state == CivilianState.FLEEING:
                    fleeing += 1
                else:
                    normal += 1

        operational_infra = sum(1 for i in self.infrastructure.values() if i.is_operational)
        total_infra = len(self.infrastructure)

        return {
            "total": len(self.civilians),
            "alive": alive,
            "injured": injured,
            "dead": dead,
            "sheltering": sheltering,
            "fleeing": fleeing,
            "normal": normal,
            "sentiment": round(self.population_sentiment, 4),
            "collateral_events": len(self.collateral_events),
            "infrastructure_operational": operational_infra,
            "infrastructure_total": total_infra,
        }

    # ----- Three.js export -----

    def to_three_js(self) -> dict[str, Any]:
        """Export state as a JSON-serializable dict for Three.js rendering."""
        civs = []
        for civ in self.civilians:
            civs.append({
                "id": civ.civilian_id,
                "x": round(civ.position[0], 2),
                "y": round(civ.position[1], 2),
                "state": civ.state.name.lower(),
                "fear": round(civ.fear, 2),
            })

        infras = []
        for infra in self.infrastructure.values():
            infras.append({
                "id": infra.infra_id,
                "type": infra.infra_type.name.lower(),
                "x": round(infra.position[0], 2),
                "y": round(infra.position[1], 2),
                "health_pct": round(infra.health / infra.max_health, 2) if infra.max_health > 0 else 0.0,
                "operational": infra.is_operational,
                "radius": round(infra.radius, 1),
            })

        report = self.get_population_report()

        # sentiment colour: red (hostile) -> yellow (neutral) -> green (supportive)
        s = self.population_sentiment
        if s < 0.5:
            # red to yellow
            r = 255
            g = int(255 * (s / 0.5))
            b = 0
        else:
            # yellow to green
            r = int(255 * ((1.0 - s) / 0.5))
            g = 255
            b = 0
        sentiment_color = f"#{r:02x}{g:02x}{b:02x}"

        return {
            "civilians": civs,
            "infrastructure": infras,
            "sentiment": round(self.population_sentiment, 4),
            "sentiment_color": sentiment_color,
            "casualties": {
                "alive": report["alive"],
                "injured": report["injured"],
                "dead": report["dead"],
            },
        }
