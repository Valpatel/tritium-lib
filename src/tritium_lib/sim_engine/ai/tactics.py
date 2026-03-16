# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Advanced combat AI tactics engine for individual and squad decision-making.

Provides tactical situation assessment, threat evaluation, and coordinated
action selection. Units make intelligent decisions based on health, ammo,
morale, cover, and personality traits.

Integrates with steering (Vec2, distance), combat_ai (cover/flanking), and
squad (squad-level coordination).

Usage::

    from tritium_lib.sim_engine.ai.tactics import TacticsEngine, AIPersonality, PERSONALITY_PRESETS

    engine = TacticsEngine()
    threats = engine.assess_threats(unit_pos, enemies, dt=0.1)
    situation = engine.evaluate_situation(unit, threats, allies, cover_positions)
    action = engine.decide_action(situation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    magnitude,
    normalize,
    _sub,
    _add,
    _scale,
)
from tritium_lib.sim_engine.ai.combat_ai import (
    find_cover,
    is_in_cover,
    is_flanking,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ThreatAssessment:
    """Assessment of a single enemy threat."""

    threat_id: str
    position: Vec2
    distance: float
    threat_level: float  # 0-1, based on damage potential / distance
    is_flanking: bool
    is_suppressing: bool
    last_seen: float  # sim time
    estimated_health: float


@dataclass
class TacticalSituation:
    """Full tactical picture for a single unit."""

    unit_pos: Vec2
    unit_health: float
    unit_ammo: float  # 0-1 ratio
    unit_morale: float
    threats: list[ThreatAssessment]
    allies_nearby: int
    in_cover: bool
    cover_positions: list[Vec2]
    has_los_to_threats: list[bool]  # per threat, parallel to threats list
    squad_order: str | None


@dataclass
class TacticalAction:
    """A decided tactical action with reasoning."""

    action_type: str  # engage, suppress, flank, retreat, advance, hold,
    # heal_ally, throw_grenade, take_cover, relocate, overwatch
    target_pos: Vec2 | None
    target_id: str | None
    priority: float  # 0-1
    reasoning: str  # human-readable explanation


@dataclass
class AIPersonality:
    """Personality traits that modify tactical decision-making."""

    aggression: float = 0.5  # 0=cautious, 1=aggressive
    discipline: float = 0.5  # 0=panics easily, 1=holds under fire
    teamwork: float = 0.5  # 0=lone wolf, 1=always supports squad
    accuracy_bonus: float = 0.0


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


PERSONALITY_PRESETS: dict[str, AIPersonality] = {
    "veteran": AIPersonality(aggression=0.6, discipline=0.9, teamwork=0.8),
    "recruit": AIPersonality(aggression=0.3, discipline=0.3, teamwork=0.5),
    "berserker": AIPersonality(aggression=1.0, discipline=0.2, teamwork=0.1),
    "sniper": AIPersonality(aggression=0.2, discipline=0.9, teamwork=0.3),
    "medic": AIPersonality(aggression=0.1, discipline=0.7, teamwork=1.0),
    "leader": AIPersonality(aggression=0.5, discipline=0.8, teamwork=0.9),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _centroid(points: list[Vec2]) -> Vec2:
    """Average position of a list of points."""
    if not points:
        return (0.0, 0.0)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return (cx, cy)


def _cluster_radius(points: list[Vec2], center: Vec2) -> float:
    """Maximum distance from center to any point."""
    if not points:
        return 0.0
    return max(distance(p, center) for p in points)


def _nearest_threat(threats: list[ThreatAssessment]) -> ThreatAssessment | None:
    """Return the closest threat by distance."""
    if not threats:
        return None
    return min(threats, key=lambda t: t.distance)


def _highest_threat(threats: list[ThreatAssessment]) -> ThreatAssessment | None:
    """Return the most dangerous threat by threat_level."""
    if not threats:
        return None
    return max(threats, key=lambda t: t.threat_level)


def _threats_clustered(threats: list[ThreatAssessment], radius: float = 10.0) -> bool:
    """Check if multiple threats are clustered within a radius."""
    if len(threats) < 2:
        return False
    positions = [t.position for t in threats]
    center = _centroid(positions)
    return _cluster_radius(positions, center) <= radius


# ---------------------------------------------------------------------------
# TacticsEngine
# ---------------------------------------------------------------------------


class TacticsEngine:
    """Tactical decision-making engine for combat units.

    Assesses threats, evaluates the tactical situation, and selects
    actions based on a priority-driven decision tree. Personality traits
    modify thresholds and preferences.
    """

    def __init__(self, personality: AIPersonality | None = None) -> None:
        self.personality = personality or AIPersonality()

    # -- Threat assessment ---------------------------------------------------

    def assess_threats(
        self,
        unit_pos: Vec2,
        enemies: list[dict],
        dt: float,
    ) -> list[ThreatAssessment]:
        """Evaluate all visible enemies and produce threat assessments.

        Each enemy dict should have:
            - id: str
            - pos: Vec2
            - damage: float (damage per second capability)
            - facing: float (radians, optional)
            - suppressing: bool (optional)
            - health: float (0-1, optional)
            - last_seen: float (sim time, optional)
        """
        assessments: list[ThreatAssessment] = []
        for enemy in enemies:
            e_pos = enemy["pos"]
            e_id = enemy["id"]
            dist = distance(unit_pos, e_pos)
            damage = enemy.get("damage", 1.0)

            # Threat level: high damage + close distance = high threat
            # Normalize: damage contribution (0-0.5) + distance contribution (0-0.5)
            damage_factor = min(damage / 10.0, 0.5)
            # Closer = more threatening; beyond 100m threat drops off
            dist_factor = max(0.0, 0.5 * (1.0 - dist / 100.0))
            threat_level = min(1.0, damage_factor + dist_factor)

            # Check flanking
            e_facing = enemy.get("facing", 0.0)
            flanking = is_flanking(unit_pos, e_pos, e_facing)

            suppressing = enemy.get("suppressing", False)
            health = enemy.get("health", 1.0)
            last_seen = enemy.get("last_seen", 0.0)

            assessments.append(ThreatAssessment(
                threat_id=e_id,
                position=e_pos,
                distance=dist,
                threat_level=threat_level,
                is_flanking=flanking,
                is_suppressing=suppressing,
                last_seen=last_seen,
                estimated_health=health,
            ))

        # Sort by threat level descending
        assessments.sort(key=lambda t: t.threat_level, reverse=True)
        return assessments

    # -- Situation evaluation ------------------------------------------------

    def evaluate_situation(
        self,
        unit: dict,
        threats: list[ThreatAssessment],
        allies: list[dict],
        cover: list[Vec2],
    ) -> TacticalSituation:
        """Build a complete tactical picture for a unit.

        unit dict should have:
            - pos: Vec2
            - health: float (0-1)
            - ammo: float (0-1)
            - morale: float (0-1)
            - facing: float (radians, optional)
            - squad_order: str | None (optional)

        allies is a list of dicts with at least:
            - pos: Vec2

        cover is a list of Vec2 positions that provide cover.
        """
        unit_pos: Vec2 = unit["pos"]
        unit_health: float = unit.get("health", 1.0)
        unit_ammo: float = unit.get("ammo", 1.0)
        unit_morale: float = unit.get("morale", 1.0)
        squad_order: str | None = unit.get("squad_order")

        # Count nearby allies (within 30m)
        allies_nearby = sum(
            1 for a in allies if distance(unit_pos, a["pos"]) <= 30.0
        )

        # Check if unit is in cover relative to any threat
        # Build obstacle list from cover positions (treat each as a small obstacle)
        obstacles = [(cp, 2.0) for cp in cover]
        in_cover_flag = False
        if threats:
            nearest = _nearest_threat(threats)
            if nearest:
                in_cover_flag = is_in_cover(unit_pos, nearest.position, obstacles)

        # Line of sight per threat: True if NOT blocked by cover
        has_los: list[bool] = []
        for threat in threats:
            # If cover blocks the line to this threat, no LOS
            blocked = is_in_cover(unit_pos, threat.position, obstacles)
            has_los.append(not blocked)

        return TacticalSituation(
            unit_pos=unit_pos,
            unit_health=unit_health,
            unit_ammo=unit_ammo,
            unit_morale=unit_morale,
            threats=threats,
            allies_nearby=allies_nearby,
            in_cover=in_cover_flag,
            cover_positions=cover,
            has_los_to_threats=has_los,
            squad_order=squad_order,
        )

    # -- Action decision -----------------------------------------------------

    def decide_action(self, situation: TacticalSituation) -> TacticalAction:
        """Select the best tactical action based on the situation and personality.

        Decision tree (checked in priority order):
        1. Health < 20% (adjusted by discipline) and no medic nearby -> retreat
        2. Ammo < 10% -> retreat to resupply
        3. Morale < 0.3 (adjusted by discipline) -> retreat
        4. Suppressed (>0.7 of threats suppressing) -> stay in cover
        5. No threats visible -> advance toward objective / patrol
        6. Threat is flanking -> relocate to cover facing flank
        7. Outnumbered 3:1 -> suppress and retreat
        8. In cover with LOS -> engage nearest threat
        9. Not in cover -> take cover first
        10. Multiple threats clustered + have grenades -> throw_grenade
        11. Single threat, good position -> engage
        12. Ally down nearby -> heal (if medic personality)
        """
        p = self.personality
        s = situation

        # Effective thresholds modified by personality
        health_retreat_threshold = 0.2 * (2.0 - p.discipline)  # disciplined units tough it out
        morale_retreat_threshold = 0.3 * (2.0 - p.discipline)

        # 1. Critical health -> retreat
        if s.unit_health < health_retreat_threshold:
            return TacticalAction(
                action_type="retreat",
                target_pos=self._find_retreat_pos(s),
                target_id=None,
                priority=1.0,
                reasoning=f"Health critical ({s.unit_health:.0%}), retreating to survive",
            )

        # 2. Out of ammo -> retreat to resupply
        if s.unit_ammo < 0.1:
            return TacticalAction(
                action_type="retreat",
                target_pos=self._find_retreat_pos(s),
                target_id=None,
                priority=0.95,
                reasoning=f"Ammo critically low ({s.unit_ammo:.0%}), retreating to resupply",
            )

        # 3. Morale broken -> retreat (unless very disciplined)
        if s.unit_morale < morale_retreat_threshold:
            return TacticalAction(
                action_type="retreat",
                target_pos=self._find_retreat_pos(s),
                target_id=None,
                priority=0.9,
                reasoning=f"Morale broken ({s.unit_morale:.0%}), retreating",
            )

        # 4. Being suppressed -> stay in cover
        if s.threats:
            suppressing_count = sum(1 for t in s.threats if t.is_suppressing)
            suppression_ratio = suppressing_count / len(s.threats)
            if suppression_ratio > 0.7:
                if s.in_cover:
                    return TacticalAction(
                        action_type="hold",
                        target_pos=None,
                        target_id=None,
                        priority=0.85,
                        reasoning="Under heavy suppression, holding in cover until it lifts",
                    )
                else:
                    return TacticalAction(
                        action_type="take_cover",
                        target_pos=self._nearest_cover(s),
                        target_id=None,
                        priority=0.85,
                        reasoning="Under heavy suppression with no cover, seeking cover immediately",
                    )

        # 5. No threats -> advance or patrol
        if not s.threats:
            action = "advance" if p.aggression > 0.5 else "hold"
            return TacticalAction(
                action_type=action,
                target_pos=None,
                target_id=None,
                priority=0.2,
                reasoning="No threats visible, " + (
                    "advancing toward objective" if action == "advance"
                    else "holding position and scanning"
                ),
            )

        # From here, we have threats
        nearest = _nearest_threat(s.threats)
        highest = _highest_threat(s.threats)

        # 6. Flanking threat -> relocate
        flanking_threats = [t for t in s.threats if t.is_flanking]
        if flanking_threats:
            ft = flanking_threats[0]
            cover_pos = self._find_cover_facing(s, ft.position)
            return TacticalAction(
                action_type="relocate",
                target_pos=cover_pos,
                target_id=ft.threat_id,
                priority=0.8,
                reasoning=f"Threat {ft.threat_id} flanking from {ft.distance:.0f}m, relocating to face",
            )

        # 7. Outnumbered 3:1 -> suppress and retreat
        enemy_count = len(s.threats)
        friendly_count = s.allies_nearby + 1  # +1 for self
        if enemy_count >= friendly_count * 3:
            # Aggressive units suppress first, cautious ones just retreat
            if p.aggression > 0.5 and s.unit_ammo > 0.2:
                return TacticalAction(
                    action_type="suppress",
                    target_pos=nearest.position if nearest else None,
                    target_id=nearest.threat_id if nearest else None,
                    priority=0.75,
                    reasoning=f"Outnumbered {enemy_count}:{friendly_count}, suppressing to cover retreat",
                )
            else:
                return TacticalAction(
                    action_type="retreat",
                    target_pos=self._find_retreat_pos(s),
                    target_id=None,
                    priority=0.75,
                    reasoning=f"Outnumbered {enemy_count}:{friendly_count}, retreating",
                )

        # 8. In cover with LOS -> engage
        if s.in_cover:
            # Find a threat we have LOS to
            for i, threat in enumerate(s.threats):
                if i < len(s.has_los_to_threats) and s.has_los_to_threats[i]:
                    return TacticalAction(
                        action_type="engage",
                        target_pos=threat.position,
                        target_id=threat.threat_id,
                        priority=0.7,
                        reasoning=f"In cover with LOS, engaging {threat.threat_id} at {threat.distance:.0f}m",
                    )
            # In cover but no LOS to any threat -> hold/overwatch
            return TacticalAction(
                action_type="overwatch",
                target_pos=None,
                target_id=None,
                priority=0.6,
                reasoning="In cover but no clear LOS, setting up overwatch",
            )

        # 9. Not in cover -> take cover first (unless very aggressive)
        if not s.in_cover and s.cover_positions:
            if p.aggression < 0.8:
                return TacticalAction(
                    action_type="take_cover",
                    target_pos=self._nearest_cover(s),
                    target_id=None,
                    priority=0.65,
                    reasoning="Exposed without cover, moving to nearest cover position",
                )

        # 10. Multiple clustered threats -> grenade
        if len(s.threats) >= 2 and _threats_clustered(s.threats, radius=10.0):
            center = _centroid([t.position for t in s.threats])
            return TacticalAction(
                action_type="throw_grenade",
                target_pos=center,
                target_id=None,
                priority=0.7,
                reasoning=f"Enemies clustered within 10m, throwing grenade at cluster center",
            )

        # 11. Medic personality + wounded ally -> heal
        if p.teamwork >= 0.9 and p.aggression <= 0.2:
            # Medic behavior: prioritize healing if we are a medic-type personality
            return TacticalAction(
                action_type="heal_ally",
                target_pos=None,
                target_id=None,
                priority=0.6,
                reasoning="Medic personality, checking for wounded allies to heal",
            )

        # 12. Default: engage the nearest/highest threat
        target = highest if highest else nearest
        if target:
            return TacticalAction(
                action_type="engage",
                target_pos=target.position,
                target_id=target.threat_id,
                priority=0.5,
                reasoning=f"Engaging {target.threat_id} (threat level {target.threat_level:.2f}) at {target.distance:.0f}m",
            )

        # Fallback: hold
        return TacticalAction(
            action_type="hold",
            target_pos=None,
            target_id=None,
            priority=0.1,
            reasoning="No clear action, holding position",
        )

    # -- Squad coordination --------------------------------------------------

    def decide_squad_action(
        self,
        squad_situation: dict,
    ) -> list[tuple[str, TacticalAction]]:
        """Produce coordinated actions for all units in a squad.

        squad_situation dict should have:
            - units: list[dict] each with id, pos, health, ammo, morale, role
            - threats: list[ThreatAssessment]
            - cover_positions: list[Vec2]
            - objective: Vec2 | None

        Returns a list of (unit_id, TacticalAction) tuples.

        Coordination logic:
        - Split into fire teams for bounding overwatch
        - Assign suppressors and movers
        - Coordinate flanking (half suppress, half move)
        - Medic prioritizes wounded
        """
        units = squad_situation.get("units", [])
        threats = squad_situation.get("threats", [])
        cover_positions = squad_situation.get("cover_positions", [])
        objective = squad_situation.get("objective")

        if not units:
            return []

        results: list[tuple[str, TacticalAction]] = []

        # Identify roles
        medics = [u for u in units if u.get("role") == "medic"]
        non_medics = [u for u in units if u.get("role") != "medic"]

        # Find wounded allies
        wounded = [u for u in units if u.get("health", 1.0) < 0.5]

        # Assign medics to heal wounded
        for medic in medics:
            if wounded:
                target = min(wounded, key=lambda w: distance(medic["pos"], w["pos"]))
                results.append((medic["id"], TacticalAction(
                    action_type="heal_ally",
                    target_pos=target["pos"],
                    target_id=target["id"],
                    priority=0.9,
                    reasoning=f"Medic moving to heal wounded ally {target['id']}",
                )))
                wounded = [w for w in wounded if w["id"] != target["id"]]
            else:
                # No wounded, medic follows squad
                results.append((medic["id"], TacticalAction(
                    action_type="hold",
                    target_pos=None,
                    target_id=None,
                    priority=0.3,
                    reasoning="Medic holding position, no wounded allies",
                )))

        if not threats:
            # No threats: everyone advances toward objective
            for u in non_medics:
                results.append((u["id"], TacticalAction(
                    action_type="advance",
                    target_pos=objective,
                    target_id=None,
                    priority=0.3,
                    reasoning="No threats, advancing toward objective",
                )))
            return results

        # Split non-medics into two fire teams for bounding overwatch
        half = max(1, len(non_medics) // 2)
        suppressors = non_medics[:half]
        movers = non_medics[half:]

        nearest = _nearest_threat(threats)

        # Suppressors: suppress the nearest threat
        for u in suppressors:
            results.append((u["id"], TacticalAction(
                action_type="suppress",
                target_pos=nearest.position if nearest else None,
                target_id=nearest.threat_id if nearest else None,
                priority=0.7,
                reasoning=f"Fire team A: suppressing {nearest.threat_id if nearest else 'area'} to cover movement",
            )))

        # Movers: flank or advance
        if len(threats) >= 1 and nearest:
            # Compute a flank position perpendicular to the threat
            squad_center = _centroid([u["pos"] for u in units])
            to_threat = _sub(nearest.position, squad_center)
            perp = (-to_threat[1], to_threat[0])
            perp_norm = normalize(perp)
            flank_target = _add(nearest.position, _scale(perp_norm, 15.0))

            for u in movers:
                results.append((u["id"], TacticalAction(
                    action_type="flank",
                    target_pos=flank_target,
                    target_id=nearest.threat_id,
                    priority=0.7,
                    reasoning=f"Fire team B: flanking {nearest.threat_id} while team A suppresses",
                )))
        else:
            for u in movers:
                results.append((u["id"], TacticalAction(
                    action_type="advance",
                    target_pos=objective,
                    target_id=None,
                    priority=0.5,
                    reasoning="Fire team B: advancing toward objective",
                )))

        return results

    # -- Internal helpers ----------------------------------------------------

    def _find_retreat_pos(self, situation: TacticalSituation) -> Vec2 | None:
        """Find a position away from threats."""
        if not situation.threats:
            return None
        # Move directly away from the centroid of all threats
        threat_center = _centroid([t.position for t in situation.threats])
        away = _sub(situation.unit_pos, threat_center)
        away_norm = normalize(away)
        # Retreat 30m away from threat center
        return _add(situation.unit_pos, _scale(away_norm, 30.0))

    def _nearest_cover(self, situation: TacticalSituation) -> Vec2 | None:
        """Find the nearest cover position."""
        if not situation.cover_positions:
            return None
        return min(situation.cover_positions, key=lambda cp: distance(situation.unit_pos, cp))

    def _find_cover_facing(self, situation: TacticalSituation, threat_pos: Vec2) -> Vec2 | None:
        """Find cover that faces a specific threat direction."""
        if not situation.cover_positions:
            return None
        # Prefer cover that is not between us and the threat
        best: Vec2 | None = None
        best_score = float("inf")
        for cp in situation.cover_positions:
            d = distance(situation.unit_pos, cp)
            # Penalize cover that is toward the threat
            to_cover = _sub(cp, situation.unit_pos)
            to_threat = _sub(threat_pos, situation.unit_pos)
            to_cover_norm = normalize(to_cover)
            to_threat_norm = normalize(to_threat)
            dot = to_cover_norm[0] * to_threat_norm[0] + to_cover_norm[1] * to_threat_norm[1]
            # Prefer cover that is NOT toward the threat (low dot product)
            score = d + dot * 20.0  # penalize cover toward the threat
            if score < best_score:
                best_score = score
                best = cp
        return best
