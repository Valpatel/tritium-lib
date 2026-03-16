# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""High-level AI strategy planner for faction-level decisions.

Operates above the tactical layer (tactics.py, squad.py) to make strategic
choices: which squads attack where, when to retreat faction-wide, how to
allocate reserves.  Pluggable strategy profiles (aggressive, defensive,
guerrilla, etc.) modify decision thresholds.

Usage::

    from tritium_lib.sim_engine.ai.strategy import StrategicAI, StrategicGoal

    ai = StrategicAI(profile="balanced")
    assessment = ai.assess(world_state)
    plan = ai.plan("alpha_faction", assessment)
    assignments = ai.assign_squads(plan, squads, positions)
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    normalize,
    _sub,
    _add,
    _scale,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StrategicGoal(Enum):
    """High-level strategic objectives for a faction."""

    ATTACK = "attack"
    DEFEND = "defend"
    FLANK = "flank"
    ENCIRCLE = "encircle"
    RETREAT = "retreat"
    REINFORCE = "reinforce"
    PROBE = "probe"
    AMBUSH = "ambush"
    SIEGE = "siege"
    PATROL = "patrol"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StrategicPlan:
    """A faction-level strategic plan assigning squads to objectives."""

    plan_id: str
    goal: StrategicGoal
    faction: str
    primary_target: Vec2 | None = None
    secondary_targets: list[Vec2] = field(default_factory=list)
    assigned_squads: list[str] = field(default_factory=list)
    reserve_squads: list[str] = field(default_factory=list)
    priority: int = 0
    confidence: float = 0.5
    reasoning: str = ""


@dataclass
class StrategyProfile:
    """Tunable parameters that define a faction's strategic personality."""

    name: str
    attack_threshold: float = 1.2      # force ratio needed to attack
    defend_threshold: float = 0.8      # force ratio below which we defend
    retreat_threshold: float = 0.4     # force ratio below which we retreat
    flank_preference: float = 0.5      # 0=never flank, 1=always try
    reserve_fraction: float = 0.2      # fraction of squads held in reserve
    aggression: float = 0.5            # general aggression modifier 0-1
    caution: float = 0.5               # general caution modifier 0-1
    probe_when_unknown: bool = True    # send probes when enemy pos unknown
    ambush_preference: float = 0.3     # tendency to set ambushes 0-1
    siege_patience: float = 0.5        # willingness to wait out sieges 0-1


# ---------------------------------------------------------------------------
# Strategy profiles
# ---------------------------------------------------------------------------


STRATEGY_PROFILES: dict[str, StrategyProfile] = {
    "aggressive": StrategyProfile(
        name="aggressive",
        attack_threshold=0.9,
        defend_threshold=0.5,
        retreat_threshold=0.25,
        flank_preference=0.3,
        reserve_fraction=0.1,
        aggression=0.9,
        caution=0.1,
        probe_when_unknown=False,
        ambush_preference=0.1,
        siege_patience=0.2,
    ),
    "defensive": StrategyProfile(
        name="defensive",
        attack_threshold=2.0,
        defend_threshold=1.0,
        retreat_threshold=0.5,
        flank_preference=0.2,
        reserve_fraction=0.35,
        aggression=0.2,
        caution=0.9,
        probe_when_unknown=True,
        ambush_preference=0.5,
        siege_patience=0.8,
    ),
    "balanced": StrategyProfile(
        name="balanced",
        attack_threshold=1.2,
        defend_threshold=0.8,
        retreat_threshold=0.4,
        flank_preference=0.5,
        reserve_fraction=0.2,
        aggression=0.5,
        caution=0.5,
        probe_when_unknown=True,
        ambush_preference=0.3,
        siege_patience=0.5,
    ),
    "guerrilla": StrategyProfile(
        name="guerrilla",
        attack_threshold=1.5,
        defend_threshold=0.6,
        retreat_threshold=0.3,
        flank_preference=0.8,
        reserve_fraction=0.15,
        aggression=0.6,
        caution=0.7,
        probe_when_unknown=True,
        ambush_preference=0.9,
        siege_patience=0.1,
    ),
    "blitz": StrategyProfile(
        name="blitz",
        attack_threshold=0.7,
        defend_threshold=0.4,
        retreat_threshold=0.2,
        flank_preference=0.7,
        reserve_fraction=0.05,
        aggression=1.0,
        caution=0.0,
        probe_when_unknown=False,
        ambush_preference=0.0,
        siege_patience=0.0,
    ),
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


def _gen_plan_id() -> str:
    """Generate a short unique plan ID."""
    return f"plan_{uuid.uuid4().hex[:8]}"


def _cluster_radius(positions: list[Vec2]) -> float:
    """Maximum distance from the centroid to any point."""
    if len(positions) < 2:
        return 0.0
    center = _centroid(positions)
    return max(distance(center, p) for p in positions)


def _flank_exposed(
    enemy_positions: list[Vec2],
    friendly_positions: list[Vec2],
) -> Vec2 | None:
    """Detect if the enemy has an exposed flank.

    Returns a flank attack position if the enemy cluster is elongated
    and one side is undefended, else None.
    """
    if len(enemy_positions) < 2 or not friendly_positions:
        return None

    e_center = _centroid(enemy_positions)
    f_center = _centroid(friendly_positions)

    # Direction from friendly to enemy
    to_enemy = _sub(e_center, f_center)
    to_enemy_mag = math.hypot(to_enemy[0], to_enemy[1])
    if to_enemy_mag < 1e-6:
        return None

    # Perpendicular direction (potential flank axis)
    perp = (-to_enemy[1], to_enemy[0])
    perp_norm = normalize(perp)

    # Check spread of enemies along the perpendicular axis
    projections = [
        (p[0] - e_center[0]) * perp_norm[0] + (p[1] - e_center[1]) * perp_norm[1]
        for p in enemy_positions
    ]
    spread = max(projections) - min(projections) if projections else 0.0

    # If the enemy is spread out along the perpendicular, their flank is exposed
    if spread > 15.0:
        # Attack from the side with fewer enemies
        left_count = sum(1 for p in projections if p < 0)
        right_count = sum(1 for p in projections if p >= 0)
        if left_count <= right_count:
            # Flank from the left (fewer enemies)
            flank_pos = _add(e_center, _scale(perp_norm, -spread * 0.6))
        else:
            flank_pos = _add(e_center, _scale(perp_norm, spread * 0.6))
        return flank_pos

    return None


def _enemy_concentrated(enemy_positions: list[Vec2], threshold: float = 15.0) -> bool:
    """True if all enemies are within a tight cluster."""
    if len(enemy_positions) < 3:
        return False
    return _cluster_radius(enemy_positions) <= threshold


# ---------------------------------------------------------------------------
# StrategicAI
# ---------------------------------------------------------------------------


class StrategicAI:
    """Faction-level strategic planner.

    Assesses the battlefield at a high level and produces StrategicPlans
    that assign squads to objectives.  Strategy profiles control how
    aggressive, cautious, or unconventional the AI behaves.

    Args:
        profile: Name of a preset profile or a StrategyProfile instance.
    """

    def __init__(self, profile: str | StrategyProfile = "balanced") -> None:
        if isinstance(profile, str):
            self.profile = STRATEGY_PROFILES.get(profile, STRATEGY_PROFILES["balanced"])
        else:
            self.profile = profile
        self._plan_history: list[StrategicPlan] = []

    @property
    def plan_history(self) -> list[StrategicPlan]:
        """All plans produced by this AI, newest last."""
        return list(self._plan_history)

    # -- Assessment ----------------------------------------------------------

    def assess(self, world_state: dict) -> dict:
        """Evaluate the strategic situation from a world state dict.

        Expected world_state keys:
            - friendly_squads: list[dict] with 'id', 'position', 'strength', 'morale', 'ammo'
            - enemy_squads: list[dict] with 'id', 'position', 'strength' (estimated)
            - objectives: list[dict] with 'position', 'owner', 'value'
            - terrain: dict (optional) with 'chokepoints', 'high_ground', 'cover_zones'
            - fog_of_war: bool (default True) — whether enemy positions are uncertain

        Returns a dict describing the strategic picture.
        """
        friendly = world_state.get("friendly_squads", [])
        enemies = world_state.get("enemy_squads", [])
        objectives = world_state.get("objectives", [])
        terrain = world_state.get("terrain", {})
        fog = world_state.get("fog_of_war", True)

        # Force calculation
        friendly_strength = sum(s.get("strength", 1.0) for s in friendly)
        enemy_strength = sum(s.get("strength", 1.0) for s in enemies)
        force_ratio = friendly_strength / max(enemy_strength, 0.01)

        # Morale average
        friendly_morale = 0.0
        if friendly:
            friendly_morale = sum(s.get("morale", 0.5) for s in friendly) / len(friendly)

        # Ammo average
        friendly_ammo = 0.0
        if friendly:
            friendly_ammo = sum(s.get("ammo", 0.5) for s in friendly) / len(friendly)

        # Casualty ratio (if provided)
        total_initial = world_state.get("initial_friendly_strength", friendly_strength)
        casualty_ratio = 1.0 - (friendly_strength / max(total_initial, 0.01))

        # Position analysis
        friendly_positions = [s["position"] for s in friendly if "position" in s]
        enemy_positions = [s["position"] for s in enemies if "position" in s]

        friendly_center = _centroid(friendly_positions)
        enemy_center = _centroid(enemy_positions)

        # Distance between forces
        front_distance = distance(friendly_center, enemy_center) if friendly_positions and enemy_positions else float("inf")

        # Flank analysis
        exposed_flank = _flank_exposed(enemy_positions, friendly_positions)

        # Enemy concentration
        enemy_is_concentrated = _enemy_concentrated(enemy_positions)

        # Objectives held
        friendly_objectives = [o for o in objectives if o.get("owner") == "friendly"]
        contested_objectives = [o for o in objectives if o.get("owner") in ("contested", None)]
        enemy_objectives = [o for o in objectives if o.get("owner") == "enemy"]

        # Terrain advantages
        chokepoints = terrain.get("chokepoints", [])
        high_ground = terrain.get("high_ground", [])

        return {
            "force_ratio": force_ratio,
            "friendly_strength": friendly_strength,
            "enemy_strength": enemy_strength,
            "friendly_morale": friendly_morale,
            "friendly_ammo": friendly_ammo,
            "casualty_ratio": casualty_ratio,
            "friendly_center": friendly_center,
            "enemy_center": enemy_center,
            "front_distance": front_distance,
            "exposed_flank": exposed_flank,
            "enemy_concentrated": enemy_is_concentrated,
            "friendly_positions": friendly_positions,
            "enemy_positions": enemy_positions,
            "friendly_objectives": len(friendly_objectives),
            "contested_objectives": len(contested_objectives),
            "enemy_objectives": len(enemy_objectives),
            "chokepoints": chokepoints,
            "high_ground": high_ground,
            "fog_of_war": fog,
            "num_friendly_squads": len(friendly),
            "num_enemy_squads": len(enemies),
        }

    # -- Planning ------------------------------------------------------------

    def plan(self, faction: str, assessment: dict) -> StrategicPlan:
        """Produce a strategic plan based on the assessment.

        Decision logic (checked in order):
        1. Casualties too high + low ammo -> RETREAT + REINFORCE
        2. Force ratio < retreat_threshold -> RETREAT
        3. Force ratio < defend_threshold -> DEFEND
        4. Unknown enemy position (fog + no enemies seen) -> PROBE or PATROL
        5. Enemy flank exposed -> FLANK
        6. Enemy concentrated + we have superiority -> ENCIRCLE
        7. Ambush opportunity (guerrilla profile, chokepoints) -> AMBUSH
        8. Siege conditions (enemy in fortified objective) -> SIEGE
        9. Force ratio >= attack_threshold -> ATTACK
        10. Default -> DEFEND
        """
        p = self.profile
        ratio = assessment.get("force_ratio", 1.0)
        morale = assessment.get("friendly_morale", 0.5)
        ammo = assessment.get("friendly_ammo", 0.5)
        casualty = assessment.get("casualty_ratio", 0.0)
        enemy_center = assessment.get("enemy_center", (0.0, 0.0))
        exposed_flank = assessment.get("exposed_flank")
        enemy_conc = assessment.get("enemy_concentrated", False)
        fog = assessment.get("fog_of_war", True)
        num_enemy = assessment.get("num_enemy_squads", 0)
        chokepoints = assessment.get("chokepoints", [])
        enemy_objectives = assessment.get("enemy_objectives", 0)
        front_distance = assessment.get("front_distance", float("inf"))

        # 1. Critical condition: heavy casualties + low ammo
        if casualty > 0.5 and ammo < 0.3:
            plan = StrategicPlan(
                plan_id=_gen_plan_id(),
                goal=StrategicGoal.RETREAT,
                faction=faction,
                primary_target=None,
                priority=10,
                confidence=0.9,
                reasoning=f"Heavy casualties ({casualty:.0%}) and low ammo ({ammo:.0%}), retreating to reinforce",
            )
            self._plan_history.append(plan)
            return plan

        # 2. Badly outnumbered -> retreat
        if ratio < p.retreat_threshold:
            plan = StrategicPlan(
                plan_id=_gen_plan_id(),
                goal=StrategicGoal.RETREAT,
                faction=faction,
                primary_target=None,
                priority=9,
                confidence=0.85,
                reasoning=f"Force ratio {ratio:.2f} below retreat threshold {p.retreat_threshold}, withdrawing",
            )
            self._plan_history.append(plan)
            return plan

        # 3. Outnumbered but not critically -> defend
        if ratio < p.defend_threshold:
            plan = StrategicPlan(
                plan_id=_gen_plan_id(),
                goal=StrategicGoal.DEFEND,
                faction=faction,
                primary_target=None,
                priority=7,
                confidence=0.7,
                reasoning=f"Force ratio {ratio:.2f} below attack threshold, defensive posture",
            )
            self._plan_history.append(plan)
            return plan

        # 4. Unknown enemy position -> PROBE or PATROL
        if fog and num_enemy == 0 and p.probe_when_unknown:
            goal = StrategicGoal.PROBE
            plan = StrategicPlan(
                plan_id=_gen_plan_id(),
                goal=goal,
                faction=faction,
                primary_target=enemy_center if enemy_center != (0.0, 0.0) else None,
                priority=4,
                confidence=0.4,
                reasoning="No enemy contact under fog of war, sending probes to locate",
            )
            self._plan_history.append(plan)
            return plan

        if not fog and num_enemy == 0:
            plan = StrategicPlan(
                plan_id=_gen_plan_id(),
                goal=StrategicGoal.PATROL,
                faction=faction,
                primary_target=None,
                priority=2,
                confidence=0.6,
                reasoning="No enemies detected, patrolling to maintain awareness",
            )
            self._plan_history.append(plan)
            return plan

        # 5. Enemy flank exposed -> FLANK
        if exposed_flank is not None and p.flank_preference > 0.3:
            conf = min(1.0, 0.5 + p.flank_preference * 0.3 + (ratio - 1.0) * 0.2)
            plan = StrategicPlan(
                plan_id=_gen_plan_id(),
                goal=StrategicGoal.FLANK,
                faction=faction,
                primary_target=exposed_flank,
                secondary_targets=[enemy_center],
                priority=8,
                confidence=max(0.3, conf),
                reasoning=f"Enemy flank exposed, executing flanking maneuver (preference {p.flank_preference:.1f})",
            )
            self._plan_history.append(plan)
            return plan

        # 6. Enemy concentrated + superiority -> ENCIRCLE
        if enemy_conc and ratio >= p.attack_threshold:
            plan = StrategicPlan(
                plan_id=_gen_plan_id(),
                goal=StrategicGoal.ENCIRCLE,
                faction=faction,
                primary_target=enemy_center,
                priority=8,
                confidence=min(1.0, 0.5 + (ratio - 1.0) * 0.3),
                reasoning=f"Enemy concentrated and we outnumber them {ratio:.1f}:1, encircling",
            )
            self._plan_history.append(plan)
            return plan

        # 7. Ambush opportunity (guerrilla style)
        if chokepoints and p.ambush_preference > 0.5 and ratio < 1.5:
            # Pick nearest chokepoint to enemy center
            best_cp = min(chokepoints, key=lambda cp: distance(cp, enemy_center))
            plan = StrategicPlan(
                plan_id=_gen_plan_id(),
                goal=StrategicGoal.AMBUSH,
                faction=faction,
                primary_target=best_cp,
                priority=7,
                confidence=0.6 + p.ambush_preference * 0.2,
                reasoning=f"Chokepoint available and ambush-favorable profile, setting ambush",
            )
            self._plan_history.append(plan)
            return plan

        # 8. Siege conditions (enemy holds objectives, we have patience)
        if enemy_objectives > 0 and p.siege_patience > 0.4 and ratio >= 1.0:
            plan = StrategicPlan(
                plan_id=_gen_plan_id(),
                goal=StrategicGoal.SIEGE,
                faction=faction,
                primary_target=enemy_center,
                priority=6,
                confidence=0.5 + p.siege_patience * 0.3,
                reasoning=f"Enemy holds {enemy_objectives} objective(s), besieging with patience {p.siege_patience:.1f}",
            )
            self._plan_history.append(plan)
            return plan

        # 9. Force ratio >= attack_threshold -> ATTACK
        if ratio >= p.attack_threshold:
            conf = min(1.0, 0.4 + (ratio - p.attack_threshold) * 0.3 + morale * 0.2)
            plan = StrategicPlan(
                plan_id=_gen_plan_id(),
                goal=StrategicGoal.ATTACK,
                faction=faction,
                primary_target=enemy_center,
                priority=8,
                confidence=max(0.3, conf),
                reasoning=f"Force ratio {ratio:.2f} favorable, attacking enemy position",
            )
            self._plan_history.append(plan)
            return plan

        # 10. Default -> DEFEND
        plan = StrategicPlan(
            plan_id=_gen_plan_id(),
            goal=StrategicGoal.DEFEND,
            faction=faction,
            primary_target=None,
            priority=5,
            confidence=0.5,
            reasoning=f"No clear advantage (ratio {ratio:.2f}), holding defensive posture",
        )
        self._plan_history.append(plan)
        return plan

    # -- Adaptation ----------------------------------------------------------

    def adapt(self, plan: StrategicPlan, events: list[dict]) -> StrategicPlan:
        """Adapt an existing plan based on battlefield events.

        Recognized event types:
            - ambush_detected: enemy ambush found -> switch to DEFEND or RETREAT
            - reinforcements_arrived: more troops -> consider escalating to ATTACK
            - objective_taken: we captured something -> shift focus
            - heavy_casualties: lost many units -> consider RETREAT
            - enemy_retreating: enemy pulling back -> pursue (ATTACK)
            - flank_threatened: our flank is at risk -> adjust
            - supply_line_cut: logistics disrupted -> RETREAT
            - intel_update: new enemy positions revealed

        Returns a new plan (does not mutate the original).
        """
        p = self.profile
        new_goal = plan.goal
        new_target = plan.primary_target
        new_secondary = list(plan.secondary_targets)
        new_priority = plan.priority
        new_confidence = plan.confidence
        reasons: list[str] = [plan.reasoning]

        for event in events:
            etype = event.get("type", "")

            if etype == "ambush_detected":
                if new_goal == StrategicGoal.ATTACK:
                    new_goal = StrategicGoal.DEFEND
                    new_confidence = max(0.3, new_confidence - 0.2)
                    reasons.append("Ambush detected, shifting to defense")
                elif new_goal == StrategicGoal.FLANK:
                    new_goal = StrategicGoal.RETREAT
                    new_confidence = max(0.2, new_confidence - 0.3)
                    reasons.append("Ambush detected during flank, retreating")

            elif etype == "reinforcements_arrived":
                strength_added = event.get("strength", 0.0)
                new_confidence = min(1.0, new_confidence + 0.15)
                if new_goal in (StrategicGoal.DEFEND, StrategicGoal.RETREAT):
                    if p.aggression > 0.4:
                        new_goal = StrategicGoal.ATTACK
                        reasons.append(f"Reinforcements (+{strength_added:.0f}), escalating to attack")
                    else:
                        new_goal = StrategicGoal.DEFEND
                        reasons.append(f"Reinforcements (+{strength_added:.0f}), strengthening defense")
                new_priority = max(new_priority, 7)

            elif etype == "objective_taken":
                obj_pos = event.get("position")
                if obj_pos and new_target:
                    # If we took our primary, shift to next target
                    if obj_pos == new_target:
                        if new_secondary:
                            new_target = new_secondary.pop(0)
                            reasons.append("Primary objective taken, shifting to next target")
                        else:
                            new_goal = StrategicGoal.DEFEND
                            reasons.append("All objectives taken, consolidating defense")
                new_confidence = min(1.0, new_confidence + 0.1)

            elif etype == "heavy_casualties":
                casualty_pct = event.get("casualty_percent", 0.3)
                new_confidence = max(0.1, new_confidence - casualty_pct)
                if casualty_pct > 0.4 or new_confidence < 0.3:
                    new_goal = StrategicGoal.RETREAT
                    new_priority = 10
                    reasons.append(f"Heavy casualties ({casualty_pct:.0%}), forced retreat")
                else:
                    reasons.append(f"Casualties sustained ({casualty_pct:.0%}), reevaluating")

            elif etype == "enemy_retreating":
                retreat_pos = event.get("position")
                if p.aggression > 0.3:
                    new_goal = StrategicGoal.ATTACK
                    if retreat_pos:
                        new_target = retreat_pos
                    new_confidence = min(1.0, new_confidence + 0.2)
                    reasons.append("Enemy retreating, pursuing")
                else:
                    new_goal = StrategicGoal.PATROL
                    reasons.append("Enemy retreating, transitioning to patrol")

            elif etype == "flank_threatened":
                threat_pos = event.get("position")
                if threat_pos:
                    new_secondary.append(threat_pos)
                if new_goal == StrategicGoal.ATTACK:
                    new_goal = StrategicGoal.DEFEND
                    reasons.append("Flank threatened, shifting to defense")
                new_confidence = max(0.2, new_confidence - 0.15)

            elif etype == "supply_line_cut":
                new_goal = StrategicGoal.RETREAT
                new_priority = 9
                new_confidence = max(0.2, new_confidence - 0.25)
                reasons.append("Supply line cut, forced retreat")

            elif etype == "intel_update":
                new_positions = event.get("enemy_positions", [])
                for pos in new_positions:
                    if pos not in new_secondary:
                        new_secondary.append(pos)
                reasons.append(f"Intel update: {len(new_positions)} new enemy position(s)")

        adapted = StrategicPlan(
            plan_id=_gen_plan_id(),
            goal=new_goal,
            faction=plan.faction,
            primary_target=new_target,
            secondary_targets=new_secondary,
            assigned_squads=list(plan.assigned_squads),
            reserve_squads=list(plan.reserve_squads),
            priority=new_priority,
            confidence=max(0.0, min(1.0, new_confidence)),
            reasoning=" | ".join(reasons),
        )
        self._plan_history.append(adapted)
        return adapted

    # -- Squad assignment ----------------------------------------------------

    def assign_squads(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
    ) -> dict[str, dict]:
        """Assign squads to roles within a strategic plan.

        Each squad dict should have at minimum:
            - id: str
            - strength: float
            - morale: float (optional)
            - specialty: str (optional) — "armor", "recon", "infantry", etc.

        positions maps squad_id -> current Vec2 position.

        Returns a dict mapping squad_id to an assignment dict:
            {
                "role": "assault" | "flanking" | "reserve" | "support" | "screening" | "siege" | "probe" | "patrol",
                "target": Vec2 | None,
                "orders": str,
            }
        """
        if not squads:
            return {}

        p = self.profile
        assignments: dict[str, dict] = {}

        # Sort squads by strength descending for priority assignment
        sorted_squads = sorted(squads, key=lambda s: s.get("strength", 1.0), reverse=True)

        # Determine reserve count
        reserve_count = max(0, int(len(sorted_squads) * p.reserve_fraction))

        # Pick reserve squads (weakest units go to reserve)
        reserve_ids: set[str] = set()
        if reserve_count > 0:
            reserve_candidates = sorted_squads[-reserve_count:]
            for sq in reserve_candidates:
                sid = sq["id"]
                reserve_ids.add(sid)
                assignments[sid] = {
                    "role": "reserve",
                    "target": None,
                    "orders": "Hold in reserve, ready to reinforce",
                }

        active_squads = [s for s in sorted_squads if s["id"] not in reserve_ids]

        if plan.goal == StrategicGoal.ATTACK:
            self._assign_attack(plan, active_squads, positions, assignments)

        elif plan.goal == StrategicGoal.DEFEND:
            self._assign_defend(plan, active_squads, positions, assignments)

        elif plan.goal == StrategicGoal.FLANK:
            self._assign_flank(plan, active_squads, positions, assignments)

        elif plan.goal == StrategicGoal.ENCIRCLE:
            self._assign_encircle(plan, active_squads, positions, assignments)

        elif plan.goal == StrategicGoal.RETREAT:
            self._assign_retreat(plan, active_squads, positions, assignments)

        elif plan.goal == StrategicGoal.REINFORCE:
            self._assign_reinforce(plan, active_squads, positions, assignments)

        elif plan.goal == StrategicGoal.PROBE:
            self._assign_probe(plan, active_squads, positions, assignments)

        elif plan.goal == StrategicGoal.AMBUSH:
            self._assign_ambush(plan, active_squads, positions, assignments)

        elif plan.goal == StrategicGoal.SIEGE:
            self._assign_siege(plan, active_squads, positions, assignments)

        elif plan.goal == StrategicGoal.PATROL:
            self._assign_patrol(plan, active_squads, positions, assignments)

        # Update plan with assigned/reserve squad lists
        plan.assigned_squads = [s["id"] for s in active_squads]
        plan.reserve_squads = list(reserve_ids)

        return assignments

    # -- Assignment helpers --------------------------------------------------

    def _assign_attack(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
        out: dict[str, dict],
    ) -> None:
        """All active squads assault the primary target."""
        target = plan.primary_target or (0.0, 0.0)
        # First half: direct assault. Second half: support/suppress.
        half = max(1, len(squads) // 2)
        for i, sq in enumerate(squads):
            if i < half:
                out[sq["id"]] = {
                    "role": "assault",
                    "target": target,
                    "orders": "Advance and engage enemy at primary target",
                }
            else:
                out[sq["id"]] = {
                    "role": "support",
                    "target": target,
                    "orders": "Provide suppressive fire and support assault element",
                }

    def _assign_defend(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
        out: dict[str, dict],
    ) -> None:
        """Distribute squads across defensive positions."""
        for sq in squads:
            sid = sq["id"]
            pos = positions.get(sid, (0.0, 0.0))
            out[sid] = {
                "role": "defense",
                "target": pos,  # Hold current position
                "orders": "Hold position and repel attackers",
            }

    def _assign_flank(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
        out: dict[str, dict],
    ) -> None:
        """Split into flanking element and fixing force."""
        flank_target = plan.primary_target or (0.0, 0.0)
        fix_target = plan.secondary_targets[0] if plan.secondary_targets else (0.0, 0.0)

        # One third flanks, two thirds fix
        flank_count = max(1, len(squads) // 3)
        for i, sq in enumerate(squads):
            if i < flank_count:
                out[sq["id"]] = {
                    "role": "flanking",
                    "target": flank_target,
                    "orders": "Execute flanking maneuver on exposed enemy side",
                }
            else:
                out[sq["id"]] = {
                    "role": "support",
                    "target": fix_target,
                    "orders": "Fix enemy in place while flanking element maneuvers",
                }

    def _assign_encircle(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
        out: dict[str, dict],
    ) -> None:
        """Distribute squads around the enemy position."""
        center = plan.primary_target or (0.0, 0.0)
        n = len(squads)
        encircle_radius = 40.0
        for i, sq in enumerate(squads):
            angle = 2.0 * math.pi * i / max(1, n)
            target = (
                center[0] + math.cos(angle) * encircle_radius,
                center[1] + math.sin(angle) * encircle_radius,
            )
            out[sq["id"]] = {
                "role": "encircle",
                "target": target,
                "orders": f"Move to encirclement position (sector {i+1}/{n})",
            }

    def _assign_retreat(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
        out: dict[str, dict],
    ) -> None:
        """All squads withdraw. Strongest covers the retreat."""
        # Strongest squad acts as rearguard
        for i, sq in enumerate(squads):
            if i == 0 and len(squads) > 1:
                out[sq["id"]] = {
                    "role": "rearguard",
                    "target": positions.get(sq["id"], (0.0, 0.0)),
                    "orders": "Cover retreat of other squads, disengage last",
                }
            else:
                out[sq["id"]] = {
                    "role": "retreat",
                    "target": None,
                    "orders": "Withdraw to rally point",
                }

    def _assign_reinforce(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
        out: dict[str, dict],
    ) -> None:
        """Move all squads toward the reinforcement target."""
        target = plan.primary_target or (0.0, 0.0)
        for sq in squads:
            out[sq["id"]] = {
                "role": "reinforce",
                "target": target,
                "orders": "Move to reinforce allied position",
            }

    def _assign_probe(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
        out: dict[str, dict],
    ) -> None:
        """Send recon squads forward, keep others ready."""
        # Send up to half as probes; rest hold
        probe_count = max(1, len(squads) // 2)
        target = plan.primary_target
        for i, sq in enumerate(squads):
            if i < probe_count:
                # Prefer recon-type squads
                out[sq["id"]] = {
                    "role": "probe",
                    "target": target,
                    "orders": "Advance cautiously and report enemy positions",
                }
            else:
                out[sq["id"]] = {
                    "role": "support",
                    "target": positions.get(sq["id"]),
                    "orders": "Hold position, ready to support probing element",
                }

    def _assign_ambush(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
        out: dict[str, dict],
    ) -> None:
        """Position squads along an ambush kill zone."""
        ambush_point = plan.primary_target or (0.0, 0.0)
        n = len(squads)
        # Spread along a line perpendicular to the expected approach
        for i, sq in enumerate(squads):
            offset = (i - (n - 1) / 2.0) * 15.0
            target = (ambush_point[0] + offset, ambush_point[1])
            out[sq["id"]] = {
                "role": "ambush",
                "target": target,
                "orders": "Conceal and wait for enemy to enter kill zone, fire on signal",
            }

    def _assign_siege(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
        out: dict[str, dict],
    ) -> None:
        """Surround and blockade the enemy position."""
        center = plan.primary_target or (0.0, 0.0)
        n = len(squads)
        siege_radius = 60.0
        for i, sq in enumerate(squads):
            angle = 2.0 * math.pi * i / max(1, n)
            target = (
                center[0] + math.cos(angle) * siege_radius,
                center[1] + math.sin(angle) * siege_radius,
            )
            out[sq["id"]] = {
                "role": "siege",
                "target": target,
                "orders": f"Maintain siege perimeter (sector {i+1}/{n}), prevent breakout",
            }

    def _assign_patrol(
        self,
        plan: StrategicPlan,
        squads: list[dict],
        positions: dict[str, Vec2],
        out: dict[str, dict],
    ) -> None:
        """Distribute squads across patrol routes."""
        for sq in squads:
            sid = sq["id"]
            out[sid] = {
                "role": "patrol",
                "target": positions.get(sid),
                "orders": "Patrol assigned sector and report contacts",
            }
