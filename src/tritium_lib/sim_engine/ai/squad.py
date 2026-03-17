"""Squad coordination AI — group-level tactics, morale, and formation control.

Squads are groups of units that share orders, threat information, and morale.
The SquadTactics class provides formation computation, fire sector assignment,
bounding overwatch, and tactical order recommendation.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    normalize,
    magnitude,
    _sub,
    _add,
    _scale,
)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------


class SquadRole(Enum):
    """Role a unit fills within a squad."""

    LEADER = "leader"
    RIFLEMAN = "rifleman"
    SUPPORT = "support"
    SCOUT = "scout"
    MEDIC = "medic"
    ENGINEER = "engineer"


@dataclass
class Order:
    """A tactical order issued to a squad."""

    order_type: str  # advance, hold, retreat, flank_left, flank_right, suppress, cover_fire, regroup, patrol, guard
    target_pos: Vec2 | None = None
    target_id: str | None = None
    priority: int = 0
    issued_at: float = 0.0


@dataclass
class SquadState:
    """Aggregate state of a squad."""

    cohesion: float = 1.0  # 0-1, how close together
    morale: float = 1.0  # average squad morale
    alert_level: float = 0.0  # 0=relaxed, 1=combat
    known_threats: list[tuple[Vec2, str]] = field(default_factory=list)
    casualties: int = 0
    ammo_status: float = 1.0  # 0-1, average ammo remaining


# ---------------------------------------------------------------------------
# Squad
# ---------------------------------------------------------------------------


class Squad:
    """A group of units that coordinate, share intel, and maintain morale."""

    # Promotion priority: leader > support > engineer > rifleman > scout > medic
    PROMOTION_PRIORITY: ClassVar[list[SquadRole]] = [
        SquadRole.SUPPORT,
        SquadRole.ENGINEER,
        SquadRole.RIFLEMAN,
        SquadRole.SCOUT,
        SquadRole.MEDIC,
    ]

    def __init__(self, squad_id: str, name: str, alliance: str) -> None:
        self.squad_id = squad_id
        self.name = name
        self.alliance = alliance
        self.members: list[str] = []
        self.roles: dict[str, SquadRole] = {}
        self.leader_id: str | None = None
        self.state = SquadState()
        self.current_order: Order | None = None
        self.order_history: list[Order] = []

    # -- Membership ----------------------------------------------------------

    def add_member(self, unit_id: str, role: SquadRole = SquadRole.RIFLEMAN) -> None:
        """Add a unit to the squad. First member with LEADER role becomes leader."""
        if unit_id in self.members:
            return
        self.members.append(unit_id)
        self.roles[unit_id] = role
        if role == SquadRole.LEADER:
            self.leader_id = unit_id
        elif self.leader_id is None:
            # First member becomes leader if none assigned
            self.leader_id = unit_id
            self.roles[unit_id] = SquadRole.LEADER

    def remove_member(self, unit_id: str) -> None:
        """Remove a unit. If it was the leader, auto-promote the best candidate."""
        if unit_id not in self.members:
            return
        self.members.remove(unit_id)
        self.roles.pop(unit_id, None)
        self.state.casualties += 1

        if unit_id == self.leader_id:
            self.leader_id = None
            self._auto_promote()

    def _auto_promote(self) -> None:
        """Promote the highest-priority remaining member to leader."""
        if not self.members:
            return
        for priority_role in self.PROMOTION_PRIORITY:
            for mid in self.members:
                if self.roles.get(mid) == priority_role:
                    self.leader_id = mid
                    self.roles[mid] = SquadRole.LEADER
                    return
        # Fallback: first remaining member
        first = self.members[0]
        self.leader_id = first
        self.roles[first] = SquadRole.LEADER

    # -- Orders --------------------------------------------------------------

    def issue_order(self, order: Order) -> None:
        """Set the current order, archiving the previous one."""
        if self.current_order is not None:
            self.order_history.append(self.current_order)
        self.current_order = order

    # -- Spatial queries -----------------------------------------------------

    def center_of_mass(self, positions: dict[str, Vec2]) -> Vec2:
        """Average position of living squad members."""
        pts = [positions[m] for m in self.members if m in positions]
        if not pts:
            return (0.0, 0.0)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return (cx, cy)

    def spread(self, positions: dict[str, Vec2]) -> float:
        """Maximum distance between any two living members."""
        pts = [positions[m] for m in self.members if m in positions]
        if len(pts) < 2:
            return 0.0
        max_dist = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = distance(pts[i], pts[j])
                if d > max_dist:
                    max_dist = d
        return max_dist

    def update_cohesion(self, positions: dict[str, Vec2]) -> None:
        """Recalculate cohesion from spread. Tighter = higher cohesion."""
        s = self.spread(positions)
        n = len([m for m in self.members if m in positions])
        if n < 2:
            self.state.cohesion = 1.0
            return
        # Ideal spread is roughly n * 3m; cohesion drops as spread exceeds that
        ideal = n * 3.0
        if s <= ideal:
            self.state.cohesion = 1.0
        else:
            self.state.cohesion = max(0.0, 1.0 - (s - ideal) / (ideal * 3.0))

    # -- Morale --------------------------------------------------------------

    def update_morale(self, unit_states: dict[str, float]) -> None:
        """Recompute average morale from individual unit morale values.

        Applies a penalty based on casualty ratio.
        """
        living = [m for m in self.members if m in unit_states]
        if not living:
            self.state.morale = 0.0
            return
        avg = sum(unit_states[m] for m in living) / len(living)
        total_ever = len(self.members) + self.state.casualties
        if total_ever > 0:
            casualty_ratio = self.state.casualties / total_ever
            avg *= max(0.0, 1.0 - casualty_ratio * 0.5)
        self.state.morale = max(0.0, min(1.0, avg))

    def should_retreat(self) -> bool:
        """Squad should retreat if morale < 0.3 or casualties > 50%."""
        total_ever = len(self.members) + self.state.casualties
        if total_ever > 0 and self.state.casualties / total_ever > 0.5:
            return True
        return self.state.morale < 0.3

    # -- Intel sharing -------------------------------------------------------

    def share_threat(self, threat_pos: Vec2, threat_id: str) -> None:
        """Propagate threat information to the whole squad."""
        # Replace existing entry for this threat_id, or add new
        self.state.known_threats = [
            (pos, tid)
            for pos, tid in self.state.known_threats
            if tid != threat_id
        ]
        self.state.known_threats.append((threat_pos, threat_id))
        # Raise alert level
        self.state.alert_level = min(1.0, self.state.alert_level + 0.2)


# ---------------------------------------------------------------------------
# SquadTactics — static/class methods for squad-level decisions
# ---------------------------------------------------------------------------


class SquadTactics:
    """Static methods for squad-level tactical computations."""

    @staticmethod
    def compute_formation(
        squad: Squad,
        positions: dict[str, Vec2],
        formation: str,
        spacing: float = 3.0,
    ) -> dict[str, Vec2]:
        """Compute target positions for each unit in the given formation.

        Formations: "line", "column", "wedge", "diamond", "circle".
        Positions are centered on the squad's center of mass.
        """
        living = [m for m in squad.members if m in positions]
        if not living:
            return {}

        com = squad.center_of_mass(positions)
        n = len(living)
        targets: dict[str, Vec2] = {}

        if formation == "line":
            # Spread along x-axis centered on COM
            for i, uid in enumerate(living):
                offset = (i - (n - 1) / 2.0) * spacing
                targets[uid] = (com[0] + offset, com[1])

        elif formation == "column":
            # Spread along y-axis centered on COM
            for i, uid in enumerate(living):
                offset = (i - (n - 1) / 2.0) * spacing
                targets[uid] = (com[0], com[1] + offset)

        elif formation == "wedge":
            # V-shape: leader at front, others fan back
            for i, uid in enumerate(living):
                if i == 0:
                    targets[uid] = (com[0], com[1] + spacing)
                else:
                    row = (i + 1) // 2
                    side = 1 if i % 2 == 1 else -1
                    targets[uid] = (
                        com[0] + side * row * spacing,
                        com[1] - row * spacing * 0.5,
                    )

        elif formation == "diamond":
            if n == 1:
                targets[living[0]] = com
            elif n == 2:
                targets[living[0]] = (com[0], com[1] + spacing)
                targets[living[1]] = (com[0], com[1] - spacing)
            elif n == 3:
                targets[living[0]] = (com[0], com[1] + spacing)
                targets[living[1]] = (com[0] - spacing, com[1])
                targets[living[2]] = (com[0] + spacing, com[1])
            else:
                # Front, left, right, back, then extras in the center
                targets[living[0]] = (com[0], com[1] + spacing)
                targets[living[1]] = (com[0] - spacing, com[1])
                targets[living[2]] = (com[0] + spacing, com[1])
                targets[living[3]] = (com[0], com[1] - spacing)
                for i in range(4, n):
                    angle = 2 * math.pi * (i - 4) / max(1, n - 4)
                    targets[living[i]] = (
                        com[0] + math.cos(angle) * spacing * 0.5,
                        com[1] + math.sin(angle) * spacing * 0.5,
                    )

        elif formation == "circle":
            for i, uid in enumerate(living):
                angle = 2 * math.pi * i / max(1, n)
                targets[uid] = (
                    com[0] + math.cos(angle) * spacing,
                    com[1] + math.sin(angle) * spacing,
                )

        else:
            # Unknown formation — just stay at current positions
            for uid in living:
                targets[uid] = positions[uid]

        return targets

    @staticmethod
    def assign_fire_sectors(
        squad: Squad,
        positions: dict[str, Vec2],
        threats: list[tuple[Vec2, str]],
    ) -> dict[str, Vec2]:
        """Assign each member a direction to watch/fire toward.

        Returns a dict of unit_id -> direction unit vector.
        If there are threats, members are distributed across threats.
        If no threats, members cover even radial sectors.
        """
        living = [m for m in squad.members if m in positions]
        if not living:
            return {}

        result: dict[str, Vec2] = {}

        if threats:
            # Round-robin assign threats to members
            for i, uid in enumerate(living):
                threat_pos, _ = threats[i % len(threats)]
                direction = _sub(threat_pos, positions[uid])
                d = magnitude(direction)
                if d < 1e-12:
                    result[uid] = (1.0, 0.0)
                else:
                    result[uid] = normalize(direction)
        else:
            # Even radial sectors
            for i, uid in enumerate(living):
                angle = 2 * math.pi * i / max(1, len(living))
                result[uid] = (math.cos(angle), math.sin(angle))

        return result

    @staticmethod
    def bounding_overwatch(
        squad: Squad,
        positions: dict[str, Vec2],
        direction: Vec2,
    ) -> tuple[list[str], list[str]]:
        """Split squad into 'moving' and 'covering' groups.

        The half closer to the direction of movement moves first,
        the other half provides cover. Returns (moving, covering).
        """
        living = [m for m in squad.members if m in positions]
        if not living:
            return [], []
        if len(living) == 1:
            return living[:], []

        d_norm = normalize(direction)
        com = squad.center_of_mass(positions)

        # Project each member's offset from COM onto the direction vector
        scored: list[tuple[float, str]] = []
        for uid in living:
            offset = _sub(positions[uid], com)
            proj = offset[0] * d_norm[0] + offset[1] * d_norm[1]
            scored.append((proj, uid))

        scored.sort(key=lambda x: x[0], reverse=True)
        half = len(scored) // 2
        if half == 0:
            half = 1

        moving = [uid for _, uid in scored[:half]]
        covering = [uid for _, uid in scored[half:]]
        return moving, covering

    @staticmethod
    def recommend_order(
        squad: Squad,
        positions: dict[str, Vec2],
        threats: list[tuple[Vec2, str]],
        terrain_info: dict | None = None,
    ) -> Order:
        """AI decision: recommend the best order based on current situation."""
        state = squad.state

        # Critical: retreat if broken
        if squad.should_retreat():
            return Order(order_type="retreat", priority=10)

        # Low ammo: fall back to cover fire / hold
        if state.ammo_status < 0.2:
            return Order(order_type="hold", priority=7)

        # No threats: patrol or hold
        if not threats and not state.known_threats:
            if state.alert_level < 0.3:
                return Order(order_type="patrol", priority=1)
            else:
                return Order(order_type="guard", priority=3)

        # Combine visible threats and known threats
        all_threats = list(threats) + list(state.known_threats)
        if not all_threats:
            return Order(order_type="hold", priority=2)

        com = squad.center_of_mass(positions)
        # Find nearest threat
        nearest_pos, nearest_id = min(
            all_threats, key=lambda t: distance(com, t[0])
        )
        nearest_dist = distance(com, nearest_pos)

        # Close range + good morale: advance / assault
        if nearest_dist < 20.0 and state.morale > 0.6:
            return Order(
                order_type="advance",
                target_pos=nearest_pos,
                target_id=nearest_id,
                priority=8,
            )

        # Medium range: suppress then flank
        if nearest_dist < 50.0:
            if state.morale > 0.5 and len(squad.members) >= 3:
                # Determine flank side
                to_threat = _sub(nearest_pos, com)
                # Perpendicular
                flank_dir = (-to_threat[1], to_threat[0])
                flank_target = _add(nearest_pos, _scale(normalize(flank_dir), 15.0))
                return Order(
                    order_type="flank_left",
                    target_pos=flank_target,
                    target_id=nearest_id,
                    priority=6,
                )
            else:
                return Order(
                    order_type="suppress",
                    target_pos=nearest_pos,
                    target_id=nearest_id,
                    priority=5,
                )

        # Long range: advance cautiously
        if state.morale > 0.4:
            return Order(
                order_type="advance",
                target_pos=nearest_pos,
                target_id=nearest_id,
                priority=4,
            )

        return Order(order_type="hold", priority=3)


# ---------------------------------------------------------------------------
# MoralePropagation
# ---------------------------------------------------------------------------


class MoralePropagation:
    """Propagate morale effects across squads over time."""

    @staticmethod
    def propagate(
        squads: list[Squad],
        unit_states: dict[str, float],
        dt: float,
    ) -> None:
        """Update morale across all squads for one time step.

        Effects:
        - Nearby friendly casualties reduce morale (handled via casualty ratio)
        - Suppression reduces morale (alert_level acts as proxy)
        - Rally near leader boosts morale recovery
        - Low cohesion reduces morale recovery rate
        """
        for squad in squads:
            living = [m for m in squad.members if m in unit_states]
            if not living:
                squad.state.morale = 0.0
                continue

            # Base morale from unit average
            avg_morale = sum(unit_states[m] for m in living) / len(living)

            # Casualty penalty
            total_ever = len(squad.members) + squad.state.casualties
            casualty_ratio = squad.state.casualties / total_ever if total_ever > 0 else 0.0
            casualty_penalty = casualty_ratio * 0.5

            # Suppression penalty (alert_level as proxy)
            suppression_penalty = squad.state.alert_level * 0.15

            # Cohesion modifier: low cohesion slows recovery
            cohesion_factor = 0.5 + 0.5 * squad.state.cohesion

            # Leader rally bonus: if leader alive, small morale boost
            leader_bonus = 0.0
            if squad.leader_id and squad.leader_id in unit_states:
                leader_bonus = 0.05 * cohesion_factor

            # Compute target morale
            target = avg_morale - casualty_penalty - suppression_penalty + leader_bonus
            target = max(0.0, min(1.0, target))

            # Lerp toward target morale, scaled by cohesion and dt
            recovery_rate = 0.3 * cohesion_factor * dt
            current = squad.state.morale
            if target > current:
                squad.state.morale = min(target, current + recovery_rate)
            else:
                # Drops are faster
                squad.state.morale = max(target, current - recovery_rate * 2.0)

            squad.state.morale = max(0.0, min(1.0, squad.state.morale))
