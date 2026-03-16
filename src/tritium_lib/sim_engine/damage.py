"""Damage and ballistics resolution for the Tritium sim engine.

Resolves combat interactions: hit rolls, damage calculations, armor,
criticals, headshots, explosions, burst fire, and stat tracking.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class DamageType(enum.Enum):
    KINETIC = "kinetic"
    EXPLOSIVE = "explosive"
    FIRE = "fire"
    ENERGY = "energy"
    MELEE = "melee"


@dataclass
class HitResult:
    """Outcome of a single attack resolution."""
    hit: bool
    damage: float
    damage_type: DamageType
    critical: bool = False
    headshot: bool = False
    armor_absorbed: float = 0.0
    suppression_caused: float = 0.0
    source_id: str = ""
    target_id: str = ""
    range_m: float = 0.0


# ---------------------------------------------------------------------------
# Core resolution
# ---------------------------------------------------------------------------

def _range_modifier(dist: float, falloff_start: float, falloff_end: float) -> float:
    """Return accuracy multiplier based on range.

    1.0 within *falloff_start*, linear decay to 0.1 at *falloff_end*,
    clamped to 0.1 beyond that.
    """
    if falloff_end <= falloff_start:
        return 1.0
    if dist <= falloff_start:
        return 1.0
    if dist >= falloff_end:
        return 0.1
    t = (dist - falloff_start) / (falloff_end - falloff_start)
    return 1.0 - 0.9 * t  # 1.0 -> 0.1


def resolve_attack(
    attacker_pos: Vec2,
    target_pos: Vec2,
    accuracy: float,
    damage: float,
    damage_type: DamageType,
    armor: float,
    range_falloff_start: float,
    range_falloff_end: float,
    critical_chance: float = 0.05,
    headshot_chance: float = 0.02,
    rng: Optional[random.Random] = None,
    source_id: str = "",
    target_id: str = "",
) -> HitResult:
    """Resolve a single attack and return the result."""
    r = rng or random.Random()
    dist = distance(attacker_pos, target_pos)
    mod = _range_modifier(dist, range_falloff_start, range_falloff_end)
    hit_chance = max(0.0, min(1.0, accuracy * mod))

    hit = r.random() < hit_chance

    # Suppression happens even on a miss (near misses)
    suppression = 0.1 + 0.3 * (damage / 50.0)

    if not hit:
        return HitResult(
            hit=False,
            damage=0.0,
            damage_type=damage_type,
            suppression_caused=suppression,
            source_id=source_id,
            target_id=target_id,
            range_m=dist,
        )

    is_critical = r.random() < critical_chance
    is_headshot = r.random() < headshot_chance

    effective_damage = damage
    if is_critical:
        effective_damage *= 2.0
    if is_headshot:
        effective_damage *= 3.0

    armor_absorbed = 0.0
    if is_headshot:
        # Headshots bypass armor entirely
        pass
    else:
        armor_clamped = max(0.0, min(1.0, armor))
        armor_absorbed = effective_damage * armor_clamped
        effective_damage -= armor_absorbed

    effective_damage = max(0.0, effective_damage)

    return HitResult(
        hit=True,
        damage=effective_damage,
        damage_type=damage_type,
        critical=is_critical,
        headshot=is_headshot,
        armor_absorbed=armor_absorbed,
        suppression_caused=suppression,
        source_id=source_id,
        target_id=target_id,
        range_m=dist,
    )


# ---------------------------------------------------------------------------
# Explosion resolution
# ---------------------------------------------------------------------------

def resolve_explosion(
    center: Vec2,
    radius: float,
    targets: list[tuple[Vec2, str]],
    base_damage: float,
    damage_falloff: str = "linear",
) -> list[HitResult]:
    """Resolve an explosion affecting all targets within *radius*.

    *targets* is a list of ``(position, target_id)`` tuples.
    *damage_falloff* can be ``"linear"`` or ``"quadratic"``.
    """
    results: list[HitResult] = []
    for pos, tid in targets:
        dist = distance(center, pos)
        if dist > radius:
            continue

        if radius <= 0:
            ratio = 0.0
        else:
            ratio = dist / radius

        if damage_falloff == "quadratic":
            dmg = base_damage * (1.0 - ratio * ratio)
        else:
            dmg = base_damage * (1.0 - ratio)

        dmg = max(0.0, dmg)
        suppression = 0.1 + 0.3 * (base_damage / 50.0)

        results.append(HitResult(
            hit=True,
            damage=dmg,
            damage_type=DamageType.EXPLOSIVE,
            suppression_caused=suppression,
            target_id=tid,
            range_m=dist,
        ))

    return results


# ---------------------------------------------------------------------------
# Burst fire resolution
# ---------------------------------------------------------------------------

def resolve_burst(
    attacker_pos: Vec2,
    target_pos: Vec2,
    rounds: int,
    accuracy: float,
    damage_per_round: float,
    spread_deg: float,
    damage_type: DamageType,
    armor: float,
    rng: Optional[random.Random] = None,
    source_id: str = "",
    target_id: str = "",
) -> list[HitResult]:
    """Fire *rounds* rounds with per-round spread applied to accuracy.

    Each round's accuracy is reduced by a random factor drawn from
    ``[0, spread_deg / 90]``.
    """
    r = rng or random.Random()
    dist = distance(attacker_pos, target_pos)
    results: list[HitResult] = []

    for _ in range(rounds):
        spread_penalty = r.uniform(0.0, spread_deg / 90.0)
        adj_accuracy = max(0.0, accuracy - spread_penalty)

        result = resolve_attack(
            attacker_pos=attacker_pos,
            target_pos=target_pos,
            accuracy=adj_accuracy,
            damage=damage_per_round,
            damage_type=damage_type,
            armor=armor,
            range_falloff_start=dist + 1.0,  # no range penalty within burst
            range_falloff_end=dist + 2.0,
            rng=r,
            source_id=source_id,
            target_id=target_id,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Damage tracker
# ---------------------------------------------------------------------------

class DamageTracker:
    """Accumulates combat statistics across an engagement."""

    def __init__(self) -> None:
        self._records: list[HitResult] = []

    def record(self, result: HitResult) -> None:
        """Store a hit result."""
        self._records.append(result)

    def record_many(self, results: list[HitResult]) -> None:
        """Store multiple hit results."""
        self._records.extend(results)

    # -- per-unit queries ---------------------------------------------------

    def total_damage_dealt(self, unit_id: str) -> float:
        """Total damage dealt *by* unit_id."""
        return sum(r.damage for r in self._records if r.source_id == unit_id and r.hit)

    def total_damage_taken(self, unit_id: str) -> float:
        """Total damage taken *by* unit_id."""
        return sum(r.damage for r in self._records if r.target_id == unit_id and r.hit)

    def kill_count(self, unit_id: str) -> int:
        """Number of lethal hits (damage >= 100) dealt by unit_id."""
        return sum(
            1 for r in self._records
            if r.source_id == unit_id and r.hit and r.damage >= 100.0
        )

    def accuracy_rate(self, unit_id: str) -> float:
        """Fraction of attempts that hit for unit_id (as attacker)."""
        attempts = [r for r in self._records if r.source_id == unit_id]
        if not attempts:
            return 0.0
        hits = sum(1 for r in attempts if r.hit)
        return hits / len(attempts)

    # -- global queries -----------------------------------------------------

    def mvp(self) -> str:
        """Unit ID with the highest total damage dealt. Empty string if none."""
        totals: dict[str, float] = {}
        for r in self._records:
            if r.source_id and r.hit:
                totals[r.source_id] = totals.get(r.source_id, 0.0) + r.damage
        if not totals:
            return ""
        return max(totals, key=totals.get)  # type: ignore[arg-type]

    def summary(self) -> dict:
        """Full combat statistics summary."""
        all_sources = {r.source_id for r in self._records if r.source_id}
        all_targets = {r.target_id for r in self._records if r.target_id}
        all_units = all_sources | all_targets

        total_hits = sum(1 for r in self._records if r.hit)
        total_misses = sum(1 for r in self._records if not r.hit)
        total_damage = sum(r.damage for r in self._records if r.hit)
        total_crits = sum(1 for r in self._records if r.critical)
        total_headshots = sum(1 for r in self._records if r.headshot)

        per_unit: dict[str, dict] = {}
        for uid in all_units:
            per_unit[uid] = {
                "damage_dealt": self.total_damage_dealt(uid),
                "damage_taken": self.total_damage_taken(uid),
                "kills": self.kill_count(uid),
                "accuracy": self.accuracy_rate(uid),
            }

        return {
            "total_attacks": len(self._records),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_damage": total_damage,
            "total_criticals": total_crits,
            "total_headshots": total_headshots,
            "mvp": self.mvp(),
            "per_unit": per_unit,
        }
