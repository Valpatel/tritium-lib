# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Rival-faction riots — two crowd factions fighting EACH OTHER (lane/riot tick 3).

Covers the full inter-faction combat contract:
  * faction identity on SimulationTarget + serialization,
  * the DiplomacyEngine as the single hostility oracle (UnitBehaviors.set_diplomacy),
  * rival-set construction (violent roles only; civilians/calmed/factionless excluded),
  * seeded red-vs-blue skirmish through the REAL combat pipeline
    (projectile_fired / projectile_hit, dispersed aim),
  * hard backward compat: factionless riots are engagement-free between rioters
    and byte-identical with or without a diplomacy handle wired,
  * police neutrality: the stand-in line engages BOTH factions and arrests work,
  * the Graphling occupancy boundary applies to factioned units.

Determinism follows the existing lib seeded-combat pattern: all combat
randomness flows through CombatSystem's injected ``random.Random(seed)``,
and PoliceTacticsController draws go through ``random.seed`` per test.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from tritium_lib.sim_engine.behavior.behaviors import UnitBehaviors
from tritium_lib.sim_engine.combat import CombatSystem
from tritium_lib.sim_engine.core.entity import SimulationTarget
from tritium_lib.sim_engine.factions import DiplomacyEngine, Faction
from tritium_lib.sim_engine.game.riot_police import PoliceTacticsController


# ---------------------------------------------------------------------------
# Stubs + builders (mirrors tests/sim_engine/test_riot_police.py)
# ---------------------------------------------------------------------------


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, topic: str, data: dict) -> None:
        self.events.append((topic, data))

    def topic(self, name: str) -> list[dict]:
        return [d for t, d in self.events if t == name]


@dataclass
class _FakeGameMode:
    de_escalation_score: int = 0
    arrest_count: int = 0
    rout_count: int = 0


def _officer(tid: str, pos: tuple[float, float]) -> SimulationTarget:
    t = SimulationTarget(
        target_id=tid, name=tid, alliance="friendly",
        asset_type="police", position=pos,
    )
    t.apply_combat_profile()
    return t


def _rioter(tid: str, pos: tuple[float, float],
            faction: str | None = None,
            route: list[tuple[float, float]] | None = None,
            **kw) -> SimulationTarget:
    # A crowd member always has a street route at spawn (like the real crowd
    # spawner) — without waypoints the entity tick parks it "idle" and the
    # behavior system's hostile filter (status == "active") drops it.
    t = SimulationTarget(
        target_id=tid, name=tid, alliance="hostile",
        asset_type="person", position=pos, crowd_role="rioter",
        faction=faction,
        waypoints=route if route is not None else [(pos[0], pos[1] + 60.0)],
    )
    t.apply_combat_profile()  # melee: range 3.0, cooldown 2.0, damage 3.0, hp 50
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _instigator(tid: str, pos: tuple[float, float],
                faction: str | None = None) -> SimulationTarget:
    t = SimulationTarget(
        target_id=tid, name=tid, alliance="hostile",
        asset_type="person", position=pos, crowd_role="instigator",
        faction=faction,
    )
    t.apply_combat_profile()
    return t


def _civilian(tid: str, pos: tuple[float, float],
              faction: str | None = None, **kw) -> SimulationTarget:
    t = SimulationTarget(
        target_id=tid, name=tid, alliance="hostile",
        asset_type="person", position=pos, crowd_role="civilian",
        faction=faction,
        waypoints=[(pos[0], pos[1] + 60.0)],
    )
    t.is_combatant = kw.pop("is_combatant", False)
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _as_dict(targets: list[SimulationTarget]) -> dict[str, SimulationTarget]:
    return {t.target_id: t for t in targets}


def _mk_diplomacy(at_war: bool = True) -> DiplomacyEngine:
    """Two street blocs; at war unless told otherwise (then NEUTRAL)."""
    diplo = DiplomacyEngine()
    diplo.add_faction(Faction(
        faction_id="red_bloc", name="Red Bloc", color="#ff2a6d",
        ideology="rebel", strength=0.5, wealth=0.2,
    ))
    diplo.add_faction(Faction(
        faction_id="blue_bloc", name="Blue Bloc", color="#00f0ff",
        ideology="rebel", strength=0.5, wealth=0.2,
    ))
    if at_war:
        diplo.declare_war("red_bloc", "blue_bloc")
    return diplo


# ---------------------------------------------------------------------------
# Skirmish runner: 4 red vs 4 blue rioters + protected/neutral bystanders,
# driven through the REAL entity/behavior/combat tick loop.
# ---------------------------------------------------------------------------


def _run_skirmish(
    seed: int,
    *,
    factioned: bool = True,
    wire_diplomacy: bool | None = None,
    at_war: bool = True,
    ticks: int = 150,
    dt: float = 0.1,
) -> tuple[_FakeBus, dict[str, SimulationTarget]]:
    wire = factioned if wire_diplomacy is None else wire_diplomacy
    bus = _FakeBus()
    combat = CombatSystem(event_bus=bus, rng=random.Random(seed))
    behaviors = UnitBehaviors(combat)
    if wire:
        behaviors.set_diplomacy(_mk_diplomacy(at_war=at_war))

    red_fac = "red_bloc" if factioned else None
    blue_fac = "blue_bloc" if factioned else None
    # Two lines 5 m apart: inside aggro (25 m), outside melee (3 m), so the
    # charge is real movement and the strikes are real combat.
    reds = [_rioter(f"red_{i}", (-2.5, (i - 1.5) * 1.2), faction=red_fac)
            for i in range(4)]
    blues = [_rioter(f"blue_{i}", (2.5, (i - 1.5) * 1.2), faction=blue_fac)
             for i in range(4)]
    # Factioned civilians NEAR the brawl-bound axis but out of melee reach:
    # protected by ROLE, not by faction or combatant flag.
    civs = [_civilian(f"civ_{i}", (0.0, 15.0 + 2.0 * i),
                      faction=red_fac, is_combatant=True) for i in range(2)]
    # Factionless bystander rioter: never attacks other rioters, never
    # becomes a rival target.  Strolls AWAY from the brawl.
    lone = _rioter("lone_0", (0.0, -8.0), faction=None,
                   route=[(0.0, -68.0)])

    targets = _as_dict(reds + blues + civs + [lone])
    for _ in range(ticks):
        for t in targets.values():
            t.tick(dt)
        behaviors.tick(dt, targets)
        combat.tick(dt, targets)
    return bus, targets


def _digest(bus: _FakeBus, targets: dict[str, SimulationTarget]):
    """Outcome fingerprint: positions/health/status + combat event counts."""
    return (
        tuple(sorted(
            (t.target_id,
             round(t.position[0], 9), round(t.position[1], 9),
             round(t.health, 9), t.status)
            for t in targets.values()
        )),
        len(bus.topic("projectile_fired")),
        len(bus.topic("projectile_hit")),
    )


def _cross_faction(events: list[dict]) -> tuple[list[dict], list[dict]]:
    red_on_blue = [e for e in events
                   if e["source_id"].startswith("red_")
                   and e["target_id"].startswith("blue_")]
    blue_on_red = [e for e in events
                   if e["source_id"].startswith("blue_")
                   and e["target_id"].startswith("red_")]
    return red_on_blue, blue_on_red


# ---------------------------------------------------------------------------
# Faction identity + serialization
# ---------------------------------------------------------------------------


def test_faction_field_default_none_and_serialized():
    plain = _rioter("r0", (0.0, 0.0))
    assert plain.faction is None
    assert plain.to_dict()["faction"] is None

    factioned = _rioter("red_0", (0.0, 0.0), faction="red_bloc")
    d = factioned.to_dict()
    assert d["faction"] == "red_bloc"
    assert d["crowd_role"] == "rioter"


# ---------------------------------------------------------------------------
# Rival-set construction: roles, factions, graceful degradation
# ---------------------------------------------------------------------------


def test_build_rival_crowd_role_and_faction_filters():
    behaviors = UnitBehaviors(CombatSystem(event_bus=_FakeBus()))

    red_r = _rioter("red_r", (0.0, 0.0), faction="red_bloc")
    red_i = _instigator("red_i", (1.0, 0.0), faction="red_bloc")
    red_c = _civilian("red_c", (2.0, 0.0), faction="red_bloc", is_combatant=True)
    red_calmed = _rioter("red_calmed", (3.0, 0.0), faction="red_bloc")
    red_calmed.crowd_role = "calmed"
    blue_r = _rioter("blue_r", (4.0, 0.0), faction="blue_bloc")
    lone = _rioter("lone", (5.0, 0.0), faction=None)
    hostiles = _as_dict([red_r, red_i, red_c, red_calmed, blue_r, lone])

    # No diplomacy oracle wired -> no rival sets at all.
    assert behaviors._build_rival_crowd(hostiles) == {}

    # Factions at peace -> still no rival sets.
    behaviors.set_diplomacy(_mk_diplomacy(at_war=False))
    assert behaviors._build_rival_crowd(hostiles) == {}

    # At war: only VIOLENT roles cross the line — a blue rioter may engage
    # the red rioter and red instigator, never the civilian / calmed /
    # factionless bystander.
    behaviors.set_diplomacy(_mk_diplomacy(at_war=True))
    rivals = behaviors._build_rival_crowd(hostiles)
    assert set(rivals["blue_bloc"].keys()) == {"red_r", "red_i"}
    assert set(rivals["red_bloc"].keys()) == {"blue_r"}
    assert "lone" not in rivals

    # Single faction on the field -> nothing to fight.
    solo = _as_dict([red_r, red_i, lone])
    assert behaviors._build_rival_crowd(solo) == {}


def test_factions_hostile_graceful_degradation():
    behaviors = UnitBehaviors(CombatSystem(event_bus=_FakeBus()))
    # No oracle
    assert behaviors._factions_hostile("red_bloc", "blue_bloc") is False
    behaviors.set_diplomacy(_mk_diplomacy(at_war=True))
    assert behaviors._factions_hostile("red_bloc", "blue_bloc") is True
    # Same faction is never self-hostile; missing factions never hostile.
    assert behaviors._factions_hostile("red_bloc", "red_bloc") is False
    assert behaviors._factions_hostile(None, "blue_bloc") is False
    assert behaviors._factions_hostile("red_bloc", None) is False

    # A raising oracle degrades to "not hostile", never crashes the tick.
    class _Broken:
        def are_hostile(self, a, b):
            raise RuntimeError("boom")

    behaviors.set_diplomacy(_Broken())
    assert behaviors._factions_hostile("red_bloc", "blue_bloc") is False


# ---------------------------------------------------------------------------
# (a) Seeded two-faction skirmish through the real combat pipeline
# ---------------------------------------------------------------------------


def test_two_faction_skirmish_engages_rivals():
    bus, targets = _run_skirmish(4242)

    fired = bus.topic("projectile_fired")
    red_on_blue, blue_on_red = _cross_faction(fired)
    assert red_on_blue, "red rioters must engage blue rioters"
    assert blue_on_red, "blue rioters must engage red rioters"

    # Real combat systems, not a parallel damage path: the standard melee
    # projectile with the standard dispersed-aim envelope.
    for e in red_on_blue + blue_on_red:
        assert e["projectile_type"] == "melee_strike"
        assert "aim_error_deg" in e

    # Hits actually landed both ways and drew blood.
    hits = bus.topic("projectile_hit")
    hit_rb, hit_br = _cross_faction(hits)
    assert hit_rb and hit_br, "cross-faction strikes must resolve to damage"
    assert any(targets[f"blue_{i}"].health < 50.0 for i in range(4))
    assert any(targets[f"red_{i}"].health < 50.0 for i in range(4))

    # ROE: civilians untouched regardless of the faction tag they carry.
    for i in range(2):
        civ = targets[f"civ_{i}"]
        assert civ.health == civ.max_health
        assert all(e["target_id"] != civ.target_id for e in fired)

    # Factionless bystander rioter: neither shooter nor target.
    assert all(e["source_id"] != "lone_0" for e in fired)
    assert all(e["target_id"] != "lone_0" for e in fired)


def test_factions_at_peace_do_not_engage():
    # Factions set + diplomacy wired, but relations NEUTRAL (no war declared):
    # rioters never fight each other.
    bus, _targets = _run_skirmish(4242, at_war=False)
    assert bus.topic("projectile_fired") == [], (
        "factions not at war must never fight each other"
    )


# ---------------------------------------------------------------------------
# Backward compat: factionless riot is bit-for-bit unchanged
# ---------------------------------------------------------------------------


def test_factionless_compat_zero_rioter_on_rioter():
    # Same geometry, no factions, no diplomacy: with no police on the field
    # there is nothing for a rioter to engage — zero shots, zero movement of
    # the combat clock beyond ticks.
    bus, targets = _run_skirmish(4242, factioned=False)
    assert bus.topic("projectile_fired") == []
    assert all(t.health == t.max_health for t in targets.values())


def test_wiring_diplomacy_alone_changes_nothing_for_factionless_crowd():
    # Diplomacy handle wired, war declared — but nobody carries a faction.
    # The digest must be identical to the fully-unwired run.
    plain = _digest(*_run_skirmish(99, factioned=False, wire_diplomacy=False))
    wired = _digest(*_run_skirmish(99, factioned=False, wire_diplomacy=True))
    assert plain == wired


# ---------------------------------------------------------------------------
# (b) Determinism: same seed twice => identical outcome digest
# ---------------------------------------------------------------------------


def test_determinism_same_seed_identical_digest():
    a = _digest(*_run_skirmish(777))
    b = _digest(*_run_skirmish(777))
    assert a == b, "seeded rival-faction skirmish diverged between runs"
    # And the digest is non-trivial: combat really happened.
    assert a[1] > 0 and a[2] > 0


# ---------------------------------------------------------------------------
# (c) Police in between: the line engages BOTH factions; arrests still work
# ---------------------------------------------------------------------------


def test_police_engage_both_factions_and_arrests_function():
    random.seed(4242)  # PoliceTacticsController module-global draws
    bus = _FakeBus()
    gm = _FakeGameMode()
    combat = CombatSystem(event_bus=bus, rng=random.Random(7))
    behaviors = UnitBehaviors(combat)
    behaviors.set_diplomacy(_mk_diplomacy(at_war=True))
    ctrl = PoliceTacticsController(bus, game_mode=gm)

    # Squad anchored slightly south of the brawl axis (non-degenerate facing);
    # red bloc west, blue bloc east, police between them.
    officers = [
        _officer("off_0", (-1.0, -4.0)), _officer("off_1", (-1.0, -2.0)),
        _officer("off_2", (1.0, -4.0)), _officer("off_3", (1.0, -2.0)),
    ]
    reds = [_rioter(f"red_{i}", (-6.0, (i - 1.5) * 1.0), faction="red_bloc")
            for i in range(4)]
    blues = [_rioter(f"blue_{i}", (6.0, (i - 1.5) * 1.0), faction="blue_bloc")
             for i in range(4)]
    targets = _as_dict(officers + reds + blues)

    for _ in range(60):
        for t in targets.values():
            t.tick(0.1)
        ctrl.tick(0.1, targets, "civil_unrest")
        behaviors.tick(0.1, targets)
        combat.tick(0.1, targets)

    # Containment runs against the COMBINED crowd (pin, don't redesign).
    assert ctrl.squad_state in ("advance", "engage")

    # Neutrality: officers fired on members of BOTH factions.
    police_shot_factions = {
        getattr(targets.get(e["target_id"]), "faction", None)
        for e in bus.topic("projectile_fired")
        if e["source_id"].startswith("off_")
    }
    assert "red_bloc" in police_shot_factions
    assert "blue_bloc" in police_shot_factions

    # Arrests still function against factioned rioters: overwhelm each bloc
    # with a 2-officer detain pair (>= 2 officers within arrest range).
    red_alive = [t for t in reds if t.status == "active" and t.crowd_role == "rioter"]
    blue_alive = [t for t in blues if t.status == "active" and t.crowd_role == "rioter"]
    assert red_alive and blue_alive, "melee phase must not wipe a faction"
    officers[0].position = officers[1].position = red_alive[0].position
    officers[2].position = officers[3].position = blue_alive[0].position
    for r in red_alive + blue_alive:
        r.health = 20.0
    ctrl.tick(0.1, targets, "civil_unrest")

    assert gm.arrest_count >= 2
    calmed_factions = {t.faction for t in targets.values()
                       if t.crowd_role == "calmed"}
    assert {"red_bloc", "blue_bloc"} <= calmed_factions, (
        "arrests must work against BOTH factions"
    )


# ---------------------------------------------------------------------------
# Graphling boundary: occupied factioned units are never puppeted
# ---------------------------------------------------------------------------


def test_occupied_factioned_rioter_is_never_puppeted():
    bus = _FakeBus()
    combat = CombatSystem(event_bus=bus, rng=random.Random(3))
    behaviors = UnitBehaviors(combat)
    behaviors.set_diplomacy(_mk_diplomacy(at_war=True))
    behaviors.set_occupancy_check(lambda tid: tid == "red_0")

    # Graphling on shift: red_0 walks its own way (route the "agent" chose).
    red0 = _rioter("red_0", (0.0, 0.0), faction="red_bloc",
                   route=[(0.0, -60.0)])
    red1 = _rioter("red_1", (0.0, 2.0), faction="red_bloc")   # stand-in
    blue0 = _rioter("blue_0", (2.0, 0.0), faction="blue_bloc")
    targets = _as_dict([red0, red1, blue0])

    for _ in range(30):
        for t in targets.values():
            t.tick(0.1)
        behaviors.tick(0.1, targets)
        combat.tick(0.1, targets)

    fired = bus.topic("projectile_fired")
    # The occupied unit's stand-in AI stays OFF: no shots, and its route is
    # never overwritten with a stand-in charge waypoint.
    assert all(e["source_id"] != "red_0" for e in fired)
    assert targets["red_0"].waypoints == [(0.0, -60.0)]
    # Free units on both sides still fight (occupied unit remains a valid
    # physical target for its enemies — occupancy is not a cloak).
    assert any(e["source_id"] == "red_1" for e in fired)
    assert any(e["source_id"] == "blue_0" for e in fired)
