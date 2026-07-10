# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for PoliceTacticsController — riot police squad stand-in AI.

Covers the crowd-control loop the civil_unrest mode is built around:
line/wedge formations, the hold->form->advance->engage FSM, non-lethal
arrests + routs, the Epstein grievance-feedback (agitation) arc, crowd-beat
transitions, the Graphling occupancy boundary, and golden-replay determinism.

Every random draw goes through the module-global RNG, so ``random.seed`` per
test keeps the seeded assertions reproducible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from tritium_lib.sim_engine.ai.formations import FormationType
from tritium_lib.sim_engine.behavior.behaviors import UnitBehaviors
from tritium_lib.sim_engine.combat import CombatSystem
from tritium_lib.sim_engine.core.entity import SimulationTarget
from tritium_lib.sim_engine.game.riot_police import (
    PoliceTacticsController,
    _INITIAL_AGITATION,
    _is_violent,
)


# ---------------------------------------------------------------------------
# Stubs + builders
# ---------------------------------------------------------------------------


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, topic: str, data: dict) -> None:
        self.events.append((topic, data))

    def topic(self, name: str) -> list[dict]:
        return [d for t, d in self.events if t == name]

    def beats(self, beat: str | None = None) -> list[dict]:
        out = self.topic("crowd_event")
        if beat is not None:
            out = [d for d in out if d.get("beat") == beat]
        return out


@dataclass
class _FakeGameMode:
    de_escalation_score: int = 0
    arrest_count: int = 0
    rout_count: int = 0


def _officer(tid: str, pos: tuple[float, float], **kw) -> SimulationTarget:
    t = SimulationTarget(
        target_id=tid, name=tid, alliance="friendly",
        asset_type="police", position=pos,
    )
    t.apply_combat_profile()
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _rioter(tid: str, pos: tuple[float, float], health: float = 50.0, **kw) -> SimulationTarget:
    t = SimulationTarget(
        target_id=tid, name=tid, alliance="hostile",
        asset_type="person", position=pos, crowd_role="rioter",
    )
    t.health = health
    t.is_combatant = True
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _civilian(tid: str, pos: tuple[float, float], **kw) -> SimulationTarget:
    t = SimulationTarget(
        target_id=tid, name=tid, alliance="hostile",
        asset_type="person", position=pos, crowd_role="civilian",
    )
    t.is_combatant = kw.pop("is_combatant", False)
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _as_dict(targets: list[SimulationTarget]) -> dict[str, SimulationTarget]:
    return {t.target_id: t for t in targets}


# ---------------------------------------------------------------------------
# (a) LINE formation perpendicular to the crowd axis + stable assignment
# ---------------------------------------------------------------------------


def test_line_slots_perpendicular_to_crowd_axis_and_stable():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus, formation_spacing=2.5)

    # Officers symmetric about origin -> squad centroid (0, 0).
    officers = [
        _officer("off_0", (-1.0, -1.0)),
        _officer("off_1", (1.0, -1.0)),
        _officer("off_2", (-1.0, 1.0)),
        _officer("off_3", (1.0, 1.0)),
    ]
    # 3 rioters (< wedge cluster) due east -> violent centroid (20, 0), facing +X.
    rioters = [
        _rioter("r0", (20.0, -2.0)),
        _rioter("r1", (20.0, 0.0)),
        _rioter("r2", (20.0, 2.0)),
    ]
    targets = _as_dict(officers + rioters)

    ctrl.tick(0.1, targets, "civil_unrest")

    assert ctrl.squad_state == "form"
    assert ctrl.formation_type == FormationType.LINE

    facing = (1.0, 0.0)  # anchor(0,0) -> violent centroid (20,0)
    ordered = sorted(officers, key=lambda o: o.target_id)
    slots = [o.waypoints[-1] for o in ordered]
    assert all(len(o.waypoints) == 1 for o in ordered), "every officer got exactly one slot"

    # The line runs perpendicular to the facing axis: the vector between any two
    # slots has (near) zero component along the facing direction.
    for i in range(len(slots) - 1):
        dx = slots[i + 1][0] - slots[i][0]
        dy = slots[i + 1][1] - slots[i][1]
        along = dx * facing[0] + dy * facing[1]
        assert abs(along) < 1e-6, "slots must be perpendicular to the crowd axis"

    # Stable assignment: officers sorted by id map to slots of monotonically
    # increasing lateral (y) coordinate.
    ys = [s[1] for s in slots]
    assert ys == sorted(ys)
    assert ys[0] < ys[-1]


# ---------------------------------------------------------------------------
# (b) WEDGE triggers on a tight cluster of >= 6 violent
# ---------------------------------------------------------------------------


def test_wedge_on_tight_cluster():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers = [_officer(f"off_{i}", (float(i), 0.0)) for i in range(6)]

    # 7 rioters packed within a few metres of (20, 0): a tight mass.
    rioters = [_rioter(f"r{i}", (20.0 + (i % 3) * 0.5, (i - 3) * 0.5)) for i in range(7)]
    targets = _as_dict(officers + rioters)

    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.formation_type == FormationType.WEDGE


def test_line_on_loose_crowd():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers = [_officer(f"off_{i}", (float(i), 0.0)) for i in range(6)]
    # Only 4 rioters (< 6) -> LINE regardless of tightness.
    rioters = [_rioter(f"r{i}", (20.0, (i - 2) * 1.0)) for i in range(4)]
    targets = _as_dict(officers + rioters)

    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.formation_type == FormationType.LINE


# ---------------------------------------------------------------------------
# (c) Full engagement loop: hold->form->advance->engage + arrests
# ---------------------------------------------------------------------------


def test_full_engagement_loop_states_then_arrests():
    random.seed(101)
    bus = _FakeBus()
    gm = _FakeGameMode()
    ctrl = PoliceTacticsController(bus, game_mode=gm)

    # 8 officers around (0,0), 12 rioters around (5,0): inside engage_range (12)
    # but outside arrest_range (4) so nothing is detained yet.
    officers = [_officer(f"off_{i}", (math.cos(i) * 0.5, math.sin(i) * 0.5)) for i in range(8)]
    rioters = [_rioter(f"r{i}", (5.0 + (i % 4) * 0.1, (i // 4 - 1) * 0.2), health=50.0)
               for i in range(12)]
    targets = _as_dict(officers + rioters)

    states: list[str] = []
    for _ in range(5):
        ctrl.tick(0.1, targets, "civil_unrest")
        states.append(ctrl.squad_state)

    assert states[0] == "form"
    assert "advance" in states and "engage" in states
    assert states.index("advance") < states.index("engage")
    assert gm.arrest_count == 0, "no arrests while officers are out of arrest range"

    # Wear the crowd down and close the distance: every officer on the mob.
    for o in officers:
        o.position = (5.0, 0.0)
    for r in rioters:
        r.health = 20.0

    ctrl.tick(0.1, targets, "civil_unrest")

    assert gm.arrest_count == 12
    assert gm.de_escalation_score == 12 * 25
    sample = targets["r0"]
    assert sample.crowd_role == "calmed"
    assert sample.alliance == "neutral"
    assert sample.is_combatant is False
    assert sample.identified is True
    assert sample.weapon_range == 0.0 and sample.weapon_damage == 0.0
    # arrest_made carries the arresting officer ids + position.
    arrests = bus.topic("arrest_made")
    assert arrests, "arrest_made event must be published"
    assert len(arrests[0]["officer_ids"]) >= 2
    assert "position" in arrests[0]


# ---------------------------------------------------------------------------
# (d) Rout path: weak, un-arrestable target flees away from the squad
# ---------------------------------------------------------------------------


def test_rout_flees_away_from_squad():
    bus = _FakeBus()
    gm = _FakeGameMode()
    ctrl = PoliceTacticsController(bus, game_mode=gm)

    # A single officer near a low-health rioter: only ONE officer in arrest range,
    # so the target cannot be arrested (needs >= 2) and is routed instead.
    officer = _officer("off_0", (8.0, 0.0))
    rioter = _rioter("r0", (12.0, 0.0), health=20.0)
    targets = _as_dict([officer, rioter])

    ctrl.tick(0.1, targets, "civil_unrest")

    assert gm.rout_count == 1
    assert gm.de_escalation_score == 10
    assert rioter.crowd_role == "civilian"
    assert rioter.is_combatant is False
    assert bus.topic("rioter_routed"), "rioter_routed event must be published"

    # Flee waypoint is ~30 m further out, directly away from the squad centroid.
    squad = (8.0, 0.0)
    flee = rioter.waypoints[-1]
    d_flee = math.hypot(flee[0] - rioter.position[0], flee[1] - rioter.position[1])
    assert d_flee == pytest.approx(30.0, abs=1e-3)
    # Away from squad: further from the squad than the rioter's own position.
    assert (math.hypot(flee[0] - squad[0], flee[1] - squad[1])
            > math.hypot(rioter.position[0] - squad[0], rioter.position[1] - squad[1]))
    assert flee[0] > rioter.position[0]  # squad is west, so flee east


# ---------------------------------------------------------------------------
# (e) ROE: _police_behavior never fires on civilians / calmed
# ---------------------------------------------------------------------------


def test_police_behavior_never_fires_on_civilian():
    behaviors = UnitBehaviors(CombatSystem(event_bus=_FakeBus()))
    officer = _officer("off_0", (0.0, 0.0))

    # Civilian is NEAREST (2 m), rioter is further (5 m). Even a combatant-flagged
    # civilian must be spared — the ROE guards on crowd_role, not is_combatant.
    civ = _civilian("civ0", (2.0, 0.0), is_combatant=True)
    rioter = _rioter("r0", (5.0, 0.0))
    hostiles = _as_dict([civ, rioter])

    behaviors._police_behavior(officer, hostiles)

    projectiles = list(behaviors._combat._projectiles.values())
    assert len(projectiles) == 1, "exactly one shot fired"
    assert projectiles[0].target_id == "r0", "police must fire at the rioter"
    assert all(p.target_id != "civ0" for p in projectiles), "never at the civilian"


def test_police_behavior_spares_calmed_target():
    behaviors = UnitBehaviors(CombatSystem(event_bus=_FakeBus()))
    officer = _officer("off_0", (0.0, 0.0))
    # A calmed (already-arrested) target nearest, an active rioter further.
    calmed = _rioter("c0", (2.0, 0.0))
    calmed.crowd_role = "calmed"
    rioter = _rioter("r0", (5.0, 0.0))
    hostiles = _as_dict([calmed, rioter])

    behaviors._police_behavior(officer, hostiles)
    projectiles = list(behaviors._combat._projectiles.values())
    assert len(projectiles) == 1
    assert projectiles[0].target_id == "r0"


# ---------------------------------------------------------------------------
# (f) Agitation: rises with shots, falls with arrests; radicalization gated
#     on engage and scaled by agitation
# ---------------------------------------------------------------------------


def test_agitation_rises_with_police_shots():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officer = _officer("off_0", (0.0, 0.0))
    targets = _as_dict([officer])

    ctrl.tick(0.1, targets, "civil_unrest")  # baseline last_fired recorded
    assert ctrl.agitation == pytest.approx(_INITIAL_AGITATION)

    officer.last_fired = 1.0  # a shot went off since last tick
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.agitation == pytest.approx(_INITIAL_AGITATION + 0.01)


def test_agitation_falls_with_arrest():
    bus = _FakeBus()
    gm = _FakeGameMode()
    ctrl = PoliceTacticsController(bus, game_mode=gm)
    officers = [_officer("off_0", (0.0, 0.0)), _officer("off_1", (0.5, 0.0))]
    rioter = _rioter("r0", (0.2, 0.0), health=20.0)
    targets = _as_dict(officers + [rioter])

    ctrl.tick(0.1, targets, "civil_unrest")
    assert gm.arrest_count == 1
    assert ctrl.agitation == pytest.approx(_INITIAL_AGITATION - 0.05)


def _radicalize_scene():
    """One officer, one live rioter in melee contact, one nearby civilian."""
    officer = _officer("off_0", (0.0, 0.0))
    rioter = _rioter("r0", (2.0, 0.0), health=50.0)   # in melee range, full HP
    civ = _civilian("civ0", (5.0, 0.0))               # within 10 m of contact
    return officer, rioter, civ


def test_radicalization_only_while_engaged():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officer, rioter, civ = _radicalize_scene()
    targets = _as_dict([officer, rioter, civ])

    with patch("tritium_lib.sim_engine.game.riot_police.random.random", return_value=0.0):
        ctrl.tick(0.1, targets, "civil_unrest")  # form
        assert ctrl.squad_state == "form"
        assert civ.crowd_role == "civilian", "no radicalization before engage"
        ctrl.tick(0.1, targets, "civil_unrest")  # advance
        assert ctrl.squad_state == "advance"
        assert civ.crowd_role == "civilian", "no radicalization before engage"
        ctrl.tick(0.1, targets, "civil_unrest")  # engage
        assert ctrl.squad_state == "engage"
        assert civ.crowd_role == "rioter", "radicalizes once engaged"


def test_radicalization_scales_with_agitation():
    # Zero agitation -> zero radicalization probability even with random()==0.
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    ctrl._agitation = 0.0
    officer, rioter, civ = _radicalize_scene()
    targets = _as_dict([officer, rioter, civ])

    with patch("tritium_lib.sim_engine.game.riot_police.random.random", return_value=0.0):
        for _ in range(4):
            ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.squad_state == "engage"
    assert civ.crowd_role == "civilian", "no radicalization when agitation is 0"


# ---------------------------------------------------------------------------
# (g) Beats published exactly once per transition
# ---------------------------------------------------------------------------


def test_crowd_beats_fire_once_per_transition():
    random.seed(202)
    bus = _FakeBus()
    gm = _FakeGameMode()
    ctrl = PoliceTacticsController(bus, game_mode=gm)

    officers = [_officer(f"off_{i}", (math.cos(i) * 0.5, math.sin(i) * 0.5)) for i in range(6)]
    rioters = [_rioter(f"r{i}", (5.0, (i - 3) * 0.4), health=50.0) for i in range(8)]
    targets = _as_dict(officers + rioters)

    # Phase A: drive to engagement (line + push established, peak violent = 8).
    for _ in range(4):
        ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.squad_state == "engage"

    # Phase B: overwhelm the mob — every officer on the crowd, all worn down.
    for o in officers:
        o.position = (5.0, 0.0)
    for r in rioters:
        r.health = 20.0
    ctrl.tick(0.1, targets, "civil_unrest")

    assert len(bus.beats("police_line")) == 1
    assert len(bus.beats("police_push")) == 1
    # 8 arrests -> arrest_surge at 3 and 6.
    surges = bus.beats("arrest_surge")
    assert [d["arrests"] for d in surges] == [3, 6]
    # Crowd broken exactly once when violent count crashes below 25% of peak(8).
    assert len(bus.beats("crowd_broken")) == 1
    assert bus.beats("crowd_broken")[0]["rioters"] == 0


# ---------------------------------------------------------------------------
# (h) Occupied officers (Graphling on shift) never get controller waypoints
# ---------------------------------------------------------------------------


def test_occupied_officer_never_assigned_waypoints():
    bus = _FakeBus()
    occupied_ids = {"off_A"}
    ctrl = PoliceTacticsController(
        bus, occupancy_check=lambda tid: tid in occupied_ids,
    )
    off_a = _officer("off_A", (0.0, 0.0))   # occupied (Graphling drives it)
    off_b = _officer("off_B", (1.0, 0.0))   # free (stand-in AI drives it)
    rioters = [_rioter(f"r{i}", (10.0, (i - 1) * 0.5)) for i in range(3)]
    targets = _as_dict([off_a, off_b] + rioters)

    ctrl.tick(0.1, targets, "civil_unrest")

    assert off_a.waypoints == [], "occupied officer must never be puppeted"
    assert off_b.waypoints, "free officer gets a formation slot"


# ---------------------------------------------------------------------------
# (i) Determinism: same seed + same setup -> identical outcomes
# ---------------------------------------------------------------------------


def _run_mixed_scenario(seed: int) -> tuple[int, int, float, int]:
    random.seed(seed)
    bus = _FakeBus()
    gm = _FakeGameMode()
    ctrl = PoliceTacticsController(bus, game_mode=gm)
    ctrl._agitation = 0.9  # high grievance so radicalization actually rolls

    officers = [_officer(f"off_{i}", (math.cos(i), math.sin(i))) for i in range(4)]
    rioters = [_rioter(f"r{i}", (3.0 + (i % 3) * 0.3, (i - 4) * 0.3), health=22.0)
               for i in range(8)]
    civs = [_civilian(f"civ{i}", (4.0, (i - 3) * 1.5)) for i in range(6)]
    targets = _as_dict(officers + rioters + civs)

    for _ in range(12):
        ctrl.tick(0.1, targets, "civil_unrest")

    radicalized = sum(1 for t in targets.values()
                      if t.target_id.startswith("civ") and t.crowd_role == "rioter")
    return gm.arrest_count, gm.rout_count, round(ctrl.agitation, 9), radicalized


def test_determinism_same_seed_same_outcome():
    a = _run_mixed_scenario(4242)
    b = _run_mixed_scenario(4242)
    assert a == b, f"seeded runs diverged: {a} != {b}"


# ---------------------------------------------------------------------------
# Guards: no-op outside civil_unrest; reset clears transitions
# ---------------------------------------------------------------------------


def test_noop_outside_civil_unrest():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers = [_officer("off_0", (0.0, 0.0))]
    rioters = [_rioter("r0", (5.0, 0.0))]
    targets = _as_dict(officers + rioters)

    ctrl.tick(0.1, targets, "battle")
    assert ctrl.squad_state == "hold"
    assert officers[0].waypoints == []
    assert not bus.events


def test_reset_clears_beats_and_state():
    bus = _FakeBus()
    gm = _FakeGameMode()
    ctrl = PoliceTacticsController(bus, game_mode=gm)
    officers = [_officer(f"off_{i}", (0.0, float(i))) for i in range(3)]
    rioters = [_rioter(f"r{i}", (5.0, float(i))) for i in range(3)]
    targets = _as_dict(officers + rioters)

    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.squad_state == "form"
    assert bus.beats("police_line")

    ctrl.reset()
    assert ctrl.squad_state == "hold"
    assert ctrl.agitation == pytest.approx(_INITIAL_AGITATION)
    assert ctrl._line_announced is False
    assert ctrl._arrest_total == 0
    assert ctrl._peak_violent == 0

    # A fresh line beat fires again after reset.
    bus.events.clear()
    ctrl.tick(0.1, targets, "civil_unrest")
    assert len(bus.beats("police_line")) == 1
