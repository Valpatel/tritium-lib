# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the operator command path, kettle cordon, and focus-fire spread.

lane/riot tick 2 additions to the riot-police stand-in AI:

  * ``command_tactic`` — the production squad-lead interface (auto / line /
    wedge / kettle), its ``police_tactic_commanded`` event, forced-formation
    overrides, and the exact ``get_status`` API contract the SC layer binds to.
  * The **kettle** cordon — ARC formation around the local cluster, the
    ``kettle_formed`` / ``corridor_flow`` beats, corridor drive out the gap,
    arrests inside the cordon, and clean exit / reset.
  * Focus-fire spread in ``_police_behavior`` — a line of officers spreads fire
    across the front instead of the whole squad volleying one rioter.

Stand-in video-game AI throughout; never Graphling cognition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from tritium_lib.sim_engine.ai.formations import FormationType
from tritium_lib.sim_engine.behavior.behaviors import UnitBehaviors
from tritium_lib.sim_engine.combat import CombatSystem
from tritium_lib.sim_engine.core.entity import SimulationTarget
from tritium_lib.sim_engine.game.riot_police import (
    PoliceTacticsController,
    _INITIAL_AGITATION,
)


# ---------------------------------------------------------------------------
# Stubs + builders (mirror test_riot_police.py, kept self-contained)
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
    t.status = "active"
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
    t.status = "active"
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def _as_dict(targets: list[SimulationTarget]) -> dict[str, SimulationTarget]:
    return {t.target_id: t for t in targets}


def _ring(prefix: str, n: int, radius: float, builder, **kw) -> list:
    return [
        builder(f"{prefix}{i}",
                (radius * math.cos(i * 1.3), radius * math.sin(i * 1.3)), **kw)
        for i in range(n)
    ]


# ===========================================================================
# command_tactic contract
# ===========================================================================


def test_command_tactic_invalid_returns_false_no_change():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    assert ctrl.command_tactic("banana") is False
    assert ctrl.commanded_tactic == "auto"
    assert ctrl.corridor is None
    assert not bus.topic("police_tactic_commanded")


def test_command_tactic_valid_publishes_once():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    assert ctrl.command_tactic("line") is True
    assert ctrl.commanded_tactic == "line"
    evts = bus.topic("police_tactic_commanded")
    assert len(evts) == 1
    assert evts[0] == {"tactic": "line", "corridor": None, "faction": None}


def test_command_tactic_kettle_carries_corridor():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    assert ctrl.command_tactic("kettle", corridor=(30.0, 10.0)) is True
    assert ctrl.corridor == (30.0, 10.0)
    evts = bus.topic("police_tactic_commanded")
    assert evts[-1]["tactic"] == "kettle"
    assert evts[-1]["corridor"] == {"x": 30.0, "y": 10.0}


def test_get_status_exact_key_contract():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    status = ctrl.get_status()
    assert set(status.keys()) == {
        "squad_state", "formation_type", "commanded_tactic",
        "agitation", "corridor", "target_faction", "arrests",
    }
    assert status["squad_state"] == "hold"
    assert status["formation_type"] is None
    assert status["commanded_tactic"] == "auto"
    assert status["agitation"] == pytest.approx(_INITIAL_AGITATION)
    assert status["corridor"] is None
    assert status["target_faction"] is None
    assert status["arrests"] == 0


def test_get_status_reflects_live_formation_string():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers = [_officer(f"off_{i}", (float(i), 0.0)) for i in range(4)]
    rioters = [_rioter(f"r{i}", (20.0, (i - 1) * 1.0)) for i in range(3)]
    targets = _as_dict(officers + rioters)
    ctrl.tick(0.1, targets, "civil_unrest")
    status = ctrl.get_status()
    assert status["formation_type"] == "line"  # string value, not the enum
    assert status["squad_state"] == "form"


# ===========================================================================
# Forced formation overrides
# ===========================================================================


def test_forced_line_beats_auto_wedge():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    ctrl.command_tactic("line")
    officers = [_officer(f"off_{i}", (float(i), 0.0)) for i in range(6)]
    # A tight mass of 7 would auto-WEDGE — the command forces LINE.
    rioters = [_rioter(f"r{i}", (20.0 + (i % 3) * 0.5, (i - 3) * 0.5)) for i in range(7)]
    targets = _as_dict(officers + rioters)
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.formation_type == FormationType.LINE


def test_forced_wedge_beats_auto_line():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    ctrl.command_tactic("wedge")
    officers = [_officer(f"off_{i}", (float(i), 0.0)) for i in range(6)]
    # Only 2 rioters would auto-LINE — the command forces WEDGE.
    rioters = [_rioter(f"r{i}", (20.0, (i - 0.5) * 1.0)) for i in range(2)]
    targets = _as_dict(officers + rioters)
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.formation_type == FormationType.WEDGE


# ===========================================================================
# Kettle cordon
# ===========================================================================


def _form_kettle(ctrl, officers, targets):
    """Drive a kettle to the "formed" state: plan slots, snap officers on."""
    ctrl.tick(0.1, targets, "civil_unrest")  # tick 1: plan ARC slots
    for o in sorted(officers, key=lambda x: x.target_id):
        if o.waypoints:
            o.position = o.waypoints[-1]
    ctrl.tick(0.1, targets, "civil_unrest")  # tick 2: formed + corridor drive


def test_kettle_officers_get_arc_slots():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers = _ring("off_", 8, 0.5, _officer)
    rioters = _ring("r", 5, 0.4, _rioter, health=50.0)
    targets = _as_dict(officers + rioters)

    ctrl.command_tactic("kettle")
    ctrl.tick(0.1, targets, "civil_unrest")

    assert ctrl.squad_state == "kettle"
    assert ctrl.formation_type == FormationType.ARC
    center = ctrl._kettle_center
    radius = ctrl._kettle_radius
    assert radius == pytest.approx(10.0)  # loose cluster -> the 10 m floor
    for o in officers:
        assert len(o.waypoints) == 1
        d = math.hypot(o.waypoints[-1][0] - center[0], o.waypoints[-1][1] - center[1])
        assert d == pytest.approx(radius, abs=1e-6)


def test_kettle_formed_beat_fires_once():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers = _ring("off_", 8, 0.5, _officer)
    rioters = _ring("r", 5, 0.4, _rioter, health=50.0)
    targets = _as_dict(officers + rioters)

    ctrl.command_tactic("kettle")
    # Before officers reach the ring: no kettle_formed.
    ctrl.tick(0.1, targets, "civil_unrest")
    assert not bus.beats("kettle_formed")

    # Snap them on and tick: >= 75% on-slot -> kettle_formed once.
    for o in sorted(officers, key=lambda x: x.target_id):
        o.position = o.waypoints[-1]
    ctrl.tick(0.1, targets, "civil_unrest")
    assert len(bus.beats("kettle_formed")) == 1

    # Held in the cordon: the beat does not re-fire.
    ctrl.tick(0.1, targets, "civil_unrest")
    assert len(bus.beats("kettle_formed")) == 1


def test_kettle_corridor_push_out_the_gap():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers = _ring("off_", 8, 0.5, _officer)
    rioters = _ring("r", 5, 0.4, _rioter, health=50.0)
    targets = _as_dict(officers + rioters)

    ctrl.command_tactic("kettle", corridor=(100.0, 0.0))  # gap toward +X
    _form_kettle(ctrl, officers, targets)

    center = ctrl._kettle_center
    gap_dir = ctrl._kettle_gap_dir
    ring = ctrl._kettle_radius
    for r in rioters:
        assert len(r.waypoints) == 1, "still-violent rioter shoved out the gap"
        wp = r.waypoints[-1]
        vx, vy = wp[0] - center[0], wp[1] - center[1]
        # On the gap side of the centre and past the cordon ring.
        assert vx * gap_dir[0] + vy * gap_dir[1] > 0
        assert math.hypot(vx, vy) > ring
    # >= 3 distinct rioters pushed -> corridor_flow once.
    assert len(bus.beats("corridor_flow")) == 1


def test_kettle_corridor_flow_requires_three_pushes():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers = _ring("off_", 8, 0.5, _officer)
    rioters = _ring("r", 2, 0.4, _rioter, health=50.0)  # only two to push
    targets = _as_dict(officers + rioters)

    ctrl.command_tactic("kettle", corridor=(100.0, 0.0))
    _form_kettle(ctrl, officers, targets)

    assert len(bus.beats("kettle_formed")) == 1
    assert len(ctrl._corridor_pushed) == 2
    assert not bus.beats("corridor_flow"), "fewer than 3 pushes -> no flow beat"


def test_arrests_land_while_kettled():
    bus = _FakeBus()
    gm = _FakeGameMode()
    ctrl = PoliceTacticsController(bus, game_mode=gm)
    # Two officers right on a worn-down rioter; kettle commanded.
    officers = [_officer("off_0", (0.0, 0.0)), _officer("off_1", (0.5, 0.0))]
    rioter = _rioter("r0", (0.2, 0.0), health=20.0)
    targets = _as_dict(officers + [rioter])

    ctrl.command_tactic("kettle")
    ctrl.tick(0.1, targets, "civil_unrest")

    assert ctrl.squad_state == "kettle"
    assert gm.arrest_count == 1
    assert rioter.crowd_role == "calmed"
    assert rioter.alliance == "neutral"
    assert ctrl.get_status()["arrests"] == 1
    assert bus.topic("arrest_made")


def test_kettle_auto_gap_faces_away_from_squad():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    # Squad due west of a cluster at the origin: auto gap should open east.
    officers = [_officer(f"off_{i}", (-20.0, (i - 3.5) * 0.5)) for i in range(8)]
    rioters = _ring("r", 5, 0.4, _rioter, health=50.0)
    targets = _as_dict(officers + rioters)

    ctrl.command_tactic("kettle")  # no corridor -> auto
    ctrl.tick(0.1, targets, "civil_unrest")

    gap_dir = ctrl._kettle_gap_dir
    # Away from the squad (which is to the west) => gap points roughly east.
    assert gap_dir[0] > 0.9


def test_kettle_exit_resumes_form_state():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers = _ring("off_", 6, 0.5, _officer)
    rioters = _ring("r", 5, 0.4, _rioter, health=50.0)
    targets = _as_dict(officers + rioters)

    ctrl.command_tactic("kettle")
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.squad_state == "kettle"

    ctrl.command_tactic("auto")
    assert ctrl._kettle_gap_dir is None  # cordon transient state cleared
    assert ctrl._corridor_pushed == {}
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.squad_state == "form"


def test_reset_clears_tactic_and_kettle_state():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers = _ring("off_", 6, 0.5, _officer)
    rioters = _ring("r", 5, 0.4, _rioter, health=50.0)
    targets = _as_dict(officers + rioters)

    ctrl.command_tactic("kettle", corridor=(50.0, 0.0))
    _form_kettle(ctrl, officers, targets)
    assert ctrl._corridor_pushed  # cordon was active

    ctrl.reset()
    assert ctrl.commanded_tactic == "auto"
    assert ctrl.corridor is None
    assert ctrl.squad_state == "hold"
    assert ctrl._kettle_gap_dir is None
    assert ctrl._corridor_pushed == {}
    assert ctrl._sim_clock == 0.0
    status = ctrl.get_status()
    assert status["commanded_tactic"] == "auto"
    assert status["corridor"] is None


def test_remove_unit_clears_corridor_throttle():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    ctrl._corridor_pushed["r0"] = 5.0
    ctrl.remove_unit("r0")
    assert "r0" not in ctrl._corridor_pushed


# ===========================================================================
# Faction-aware kettle — cordon ONE bloc, leave the rival be
# ===========================================================================
#
# The three-way headline: with two rival blocs on the street, an operator can
# command "kettle the RED bloc" and the cordon forms around the RED cluster
# only.  The CYAN bloc — massed elsewhere — is never cordoned, never pushed
# out the corridor, and never arrested by this command.  Deterministic
# geometry (no RNG in the kettle path): red bloc massed in the WEST, cyan bloc
# massed in the EAST, officers south of the red cluster.


def _two_bloc_field():
    """8 officers + a 4-strong RED bloc (west) + a 4-strong CYAN bloc (east)."""
    officers = [_officer(f"off_{i}", (-40.0 + (i - 3.5) * 0.6, -13.0))
                for i in range(8)]
    reds = [_rioter(f"red_{i}", (-40.0 + math.cos(i * 1.3) * 1.5,
                                 math.sin(i * 1.3) * 1.5), faction="red_bloc")
            for i in range(4)]
    cyans = [_rioter(f"cyan_{i}", (40.0 + math.cos(i * 1.3) * 1.5,
                                   math.sin(i * 1.3) * 1.5), faction="cyan_bloc")
             for i in range(4)]
    return officers, reds, cyans


def test_command_tactic_kettle_carries_faction():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    assert ctrl.command_tactic("kettle", faction="red_bloc") is True
    assert ctrl.target_faction == "red_bloc"
    assert ctrl.get_status()["target_faction"] == "red_bloc"
    evt = bus.topic("police_tactic_commanded")[-1]
    assert evt["tactic"] == "kettle"
    assert evt["faction"] == "red_bloc"


def test_faction_target_cleared_by_non_kettle_command():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    ctrl.command_tactic("kettle", faction="red_bloc")
    assert ctrl.target_faction == "red_bloc"
    # Any non-kettle tactic drops the faction scope.
    ctrl.command_tactic("line")
    assert ctrl.target_faction is None
    assert ctrl.get_status()["target_faction"] is None


def test_faction_target_ignored_for_non_kettle_tactic():
    # A faction handed to line/wedge/auto is meaningless and dropped.
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    ctrl.command_tactic("line", faction="red_bloc")
    assert ctrl.target_faction is None


def test_faction_kettle_cordons_targeted_bloc_not_the_rival():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers, reds, cyans = _two_bloc_field()
    targets = _as_dict(officers + reds + cyans)

    ctrl.command_tactic("kettle", faction="red_bloc")
    ctrl.tick(0.1, targets, "civil_unrest")

    assert ctrl.squad_state == "kettle"
    assert ctrl.formation_type == FormationType.ARC
    center = ctrl._kettle_center
    radius = ctrl._kettle_radius
    # The cordon centres on the RED cluster (west, x ~ -40), NOT the cyan
    # cluster (east, x ~ +40): the operator kettled the bloc they named.
    assert center[0] < -30.0, f"kettle centred at {center}, not on the red bloc"

    # Officers ring the RED cluster centre.
    for o in officers:
        d = math.hypot(o.waypoints[-1][0] - center[0],
                       o.waypoints[-1][1] - center[1])
        assert d == pytest.approx(radius, abs=1e-6)

    # The CYAN bloc is nowhere near the cordon — every cyan is far outside the
    # ring, so it is structurally NOT contained by this command.
    for c in cyans:
        d = math.hypot(c.position[0] - center[0], c.position[1] - center[1])
        assert d > radius * 3, "the untargeted cyan bloc must not be cordoned"


def test_faction_kettle_corridor_pushes_only_targeted_bloc():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers, reds, cyans = _two_bloc_field()
    targets = _as_dict(officers + reds + cyans)

    ctrl.command_tactic("kettle", faction="red_bloc")
    # Snap officers onto their ARC slots and re-tick so the cordon is "formed"
    # and the corridor drive runs.
    ctrl.tick(0.1, targets, "civil_unrest")
    for o in sorted(officers, key=lambda x: x.target_id):
        if o.waypoints:
            o.position = o.waypoints[-1]
    ctrl.tick(0.1, targets, "civil_unrest")

    # Only RED members were ever handed a corridor-exit waypoint; the CYAN
    # bloc's throttle map stays empty — the rival is never pushed.
    pushed = set(ctrl._corridor_pushed.keys())
    assert pushed, "the targeted bloc should be driven out the corridor"
    assert all(tid.startswith("red_") for tid in pushed), pushed
    assert not any(tid.startswith("cyan_") for tid in pushed)


def test_faction_kettle_arrests_only_targeted_bloc():
    bus = _FakeBus()
    gm = _FakeGameMode()
    ctrl = PoliceTacticsController(bus, game_mode=gm)
    officers, reds, cyans = _two_bloc_field()
    targets = _as_dict(officers + reds + cyans)

    # Wear BOTH blocs down below arrest health, and overwhelm ONE red and ONE
    # cyan with a 2-officer detain pair each.  Only the RED arrest may land —
    # the command is scoped to the red bloc, so cyan is never a candidate.
    for r in reds + cyans:
        r.health = 20.0
    officers[0].position = officers[1].position = reds[0].position
    officers[2].position = officers[3].position = cyans[0].position

    ctrl.command_tactic("kettle", faction="red_bloc")
    ctrl.tick(0.1, targets, "civil_unrest")

    assert reds[0].crowd_role == "calmed", "the targeted red bloc member arrested"
    assert cyans[0].crowd_role == "rioter", "the untargeted cyan bloc untouched"
    calmed_factions = {t.faction for t in targets.values()
                       if t.crowd_role == "calmed"}
    assert calmed_factions == {"red_bloc"}, calmed_factions


def test_faction_kettle_disbands_when_target_bloc_cleared():
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers, reds, cyans = _two_bloc_field()
    targets = _as_dict(officers + reds + cyans)

    ctrl.command_tactic("kettle", faction="red_bloc")
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.squad_state == "kettle"

    # The whole red bloc is calmed (contained) — but cyan still riots in the
    # east.  The command was to kettle RED, so the squad now stands easy; it
    # does not silently re-target the rival bloc.
    for r in reds:
        r.crowd_role = "calmed"
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.squad_state == "hold"
    assert ctrl._kettle_gap_dir is None  # cordon transient state cleared


def test_legacy_kettle_without_faction_takes_nearest_bloc():
    # No faction => the cordon forms on whatever violent cluster is nearest,
    # regardless of bloc (backward-compatible with the pre-faction kettle).
    bus = _FakeBus()
    ctrl = PoliceTacticsController(bus)
    officers, reds, cyans = _two_bloc_field()  # officers sit by the RED cluster
    targets = _as_dict(officers + reds + cyans)

    ctrl.command_tactic("kettle")  # no faction
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.target_faction is None
    # Nearest cluster to the squad is red (west) — the legacy kettle centres
    # there, but purely by distance, not by faction scoping.
    assert ctrl._kettle_center[0] < -30.0


# ===========================================================================
# Focus-fire spread (behaviors._police_behavior)
# ===========================================================================


def test_focus_fire_spreads_across_the_front():
    """4 officers + 2 in-range rioters: fire spreads, not all on one rioter."""
    behaviors = UnitBehaviors(CombatSystem(event_bus=_FakeBus()))
    officers = [
        _officer("off_0", (0.0, 0.0)),
        _officer("off_1", (0.0, 1.0)),
        _officer("off_2", (1.0, 0.0)),
        _officer("off_3", (1.0, 1.0)),
    ]
    officer_ids = {o.target_id for o in officers}
    # Both rioters within the 8 m pepper-ball range of every officer; r0 nearer.
    rioters = [_rioter("r0", (4.0, 0.0)), _rioter("r1", (5.0, 0.0))]
    targets = _as_dict(officers + rioters)

    behaviors.tick(0.1, targets)

    officer_shots = [
        p.target_id for p in behaviors._combat._projectiles.values()
        if p.source_id in officer_ids
    ]
    assert len(officer_shots) == 4, "every officer fired once"
    # Both rioters drew fire — the line did not volley a single target.
    assert set(officer_shots) == {"r0", "r1"}
    # No rioter soaked the whole squad.
    assert officer_shots.count("r0") < 4
    assert officer_shots.count("r1") < 4


def test_focus_fire_single_target_still_engaged():
    """One in-range rioter: all officers still (correctly) fire at it."""
    behaviors = UnitBehaviors(CombatSystem(event_bus=_FakeBus()))
    officers = [_officer(f"off_{i}", (float(i) * 0.3, 0.0)) for i in range(4)]
    officer_ids = {o.target_id for o in officers}
    rioter = _rioter("r0", (3.0, 0.0))
    targets = _as_dict(officers + [rioter])

    behaviors.tick(0.1, targets)

    officer_shots = [
        p.target_id for p in behaviors._combat._projectiles.values()
        if p.source_id in officer_ids
    ]
    assert officer_shots == ["r0"] * 4


def test_focus_fire_respects_hard_roe():
    """Focus-fire never spreads onto a protected civilian / calmed target."""
    behaviors = UnitBehaviors(CombatSystem(event_bus=_FakeBus()))
    officers = [_officer(f"off_{i}", (float(i) * 0.3, 0.0)) for i in range(3)]
    officer_ids = {o.target_id for o in officers}
    rioter = _rioter("r0", (3.0, 0.0))
    civ = _rioter("civ0", (2.0, 0.0))
    civ.crowd_role = "civilian"
    calmed = _rioter("c0", (2.5, 0.0))
    calmed.crowd_role = "calmed"
    targets = _as_dict(officers + [rioter, civ, calmed])

    behaviors.tick(0.1, targets)

    officer_shots = {
        p.target_id for p in behaviors._combat._projectiles.values()
        if p.source_id in officer_ids
    }
    assert officer_shots == {"r0"}, "only the rioter is ever a valid target"
