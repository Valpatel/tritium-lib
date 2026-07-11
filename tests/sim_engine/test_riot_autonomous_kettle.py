# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Autonomous faction-kettle doctrine — the stand-in commander decides on its own.

lane/riot: the police FSM no longer waits for an operator to kettle a bloc.
For a rival-faction riot ``enable_auto_kettle(blocs)`` arms a doctrine that
reads the live per-bloc violent strength each tick and AUTONOMOUSLY cordons a
bloc that is decisively overwhelming its rival (or massing violently), switches
the cordon when the balance flips, and releases it when dominance decays — with
hysteresis + a minimum hold so it never thrashes.  Deterministic (integer count
thresholds, no RNG).  An operator command hard-overrides the doctrine.

This is the production-half essence of "AI battling": the squad stand-in makes
the tactical decision a real commander would.  Stand-in video-game AI; never
Graphling cognition.
"""

from __future__ import annotations

import math

from tritium_lib.sim_engine.ai.formations import FormationType
from tritium_lib.sim_engine.core.entity import SimulationTarget
from tritium_lib.sim_engine.game.riot_police import PoliceTacticsController


# ---------------------------------------------------------------------------
# Stubs + builders (self-contained; mirror test_riot_police_tactics.py)
# ---------------------------------------------------------------------------


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, topic: str, data: dict) -> None:
        self.events.append((topic, data))

    def topic(self, name: str) -> list[dict]:
        return [d for t, d in self.events if t == name]


def _officer(tid: str, pos: tuple[float, float]) -> SimulationTarget:
    t = SimulationTarget(
        target_id=tid, name=tid, alliance="friendly",
        asset_type="police", position=pos,
    )
    t.apply_combat_profile()
    t.status = "active"
    return t


def _rioter(tid: str, pos: tuple[float, float], faction: str,
            health: float = 50.0) -> SimulationTarget:
    t = SimulationTarget(
        target_id=tid, name=tid, alliance="hostile",
        asset_type="person", position=pos, crowd_role="rioter",
    )
    t.health = health
    t.is_combatant = True
    t.status = "active"
    t.faction = faction
    return t


def _as_dict(targets: list[SimulationTarget]) -> dict[str, SimulationTarget]:
    return {t.target_id: t for t in targets}


def _field(n_red: int, n_cyan: int):
    """8 officers + n_red RED (west, x~-40) + n_cyan CYAN (east, x~+40)."""
    officers = [_officer(f"off_{i}", (-40.0 + (i - 3.5) * 0.6, -13.0))
                for i in range(8)]
    reds = [_rioter(f"red_{i}", (-40.0 + math.cos(i * 1.3) * 1.5,
                                 math.sin(i * 1.3) * 1.5), "red_bloc")
            for i in range(n_red)]
    cyans = [_rioter(f"cyan_{i}", (40.0 + math.cos(i * 1.3) * 1.5,
                                   math.sin(i * 1.3) * 1.5), "cyan_bloc")
             for i in range(n_cyan)]
    return officers, reds, cyans


def _armed_ctrl(**kw) -> PoliceTacticsController:
    ctrl = PoliceTacticsController(_FakeBus())
    ctrl.enable_auto_kettle({"red_bloc", "cyan_bloc"}, **kw)
    return ctrl


# ===========================================================================
# Gating — single-faction riots stay byte-identical (doctrine never armed)
# ===========================================================================


def test_doctrine_disabled_by_default_no_auto_kettle():
    # No enable_auto_kettle: even a lopsided two-bloc field never auto-kettles.
    ctrl = PoliceTacticsController(_FakeBus())
    officers, reds, cyans = _field(6, 2)
    ctrl.tick(0.1, _as_dict(officers + reds + cyans), "civil_unrest")
    assert ctrl.commanded_tactic == "auto"
    assert ctrl.tactic_source == "auto"
    assert ctrl.auto_kettle_target is None
    assert ctrl.squad_state != "kettle"


def test_enable_requires_two_distinct_blocs():
    ctrl = PoliceTacticsController(_FakeBus())
    ctrl.enable_auto_kettle({"red_bloc"})            # only one bloc
    assert ctrl._auto_kettle_enabled is False
    ctrl.enable_auto_kettle([])                       # none
    assert ctrl._auto_kettle_enabled is False
    ctrl.enable_auto_kettle({"red_bloc", "cyan_bloc"})
    assert ctrl._auto_kettle_enabled is True


# ===========================================================================
# The headline: the AI kettles the dominant bloc ON ITS OWN
# ===========================================================================


def test_autonomous_kettles_dominant_bloc():
    ctrl = _armed_ctrl()
    officers, reds, cyans = _field(6, 2)     # RED overwhelms CYAN (6 vs 2)
    ctrl.tick(0.1, _as_dict(officers + reds + cyans), "civil_unrest")

    # The stand-in commander cordoned RED with no operator command.
    assert ctrl.commanded_tactic == "kettle"
    assert ctrl.target_faction == "red_bloc"
    assert ctrl.auto_kettle_target == "red_bloc"
    assert ctrl.tactic_source == "autonomous"
    assert ctrl.squad_state == "kettle"
    assert ctrl.formation_type == FormationType.ARC
    # The cordon centres on the RED cluster (west, x ~ -40), NOT cyan (east).
    assert ctrl._kettle_center[0] < -30.0, ctrl._kettle_center
    # A single police_tactic_commanded event was auto-issued for red.
    evts = ctrl._event_bus.topic("police_tactic_commanded")
    assert len(evts) == 1 and evts[0]["faction"] == "red_bloc"


def test_autonomous_kettles_massing_bloc_with_no_rival():
    # "or a bloc massing violently": one bloc past the mass floor, rival gone.
    ctrl = _armed_ctrl()
    officers, reds, _ = _field(6, 0)
    ctrl.tick(0.1, _as_dict(officers + reds), "civil_unrest")
    assert ctrl.target_faction == "red_bloc"
    assert ctrl.squad_state == "kettle"


def test_no_kettle_when_blocs_balanced():
    ctrl = _armed_ctrl()
    officers, reds, cyans = _field(4, 4)     # even street — no dominant bloc
    ctrl.tick(0.1, _as_dict(officers + reds + cyans), "civil_unrest")
    assert ctrl.commanded_tactic == "auto"
    assert ctrl.auto_kettle_target is None
    assert ctrl.squad_state != "kettle"


def test_no_kettle_below_mass_floor():
    # RED "dominates" cyan 3-0 by ratio/margin, but 3 < min_strength: a stray
    # knot is not kettled, only a real mass is.
    ctrl = _armed_ctrl()
    officers, reds, _ = _field(3, 0)
    ctrl.tick(0.1, _as_dict(officers + reds), "civil_unrest")
    assert ctrl.commanded_tactic == "auto"
    assert ctrl.auto_kettle_target is None


# ===========================================================================
# Operator override — a human command wins over the doctrine
# ===========================================================================


def test_operator_command_overrides_the_doctrine():
    ctrl = _armed_ctrl()
    officers, reds, cyans = _field(6, 2)     # doctrine WANTS to kettle red
    targets = _as_dict(officers + reds + cyans)

    # But the operator explicitly kettles CYAN first.
    ctrl.command_tactic("kettle", faction="cyan_bloc")   # _source="operator"
    assert ctrl.tactic_source == "operator"
    ctrl.tick(0.1, targets, "civil_unrest")

    # The operator's choice stands: cyan cordoned (east), red left alone; the
    # doctrine never armed a target of its own.
    assert ctrl.target_faction == "cyan_bloc"
    assert ctrl.auto_kettle_target is None
    assert ctrl._kettle_center[0] > 30.0, ctrl._kettle_center
    # And it does not silently flip to red on the next tick either.
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.target_faction == "cyan_bloc"


def test_operator_line_command_blocks_autonomous_kettle():
    ctrl = _armed_ctrl()
    officers, reds, cyans = _field(6, 2)
    targets = _as_dict(officers + reds + cyans)
    ctrl.command_tactic("line")              # operator forces a plain line
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.commanded_tactic == "line"
    assert ctrl.formation_type == FormationType.LINE
    assert ctrl.auto_kettle_target is None


def test_operator_return_to_auto_resumes_doctrine():
    ctrl = _armed_ctrl()
    officers, reds, cyans = _field(6, 2)
    targets = _as_dict(officers + reds + cyans)

    ctrl.command_tactic("line")              # lock the doctrine out
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.auto_kettle_target is None

    ctrl.command_tactic("auto")              # operator hands control back
    assert ctrl.tactic_source == "auto"
    ctrl.tick(0.1, targets, "civil_unrest")
    # The doctrine resumes and kettles the still-dominant red bloc.
    assert ctrl.target_faction == "red_bloc"
    assert ctrl.tactic_source == "autonomous"


# ===========================================================================
# Hysteresis — min-hold, switch on flip, release on decay (anti-thrash)
# ===========================================================================


def test_min_hold_prevents_immediate_release():
    ctrl = _armed_ctrl(min_hold=5.0)
    officers, reds, cyans = _field(6, 2)
    targets = _as_dict(officers + reds + cyans)

    ctrl.tick(0.1, targets, "civil_unrest")  # arm red (since = 0.1)
    assert ctrl.auto_kettle_target == "red_bloc"

    # RED collapses below dominance the very next tick, but inside the hold
    # window the doctrine keeps its cordon (no thrash).
    for r in reds[3:]:
        r.crowd_role = "calmed"              # red violent 6 -> 3
    ctrl.tick(0.1, targets, "civil_unrest")  # held = 0.1 < 5.0
    assert ctrl.auto_kettle_target == "red_bloc"


def test_release_to_auto_when_dominance_decays():
    ctrl = _armed_ctrl(min_hold=0.0)         # re-decide immediately
    officers, reds, cyans = _field(6, 2)
    targets = _as_dict(officers + reds + cyans)

    ctrl.tick(0.1, targets, "civil_unrest")  # arm red
    assert ctrl.auto_kettle_target == "red_bloc"

    # RED contained below the mass floor; cyan is no threat -> release to FSM.
    for r in reds:
        r.crowd_role = "calmed"
    ctrl.tick(0.1, targets, "civil_unrest")
    assert ctrl.auto_kettle_target is None
    assert ctrl.commanded_tactic == "auto"
    assert ctrl.tactic_source == "auto"


def test_balance_flip_switches_the_cordon():
    ctrl = _armed_ctrl(min_hold=0.0)
    officers, reds, cyans = _field(6, 6)
    for c in cyans[2:]:
        c.crowd_role = "calmed"              # cyan violent 6 -> 2 at start
    targets = _as_dict(officers + reds + cyans)

    ctrl.tick(0.1, targets, "civil_unrest")  # red 6 vs cyan 2 -> arm red
    assert ctrl.auto_kettle_target == "red_bloc"

    # The street flips: red collapses, cyan surges.
    for r in reds[2:]:
        r.crowd_role = "calmed"              # red violent -> 2
    for c in cyans[2:]:
        c.crowd_role = "rioter"             # cyan violent -> 6
    ctrl.tick(0.1, targets, "civil_unrest")  # cyan 6 vs red 2 -> switch to cyan
    assert ctrl.auto_kettle_target == "cyan_bloc"
    assert ctrl.target_faction == "cyan_bloc"
    # Two auto-issued commands: the initial red kettle + the switch to cyan.
    factions = [e["faction"]
                for e in ctrl._event_bus.topic("police_tactic_commanded")]
    assert factions == ["red_bloc", "cyan_bloc"], factions


# ===========================================================================
# Terrain-aware slot-validation seam (the cordon closes on real AO terrain)
# ===========================================================================


def _kettle_waypoints(ctrl, officers, reds, cyans):
    """Command a red-bloc kettle, tick once, return officer waypoint ends."""
    targets = _as_dict(officers + reds + cyans)
    ctrl.command_tactic("kettle", faction="red_bloc")
    ctrl.tick(0.1, targets, "civil_unrest")
    return {o.target_id: tuple(o.waypoints[-1]) for o in officers if o.waypoints}


def test_slot_validator_none_equals_identity():
    # No validator vs an identity validator produce byte-identical cordon slots.
    ctrl_a = PoliceTacticsController(_FakeBus())              # validator None
    wp_a = _kettle_waypoints(ctrl_a, *_field(4, 4))

    ctrl_b = PoliceTacticsController(
        _FakeBus(), slot_validator=lambda frm, slot: slot,   # identity
    )
    wp_b = _kettle_waypoints(ctrl_b, *_field(4, 4))
    assert wp_a == wp_b, "identity validator perturbed the cordon slots"


def test_slot_validator_nudges_blocked_kettle_slots():
    # A validator standing in for "this slot is in a wall" lifts any slot with
    # y < 0 up to y = 5.0.  Every officer waypoint must reflect the nudge, and
    # the validator must actually have been consulted for the cordon.
    calls: list = []

    def validator(frm, slot):
        calls.append((frm, slot))
        return (slot[0], 5.0) if slot[1] < 0.0 else slot

    ctrl = PoliceTacticsController(_FakeBus(), slot_validator=validator)
    wp = _kettle_waypoints(ctrl, *_field(4, 4))
    assert calls, "the slot validator was never consulted for the cordon"
    assert all(y >= 0.0 for (_x, y) in wp.values()), (
        f"a cordon slot stayed in the 'wall' (y<0): {wp}"
    )


def test_slot_validator_applies_to_line_formation():
    # The seam covers plain formations too, not just kettle cordons.
    seen: list = []

    def validator(frm, slot):
        seen.append(slot)
        return slot

    ctrl = PoliceTacticsController(_FakeBus(), slot_validator=validator)
    officers, reds, cyans = _field(4, 0)
    ctrl.command_tactic("line")
    ctrl.tick(0.1, _as_dict(officers + reds), "civil_unrest")
    ctrl.tick(0.1, _as_dict(officers + reds), "civil_unrest")  # form -> advance
    assert seen, "the slot validator was never consulted for a line formation"


def test_slot_validator_error_falls_back_to_raw_slot():
    # A throwing validator never breaks the tick; slots fall back to raw.
    def boom(frm, slot):
        raise RuntimeError("costmap unavailable")

    ctrl = PoliceTacticsController(_FakeBus(), slot_validator=boom)
    wp = _kettle_waypoints(ctrl, *_field(4, 4))
    assert wp, "a throwing validator must not stop the cordon from forming"


def test_reset_clears_doctrine_target_but_keeps_arming():
    ctrl = _armed_ctrl()
    officers, reds, cyans = _field(6, 2)
    ctrl.tick(0.1, _as_dict(officers + reds + cyans), "civil_unrest")
    assert ctrl.auto_kettle_target == "red_bloc"

    ctrl.reset()
    assert ctrl.auto_kettle_target is None
    assert ctrl.commanded_tactic == "auto"
    assert ctrl.tactic_source == "auto"
    # Still armed after reset (a re-run of the same rival riot re-decides).
    assert ctrl._auto_kettle_enabled is True
    ctrl.tick(0.1, _as_dict(officers + reds + cyans), "civil_unrest")
    assert ctrl.auto_kettle_target == "red_bloc"
