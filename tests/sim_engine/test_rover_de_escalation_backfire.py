# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Regression: the shoot-vs-de-escalate dilemma must be REACHABLE in live play.

Civil-unrest audit (2026-06-14) top gap #1: `_rover_de_escalation` has a fully
implemented backfire branch (firing near a crowd resets de-escalation timers and
30%-radicalizes nearby civilians), but the dispatch call site in
`UnitBehaviors.tick` never passed `rover_fired`, so the branch was dead code and
the single most interesting operator choice never existed in live play.

These tests drive the real `UnitBehaviors.tick` dispatch (not the helper in
isolation) so they fail if the wiring regresses again.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tritium_lib.sim_engine.behavior.behaviors import UnitBehaviors
from tritium_lib.sim_engine.combat import CombatSystem
from tritium_lib.sim_engine.core.entity import SimulationTarget


def _mk_behaviors() -> UnitBehaviors:
    behaviors = UnitBehaviors(CombatSystem(event_bus=MagicMock()))
    behaviors.set_game_mode_type("civil_unrest")
    return behaviors


def _scene() -> dict[str, SimulationTarget]:
    """A rover with a rioter and a hostile civilian both within 15 m and in
    weapon range — so the rover WILL fire this tick."""
    rover = SimulationTarget(
        target_id="rover1", name="Rover", alliance="friendly",
        asset_type="rover", position=(0.0, 0.0), weapon_range=50.0,
    )
    rioter = SimulationTarget(
        target_id="rioter1", name="Rioter", alliance="hostile",
        asset_type="person", position=(10.0, 0.0), crowd_role="rioter",
    )
    civ = SimulationTarget(
        target_id="civ1", name="Civ", alliance="hostile",
        asset_type="person", position=(8.0, 0.0), crowd_role="civilian",
    )
    return {"rover1": rover, "rioter1": rioter, "civ1": civ}


def test_dispatch_passes_rover_fired_when_rover_fires():
    """WIRING: when the rover fires this tick, tick() must tell
    _rover_de_escalation that it fired (rover_fired=True)."""
    behaviors = _mk_behaviors()
    targets = _scene()

    captured: list[bool] = []
    orig = behaviors._rover_de_escalation

    def spy(rover, tgts, dt=0.1, rover_fired=False):
        captured.append(rover_fired)
        return orig(rover, tgts, dt=dt, rover_fired=rover_fired)

    with patch.object(behaviors, "_rover_de_escalation", spy):
        behaviors.tick(0.1, targets)

    assert captured, "_rover_de_escalation was never called for the rover"
    assert captured[0] is True, (
        "tick() did not propagate that the rover fired — backfire is dead code"
    )


def test_firing_into_crowd_resets_timers_and_radicalizes_civilian():
    """EFFECT: a rover that fires mid-de-escalation clears its proximity timers
    and can convert a nearby hostile civilian into a rioter (the backfire)."""
    behaviors = _mk_behaviors()
    targets = _scene()

    # Pre-seed a de-escalation timer that was accumulating on the rioter.
    behaviors._de_escalation_timers = {"rioter1": 2.0}

    # Force the 30% radicalization roll to succeed deterministically.
    with patch("tritium_lib.sim_engine.behavior.behaviors.random.random", return_value=0.0):
        behaviors.tick(0.1, targets)

    # Backfire branch ran: timers cleared (NOT accumulated to 2.1), civ flipped.
    assert behaviors._de_escalation_timers == {}, (
        "firing should reset de-escalation timers (backfire), not accumulate them"
    )
    assert targets["civ1"].crowd_role == "rioter", (
        "firing into the crowd should radicalize a nearby civilian"
    )
    assert targets["civ1"].is_combatant is True
