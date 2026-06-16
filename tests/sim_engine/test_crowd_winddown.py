# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Riot wind-down: a fled crowd goes HOME (agent sink at the boundary exit).

TDD spec for the CrowdSimulator agent-sink fix (G1). Before the fix a FLEEING
crowd reached the boundary exits and MILLED there forever — 120 in, 120 out,
0 ever removed, never empties. After the fix, a member that reaches within
``_EXIT_HOME_RADIUS`` of its picked exit is removed (gone home), so the visible
crowd DRAINS to empty.

These tests assert the FIX (unlike the SC measurement harness, which only
asserts invariants).  They cover:

  (a) WINDS DOWN — a crowd driven fully to FLEEING drains to empty.
  (b) DOES NOT spuriously despawn — an AGITATED/RIOTING riot crowd (no flee)
      keeps every member, so the riot still BUILDS, and battle-style crowds
      are untouched.
  (c) PANICKED members are NOT instantly sunk (they flee a live threat, they
      have not "gone home").
  (d) gradual / not-instant — the crowd shrinks over many ticks as bodies
      actually reach the perimeter, not in a single tick.
"""

from __future__ import annotations

import random

import pytest

from tritium_lib.sim_engine.crowd import (
    CrowdSimulator,
    CrowdEvent,
    CrowdMood,
    _EXIT_HOME_RADIUS,
)

pytestmark = pytest.mark.unit

BOUNDS = (0.0, 0.0, 200.0, 200.0)


def _spawn_rioting_blob(sim: CrowdSimulator, count: int = 120) -> list[str]:
    cx = (BOUNDS[0] + BOUNDS[2]) / 2.0
    cy = (BOUNDS[1] + BOUNDS[3]) / 2.0
    return sim.spawn_crowd((cx, cy), count=count, radius=30.0,
                           mood=CrowdMood.RIOTING, leader_ratio=0.05)


class TestCrowdGoesHome:
    """A crowd that flees must DRAIN to empty (agent sink at the exit)."""

    def test_fleeing_crowd_drains_to_empty(self):
        random.seed(1234)
        sim = CrowdSimulator(BOUNDS, max_members=500)
        _spawn_rioting_blob(sim, 120)
        start = len(sim.members)
        assert start == 120

        # Slam everyone to maximum fear so _resolve_mood -> FLEEING and they
        # route to the dispersed boundary exits.
        for m in sim.members:
            m.fear = 1.0
            m.aggression = 0.0
            m.mood = CrowdMood.FLEEING
        sim.inject_event(CrowdEvent(
            event_type="stampede", position=(100.0, 100.0),
            radius=300.0, intensity=1.0, timestamp=0.0,
        ))

        dt = 0.1
        empty_tick = -1
        for tick_i in range(1500):  # 150 sim-seconds — ample to cross + sink
            sim.tick(dt)
            if empty_tick < 0 and len(sim.members) == 0:
                empty_tick = tick_i
                break

        # The FIX: the crowd drained to (near) empty and members were sunk.
        assert sim.members_gone_home > 0, "no member ever went home — sink dead"
        assert len(sim.members) == 0, (
            f"crowd never emptied: {len(sim.members)} still milling "
            f"(gone_home={sim.members_gone_home})"
        )
        assert empty_tick >= 0
        # Conservation: everyone is accounted for (home + still present).
        assert sim.members_gone_home == start

    def test_does_not_empty_instantly(self):
        """The sink must be gated on ARRIVAL, not on the FLEEING label — a crowd
        starting at the centre cannot all be home on tick 0/1."""
        random.seed(99)
        sim = CrowdSimulator(BOUNDS, max_members=500)
        _spawn_rioting_blob(sim, 120)
        for m in sim.members:
            m.fear = 1.0
            m.aggression = 0.0
            m.mood = CrowdMood.FLEEING

        sim.tick(0.1)
        # After one tick almost nobody has crossed ~90 m to a perimeter exit.
        assert len(sim.members) >= 110, (
            f"too many sunk instantly: {120 - len(sim.members)} gone in 1 tick"
        )


class TestNoSpuriousDespawn:
    """A NON-fleeing riot crowd must keep every member (riot still BUILDS)."""

    def test_rioting_crowd_does_not_shrink(self):
        random.seed(7)
        sim = CrowdSimulator(BOUNDS, max_members=500)
        _spawn_rioting_blob(sim, 120)
        start = len(sim.members)

        # Keep them rioting/agitated (no fear) for a long run.
        dt = 0.1
        for _ in range(900):  # 90 sim-seconds
            sim.tick(dt)

        assert len(sim.members) == start, (
            f"rioting crowd shrank: {start} -> {len(sim.members)} "
            f"(gone_home={sim.members_gone_home})"
        )
        assert sim.members_gone_home == 0

    def test_panicked_members_not_sunk(self):
        """PANICKED members flee a live threat but have NOT gone home — even a
        panicked member parked on an exit must not be despawned."""
        random.seed(3)
        sim = CrowdSimulator(BOUNDS, max_members=500)
        sim.spawn_crowd((100.0, 100.0), count=60, radius=20.0,
                        mood=CrowdMood.PANICKED, leader_ratio=0.0)
        start = len(sim.members)
        # Force every member onto an exit position AND mark them PANICKED with a
        # picked exit_target right where they stand — the ONLY thing keeping them
        # alive is the mood gate (PANICKED != FLEEING), not their distance.
        for m in sim.members:
            m.mood = CrowdMood.PANICKED
            m.exit_target = m.position  # distance 0 to "exit"
        sim.tick(0.1)
        assert len(sim.members) == start, "PANICKED members were wrongly sunk"
        assert sim.members_gone_home == 0


class TestSinkBookkeeping:
    """The sink only fires on FLEEING + arrival, and is exact."""

    def test_sink_on_arrival_exact(self):
        sim = CrowdSimulator(BOUNDS, max_members=500)
        sim.spawn_crowd((100.0, 100.0), count=4, radius=1.0,
                        mood=CrowdMood.FLEEING, leader_ratio=0.0)
        # Two members are AT their exit (within home radius); two are far.
        members = sim.members
        for i, m in enumerate(members):
            if i < 2:
                m.mood = CrowdMood.FLEEING
                m.exit_target = (m.position[0] + _EXIT_HOME_RADIUS * 0.5,
                                 m.position[1])
            else:
                m.mood = CrowdMood.FLEEING
                m.exit_target = (m.position[0] + 50.0, m.position[1])
        sim._sink_arrived_fleers()
        assert sim.members_gone_home == 2
        assert len(sim.members) == 2
