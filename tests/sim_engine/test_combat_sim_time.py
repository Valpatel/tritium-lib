# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Combat timing must run on SIM time, not wall-clock (gap G-1).

Weapon cooldowns (`SimulationTarget.can_fire`/`last_fired`) and
projectile flight expiry (`Projectile.created_at`) used `time.time()`.
Consequences: replaying faster than real time distorts weapon cadence
vs movement (blocks the Master Plan P1 step-4 golden replay and makes
the step-5 tick multiplier dishonest), and even real-time runs are
non-deterministic at cooldown boundaries.

Contract pinned here: a target carries `sim_time` advanced only by
`tick(dt)`; cooldowns and projectile lifetimes elapse in that clock.
Wall-clock waiting changes NOTHING.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from tritium_lib.sim_engine.combat.combat import CombatSystem
from tritium_lib.sim_engine.core.entity import SimulationTarget


def _target(tid: str, alliance: str = "friendly", pos=(0.0, 0.0),
            cooldown: float = 1.0) -> SimulationTarget:
    return SimulationTarget(
        target_id=tid, name=tid, asset_type="turret", alliance=alliance,
        position=pos, weapon_cooldown=cooldown, weapon_range=50.0,
        weapon_damage=10.0, speed=0.0,
    )


class TestSimTimeCooldown:
    @pytest.mark.unit
    def test_fresh_target_can_fire_immediately(self):
        t = _target("t1", cooldown=5.0)
        assert t.can_fire(), "a never-fired unit must not start on cooldown"

    @pytest.mark.unit
    def test_cooldown_elapses_in_sim_time_not_wall_clock(self):
        t = _target("t1", cooldown=1.0)
        t.last_fired = t.sim_time  # just fired, in sim terms
        assert not t.can_fire()
        # 1.2 s of SIM time in microseconds of wall time
        for _ in range(12):
            t.tick(0.1)
        assert t.can_fire(), (
            "cooldown must elapse with ticked sim dt — faster-than-real-time "
            "replay would otherwise freeze every weapon"
        )

    @pytest.mark.unit
    def test_wall_clock_waiting_does_not_unlock_cooldown(self):
        t = _target("t1", cooldown=0.05)
        t.last_fired = t.sim_time
        time.sleep(0.12)  # wall-clock passes, sim time does not
        assert not t.can_fire(), (
            "wall-clock must not advance the cooldown — that is the G-1 bug"
        )

    @pytest.mark.unit
    def test_fire_stamps_last_fired_with_sim_time(self):
        bus = MagicMock()
        cs = CombatSystem(bus)
        src = _target("src", cooldown=1.0)
        tgt = _target("tgt", alliance="hostile", pos=(10.0, 0.0))
        for _ in range(5):
            src.tick(0.1)  # sim_time = 0.5
        proj = cs.fire(src, tgt)
        assert proj is not None
        assert src.last_fired == pytest.approx(src.sim_time), (
            "last_fired must be stamped in the target's sim clock, "
            "not time.time()"
        )


class TestProjectileSimTime:
    @pytest.mark.unit
    def test_projectile_expires_by_sim_time_not_wall_clock(self):
        bus = MagicMock()
        cs = CombatSystem(bus)
        src = _target("src")
        # Aim at a far-away hostile so the dart cannot arrive quickly
        tgt = _target("tgt", alliance="hostile", pos=(45.0, 0.0))
        proj = cs.fire(src, tgt)
        assert proj is not None
        targets = {"src": src, "tgt": tgt}
        # 6 s of SIM time in fast ticks: the 5 s max flight must expire
        for _ in range(60):
            cs.tick(0.1, targets)
        assert proj.id not in cs._projectiles, (
            "projectile flight lifetime must elapse in sim time — at "
            "faster-than-real-time replay, wall-clock expiry never fires"
        )

    @pytest.mark.unit
    def test_projectile_does_not_expire_from_wall_clock_alone(self):
        bus = MagicMock()
        cs = CombatSystem(bus)
        src = _target("src")
        tgt = _target("tgt", alliance="hostile", pos=(45.0, 0.0))
        proj = cs.fire(src, tgt)
        assert proj is not None
        # Backdate creation by 10 wall-clock seconds: under sim-time
        # accounting this must NOT expire the projectile on a small tick.
        proj.created_at -= 10.0 if proj.created_at > 1e6 else 0.0
        cs.tick(0.001, {"src": src, "tgt": tgt})
        # With sim-time accounting (created_at stamped from the combat
        # sim clock), the projectile is 0.001 s old and must still fly.
        assert proj.id in cs._projectiles or proj.hit, (
            "wall-clock age must be irrelevant to flight expiry"
        )
