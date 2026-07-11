# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Regression tests for the sim-clock determinism fix.

Background: the swarm_attack golden replay was intermittently non-deterministic
under CPU load.  The root cause was wall-clock coupling in the morale recovery
window (``time.time() - last_hit > 3.0``) and the stats assist window: in a
fast headless replay wall-clock barely advances, so the window was crossed on
different ticks run-to-run once the process was descheduled under load.  Routing
both through the sim clock (SimClockMixin) removes any wall-clock input to the
compared metrics, so determinism is guaranteed by construction.

These tests lock the behaviour in: morale recovery and the stats assist window
must follow the *attached sim clock* and ignore *wall time*.
"""
from __future__ import annotations

import time

from tritium_lib.sim_engine.core.sim_clock import SimClockMixin
from tritium_lib.sim_engine.game.morale import MoraleSystem, DEFAULT_MORALE
from tritium_lib.sim_engine.game.stats import StatsTracker


class _Clock:
    """Minimal engine-like clock source exposing a mutable ``_sim_clock``."""

    def __init__(self, t: float = 0.0) -> None:
        self._sim_clock = t


class _Unit:
    status = "active"


# --- SimClockMixin ---------------------------------------------------------

class TestSimClockMixin:
    def test_standalone_falls_back_to_wall_clock(self):
        m = SimClockMixin()
        # Wall-clock epoch is a large positive number; sim clock starts near 0.
        assert m._now() > 1_000_000_000.0

    def test_attached_reads_sim_clock(self):
        m = SimClockMixin()
        clk = _Clock(12.5)
        m.attach_clock(clk)
        assert m._now() == 12.5
        clk._sim_clock = 99.0          # live read, not a snapshot
        assert m._now() == 99.0

    def test_attached_source_without_sim_clock_falls_back(self):
        m = SimClockMixin()
        m.attach_clock(object())        # no _sim_clock attribute
        assert m._now() > 1_000_000_000.0


# --- Morale recovery on sim time -------------------------------------------

class TestMoraleSimClock:
    def _hit_unit(self) -> tuple[MoraleSystem, _Clock, dict]:
        m = MoraleSystem()
        clk = _Clock(0.0)
        m.attach_clock(clk)
        # 100 dmg -> morale 0.5 (below DEFAULT so recovery is eligible), last
        # hit stamped at sim t=0.
        m.on_damage_taken("u1", 100.0)
        return m, clk, {"u1": _Unit()}

    def test_recovery_follows_sim_time(self):
        m, clk, targets = self._hit_unit()
        before = m.get_morale("u1")
        assert before < DEFAULT_MORALE
        # Not enough sim time elapsed (< 3.0s window) -> no recovery.
        clk._sim_clock = 2.0
        m.tick(0.1, targets)
        assert m.get_morale("u1") == before
        # Past the window in SIM time -> recovers.
        clk._sim_clock = 3.5
        m.tick(0.1, targets)
        assert m.get_morale("u1") > before

    def test_recovery_ignores_wall_time(self):
        m, clk, targets = self._hit_unit()
        before = m.get_morale("u1")
        # Real wall time passes, but the SIM clock is frozen inside the window.
        time.sleep(0.05)
        clk._sim_clock = 1.0            # still < 3.0 sim-seconds
        m.tick(0.1, targets)
        assert m.get_morale("u1") == before, "recovery must not depend on wall time"

    def test_stamp_and_read_share_the_sim_clock(self):
        # on_damage_taken stamps last_hit via _now(); tick reads via _now().
        # Both must use the attached sim clock so the window is self-consistent.
        m = MoraleSystem()
        clk = _Clock(100.0)
        m.attach_clock(clk)
        m.on_damage_taken("u1", 100.0)         # last_hit = 100.0 (sim)
        assert m._last_hit_time["u1"] == 100.0
        before = m.get_morale("u1")
        clk._sim_clock = 102.0                 # +2s < window
        m.tick(0.1, {"u1": _Unit()})
        assert m.get_morale("u1") == before
        clk._sim_clock = 104.0                 # +4s > window
        m.tick(0.1, {"u1": _Unit()})
        assert m.get_morale("u1") > before


# --- Stats assist window on sim time ---------------------------------------

class TestStatsSimClock:
    def test_assist_window_follows_sim_time(self):
        st = StatsTracker()
        clk = _Clock(0.0)
        st.attach_clock(clk)
        st.register_unit("a", "A", "friendly", "rover")
        st.register_unit("b", "B", "friendly", "rover")
        st.register_unit("v", "V", "hostile", "person")
        # Hit at sim t=0, kill within the 5s assist window (sim t=3) -> assist.
        st.on_shot_hit("a", "v", 20.0)         # ts defaults to _now() = 0.0
        clk._sim_clock = 3.0
        st.on_kill("b", "v")
        assert st.get_unit_stats("a").assists == 1

    def test_assist_window_expires_on_sim_time_not_wall(self):
        st = StatsTracker()
        clk = _Clock(0.0)
        st.attach_clock(clk)
        st.register_unit("a", "A", "friendly", "rover")
        st.register_unit("b", "B", "friendly", "rover")
        st.register_unit("v", "V", "hostile", "person")
        st.on_shot_hit("a", "v", 20.0)         # ts = 0.0 (sim)
        # Real wall time passes, but sim time is well past the 5s window.
        time.sleep(0.02)
        clk._sim_clock = 10.0                  # 10s sim > 5s window
        st.on_kill("b", "v")
        assert st.get_unit_stats("a").assists == 0
