# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SimClockMixin -- deterministic sim-time for subsystems that gate on elapsed time.

Any simulation subsystem whose *state* depends on elapsed time (morale recovery
windows, assist windows, cooldowns) must read **sim time**, not wall-clock.  A
headless golden replay runs ~500x real-time: wall-clock barely advances between
ticks, so a wall-clock "3 seconds since last hit" window is never crossed in a
fast run -- until the process is descheduled under CPU load, when it crosses on
some ticks and not others.  That makes the same seed produce a different battle
run-to-run (the swarm_attack golden drifted ~1-in-4 under load: sim_seconds +-0.8,
projectiles +-1, eliminations +-1).  Routing the window through sim time makes the
window mean *simulated* seconds (deterministic, and the behaviour the design
actually intends) and removes the wall-clock coupling entirely.

This mirrors ``engine.simulation.behavior.hostile._now()``: read the attached
engine's monotonically-advancing ``_sim_clock`` when an engine (or any object
exposing a float ``_sim_clock``) is attached; fall back to wall-clock ONLY when
standalone, so bare-unit tests that construct the subsystem without an engine
keep working.

Usage::

    class MoraleSystem(SimClockMixin):
        def tick(self, dt, targets):
            now = self._now()          # sim seconds when attached
            ...

    # in the engine, once the subsystem is built:
    self.morale_system.attach_clock(self)   # self exposes ._sim_clock
"""

from __future__ import annotations

import time
from typing import Any


class SimClockMixin:
    """Provides ``_now()`` returning sim-time from an attached clock source.

    The clock source is any object exposing a float ``_sim_clock`` attribute
    (the SimulationEngine).  Until one is attached -- or if it has no
    ``_sim_clock`` -- ``_now()`` returns wall-clock ``time.time()`` so unit
    tests that never attach an engine behave as before.
    """

    _clock_source: Any = None

    def attach_clock(self, source: Any) -> None:
        """Attach a clock source exposing a float ``_sim_clock`` attribute.

        Store the *object* (not the value) so ``_now()`` reads the live,
        monotonically-advancing sim clock at call time.
        """
        self._clock_source = source

    def _now(self) -> float:
        """Return sim-time (seconds) from the attached clock, else wall-clock.

        Wall-clock fallback fires only when standalone (no clock attached, or
        the attached object exposes no ``_sim_clock``) -- e.g. bare-unit tests.
        """
        src = self._clock_source
        clk = getattr(src, "_sim_clock", None) if src is not None else None
        return clk if clk is not None else time.time()
