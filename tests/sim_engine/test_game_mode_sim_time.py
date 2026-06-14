# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""GameMode wave pacing must run on SIM time, not wall-clock (gap G-1).

Wave spawning ran in a daemon Thread with time.sleep(0.5) stagger, and
the stalemate / wave-advance / wave-duration timers used time.time().
Consequence, observed in a real headless run: 600 sim-seconds of
fast replay produced exactly ONE hostile — the wall-clock spawner
can't keep up when sim time outruns real time, so the golden nightly
replay (Master Plan P1 step 4) and the tick multiplier (step 5) are
impossible until pacing is tick-driven.

Contract pinned here: spawns come from a queue drained inside
game_mode.tick(dt) at _SPAWN_STAGGER intervals of GAME SIM TIME;
stalemate and wave-advance timers elapse in the same clock. No
threads, no sleeps, no wall-clock.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tritium_lib.sim_engine.game.game_mode import (
    GameMode,
    WAVE_CONFIGS,
    _COUNTDOWN_DURATION,
    _SPAWN_STAGGER,
    _STALEMATE_TIMEOUT,
    _WAVE_ADVANCE_DELAY,
    _WAVE_HARD_TIMEOUT,
)


class FakeHostile:
    def __init__(self, n: int) -> None:
        self.target_id = f"hostile-{n}"
        self.name = f"Hostile {n}"
        self.position = (10.0, 10.0)
        self.speed = 1.5
        self.health = 80.0
        self.max_health = 80.0
        self.status = "active"
        self.alliance = "hostile"
        self.is_combatant = True


class FakeFriendly:
    def __init__(self) -> None:
        self.target_id = "turret-1"
        self.alliance = "friendly"
        self.is_combatant = True
        self.status = "active"
        self.battery = 1.0
        self.health = 200.0
        self.max_health = 200.0


class FakeEngine:
    """Records spawns; one immortal friendly keeps defeat away."""

    def __init__(self) -> None:
        self.spawned: list[FakeHostile] = []
        self._friendly = FakeFriendly()

    def spawn_hostile(self, direction: str = "random") -> FakeHostile:
        h = FakeHostile(len(self.spawned))
        self.spawned.append(h)
        return h

    def spawn_hostile_typed(self, asset_type: str = "person", speed=None,
                            health=None, direction: str = "random",
                            drone_variant=None) -> FakeHostile:
        return self.spawn_hostile(direction)

    def get_targets(self):
        return [self._friendly] + self.spawned


def _game() -> tuple[GameMode, FakeEngine]:
    engine = FakeEngine()
    gm = GameMode(MagicMock(), engine, MagicMock())
    return gm, engine


def _run_sim(gm: GameMode, seconds: float, dt: float = 0.1) -> None:
    for _ in range(int(round(seconds / dt))):
        gm.tick(dt)


class TestSimTimeSpawning:
    @pytest.mark.unit
    def test_full_wave_spawns_in_fast_replay(self):
        gm, engine = _game()
        gm.begin_war()
        wave1_count = WAVE_CONFIGS[0].count
        # Countdown + full stagger window + slack, all in fast sim time
        _run_sim(gm, _COUNTDOWN_DURATION + wave1_count * _SPAWN_STAGGER + 2.0)
        assert len(engine.spawned) >= wave1_count, (
            f"only {len(engine.spawned)}/{wave1_count} hostiles spawned — "
            "wall-clock spawner cannot keep up with fast replay (G-1)"
        )

    @pytest.mark.unit
    def test_spawns_are_stagger_paced_in_sim_time(self):
        gm, engine = _game()
        gm.begin_war()
        # Countdown accumulates float drift (50 x 0.1 != exactly 5.0),
        # so give it two extra frames: one to flip to active, one for
        # the first drain.
        _run_sim(gm, _COUNTDOWN_DURATION + 0.2)
        first = len(engine.spawned)
        assert first >= 1, "first hostile must spawn on the first active tick"
        # One full stagger interval later exactly one more arrives
        _run_sim(gm, _SPAWN_STAGGER)
        assert len(engine.spawned) == first + 1, (
            "spawns must pace at _SPAWN_STAGGER intervals of sim time"
        )

    @pytest.mark.unit
    def test_no_spawn_thread_is_used(self):
        gm, engine = _game()
        gm.begin_war()
        _run_sim(gm, _COUNTDOWN_DURATION + 1.0)
        assert gm._spawn_thread is None, (
            "spawning must be tick-driven, not a wall-clock thread"
        )


class TestSimTimeTimers:
    @pytest.mark.unit
    def test_wave_advance_delay_elapses_in_sim_time(self):
        gm, engine = _game()
        gm.begin_war()
        wave1_count = WAVE_CONFIGS[0].count
        _run_sim(gm, _COUNTDOWN_DURATION + wave1_count * _SPAWN_STAGGER + 2.0)
        # Kill everything spawned so far → wave completes
        for h in engine.spawned:
            h.status = "eliminated"
        _run_sim(gm, 0.2)
        assert gm.state == "wave_complete"
        # The advance delay must elapse in sim time (fast loop)
        _run_sim(gm, _WAVE_ADVANCE_DELAY + 0.3)
        assert gm.state == "active" and gm.wave == 2, (
            "wave-advance delay must elapse in sim time, not wall-clock"
        )

    @pytest.mark.unit
    def test_stalemate_fires_in_sim_time(self):
        gm, engine = _game()
        gm.begin_war()
        wave1_count = WAVE_CONFIGS[0].count
        _run_sim(gm, _COUNTDOWN_DURATION + wave1_count * _SPAWN_STAGGER + 2.0)
        alive_before = sum(1 for h in engine.spawned if h.status == "active")
        assert alive_before > 0
        # No eliminations happen; > _STALEMATE_TIMEOUT sim seconds pass fast
        _run_sim(gm, _STALEMATE_TIMEOUT + 2.0)
        assert gm.state in ("wave_complete", "active", "victory"), (
            f"stalemate must force wave resolution in sim time "
            f"(state={gm.state}, alive hostiles never cleared)"
        )


class TestStalemateProgressReset:
    """The stalemate timer must treat ACTIVE COMBAT as progress.

    FEATURE-AUDIT 2026-06-14: a fresh-player trace of the 10-wave battle found
    wave 1 was "won" by the 60s stalemate force-eliminator, not by the
    defenders -- the clock only reset on a KILL, so defenders steadily chipping
    hostiles (real combat, just slow) were force-removed by the timer before a
    single hostile died.  The fix resets the clock whenever total wave-hostile
    HP drops (damage = progress), with an absolute hard ceiling as a backstop.
    """

    def _active_wave(self):
        gm, engine = _game()
        gm.begin_war()
        wave1_count = WAVE_CONFIGS[0].count
        _run_sim(gm, _COUNTDOWN_DURATION + wave1_count * _SPAWN_STAGGER + 2.0)
        assert gm.state == "active"
        assert any(h.status == "active" for h in engine.spawned)
        return gm, engine

    @pytest.mark.unit
    def test_ongoing_damage_prevents_premature_force_elimination(self):
        """While hostiles keep taking damage, the timer must NOT force-clear."""
        gm, engine = self._active_wave()
        # Make hostiles tanky so steady damage does not finish them inside the
        # window -- we are testing the timer, not wave completion.
        for h in engine.spawned:
            h.health = 100_000.0
            h.max_health = 100_000.0
        calls = []
        gm._force_eliminate_wave_hostiles = lambda: calls.append(gm._sim_time)
        dt = 0.1
        # Run well past the old stalemate timeout, applying real damage each tick.
        for _ in range(int((_STALEMATE_TIMEOUT * 1.5) / dt)):
            alive = [h for h in engine.spawned if h.status == "active"]
            if alive:
                alive[0].health -= 2.0  # defenders chipping a hostile
            gm.tick(dt)
        assert not calls, (
            f"stalemate force-elimination fired at {calls} despite continuous "
            "combat damage -- active combat must reset the stalemate clock"
        )

    @pytest.mark.unit
    def test_true_stalemate_still_force_eliminates(self):
        """No damage at all for the timeout must still break the stalemate."""
        gm, engine = self._active_wave()
        for h in engine.spawned:
            h.health = 100_000.0  # never die on their own
        orig = gm._force_eliminate_wave_hostiles
        calls = []

        def spy():
            calls.append(gm._sim_time)
            orig()

        gm._force_eliminate_wave_hostiles = spy
        # No damage applied -> genuine stalemate.
        _run_sim(gm, _STALEMATE_TIMEOUT + 5.0)
        assert calls, (
            "a true stalemate (no damage dealt) must still force-eliminate to "
            "prevent an infinite game"
        )

    @pytest.mark.unit
    def test_hard_ceiling_bounds_total_wave_duration(self):
        """A perpetual damage trickle cannot keep a wave alive forever."""
        gm, engine = self._active_wave()
        for h in engine.spawned:
            h.health = 100_000.0
            h.max_health = 100_000.0
        orig = gm._force_eliminate_wave_hostiles
        calls = []

        def spy():
            calls.append(gm._sim_time)
            orig()

        gm._force_eliminate_wave_hostiles = spy
        dt = 0.1
        # Trickle damage every tick (resets the soft clock forever); only the
        # hard ceiling can resolve the wave.
        for _ in range(int((_WAVE_HARD_TIMEOUT + 20.0) / dt)):
            alive = [h for h in engine.spawned if h.status == "active"]
            if alive:
                alive[0].health -= 0.1
            gm.tick(dt)
        assert calls, (
            "the absolute hard ceiling must force-resolve a wave even when a "
            "damage trickle keeps resetting the soft stalemate clock"
        )
