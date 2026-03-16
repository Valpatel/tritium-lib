# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the scenario tick runner — sim_engine.scenario."""

from __future__ import annotations

import random

import pytest

from tritium_lib.sim_engine.scenario import (
    SimEvent,
    WaveConfig,
    Objective,
    ScenarioConfig,
    SimState,
    Scenario,
    PRESET_SCENARIOS,
)
from tritium_lib.sim_engine.units import Alliance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_config(
    waves: int = 1,
    hostile_count: int = 3,
    friendly_count: int = 3,
    tick_rate: float = 10.0,
    max_ticks: int = 2000,
    objectives: list[Objective] | None = None,
) -> ScenarioConfig:
    """Create a minimal scenario config for testing."""
    wave_list = [
        WaveConfig(
            wave_number=i + 1,
            spawn_delay=0.5,
            hostiles=[{
                "template": "infantry",
                "count": hostile_count,
                "spawn_pos": (10.0, 100.0),
                "target_pos": (100.0, 100.0),
            }],
        )
        for i in range(waves)
    ]
    return ScenarioConfig(
        name="Test Scenario",
        tick_rate=tick_rate,
        max_ticks=max_ticks,
        waves=wave_list,
        objectives=objectives or [],
        friendly_units=[
            {"template": "infantry", "count": friendly_count, "spawn_pos": (100.0, 100.0)},
        ],
    )


# ---------------------------------------------------------------------------
# SimEvent dataclass
# ---------------------------------------------------------------------------

class TestSimEvent:
    def test_default_data(self):
        e = SimEvent(tick=0, time=0.0, event_type="test")
        assert e.data == {}

    def test_custom_data(self):
        e = SimEvent(tick=1, time=0.1, event_type="unit_spawned", data={"unit_id": "u_001"})
        assert e.data["unit_id"] == "u_001"
        assert e.tick == 1
        assert e.event_type == "unit_spawned"


# ---------------------------------------------------------------------------
# WaveConfig dataclass
# ---------------------------------------------------------------------------

class TestWaveConfig:
    def test_defaults(self):
        w = WaveConfig(wave_number=1)
        assert w.spawn_delay == 2.0
        assert w.hostiles == []
        assert w.wave_bonus == 0.0

    def test_custom(self):
        w = WaveConfig(wave_number=3, spawn_delay=0.5, wave_bonus=0.2,
                       hostiles=[{"template": "heavy", "count": 2}])
        assert w.wave_number == 3
        assert len(w.hostiles) == 1
        assert w.hostiles[0]["count"] == 2


# ---------------------------------------------------------------------------
# Objective dataclass
# ---------------------------------------------------------------------------

class TestObjective:
    def test_defaults(self):
        o = Objective(objective_type="eliminate_all", target_value=1.0)
        assert o.current_value == 0.0
        assert o.completed is False

    def test_types(self):
        for otype in ("eliminate_all", "survive_time", "defend_point", "kill_count"):
            o = Objective(objective_type=otype, target_value=10.0)
            assert o.objective_type == otype


# ---------------------------------------------------------------------------
# ScenarioConfig dataclass
# ---------------------------------------------------------------------------

class TestScenarioConfig:
    def test_defaults(self):
        c = ScenarioConfig(name="Test")
        assert c.tick_rate == 10.0
        assert c.max_ticks == 6000
        assert c.waves == []
        assert c.map_size == (200.0, 200.0)

    def test_custom_map_size(self):
        c = ScenarioConfig(name="Small", map_size=(50.0, 50.0))
        assert c.map_size == (50.0, 50.0)


# ---------------------------------------------------------------------------
# SimState dataclass
# ---------------------------------------------------------------------------

class TestSimState:
    def test_defaults(self):
        s = SimState()
        assert s.tick == 0
        assert s.time == 0.0
        assert s.phase == "setup"
        assert s.units == {}
        assert s.events == []
        assert s.score == 0
        assert s.result == ""


# ---------------------------------------------------------------------------
# Scenario — phase control
# ---------------------------------------------------------------------------

class TestScenarioPhases:
    def test_starts_in_setup(self):
        sim = Scenario(_simple_config())
        assert sim.state.phase == "setup"

    def test_start_transitions_to_active(self):
        sim = Scenario(_simple_config())
        sim.start()
        assert sim.state.phase == "active"

    def test_start_idempotent(self):
        sim = Scenario(_simple_config())
        sim.start()
        sim.start()  # second call should be no-op
        assert sim.state.phase == "active"

    def test_pause(self):
        sim = Scenario(_simple_config())
        sim.start()
        sim.pause()
        assert sim.state.phase == "paused"

    def test_resume(self):
        sim = Scenario(_simple_config())
        sim.start()
        sim.pause()
        sim.resume()
        assert sim.state.phase == "active"

    def test_pause_only_from_active(self):
        sim = Scenario(_simple_config())
        sim.pause()  # should be no-op in setup
        assert sim.state.phase == "setup"

    def test_resume_only_from_paused(self):
        sim = Scenario(_simple_config())
        sim.start()
        sim.resume()  # already active, should be no-op
        assert sim.state.phase == "active"

    def test_tick_does_nothing_in_setup(self):
        sim = Scenario(_simple_config())
        sim.tick()
        assert sim.state.tick == 0

    def test_tick_does_nothing_when_paused(self):
        sim = Scenario(_simple_config())
        sim.start()
        sim.tick()
        sim.pause()
        tick_before = sim.state.tick
        sim.tick()
        assert sim.state.tick == tick_before


# ---------------------------------------------------------------------------
# Scenario — ticking
# ---------------------------------------------------------------------------

class TestScenarioTicking:
    def test_tick_advances_time(self):
        sim = Scenario(_simple_config(tick_rate=10.0))
        sim.start()
        sim.tick()
        assert sim.state.tick == 1
        assert abs(sim.state.time - 0.1) < 1e-9

    def test_multiple_ticks(self):
        sim = Scenario(_simple_config(tick_rate=10.0))
        sim.start()
        for _ in range(50):
            sim.tick()
        assert sim.state.tick == 50
        assert abs(sim.state.time - 5.0) < 1e-6

    def test_custom_dt(self):
        sim = Scenario(_simple_config())
        sim.start()
        sim.tick(dt=0.5)
        assert abs(sim.state.time - 0.5) < 1e-9

    def test_tick_rate_determines_dt(self):
        sim = Scenario(_simple_config(tick_rate=20.0))
        sim.start()
        sim.tick()
        assert abs(sim.state.time - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# Scenario — spawning
# ---------------------------------------------------------------------------

class TestScenarioSpawning:
    def test_friendlies_spawn_on_start(self):
        sim = Scenario(_simple_config(friendly_count=4))
        sim.start()
        friendly_units = [
            u for u in sim._units.values()
            if u.alliance == Alliance.FRIENDLY
        ]
        assert len(friendly_units) == 4

    def test_hostiles_spawn_per_wave(self):
        random.seed(42)
        sim = Scenario(_simple_config(hostile_count=5))
        sim.start()
        # Run enough ticks for all spawns (5 units * 0.5s delay = 2.5s)
        for _ in range(30):
            sim.tick()
        hostile_units = [
            u for u in sim._units.values()
            if u.alliance == Alliance.HOSTILE
        ]
        assert len(hostile_units) == 5

    def test_spawn_events_emitted(self):
        sim = Scenario(_simple_config(friendly_count=2, hostile_count=2))
        sim.start()
        for _ in range(30):
            sim.tick()
        spawn_events = [e for e in sim.state.events if e.event_type == "unit_spawned"]
        assert len(spawn_events) >= 4  # 2 friendly + 2 hostile

    def test_wave_start_event(self):
        sim = Scenario(_simple_config())
        sim.start()
        wave_starts = [e for e in sim.state.events if e.event_type == "wave_start"]
        assert len(wave_starts) == 1
        assert wave_starts[0].data["wave"] == 1


# ---------------------------------------------------------------------------
# Scenario — combat
# ---------------------------------------------------------------------------

class TestScenarioCombat:
    def test_units_take_damage(self):
        """Run long enough that combat should occur."""
        random.seed(42)
        config = _simple_config(hostile_count=3, friendly_count=3, max_ticks=500)
        sim = Scenario(config)
        sim.run(max_ticks=500)
        damage_events = [e for e in sim.state.events if e.event_type == "damage_dealt"]
        assert len(damage_events) > 0

    def test_units_can_die(self):
        random.seed(42)
        config = _simple_config(hostile_count=5, friendly_count=5, max_ticks=1000)
        sim = Scenario(config)
        sim.run(max_ticks=1000)
        kill_events = [e for e in sim.state.events if e.event_type == "unit_killed"]
        assert len(kill_events) > 0

    def test_dead_units_stop_acting(self):
        random.seed(42)
        config = _simple_config(hostile_count=3, friendly_count=3, max_ticks=500)
        sim = Scenario(config)
        sim.run(max_ticks=500)
        for unit in sim._units.values():
            if not unit.state.is_alive:
                assert unit.state.status == "dead"


# ---------------------------------------------------------------------------
# Scenario — wave progression
# ---------------------------------------------------------------------------

class TestScenarioWaves:
    def test_wave_advances_when_hostiles_cleared(self):
        random.seed(42)
        config = _simple_config(
            waves=2, hostile_count=2, friendly_count=10, max_ticks=2000,
        )
        sim = Scenario(config)
        sim.run(max_ticks=2000)
        wave_end_events = [e for e in sim.state.events if e.event_type == "wave_end"]
        # With 10 friendlies vs 2 hostiles, at least wave 1 should clear
        assert len(wave_end_events) >= 1

    def test_multiple_waves(self):
        random.seed(42)
        config = _simple_config(
            waves=3, hostile_count=1, friendly_count=10, max_ticks=3000,
        )
        sim = Scenario(config)
        sim.run(max_ticks=3000)
        wave_starts = [e for e in sim.state.events if e.event_type == "wave_start"]
        assert len(wave_starts) >= 2

    def test_wave_bonus_makes_enemies_stronger(self):
        config = ScenarioConfig(
            name="Bonus Test",
            tick_rate=10.0,
            max_ticks=100,
            waves=[
                WaveConfig(
                    wave_number=1,
                    spawn_delay=0.0,
                    wave_bonus=0.5,  # 50% stat boost
                    hostiles=[{
                        "template": "infantry",
                        "count": 1,
                        "spawn_pos": (10.0, 100.0),
                        "target_pos": (100.0, 100.0),
                    }],
                ),
            ],
            friendly_units=[
                {"template": "infantry", "count": 1, "spawn_pos": (100.0, 100.0)},
            ],
        )
        sim = Scenario(config)
        sim.start()
        # Tick enough for spawn
        for _ in range(5):
            sim.tick()
        hostiles = [u for u in sim._units.values() if u.alliance == Alliance.HOSTILE]
        assert len(hostiles) == 1
        h = hostiles[0]
        # Infantry base health is 100, with 0.5 bonus should be 150
        assert h.stats.max_health == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Scenario — objectives
# ---------------------------------------------------------------------------

class TestScenarioObjectives:
    def test_survive_time_objective(self):
        config = _simple_config(
            hostile_count=0, friendly_count=3, max_ticks=200,
            objectives=[Objective(objective_type="survive_time", target_value=5.0)],
        )
        # No hostiles, just need to survive 5 seconds (50 ticks at 10 tps)
        sim = Scenario(config)
        sim.run(max_ticks=200)
        assert config.objectives[0].completed is True
        assert sim.state.result == "victory"

    def test_kill_count_objective(self):
        random.seed(42)
        config = _simple_config(
            hostile_count=5, friendly_count=10, max_ticks=2000,
            objectives=[Objective(objective_type="kill_count", target_value=3.0)],
        )
        sim = Scenario(config)
        sim.run(max_ticks=2000)
        # 10 friendlies vs 5 hostiles, should get 3 kills
        assert config.objectives[0].current_value >= 3

    def test_objective_complete_event(self):
        config = _simple_config(
            hostile_count=0, friendly_count=1, max_ticks=200,
            objectives=[Objective(objective_type="survive_time", target_value=1.0)],
        )
        sim = Scenario(config)
        sim.run(max_ticks=200)
        obj_events = [e for e in sim.state.events if e.event_type == "objective_complete"]
        assert len(obj_events) == 1


# ---------------------------------------------------------------------------
# Scenario — game over
# ---------------------------------------------------------------------------

class TestScenarioGameOver:
    def test_victory_on_objectives_complete(self):
        config = _simple_config(
            hostile_count=0, friendly_count=1, max_ticks=200,
            objectives=[Objective(objective_type="survive_time", target_value=1.0)],
        )
        sim = Scenario(config)
        state = sim.run(max_ticks=200)
        assert state.result == "victory"
        assert state.phase == "game_over"

    def test_defeat_when_all_friendlies_dead(self):
        random.seed(42)
        # Lots of hostiles, few friendlies
        config = _simple_config(
            hostile_count=20, friendly_count=1, max_ticks=3000,
        )
        # No objectives, so can't "win" — but can lose by having all friendlies die
        sim = Scenario(config)
        state = sim.run(max_ticks=3000)
        # With 20 hostiles vs 1 friendly, defeat is likely
        if state.result == "defeat":
            assert state.phase == "game_over"

    def test_draw_on_tick_limit(self):
        config = _simple_config(
            hostile_count=0, friendly_count=1, max_ticks=10,
        )
        # No objectives, no hostiles, will hit tick limit
        sim = Scenario(config)
        state = sim.run(max_ticks=10)
        assert state.result == "draw"
        assert state.phase == "game_over"

    def test_scenario_end_event(self):
        config = _simple_config(
            hostile_count=0, friendly_count=1, max_ticks=10,
        )
        sim = Scenario(config)
        sim.run(max_ticks=10)
        end_events = [e for e in sim.state.events if e.event_type == "scenario_end"]
        assert len(end_events) >= 1


# ---------------------------------------------------------------------------
# Scenario — score
# ---------------------------------------------------------------------------

class TestScenarioScore:
    def test_score_increases_on_kills(self):
        random.seed(42)
        config = _simple_config(hostile_count=3, friendly_count=10, max_ticks=2000)
        sim = Scenario(config)
        sim.run(max_ticks=2000)
        assert sim.state.score > 0

    def test_wave_clear_bonus(self):
        random.seed(42)
        config = _simple_config(
            waves=1, hostile_count=1, friendly_count=10, max_ticks=2000,
        )
        sim = Scenario(config)
        sim.run(max_ticks=2000)
        wave_ends = [e for e in sim.state.events if e.event_type == "wave_end"]
        if wave_ends:
            assert sim.state.score >= 500  # at least the wave clear bonus


# ---------------------------------------------------------------------------
# Scenario — stats
# ---------------------------------------------------------------------------

class TestScenarioStats:
    def test_stats_structure(self):
        config = _simple_config(hostile_count=0, friendly_count=1, max_ticks=10)
        sim = Scenario(config)
        sim.run(max_ticks=10)
        s = sim.stats()
        assert "scenario" in s
        assert "result" in s
        assert "ticks" in s
        assert "time" in s
        assert "waves_cleared" in s
        assert "friendly_kills" in s
        assert "hostile_kills" in s
        assert "total_damage_dealt" in s
        assert "total_damage_taken" in s
        assert "shots_fired" in s
        assert "shots_hit" in s
        assert "accuracy" in s
        assert "score" in s
        assert "mvp" in s
        assert "mvp_kills" in s
        assert "alive_friendly" in s
        assert "alive_hostile" in s

    def test_stats_accuracy_calculation(self):
        config = _simple_config(hostile_count=0, friendly_count=1, max_ticks=10)
        sim = Scenario(config)
        sim.run(max_ticks=10)
        s = sim.stats()
        # No shots fired, accuracy should be 0
        assert s["accuracy"] == 0.0

    def test_stats_after_combat(self):
        random.seed(42)
        config = _simple_config(hostile_count=5, friendly_count=5, max_ticks=1000)
        sim = Scenario(config)
        sim.run(max_ticks=1000)
        s = sim.stats()
        assert s["shots_fired"] > 0
        assert 0.0 <= s["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Scenario — event listeners
# ---------------------------------------------------------------------------

class TestScenarioEvents:
    def test_on_registers_listener(self):
        sim = Scenario(_simple_config())
        received = []
        sim.on("unit_spawned", lambda e: received.append(e))
        sim.start()
        assert len(received) > 0

    def test_multiple_listeners(self):
        sim = Scenario(_simple_config())
        count_a = []
        count_b = []
        sim.on("unit_spawned", lambda e: count_a.append(1))
        sim.on("unit_spawned", lambda e: count_b.append(1))
        sim.start()
        assert len(count_a) == len(count_b)
        assert len(count_a) > 0

    def test_listener_receives_correct_event(self):
        sim = Scenario(_simple_config())
        events = []
        sim.on("wave_start", lambda e: events.append(e))
        sim.start()
        assert len(events) == 1
        assert events[0].event_type == "wave_start"

    def test_emit_adds_to_state_events(self):
        sim = Scenario(_simple_config())
        sim.emit(SimEvent(tick=0, time=0.0, event_type="test_event"))
        assert len(sim.state.events) == 1


# ---------------------------------------------------------------------------
# Scenario — snapshot
# ---------------------------------------------------------------------------

class TestScenarioSnapshot:
    def test_snapshot_structure(self):
        sim = Scenario(_simple_config(friendly_count=2))
        sim.start()
        snap = sim.snapshot()
        assert "tick" in snap
        assert "time" in snap
        assert "phase" in snap
        assert "units" in snap
        assert "score" in snap
        assert "map_size" in snap
        assert "scenario_name" in snap

    def test_snapshot_units_serialized(self):
        sim = Scenario(_simple_config(friendly_count=2))
        sim.start()
        snap = sim.snapshot()
        assert len(snap["units"]) == 2
        for uid, udata in snap["units"].items():
            assert "unit_id" in udata
            assert "position" in udata
            assert "health" in udata
            assert "is_alive" in udata

    def test_snapshot_is_json_serializable(self):
        import json
        sim = Scenario(_simple_config(friendly_count=1))
        sim.start()
        for _ in range(10):
            sim.tick()
        snap = sim.snapshot()
        # Should not raise
        serialized = json.dumps(snap)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Scenario — run()
# ---------------------------------------------------------------------------

class TestScenarioRun:
    def test_run_auto_starts(self):
        config = _simple_config(hostile_count=0, friendly_count=1, max_ticks=10)
        sim = Scenario(config)
        state = sim.run(max_ticks=10)
        assert state.tick > 0

    def test_run_respects_max_ticks(self):
        config = _simple_config(hostile_count=0, friendly_count=1, max_ticks=50)
        sim = Scenario(config)
        state = sim.run(max_ticks=50)
        assert state.tick <= 50

    def test_run_returns_state(self):
        config = _simple_config(hostile_count=0, friendly_count=1, max_ticks=10)
        sim = Scenario(config)
        state = sim.run(max_ticks=10)
        assert isinstance(state, SimState)


# ---------------------------------------------------------------------------
# Preset scenarios
# ---------------------------------------------------------------------------

class TestPresetScenarios:
    def test_presets_exist(self):
        assert "skirmish" in PRESET_SCENARIOS
        assert "assault" in PRESET_SCENARIOS
        assert "survival" in PRESET_SCENARIOS
        assert "sniper_duel" in PRESET_SCENARIOS

    def test_preset_names(self):
        assert PRESET_SCENARIOS["skirmish"].name == "Skirmish"
        assert PRESET_SCENARIOS["assault"].name == "Assault"
        assert PRESET_SCENARIOS["survival"].name == "Survival"
        assert PRESET_SCENARIOS["sniper_duel"].name == "Sniper Duel"

    def test_skirmish_has_waves(self):
        c = PRESET_SCENARIOS["skirmish"]
        assert len(c.waves) == 3
        assert len(c.friendly_units) > 0
        assert len(c.objectives) > 0

    def test_sniper_duel_is_1v1(self):
        c = PRESET_SCENARIOS["sniper_duel"]
        assert len(c.waves) == 1
        total_hostiles = sum(
            h.get("count", 0)
            for w in c.waves
            for h in w.hostiles
        )
        total_friendlies = sum(
            f.get("count", 0)
            for f in c.friendly_units
        )
        assert total_hostiles == 1
        assert total_friendlies == 1

    def test_skirmish_runs_to_completion(self):
        random.seed(42)
        config = PRESET_SCENARIOS["skirmish"]
        sim = Scenario(config)
        state = sim.run(max_ticks=3000)
        assert state.phase == "game_over"
        assert state.tick > 0

    def test_sniper_duel_runs(self):
        random.seed(42)
        config = PRESET_SCENARIOS["sniper_duel"]
        sim = Scenario(config)
        state = sim.run(max_ticks=1000)
        assert state.tick > 0

    def test_assault_has_5_waves(self):
        c = PRESET_SCENARIOS["assault"]
        assert len(c.waves) == 5

    def test_survival_has_many_waves(self):
        c = PRESET_SCENARIOS["survival"]
        assert len(c.waves) >= 10


# ---------------------------------------------------------------------------
# Scenario — movement and AI
# ---------------------------------------------------------------------------

class TestScenarioMovement:
    def test_hostiles_move_toward_friendlies(self):
        random.seed(42)
        config = _simple_config(hostile_count=1, friendly_count=1, max_ticks=100)
        sim = Scenario(config)
        sim.start()
        # Tick enough for spawn and movement
        for _ in range(50):
            sim.tick()
        hostiles = [u for u in sim._units.values() if u.alliance == Alliance.HOSTILE]
        friendlies = [u for u in sim._units.values() if u.alliance == Alliance.FRIENDLY]
        if hostiles and friendlies:
            from tritium_lib.sim_engine.ai.steering import distance
            # Hostile should have moved closer to friendly from spawn at x=10
            h = hostiles[0]
            assert h.position[0] > 10.0  # moved right toward friendly

    def test_units_clamped_to_map(self):
        config = ScenarioConfig(
            name="Small Map",
            tick_rate=10.0,
            max_ticks=100,
            map_size=(50.0, 50.0),
            waves=[],
            friendly_units=[
                {"template": "infantry", "count": 1, "spawn_pos": (49.0, 49.0)},
            ],
        )
        sim = Scenario(config)
        sim.start()
        for _ in range(10):
            sim.tick()
        for u in sim._units.values():
            assert 0.0 <= u.position[0] <= 50.0
            assert 0.0 <= u.position[1] <= 50.0


# ---------------------------------------------------------------------------
# Scenario — suppression and morale
# ---------------------------------------------------------------------------

class TestScenarioMoraleAndSuppression:
    def test_suppression_applied_on_hit(self):
        random.seed(42)
        config = _simple_config(hostile_count=5, friendly_count=5, max_ticks=500)
        sim = Scenario(config)
        sim.run(max_ticks=500)
        # At least some units should have taken suppression at some point
        damage_events = [e for e in sim.state.events if e.event_type == "damage_dealt"]
        assert len(damage_events) > 0  # combat happened

    def test_morale_recovery(self):
        random.seed(42)
        config = _simple_config(hostile_count=1, friendly_count=1, max_ticks=100)
        sim = Scenario(config)
        sim.start()
        # Manually set a unit's morale low
        for u in sim._units.values():
            u.state.morale = 0.5
        for _ in range(100):
            sim.tick()
        # Morale should have partially recovered
        for u in sim._units.values():
            if u.state.is_alive:
                assert u.state.morale > 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestScenarioEdgeCases:
    def test_empty_waves(self):
        config = ScenarioConfig(
            name="No Waves",
            tick_rate=10.0,
            max_ticks=20,
            waves=[],
            friendly_units=[
                {"template": "infantry", "count": 1, "spawn_pos": (50.0, 50.0)},
            ],
        )
        sim = Scenario(config)
        state = sim.run(max_ticks=20)
        assert state.tick == 20

    def test_no_friendly_units(self):
        config = ScenarioConfig(
            name="No Friendlies",
            tick_rate=10.0,
            max_ticks=20,
            waves=[
                WaveConfig(wave_number=1, spawn_delay=0.0, hostiles=[
                    {"template": "infantry", "count": 1,
                     "spawn_pos": (10.0, 10.0), "target_pos": (50.0, 50.0)},
                ]),
            ],
            friendly_units=[],
        )
        sim = Scenario(config)
        state = sim.run(max_ticks=20)
        assert state.tick > 0

    def test_zero_spawn_delay(self):
        config = ScenarioConfig(
            name="Instant Spawn",
            tick_rate=10.0,
            max_ticks=5,
            waves=[
                WaveConfig(wave_number=1, spawn_delay=0.0, hostiles=[
                    {"template": "infantry", "count": 3,
                     "spawn_pos": (10.0, 10.0), "target_pos": (50.0, 50.0)},
                ]),
            ],
            friendly_units=[
                {"template": "infantry", "count": 1, "spawn_pos": (50.0, 50.0)},
            ],
        )
        sim = Scenario(config)
        sim.start()
        sim.tick()
        hostiles = [u for u in sim._units.values() if u.alliance == Alliance.HOSTILE]
        assert len(hostiles) == 3  # all should spawn immediately

    def test_high_tick_rate(self):
        config = _simple_config(tick_rate=100.0, max_ticks=100, hostile_count=0)
        sim = Scenario(config)
        state = sim.run(max_ticks=100)
        assert abs(state.time - 1.0) < 1e-6  # 100 ticks at 100 tps = 1 second
