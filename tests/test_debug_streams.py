# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for game debug data streams."""

import time

import pytest

from tritium_lib.game_debug.streams import DebugFrame, DebugStream, DebugOverlay


class TestDebugStream:
    """Tests for DebugStream core behavior."""

    def test_disabled_by_default(self):
        stream = DebugStream("test_sys")
        assert stream.enabled is False

    def test_disabled_returns_none(self):
        stream = DebugStream("test_sys")
        frame = stream.begin_frame()
        assert frame is None

    def test_disabled_zero_overhead(self):
        """When disabled, begin_frame returns None and no entries are created."""
        stream = DebugStream("test_sys")
        for _ in range(100):
            frame = stream.begin_frame()
            stream.end_frame(frame)
        assert stream.latest is None
        assert len(stream.history) == 0

    def test_enabled_creates_frames(self):
        stream = DebugStream("test_sys")
        stream.enabled = True
        frame = stream.begin_frame()
        assert frame is not None
        assert frame.system == "test_sys"
        assert frame.tick == 1
        frame.entries.append({"type": "test", "value": 42})
        stream.end_frame(frame)
        assert stream.latest is frame
        assert len(stream.latest.entries) == 1
        assert stream.latest.entries[0]["value"] == 42

    def test_entries_match_active_entities(self):
        """Entries added to frame match what was explicitly appended."""
        stream = DebugStream("steering")
        stream.enabled = True
        frame = stream.begin_frame()
        for i in range(5):
            frame.entries.append({
                "type": "agent",
                "id": i,
                "pos": [float(i), float(i)],
                "speed": float(i * 0.5),
            })
        stream.end_frame(frame)

        latest = stream.latest
        assert len(latest.entries) == 5
        agents = [e for e in latest.entries if e["type"] == "agent"]
        assert len(agents) == 5
        assert agents[3]["id"] == 3

    def test_history_capped_at_max(self):
        stream = DebugStream("test_sys", max_history=5)
        stream.enabled = True
        for _ in range(10):
            frame = stream.begin_frame()
            stream.end_frame(frame)
        assert len(stream.history) == 5
        # Oldest frame should be tick 6 (first 5 were evicted)
        assert stream.history[0].tick == 6

    def test_tick_increments(self):
        stream = DebugStream("test_sys")
        stream.enabled = True
        for i in range(3):
            frame = stream.begin_frame()
            assert frame.tick == i + 1
            stream.end_frame(frame)

    def test_listeners_called_on_frame(self):
        stream = DebugStream("test_sys")
        stream.enabled = True

        received = []
        stream.on_frame(lambda f: received.append(f))

        frame = stream.begin_frame()
        frame.entries.append({"val": "hello"})
        stream.end_frame(frame)

        assert len(received) == 1
        assert received[0] is frame
        assert received[0].entries[0]["val"] == "hello"

    def test_multiple_listeners(self):
        stream = DebugStream("test_sys")
        stream.enabled = True

        count_a = []
        count_b = []
        stream.on_frame(lambda f: count_a.append(1))
        stream.on_frame(lambda f: count_b.append(1))

        frame = stream.begin_frame()
        stream.end_frame(frame)

        assert len(count_a) == 1
        assert len(count_b) == 1

    def test_end_frame_none_is_noop(self):
        stream = DebugStream("test_sys")
        stream.end_frame(None)  # should not raise
        assert stream.latest is None

    def test_frame_timestamp(self):
        stream = DebugStream("test_sys")
        stream.enabled = True
        before = time.time()
        frame = stream.begin_frame()
        after = time.time()
        assert before <= frame.timestamp <= after


class TestDebugOverlay:
    """Tests for DebugOverlay multi-stream aggregation."""

    def test_register_and_get_snapshot(self):
        overlay = DebugOverlay()
        s1 = DebugStream("physics")
        s2 = DebugStream("steering")
        overlay.register(s1)
        overlay.register(s2)

        assert "physics" in overlay.streams
        assert "steering" in overlay.streams
        assert overlay.get_snapshot() == {}

    def test_enable_all(self):
        overlay = DebugOverlay()
        s1 = DebugStream("a")
        s2 = DebugStream("b")
        overlay.register(s1)
        overlay.register(s2)

        assert not s1.enabled
        assert not s2.enabled
        overlay.enable_all()
        assert s1.enabled
        assert s2.enabled

    def test_disable_all(self):
        overlay = DebugOverlay()
        s1 = DebugStream("a")
        s1.enabled = True
        overlay.register(s1)
        overlay.disable_all()
        assert not s1.enabled

    def test_snapshot_aggregates_latest(self):
        overlay = DebugOverlay()
        s1 = DebugStream("physics")
        s2 = DebugStream("steering")
        overlay.register(s1)
        overlay.register(s2)
        overlay.enable_all()

        f1 = s1.begin_frame()
        f1.entries.append({"type": "body", "id": 0})
        s1.end_frame(f1)

        f2 = s2.begin_frame()
        f2.entries.append({"type": "agent", "id": 0})
        s2.end_frame(f2)

        snap = overlay.get_snapshot()
        assert "physics" in snap
        assert "steering" in snap
        assert snap["physics"].entries[0]["type"] == "body"
        assert snap["steering"].entries[0]["type"] == "agent"

    def test_to_dict_format(self):
        overlay = DebugOverlay()
        s1 = DebugStream("effects")
        overlay.register(s1)
        s1.enabled = True

        frame = s1.begin_frame()
        frame.entries.append({"type": "emitter", "id": 0})
        frame.entries.append({"type": "emitter", "id": 1})
        s1.end_frame(frame)

        result = overlay.to_dict()
        assert "effects" in result
        assert result["effects"]["tick"] == 1
        assert result["effects"]["entry_count"] == 2
        assert len(result["effects"]["entries"]) == 2

    def test_to_dict_excludes_empty_streams(self):
        overlay = DebugOverlay()
        s1 = DebugStream("empty")
        overlay.register(s1)
        result = overlay.to_dict()
        assert result == {}


class TestGameModuleDebugIntegration:
    """Test that actual game modules have debug streams wired in."""

    def test_steering_system_debug(self):
        import numpy as np
        from tritium_lib.game_ai.steering_np import SteeringSystem

        ss = SteeringSystem(max_agents=16)
        assert hasattr(ss, 'debug')
        assert isinstance(ss.debug, DebugStream)
        assert ss.debug.system == "steering"

        # Disabled -- no frames
        ss.add_agent((0, 0), behavior=SteeringSystem.SEEK)
        ss.targets[0] = [10, 10]
        ss.tick(0.1)
        assert ss.debug.latest is None

        # Enabled
        ss.debug.enabled = True
        ss.tick(0.1)
        assert ss.debug.latest is not None
        agents = [e for e in ss.debug.latest.entries if e["type"] == "agent"]
        assert len(agents) == 1
        assert "pos" in agents[0]
        assert "vel" in agents[0]
        assert "speed" in agents[0]

    def test_physics_world_debug(self):
        from tritium_lib.game_physics.collision import PhysicsWorld

        pw = PhysicsWorld(max_bodies=16)
        assert hasattr(pw, 'debug')
        assert isinstance(pw.debug, DebugStream)
        assert pw.debug.system == "physics"

        pw.add_body((0, 0), radius=1.0)
        pw.add_body((1.5, 0), vel=(-5, 0), radius=1.0)

        # Disabled
        events = pw.tick(0.1)
        assert pw.debug.latest is None

        # Enabled
        pw.debug.enabled = True
        events = pw.tick(0.1)
        assert pw.debug.latest is not None
        bodies = [e for e in pw.debug.latest.entries if e["type"] == "body"]
        assert len(bodies) == 2
        assert "radius" in bodies[0]
        collisions = [e for e in pw.debug.latest.entries if e["type"] == "collision"]
        # May or may not have collisions depending on positions
        for c in collisions:
            assert "impulse" in c
            assert "point" in c

    def test_effects_manager_debug(self):
        from tritium_lib.game_effects.particles import EffectsManager, explosion

        mgr = EffectsManager()
        assert hasattr(mgr, 'debug')
        assert isinstance(mgr.debug, DebugStream)
        assert mgr.debug.system == "effects"

        mgr.add(explosion((50, 50)))

        # Disabled
        mgr.tick(0.016)
        assert mgr.debug.latest is None

        # Enabled
        mgr.debug.enabled = True
        mgr.tick(0.016)
        assert mgr.debug.latest is not None
        summary = [e for e in mgr.debug.latest.entries if e["type"] == "effects_summary"]
        assert len(summary) == 1
        assert summary[0]["active_emitters"] >= 1
        assert summary[0]["total_particles"] >= 0

    def test_weapon_firer_debug(self):
        from tritium_lib.game_effects.weapons import WeaponFirer, WEAPONS

        firer = WeaponFirer(WEAPONS["pistol_9mm"], ammo=5)
        assert hasattr(firer, 'debug')
        assert isinstance(firer.debug, DebugStream)
        assert firer.debug.system == "weapons"

        firer.pull_trigger()

        # Disabled
        firer.tick(0.1)
        assert firer.debug.latest is None

        # Enabled -- need to re-pull after semi consumed
        firer.release_trigger()
        firer.debug.enabled = True
        firer.pull_trigger()
        rounds = firer.tick(1.0)  # long dt to ensure fire
        if rounds:  # should fire
            assert firer.debug.latest is not None
            fire_events = [e for e in firer.debug.latest.entries if e["type"] == "fire_event"]
            assert len(fire_events) >= 1
            assert fire_events[0]["weapon"] == "9mm Pistol"
            state = [e for e in firer.debug.latest.entries if e["type"] == "firer_state"]
            assert len(state) == 1
            assert "ammo" in state[0]

    def test_sound_event_debug(self):
        from tritium_lib.game_audio.spatial import SoundEvent

        assert hasattr(SoundEvent, 'debug')
        assert isinstance(SoundEvent.debug, DebugStream)
        assert SoundEvent.debug.system == "audio"

        se = SoundEvent("test_shot", (10, 20), volume=0.8)

        # Disabled
        SoundEvent.debug.enabled = False
        se.compute_for_listener((0, 0))
        # Can't assert latest is None because other tests may have run

        # Enabled
        SoundEvent.debug.enabled = True
        old_history_len = len(SoundEvent.debug.history)
        result = se.compute_for_listener((0, 0), listener_heading=0.5)
        assert len(SoundEvent.debug.history) > old_history_len
        latest = SoundEvent.debug.latest
        computed = [e for e in latest.entries if e["type"] == "sound_computed"]
        assert len(computed) == 1
        assert computed[0]["sound_id"] == "test_shot"
        assert "gain" in computed[0]
        assert "pan" in computed[0]

        # Clean up class-level state
        SoundEvent.debug.enabled = False

    def test_city_sim_debug(self):
        from tritium_lib.game_ai.city_sim import NeighborhoodSim

        sim = NeighborhoodSim(num_residents=5, bounds=((0, 0), (200, 200)), seed=42)
        assert hasattr(sim, 'debug')
        assert isinstance(sim.debug, DebugStream)
        assert sim.debug.system == "city_sim"

        sim.populate()

        # Disabled
        sim.tick(1.0, current_time=8.0)
        assert sim.debug.latest is None

        # Enabled
        sim.debug.enabled = True
        sim.tick(1.0, current_time=8.5)
        assert sim.debug.latest is not None
        residents = [e for e in sim.debug.latest.entries if e["type"] == "resident"]
        vehicles = [e for e in sim.debug.latest.entries if e["type"] == "vehicle"]
        assert len(residents) > 0
        assert "pos" in residents[0]
        assert "activity" in residents[0]
        assert "state" in residents[0]
