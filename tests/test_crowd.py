# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the crowd / riot simulation module."""

from __future__ import annotations

import json
import math

import pytest

from tritium_lib.sim_engine.crowd import (
    CrowdEvent,
    CrowdMember,
    CrowdMood,
    CrowdSimulator,
    CROWD_SCENARIOS,
    _EVENT_EFFECTS,
)
from tritium_lib.sim_engine.ai.steering import distance


BOUNDS = (0.0, 0.0, 100.0, 100.0)


# ── CrowdMood enum ──────────────────────────────────────────────────────────

class TestCrowdMood:
    def test_ordering(self):
        assert CrowdMood.CALM < CrowdMood.UNEASY < CrowdMood.AGITATED
        assert CrowdMood.AGITATED < CrowdMood.RIOTING < CrowdMood.PANICKED
        assert CrowdMood.PANICKED < CrowdMood.FLEEING

    def test_values(self):
        assert CrowdMood.CALM.value == 0
        assert CrowdMood.FLEEING.value == 5

    def test_all_moods_exist(self):
        names = {m.name for m in CrowdMood}
        assert names == {"CALM", "UNEASY", "AGITATED", "RIOTING", "PANICKED", "FLEEING"}


# ── CrowdMember dataclass ───────────────────────────────────────────────────

class TestCrowdMember:
    def test_defaults(self):
        m = CrowdMember(member_id="m0", position=(5.0, 10.0))
        assert m.velocity == (0.0, 0.0)
        assert m.mood == CrowdMood.CALM
        assert m.aggression == 0.0
        assert m.fear == 0.0
        assert m.has_weapon is False
        assert m.is_leader is False
        assert m.group_id == ""

    def test_custom_values(self):
        m = CrowdMember(
            member_id="m1", position=(1.0, 2.0), velocity=(0.5, 0.5),
            mood=CrowdMood.RIOTING, aggression=0.8, fear=0.1,
            has_weapon=True, is_leader=True, group_id="grp1",
        )
        assert m.has_weapon is True
        assert m.is_leader is True
        assert m.group_id == "grp1"


# ── CrowdEvent dataclass ────────────────────────────────────────────────────

class TestCrowdEvent:
    def test_creation(self):
        e = CrowdEvent(event_type="gunshot", position=(10.0, 10.0), radius=20.0, intensity=0.9, timestamp=1.0)
        assert e.event_type == "gunshot"
        assert e.radius == 20.0

    def test_all_event_types_have_effects(self):
        expected = {"gunshot", "teargas", "flashbang", "arrest", "speech", "chant",
                    "throw_object", "charge", "retreat", "stampede"}
        assert set(_EVENT_EFFECTS.keys()) == expected


# ── Spawning ─────────────────────────────────────────────────────────────────

class TestSpawn:
    def test_spawn_count(self):
        sim = CrowdSimulator(BOUNDS)
        ids = sim.spawn_crowd((50.0, 50.0), 20, radius=10.0)
        assert len(ids) == 20
        assert len(sim.members) == 20

    def test_spawn_ids_unique(self):
        sim = CrowdSimulator(BOUNDS)
        ids = sim.spawn_crowd((50.0, 50.0), 50, radius=10.0)
        assert len(set(ids)) == 50

    def test_spawn_respects_max(self):
        sim = CrowdSimulator(BOUNDS, max_members=10)
        ids = sim.spawn_crowd((50.0, 50.0), 20, radius=10.0)
        assert len(ids) == 10
        assert len(sim.members) == 10

    def test_spawn_within_radius(self):
        sim = CrowdSimulator(BOUNDS)
        center = (50.0, 50.0)
        sim.spawn_crowd(center, 100, radius=15.0)
        for m in sim.members:
            d = distance(m.position, center)
            # Allow small margin for clamping
            assert d <= 15.1, f"Member at {m.position} is {d:.1f}m from center"

    def test_spawn_in_bounds(self):
        sim = CrowdSimulator(BOUNDS)
        # Spawn near edge to test clamping
        sim.spawn_crowd((5.0, 5.0), 50, radius=20.0)
        for m in sim.members:
            assert 0.0 <= m.position[0] <= 100.0
            assert 0.0 <= m.position[1] <= 100.0

    def test_spawn_calm_mood(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 30, radius=10.0, mood=CrowdMood.CALM)
        for m in sim.members:
            assert m.mood in (CrowdMood.CALM, CrowdMood.UNEASY)  # slight randomness

    def test_spawn_agitated_mood(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 30, radius=10.0, mood=CrowdMood.AGITATED)
        agitated_count = sum(1 for m in sim.members if m.mood.value >= CrowdMood.AGITATED.value)
        assert agitated_count > 0

    def test_spawn_with_leaders(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 100, radius=10.0, leader_ratio=0.1)
        leaders = [m for m in sim.members if m.is_leader]
        assert len(leaders) >= 1

    def test_spawn_group_ids_assigned(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 10, radius=10.0)
        group_ids = {m.group_id for m in sim.members}
        assert len(group_ids) == 1
        assert "" not in group_ids

    def test_spawn_multiple_groups(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((30.0, 30.0), 10, radius=5.0)
        sim.spawn_crowd((70.0, 70.0), 10, radius=5.0)
        group_ids = {m.group_id for m in sim.members}
        assert len(group_ids) == 2


# ── Event injection ──────────────────────────────────────────────────────────

class TestEventInjection:
    def test_teargas_increases_fear(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 20, radius=5.0, mood=CrowdMood.CALM)
        initial_fear = [m.fear for m in sim.members]

        sim.inject_event(CrowdEvent(
            event_type="teargas", position=(50.0, 50.0), radius=30.0, intensity=0.8, timestamp=0.0,
        ))

        for i, m in enumerate(sim.members):
            assert m.fear >= initial_fear[i], f"Member {m.member_id} fear didn't increase"

    def test_teargas_causes_fleeing(self):
        sim = CrowdSimulator(BOUNDS)
        # Use multiple teargas events to push fear above panic threshold
        sim.spawn_crowd((50.0, 50.0), 30, radius=2.0, mood=CrowdMood.CALM)

        sim.inject_event(CrowdEvent(
            event_type="teargas", position=(50.0, 50.0), radius=20.0, intensity=1.0, timestamp=0.0,
        ))
        sim.inject_event(CrowdEvent(
            event_type="teargas", position=(50.0, 50.0), radius=20.0, intensity=1.0, timestamp=0.0,
        ))

        fearful = sum(1 for m in sim.members if m.mood.value >= CrowdMood.PANICKED.value)
        assert fearful > 0, "Teargas should cause some panic/fleeing"

    def test_gunshot_causes_panic(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 50, radius=5.0, mood=CrowdMood.CALM)

        sim.inject_event(CrowdEvent(
            event_type="gunshot", position=(50.0, 50.0), radius=40.0, intensity=1.0, timestamp=0.0,
        ))

        panicked = sum(1 for m in sim.members if m.mood.value >= CrowdMood.PANICKED.value)
        assert panicked > 0, "Gunshot should cause panic"

    def test_speech_calms(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 20, radius=5.0, mood=CrowdMood.AGITATED)

        sim.inject_event(CrowdEvent(
            event_type="speech", position=(50.0, 50.0), radius=30.0, intensity=1.0, timestamp=0.0,
        ))

        # Speech reduces fear (negative delta)
        # Some members should have reduced fear
        calm_count = sum(1 for m in sim.members if m.fear < 0.1)
        assert calm_count > 0

    def test_event_outside_radius_no_effect(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((10.0, 10.0), 10, radius=3.0, mood=CrowdMood.CALM)
        fears_before = [m.fear for m in sim.members]

        sim.inject_event(CrowdEvent(
            event_type="gunshot", position=(90.0, 90.0), radius=5.0, intensity=1.0, timestamp=0.0,
        ))

        for i, m in enumerate(sim.members):
            assert m.fear == fears_before[i], "Event outside radius should not affect members"

    def test_charge_increases_aggression(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 20, radius=5.0, mood=CrowdMood.CALM)
        initial_agg = [m.aggression for m in sim.members]

        sim.inject_event(CrowdEvent(
            event_type="charge", position=(50.0, 50.0), radius=30.0, intensity=1.0, timestamp=0.0,
        ))

        increased = sum(1 for i, m in enumerate(sim.members) if m.aggression > initial_agg[i])
        assert increased > 0


# ── Tick / simulation ────────────────────────────────────────────────────────

class TestTick:
    def test_tick_moves_members(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 10, radius=5.0)
        positions_before = [m.position for m in sim.members]

        for _ in range(10):
            sim.tick(0.1)

        moved = sum(1 for i, m in enumerate(sim.members) if m.position != positions_before[i])
        assert moved > 0, "At least some members should move after tick"

    def test_tick_keeps_in_bounds(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((5.0, 5.0), 20, radius=3.0, mood=CrowdMood.FLEEING)

        for _ in range(50):
            sim.tick(0.2)

        for m in sim.members:
            assert 0.0 <= m.position[0] <= 100.0, f"Out of bounds x: {m.position[0]}"
            assert 0.0 <= m.position[1] <= 100.0, f"Out of bounds y: {m.position[1]}"

    def test_time_advances(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 5, radius=5.0)
        sim.tick(0.5)
        sim.tick(0.3)
        assert abs(sim._time - 0.8) < 1e-6


# ── Mood propagation ────────────────────────────────────────────────────────

class TestMoodPropagation:
    def test_agitated_leader_spreads_aggression(self):
        sim = CrowdSimulator(BOUNDS)
        # Create a cluster: one aggressive leader + calm members nearby
        leader = CrowdMember(
            member_id="leader", position=(50.0, 50.0), aggression=0.9,
            fear=0.0, is_leader=True, group_id="g1", mood=CrowdMood.RIOTING,
        )
        sim.members.append(leader)
        for i in range(5):
            m = CrowdMember(
                member_id=f"follower_{i}",
                position=(50.0 + (i + 1) * 0.5, 50.0),
                aggression=0.0, fear=0.0, group_id="g1",
            )
            sim.members.append(m)

        # Tick several times to propagate
        for _ in range(20):
            sim.tick(0.5)

        followers = [m for m in sim.members if not m.is_leader]
        avg_agg = sum(m.aggression for m in followers) / len(followers)
        assert avg_agg > 0.01, f"Leader should propagate aggression, got avg {avg_agg}"

    def test_leaders_have_wider_influence(self):
        """Leaders influence at 3x the normal radius."""
        sim = CrowdSimulator(BOUNDS)
        # Leader at center
        leader = CrowdMember(
            member_id="leader", position=(50.0, 50.0), aggression=0.9,
            is_leader=True, group_id="g1", mood=CrowdMood.RIOTING,
        )
        # Member at 10m — beyond normal 5m but within leader's 15m
        far_member = CrowdMember(
            member_id="far", position=(60.0, 50.0), aggression=0.0, group_id="g1",
        )
        sim.members = [leader, far_member]

        for _ in range(30):
            sim.tick(0.5)

        assert far_member.aggression > 0.01, "Leader should influence at extended range"


# ── Escalation ───────────────────────────────────────────────────────────────

class TestEscalation:
    def test_30_percent_agitated_cascade(self):
        sim = CrowdSimulator(BOUNDS)
        # 35% agitated, rest uneasy
        for i in range(35):
            sim.members.append(CrowdMember(
                member_id=f"agg_{i}", position=(50.0 + i * 0.2, 50.0),
                aggression=0.4, mood=CrowdMood.AGITATED, group_id="g1",
            ))
        for i in range(65):
            sim.members.append(CrowdMember(
                member_id=f"calm_{i}", position=(50.0 + i * 0.2, 52.0),
                aggression=0.12, mood=CrowdMood.UNEASY, group_id="g1",
            ))

        sim._check_escalation()
        # Some UNEASY should get boosted aggression
        uneasy_agg = [m.aggression for m in sim.members if m.member_id.startswith("calm_")]
        assert any(a > 0.12 for a in uneasy_agg), "30% cascade should boost uneasy members"

    def test_50_percent_rioting_stampede_risk(self):
        sim = CrowdSimulator(BOUNDS)
        # 55% rioting, rest calm
        for i in range(55):
            sim.members.append(CrowdMember(
                member_id=f"riot_{i}", position=(50.0 + i * 0.2, 50.0),
                aggression=0.7, mood=CrowdMood.RIOTING, group_id="g1",
            ))
        for i in range(45):
            sim.members.append(CrowdMember(
                member_id=f"calm_{i}", position=(50.0 + i * 0.2, 52.0),
                aggression=0.0, fear=0.0, mood=CrowdMood.CALM, group_id="g1",
            ))

        sim._check_escalation()
        calm_fears = [m.fear for m in sim.members if m.member_id.startswith("calm_")]
        assert any(f > 0.0 for f in calm_fears), "50% rioting should inject fear into calm"


# ── Event decay ──────────────────────────────────────────────────────────────

class TestEventDecay:
    def test_events_decay_over_time(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 5, radius=5.0)
        sim.inject_event(CrowdEvent(
            event_type="teargas", position=(50.0, 50.0), radius=20.0, intensity=0.5, timestamp=0.0,
        ))
        assert len(sim.events) == 1
        initial_intensity = sim.events[0].intensity

        sim.tick(1.0)
        if sim.events:
            assert sim.events[0].intensity < initial_intensity

    def test_events_removed_when_fully_decayed(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 5, radius=5.0)
        sim.inject_event(CrowdEvent(
            event_type="flashbang", position=(50.0, 50.0), radius=10.0, intensity=0.1, timestamp=0.0,
        ))

        # Tick enough for full decay
        for _ in range(20):
            sim.tick(1.0)

        assert len(sim.events) == 0, "Fully decayed events should be removed"


# ── to_three_js output ───────────────────────────────────────────────────────

class TestThreeJsOutput:
    def test_structure(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 10, radius=5.0)
        sim.inject_event(CrowdEvent(
            event_type="chant", position=(50.0, 50.0), radius=15.0, intensity=0.5, timestamp=0.0,
        ))

        out = sim.to_three_js()
        assert "members" in out
        assert "events" in out
        assert "hotspots" in out
        assert "stats" in out

    def test_member_fields(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 5, radius=5.0)
        out = sim.to_three_js()

        for m in out["members"]:
            assert "id" in m
            assert "x" in m
            assert "y" in m
            assert "mood" in m
            assert "aggression" in m
            assert "heading" in m
            assert isinstance(m["mood"], str)

    def test_event_fields(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 5, radius=5.0)
        sim.inject_event(CrowdEvent(
            event_type="teargas", position=(30.0, 30.0), radius=10.0, intensity=0.7, timestamp=0.0,
        ))
        out = sim.to_three_js()

        assert len(out["events"]) >= 1
        e = out["events"][0]
        assert e["type"] == "teargas"
        assert "x" in e
        assert "y" in e
        assert "radius" in e
        assert "intensity" in e
        assert "age" in e

    def test_stats_fields(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 20, radius=5.0)
        out = sim.to_three_js()

        stats = out["stats"]
        assert stats["total"] == 20
        assert "calm" in stats
        assert "agitated" in stats
        assert "rioting" in stats
        assert "fleeing" in stats
        assert sum(stats[k] for k in ["calm", "uneasy", "agitated", "rioting", "panicked", "fleeing"]) == 20

    def test_json_serializable(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 30, radius=10.0, mood=CrowdMood.AGITATED)
        sim.inject_event(CrowdEvent(
            event_type="charge", position=(50.0, 50.0), radius=20.0, intensity=0.8, timestamp=0.0,
        ))
        sim.tick(0.5)

        out = sim.to_three_js()
        # Must not raise
        serialized = json.dumps(out)
        assert len(serialized) > 0

    def test_snapshot_json_serializable(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 15, radius=5.0)
        sim.tick(0.1)
        snap = sim.snapshot()
        serialized = json.dumps(snap)
        assert len(serialized) > 0


# ── Hotspots ─────────────────────────────────────────────────────────────────

class TestHotspots:
    def test_no_hotspots_when_calm(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 20, radius=5.0, mood=CrowdMood.CALM)
        hotspots = sim.get_hotspots()
        # Calm crowd should have low aggression, few or no hotspots
        for h in hotspots:
            assert h["intensity"] < 0.5

    def test_hotspots_for_rioting_cluster(self):
        sim = CrowdSimulator(BOUNDS)
        # Tight cluster of high-aggression members
        for i in range(20):
            sim.members.append(CrowdMember(
                member_id=f"r_{i}", position=(50.0 + i * 0.3, 50.0),
                aggression=0.8, mood=CrowdMood.RIOTING, group_id="g1",
            ))
        hotspots = sim.get_hotspots()
        assert len(hotspots) >= 1
        assert any(h["intensity"] >= 0.5 for h in hotspots)

    def test_hotspot_fields(self):
        sim = CrowdSimulator(BOUNDS)
        for i in range(10):
            sim.members.append(CrowdMember(
                member_id=f"h_{i}", position=(50.0 + i * 0.2, 50.0),
                aggression=0.7, mood=CrowdMood.RIOTING, group_id="g1",
            ))
        hotspots = sim.get_hotspots()
        if hotspots:
            h = hotspots[0]
            assert "x" in h
            assert "y" in h
            assert "radius" in h
            assert "intensity" in h


# ── Scenario presets ─────────────────────────────────────────────────────────

class TestScenarios:
    @pytest.mark.parametrize("name", list(CROWD_SCENARIOS.keys()))
    def test_scenario_builds(self, name: str):
        factory = CROWD_SCENARIOS[name]
        sim = factory(BOUNDS)
        assert isinstance(sim, CrowdSimulator)
        assert len(sim.members) > 0

    @pytest.mark.parametrize("name", list(CROWD_SCENARIOS.keys()))
    def test_scenario_ticks_without_crash(self, name: str):
        factory = CROWD_SCENARIOS[name]
        sim = factory(BOUNDS)
        for _ in range(10):
            sim.tick(0.1)
        # Should not raise

    @pytest.mark.parametrize("name", list(CROWD_SCENARIOS.keys()))
    def test_scenario_produces_valid_threejs(self, name: str):
        factory = CROWD_SCENARIOS[name]
        sim = factory(BOUNDS)
        sim.tick(0.5)
        out = sim.to_three_js()
        assert "members" in out
        assert "stats" in out
        # JSON-serializable
        json.dumps(out)

    def test_peaceful_protest_starts_calm(self):
        sim = CROWD_SCENARIOS["peaceful_protest"](BOUNDS)
        calm = sum(1 for m in sim.members if m.mood == CrowdMood.CALM)
        assert calm > len(sim.members) * 0.5

    def test_riot_has_rioting_members(self):
        sim = CROWD_SCENARIOS["riot"](BOUNDS)
        rioting = sum(1 for m in sim.members if m.mood.value >= CrowdMood.AGITATED.value)
        assert rioting > 0

    def test_stampede_has_panicked(self):
        sim = CROWD_SCENARIOS["stampede"](BOUNDS)
        panicked = sum(1 for m in sim.members if m.mood.value >= CrowdMood.PANICKED.value)
        assert panicked > 0

    def test_standoff_has_agitated(self):
        sim = CROWD_SCENARIOS["standoff"](BOUNDS)
        agitated = sum(1 for m in sim.members if m.mood.value >= CrowdMood.AGITATED.value)
        assert agitated > 0


# ── Overall mood ─────────────────────────────────────────────────────────────

class TestOverallMood:
    def test_calm_crowd_overall_calm(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 50, radius=10.0, mood=CrowdMood.CALM)
        sim.tick(0.1)
        assert sim.overall_mood in (CrowdMood.CALM, CrowdMood.UNEASY)

    def test_empty_sim_is_calm(self):
        sim = CrowdSimulator(BOUNDS)
        sim.tick(0.1)
        assert sim.overall_mood == CrowdMood.CALM


# ── Snapshot ─────────────────────────────────────────────────────────────────

class TestSnapshot:
    def test_snapshot_has_all_fields(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 10, radius=5.0)
        sim.tick(0.5)
        snap = sim.snapshot()
        assert "time" in snap
        assert "bounds" in snap
        assert "members" in snap
        assert "events" in snap
        assert "stats" in snap
        assert "overall_mood" in snap
        assert snap["time"] == pytest.approx(0.5)

    def test_snapshot_member_format(self):
        sim = CrowdSimulator(BOUNDS)
        sim.spawn_crowd((50.0, 50.0), 5, radius=5.0)
        snap = sim.snapshot()
        m = snap["members"][0]
        assert "member_id" in m
        assert "position" in m
        assert "velocity" in m
        assert "mood" in m
        assert "aggression" in m
        assert "fear" in m
        assert "is_leader" in m
        assert "group_id" in m
