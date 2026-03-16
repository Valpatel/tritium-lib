# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the sim engine global event bus and timeline system."""

import math

import pytest

from tritium_lib.sim_engine.event_bus import (
    EventFilter,
    EventListener,
    SimEvent,
    SimEventBus,
    SimEventType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: SimEventType = SimEventType.UNIT_SPAWNED,
    tick: int = 0,
    time: float = 0.0,
    source_id: str = "",
    target_id: str = "",
    position: tuple[float, float] | None = None,
    data: dict | None = None,
    priority: int = 5,
) -> SimEvent:
    return SimEvent(
        event_type=event_type,
        tick=tick,
        time=time,
        source_id=source_id,
        target_id=target_id,
        position=position,
        data=data if data is not None else {},
        priority=priority,
    )


# ===========================================================================
# SimEventType enum
# ===========================================================================


class TestSimEventType:
    """Verify the enum is complete and usable."""

    def test_unit_events_exist(self):
        assert SimEventType.UNIT_SPAWNED
        assert SimEventType.UNIT_KILLED
        assert SimEventType.UNIT_DAMAGED
        assert SimEventType.UNIT_HEALED
        assert SimEventType.UNIT_MOVED

    def test_combat_events_exist(self):
        assert SimEventType.SHOT_FIRED
        assert SimEventType.PROJECTILE_IMPACT
        assert SimEventType.EXPLOSION

    def test_vehicle_events_exist(self):
        assert SimEventType.VEHICLE_SPAWNED
        assert SimEventType.VEHICLE_DESTROYED
        assert SimEventType.VEHICLE_BOARDED

    def test_structure_events_exist(self):
        assert SimEventType.STRUCTURE_DAMAGED
        assert SimEventType.STRUCTURE_DESTROYED
        assert SimEventType.FIRE_STARTED
        assert SimEventType.FIRE_EXTINGUISHED

    def test_objective_events_exist(self):
        assert SimEventType.OBJECTIVE_UPDATED
        assert SimEventType.OBJECTIVE_COMPLETED
        assert SimEventType.OBJECTIVE_FAILED

    def test_flow_events_exist(self):
        assert SimEventType.WAVE_STARTED
        assert SimEventType.WAVE_COMPLETED
        assert SimEventType.GAME_OVER

    def test_detection_events_exist(self):
        assert SimEventType.DETECTION_NEW
        assert SimEventType.DETECTION_LOST

    def test_radio_events_exist(self):
        assert SimEventType.RADIO_TRANSMITTED
        assert SimEventType.RADIO_INTERCEPTED
        assert SimEventType.RADIO_JAMMED

    def test_supply_events_exist(self):
        assert SimEventType.SUPPLY_LOW
        assert SimEventType.SUPPLY_DEPLETED
        assert SimEventType.RESUPPLY

    def test_trap_events_exist(self):
        assert SimEventType.MINE_TRIGGERED
        assert SimEventType.IED_DETONATED
        assert SimEventType.TRAP_DETECTED

    def test_medical_events_exist(self):
        assert SimEventType.CASUALTY
        assert SimEventType.EVAC_REQUESTED
        assert SimEventType.TREATMENT_COMPLETE

    def test_scoring_events_exist(self):
        assert SimEventType.ACHIEVEMENT_EARNED
        assert SimEventType.SCORE_UPDATED

    def test_environment_events_exist(self):
        assert SimEventType.WEATHER_CHANGED
        assert SimEventType.TIME_ADVANCED

    def test_diplomacy_events_exist(self):
        assert SimEventType.FACTION_RELATION_CHANGED
        assert SimEventType.CEASEFIRE
        assert SimEventType.WAR_DECLARED

    def test_civilian_events_exist(self):
        assert SimEventType.CROWD_ESCALATION
        assert SimEventType.CIVILIAN_CASUALTY

    def test_ability_events_exist(self):
        assert SimEventType.ABILITY_ACTIVATED
        assert SimEventType.EFFECT_APPLIED
        assert SimEventType.EFFECT_EXPIRED

    def test_all_values_unique(self):
        values = [e.value for e in SimEventType]
        assert len(values) == len(set(values))

    def test_total_event_count(self):
        # 46 event types as specified
        assert len(SimEventType) >= 46


# ===========================================================================
# SimEvent dataclass
# ===========================================================================


class TestSimEvent:
    """Verify SimEvent fields and defaults."""

    def test_required_fields(self):
        e = SimEvent(event_type=SimEventType.SHOT_FIRED, tick=10, time=5.0)
        assert e.event_type == SimEventType.SHOT_FIRED
        assert e.tick == 10
        assert e.time == 5.0

    def test_defaults(self):
        e = SimEvent(event_type=SimEventType.UNIT_MOVED, tick=0, time=0.0)
        assert e.source_id == ""
        assert e.target_id == ""
        assert e.position is None
        assert e.data == {}
        assert e.priority == 5

    def test_full_construction(self):
        e = SimEvent(
            event_type=SimEventType.UNIT_DAMAGED,
            tick=42,
            time=21.0,
            source_id="shooter_01",
            target_id="target_02",
            position=(100.0, 200.0),
            data={"damage": 30, "weapon": "rifle"},
            priority=2,
        )
        assert e.source_id == "shooter_01"
        assert e.target_id == "target_02"
        assert e.position == (100.0, 200.0)
        assert e.data["damage"] == 30
        assert e.priority == 2

    def test_data_dict_independence(self):
        """Each event gets its own data dict."""
        e1 = _make_event()
        e2 = _make_event()
        e1.data["foo"] = "bar"
        assert "foo" not in e2.data


# ===========================================================================
# SimEventBus — subscribe/publish
# ===========================================================================


class TestSimEventBusSubscribe:
    """Test on() / off() / emit() subscription mechanics."""

    def test_on_receives_matching_events(self):
        bus = SimEventBus()
        received: list[SimEvent] = []
        bus.on(SimEventType.UNIT_SPAWNED, received.append)
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        assert len(received) == 1

    def test_on_ignores_non_matching(self):
        bus = SimEventBus()
        received: list[SimEvent] = []
        bus.on(SimEventType.UNIT_SPAWNED, received.append)
        bus.emit(_make_event(SimEventType.UNIT_KILLED))
        assert len(received) == 0

    def test_multiple_listeners_same_type(self):
        bus = SimEventBus()
        r1: list[SimEvent] = []
        r2: list[SimEvent] = []
        bus.on(SimEventType.EXPLOSION, r1.append)
        bus.on(SimEventType.EXPLOSION, r2.append)
        bus.emit(_make_event(SimEventType.EXPLOSION))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_multiple_types(self):
        bus = SimEventBus()
        spawns: list[SimEvent] = []
        kills: list[SimEvent] = []
        bus.on(SimEventType.UNIT_SPAWNED, spawns.append)
        bus.on(SimEventType.UNIT_KILLED, kills.append)
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        bus.emit(_make_event(SimEventType.UNIT_KILLED))
        assert len(spawns) == 1
        assert len(kills) == 1

    def test_off_unsubscribes(self):
        bus = SimEventBus()
        received: list[SimEvent] = []
        cb = received.append
        bus.on(SimEventType.SHOT_FIRED, cb)
        bus.emit(_make_event(SimEventType.SHOT_FIRED))
        assert len(received) == 1
        bus.off(SimEventType.SHOT_FIRED, cb)
        bus.emit(_make_event(SimEventType.SHOT_FIRED))
        assert len(received) == 1

    def test_off_nonexistent_is_safe(self):
        bus = SimEventBus()
        bus.off(SimEventType.UNIT_MOVED, lambda e: None)  # no crash

    def test_off_only_removes_specific_callback(self):
        bus = SimEventBus()
        r1: list[SimEvent] = []
        r2: list[SimEvent] = []
        cb1 = r1.append
        cb2 = r2.append
        bus.on(SimEventType.EXPLOSION, cb1)
        bus.on(SimEventType.EXPLOSION, cb2)
        bus.off(SimEventType.EXPLOSION, cb1)
        bus.emit(_make_event(SimEventType.EXPLOSION))
        assert len(r1) == 0
        assert len(r2) == 1

    def test_bad_listener_does_not_break_bus(self):
        bus = SimEventBus()
        received: list[SimEvent] = []

        def bad(e: SimEvent) -> None:
            raise RuntimeError("boom")

        bus.on(SimEventType.UNIT_DAMAGED, bad)
        bus.on(SimEventType.UNIT_DAMAGED, received.append)
        bus.emit(_make_event(SimEventType.UNIT_DAMAGED))
        assert len(received) == 1


# ===========================================================================
# SimEventBus — global listeners
# ===========================================================================


class TestSimEventBusGlobal:
    """Test on_any() / off_any() global listener mechanics."""

    def test_on_any_receives_all(self):
        bus = SimEventBus()
        received: list[SimEvent] = []
        bus.on_any(received.append)
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        bus.emit(_make_event(SimEventType.EXPLOSION))
        bus.emit(_make_event(SimEventType.WEATHER_CHANGED))
        assert len(received) == 3

    def test_off_any_stops_delivery(self):
        bus = SimEventBus()
        received: list[SimEvent] = []
        cb = received.append
        bus.on_any(cb)
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        bus.off_any(cb)
        bus.emit(_make_event(SimEventType.UNIT_KILLED))
        assert len(received) == 1

    def test_global_and_typed_both_fire(self):
        bus = SimEventBus()
        global_r: list[SimEvent] = []
        typed_r: list[SimEvent] = []
        bus.on_any(global_r.append)
        bus.on(SimEventType.SHOT_FIRED, typed_r.append)
        bus.emit(_make_event(SimEventType.SHOT_FIRED))
        assert len(global_r) == 1
        assert len(typed_r) == 1

    def test_bad_global_listener_does_not_break_bus(self):
        bus = SimEventBus()
        received: list[SimEvent] = []

        def bad(e: SimEvent) -> None:
            raise ValueError("kaboom")

        bus.on_any(bad)
        bus.on_any(received.append)
        bus.emit(_make_event(SimEventType.GAME_OVER))
        assert len(received) == 1


# ===========================================================================
# SimEventBus — emit_many
# ===========================================================================


class TestSimEventBusEmitMany:
    """Test batch publishing."""

    def test_emit_many_delivers_all(self):
        bus = SimEventBus()
        received: list[SimEvent] = []
        bus.on_any(received.append)
        events = [
            _make_event(SimEventType.WAVE_STARTED, tick=1),
            _make_event(SimEventType.UNIT_SPAWNED, tick=1),
            _make_event(SimEventType.UNIT_SPAWNED, tick=1),
        ]
        bus.emit_many(events)
        assert len(received) == 3

    def test_emit_many_preserves_order(self):
        bus = SimEventBus()
        received: list[SimEvent] = []
        bus.on_any(received.append)
        events = [
            _make_event(SimEventType.WAVE_STARTED, tick=i) for i in range(5)
        ]
        bus.emit_many(events)
        assert [e.tick for e in received] == [0, 1, 2, 3, 4]

    def test_emit_many_empty_list(self):
        bus = SimEventBus()
        bus.emit_many([])  # no crash


# ===========================================================================
# SimEventBus — log / timeline
# ===========================================================================


class TestSimEventBusLog:
    """Test event log, timeline, and query methods."""

    def test_events_are_logged(self):
        bus = SimEventBus()
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED, tick=1))
        bus.emit(_make_event(SimEventType.UNIT_KILLED, tick=2))
        assert len(bus.get_log()) == 2

    def test_get_log_filter_by_type(self):
        bus = SimEventBus()
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED, tick=1))
        bus.emit(_make_event(SimEventType.UNIT_KILLED, tick=2))
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED, tick=3))
        result = bus.get_log(event_type=SimEventType.UNIT_SPAWNED)
        assert len(result) == 2
        assert all(e.event_type == SimEventType.UNIT_SPAWNED for e in result)

    def test_get_log_filter_by_tick(self):
        bus = SimEventBus()
        for i in range(10):
            bus.emit(_make_event(tick=i))
        result = bus.get_log(since_tick=7)
        assert len(result) == 3
        assert result[0].tick == 7

    def test_get_log_combined_filter(self):
        bus = SimEventBus()
        bus.emit(_make_event(SimEventType.SHOT_FIRED, tick=1))
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED, tick=2))
        bus.emit(_make_event(SimEventType.SHOT_FIRED, tick=3))
        bus.emit(_make_event(SimEventType.SHOT_FIRED, tick=5))
        result = bus.get_log(event_type=SimEventType.SHOT_FIRED, since_tick=3)
        assert len(result) == 2

    def test_get_log_limit(self):
        bus = SimEventBus()
        for i in range(50):
            bus.emit(_make_event(tick=i))
        result = bus.get_log(limit=10)
        assert len(result) == 10
        # Should be the LAST 10
        assert result[0].tick == 40

    def test_get_timeline(self):
        bus = SimEventBus()
        for i in range(20):
            bus.emit(_make_event(tick=i))
        result = bus.get_timeline(5, 10)
        assert len(result) == 6
        assert result[0].tick == 5
        assert result[-1].tick == 10

    def test_get_timeline_empty_range(self):
        bus = SimEventBus()
        bus.emit(_make_event(tick=1))
        result = bus.get_timeline(100, 200)
        assert result == []

    def test_clear_log(self):
        bus = SimEventBus()
        for i in range(10):
            bus.emit(_make_event(tick=i))
        assert len(bus.get_log(limit=10000)) == 10
        bus.clear_log()
        assert len(bus.get_log()) == 0

    def test_log_ring_buffer_trim(self):
        bus = SimEventBus(max_log=100)
        for i in range(200):
            bus.emit(_make_event(tick=i))
        log = bus.get_log(limit=10000)
        assert len(log) <= 100
        # Most recent events should still be present
        assert log[-1].tick == 199


# ===========================================================================
# SimEventBus — stats
# ===========================================================================


class TestSimEventBusStats:
    """Test the stats() summary."""

    def test_stats_empty(self):
        bus = SimEventBus()
        assert bus.stats() == {}

    def test_stats_counts_per_type(self):
        bus = SimEventBus()
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        bus.emit(_make_event(SimEventType.UNIT_KILLED))
        s = bus.stats()
        assert s["UNIT_SPAWNED"] == 2
        assert s["UNIT_KILLED"] == 1

    def test_stats_keys_are_enum_names(self):
        bus = SimEventBus()
        bus.emit(_make_event(SimEventType.EXPLOSION))
        s = bus.stats()
        assert "EXPLOSION" in s


# ===========================================================================
# SimEventBus — to_three_js
# ===========================================================================


class TestSimEventBusThreeJs:
    """Test HUD / Three.js export."""

    def test_to_three_js_empty(self):
        bus = SimEventBus()
        assert bus.to_three_js() == []

    def test_to_three_js_basic_fields(self):
        bus = SimEventBus()
        bus.emit(_make_event(
            SimEventType.SHOT_FIRED, tick=10, time=5.5,
            source_id="unit_01", target_id="unit_02", priority=3,
        ))
        result = bus.to_three_js(last_n=1)
        assert len(result) == 1
        entry = result[0]
        assert entry["type"] == "SHOT_FIRED"
        assert entry["tick"] == 10
        assert entry["time"] == 5.5
        assert entry["source"] == "unit_01"
        assert entry["target"] == "unit_02"
        assert entry["priority"] == 3

    def test_to_three_js_includes_position(self):
        bus = SimEventBus()
        bus.emit(_make_event(position=(12.345, 67.891)))
        result = bus.to_three_js(last_n=1)
        assert result[0]["x"] == 12.35  # rounded to 2 dp
        assert result[0]["y"] == 67.89

    def test_to_three_js_excludes_position_when_none(self):
        bus = SimEventBus()
        bus.emit(_make_event())
        result = bus.to_three_js(last_n=1)
        assert "x" not in result[0]
        assert "y" not in result[0]

    def test_to_three_js_includes_data_when_present(self):
        bus = SimEventBus()
        bus.emit(_make_event(data={"weapon": "rpg"}))
        result = bus.to_three_js(last_n=1)
        assert result[0]["data"] == {"weapon": "rpg"}

    def test_to_three_js_excludes_data_when_empty(self):
        bus = SimEventBus()
        bus.emit(_make_event(data={}))
        result = bus.to_three_js(last_n=1)
        assert "data" not in result[0]

    def test_to_three_js_last_n_limit(self):
        bus = SimEventBus()
        for i in range(50):
            bus.emit(_make_event(tick=i))
        result = bus.to_three_js(last_n=5)
        assert len(result) == 5
        assert result[0]["tick"] == 45

    def test_to_three_js_last_n_zero(self):
        bus = SimEventBus()
        bus.emit(_make_event())
        assert bus.to_three_js(last_n=0) == []


# ===========================================================================
# EventFilter — filter_by_type
# ===========================================================================


class TestEventFilterByType:
    """Test EventFilter.filter_by_type."""

    def test_filter_single_type(self):
        events = [
            _make_event(SimEventType.UNIT_SPAWNED),
            _make_event(SimEventType.UNIT_KILLED),
            _make_event(SimEventType.UNIT_SPAWNED),
        ]
        result = EventFilter.filter_by_type(events, [SimEventType.UNIT_SPAWNED])
        assert len(result) == 2

    def test_filter_multiple_types(self):
        events = [
            _make_event(SimEventType.UNIT_SPAWNED),
            _make_event(SimEventType.UNIT_KILLED),
            _make_event(SimEventType.EXPLOSION),
        ]
        types = {SimEventType.UNIT_SPAWNED, SimEventType.EXPLOSION}
        result = EventFilter.filter_by_type(events, types)
        assert len(result) == 2

    def test_filter_no_match(self):
        events = [_make_event(SimEventType.UNIT_MOVED)]
        result = EventFilter.filter_by_type(events, [SimEventType.GAME_OVER])
        assert result == []

    def test_filter_empty_input(self):
        assert EventFilter.filter_by_type([], [SimEventType.UNIT_SPAWNED]) == []


# ===========================================================================
# EventFilter — filter_by_area
# ===========================================================================


class TestEventFilterByArea:
    """Test EventFilter.filter_by_area."""

    def test_events_within_radius(self):
        events = [
            _make_event(position=(10.0, 10.0)),
            _make_event(position=(100.0, 100.0)),
            _make_event(position=(11.0, 10.0)),
        ]
        result = EventFilter.filter_by_area(events, center=(10.0, 10.0), radius=5.0)
        assert len(result) == 2

    def test_excludes_events_without_position(self):
        events = [
            _make_event(position=(0.0, 0.0)),
            _make_event(position=None),
        ]
        result = EventFilter.filter_by_area(events, center=(0.0, 0.0), radius=100.0)
        assert len(result) == 1

    def test_exact_boundary(self):
        events = [_make_event(position=(5.0, 0.0))]
        result = EventFilter.filter_by_area(events, center=(0.0, 0.0), radius=5.0)
        assert len(result) == 1  # on the boundary = included

    def test_just_outside(self):
        events = [_make_event(position=(5.01, 0.0))]
        result = EventFilter.filter_by_area(events, center=(0.0, 0.0), radius=5.0)
        assert len(result) == 0

    def test_empty_input(self):
        assert EventFilter.filter_by_area([], (0, 0), 10) == []


# ===========================================================================
# EventFilter — filter_by_alliance
# ===========================================================================


class TestEventFilterByAlliance:
    """Test EventFilter.filter_by_alliance."""

    def test_match_alliance(self):
        events = [
            _make_event(data={"alliance": "friendly"}),
            _make_event(data={"alliance": "hostile"}),
            _make_event(data={"alliance": "friendly"}),
        ]
        result = EventFilter.filter_by_alliance(events, "friendly")
        assert len(result) == 2

    def test_no_alliance_key_excluded(self):
        events = [_make_event(data={"damage": 10})]
        result = EventFilter.filter_by_alliance(events, "friendly")
        assert result == []

    def test_empty_input(self):
        assert EventFilter.filter_by_alliance([], "hostile") == []


# ===========================================================================
# EventFilter — filter_by_priority
# ===========================================================================


class TestEventFilterByPriority:
    """Test EventFilter.filter_by_priority."""

    def test_filter_critical_only(self):
        events = [
            _make_event(priority=1),
            _make_event(priority=5),
            _make_event(priority=10),
        ]
        result = EventFilter.filter_by_priority(events, max_priority=1)
        assert len(result) == 1

    def test_filter_medium(self):
        events = [
            _make_event(priority=1),
            _make_event(priority=3),
            _make_event(priority=5),
            _make_event(priority=7),
        ]
        result = EventFilter.filter_by_priority(events, max_priority=5)
        assert len(result) == 3

    def test_filter_all(self):
        events = [_make_event(priority=p) for p in range(1, 11)]
        result = EventFilter.filter_by_priority(events, max_priority=10)
        assert len(result) == 10

    def test_empty_input(self):
        assert EventFilter.filter_by_priority([], max_priority=5) == []


# ===========================================================================
# Integration: bus + filter pipeline
# ===========================================================================


class TestIntegration:
    """End-to-end tests combining bus and filters."""

    def test_log_then_filter_pipeline(self):
        bus = SimEventBus()
        # Simulate a small battle
        bus.emit(_make_event(SimEventType.WAVE_STARTED, tick=0, priority=2))
        bus.emit(_make_event(
            SimEventType.UNIT_SPAWNED, tick=1, source_id="inf_01",
            position=(50, 50), data={"alliance": "friendly"}, priority=3,
        ))
        bus.emit(_make_event(
            SimEventType.UNIT_SPAWNED, tick=1, source_id="inf_02",
            position=(200, 200), data={"alliance": "hostile"}, priority=3,
        ))
        bus.emit(_make_event(
            SimEventType.SHOT_FIRED, tick=5, source_id="inf_01",
            target_id="inf_02", position=(55, 55), priority=4,
        ))
        bus.emit(_make_event(
            SimEventType.UNIT_KILLED, tick=6, source_id="inf_01",
            target_id="inf_02", position=(200, 200), priority=1,
        ))
        bus.emit(_make_event(SimEventType.WAVE_COMPLETED, tick=10, priority=2))

        # All events logged
        assert len(bus.get_log(limit=10000)) == 6

        # Timeline slice
        timeline = bus.get_timeline(0, 5)
        assert len(timeline) == 4

        # Filter by area around (50, 50) radius 20
        nearby = EventFilter.filter_by_area(bus.get_log(limit=10000), (50, 50), 20.0)
        assert len(nearby) == 2  # spawn + shot

        # Filter by alliance
        friendly = EventFilter.filter_by_alliance(
            bus.get_log(limit=10000), "friendly"
        )
        assert len(friendly) == 1

        # Filter critical
        critical = EventFilter.filter_by_priority(bus.get_log(limit=10000), 2)
        assert len(critical) == 3  # wave_started, unit_killed, wave_completed

        # Stats
        s = bus.stats()
        assert s["UNIT_SPAWNED"] == 2
        assert s["SHOT_FIRED"] == 1

    def test_hud_export_after_battle(self):
        bus = SimEventBus()
        for i in range(30):
            bus.emit(_make_event(tick=i, time=float(i) * 0.5))
        hud = bus.to_three_js(last_n=10)
        assert len(hud) == 10
        assert hud[0]["tick"] == 20

    def test_high_volume(self):
        """Bus handles 10k events without issues."""
        bus = SimEventBus(max_log=5000)
        received = [0]
        bus.on(SimEventType.UNIT_MOVED, lambda e: received.__setitem__(0, received[0] + 1))
        for i in range(10_000):
            bus.emit(_make_event(SimEventType.UNIT_MOVED, tick=i))
        assert received[0] == 10_000
        # Log should be trimmed
        log = bus.get_log(limit=100_000)
        assert len(log) <= 5000

    def test_chained_filters(self):
        events = [
            _make_event(
                SimEventType.SHOT_FIRED, priority=2,
                position=(10, 10), data={"alliance": "friendly"},
            ),
            _make_event(
                SimEventType.SHOT_FIRED, priority=8,
                position=(10, 10), data={"alliance": "friendly"},
            ),
            _make_event(
                SimEventType.EXPLOSION, priority=1,
                position=(10, 10), data={"alliance": "hostile"},
            ),
        ]
        # Chain: type -> area -> priority
        step1 = EventFilter.filter_by_type(events, [SimEventType.SHOT_FIRED])
        step2 = EventFilter.filter_by_area(step1, (10, 10), 1.0)
        step3 = EventFilter.filter_by_priority(step2, max_priority=5)
        assert len(step3) == 1
        assert step3[0].priority == 2
