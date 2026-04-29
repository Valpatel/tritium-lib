# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.event_bus — centralized pub/sub system.

Covers subscription, emission, filtering, timeline queries, log trimming,
error isolation, and Three.js export.
"""

import math

import pytest

from tritium_lib.sim_engine.event_bus import (
    EventFilter,
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
        data=data or {},
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Subscription and emission
# ---------------------------------------------------------------------------


class TestSubscription:
    """Tests for event subscription, dispatch, and unsubscription."""

    def test_on_receives_matching_events(self):
        bus = SimEventBus()
        received = []
        bus.on(SimEventType.UNIT_SPAWNED, lambda e: received.append(e))
        event = _make_event(SimEventType.UNIT_SPAWNED, tick=1, source_id="u1")
        bus.emit(event)
        assert len(received) == 1
        assert received[0].source_id == "u1"

    def test_on_ignores_non_matching_events(self):
        bus = SimEventBus()
        received = []
        bus.on(SimEventType.UNIT_KILLED, lambda e: received.append(e))
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        assert len(received) == 0

    def test_on_any_receives_all_events(self):
        bus = SimEventBus()
        received = []
        bus.on_any(lambda e: received.append(e))
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        bus.emit(_make_event(SimEventType.SHOT_FIRED))
        bus.emit(_make_event(SimEventType.EXPLOSION))
        assert len(received) == 3

    def test_off_unsubscribes_callback(self):
        bus = SimEventBus()
        received = []
        cb = lambda e: received.append(e)
        bus.on(SimEventType.UNIT_SPAWNED, cb)
        bus.off(SimEventType.UNIT_SPAWNED, cb)
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        assert len(received) == 0

    def test_off_any_unsubscribes_global(self):
        bus = SimEventBus()
        received = []
        cb = lambda e: received.append(e)
        bus.on_any(cb)
        bus.off_any(cb)
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        assert len(received) == 0

    def test_multiple_listeners_same_event(self):
        bus = SimEventBus()
        a, b = [], []
        bus.on(SimEventType.SHOT_FIRED, lambda e: a.append(e))
        bus.on(SimEventType.SHOT_FIRED, lambda e: b.append(e))
        bus.emit(_make_event(SimEventType.SHOT_FIRED))
        assert len(a) == 1
        assert len(b) == 1

    def test_emit_many_dispatches_in_order(self):
        bus = SimEventBus()
        received = []
        bus.on_any(lambda e: received.append(e.tick))
        events = [_make_event(tick=i) for i in range(5)]
        bus.emit_many(events)
        assert received == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    """A bad listener must not break other listeners or the bus."""

    def test_bad_listener_does_not_crash_bus(self):
        bus = SimEventBus()
        good = []

        def bad_cb(e):
            raise RuntimeError("I'm broken")

        bus.on(SimEventType.UNIT_SPAWNED, bad_cb)
        bus.on(SimEventType.UNIT_SPAWNED, lambda e: good.append(e))
        # Should not raise
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        assert len(good) == 1

    def test_bad_global_listener_does_not_crash_bus(self):
        bus = SimEventBus()
        good = []

        def bad_cb(e):
            raise ValueError("broken global")

        bus.on_any(bad_cb)
        bus.on_any(lambda e: good.append(e))
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        assert len(good) == 1


# ---------------------------------------------------------------------------
# Event log and queries
# ---------------------------------------------------------------------------


class TestEventLog:
    """Tests for the internal event log, timeline queries, and stats."""

    def test_events_are_logged(self):
        bus = SimEventBus()
        bus.emit(_make_event(tick=1))
        bus.emit(_make_event(tick=2))
        log = bus.get_log()
        assert len(log) == 2

    def test_get_log_filter_by_type(self):
        bus = SimEventBus()
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED, tick=1))
        bus.emit(_make_event(SimEventType.SHOT_FIRED, tick=2))
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED, tick=3))
        log = bus.get_log(event_type=SimEventType.UNIT_SPAWNED)
        assert len(log) == 2

    def test_get_log_filter_by_tick(self):
        bus = SimEventBus()
        for i in range(10):
            bus.emit(_make_event(tick=i))
        log = bus.get_log(since_tick=7)
        assert len(log) == 3  # ticks 7, 8, 9

    def test_get_log_limit(self):
        bus = SimEventBus()
        for i in range(20):
            bus.emit(_make_event(tick=i))
        log = bus.get_log(limit=5)
        assert len(log) == 5
        # Should be the most recent 5
        assert log[0].tick == 15

    def test_get_timeline_range(self):
        bus = SimEventBus()
        for i in range(10):
            bus.emit(_make_event(tick=i))
        timeline = bus.get_timeline(3, 6)
        assert len(timeline) == 4  # ticks 3, 4, 5, 6
        assert all(3 <= e.tick <= 6 for e in timeline)

    def test_clear_log(self):
        bus = SimEventBus()
        bus.emit(_make_event())
        bus.emit(_make_event())
        bus.clear_log()
        assert len(bus.get_log()) == 0

    def test_stats_counts_by_type(self):
        bus = SimEventBus()
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        bus.emit(_make_event(SimEventType.UNIT_SPAWNED))
        bus.emit(_make_event(SimEventType.SHOT_FIRED))
        stats = bus.stats()
        assert stats["UNIT_SPAWNED"] == 2
        assert stats["SHOT_FIRED"] == 1

    def test_log_trimming_at_max(self):
        """When max_log is exceeded, oldest events should be trimmed."""
        bus = SimEventBus(max_log=100)
        for i in range(150):
            bus.emit(_make_event(tick=i))
        log = bus.get_log(limit=200)
        # After trimming, should be less than 150 but log still contains events
        assert len(log) <= 150
        # The oldest events should be gone
        assert log[0].tick > 0


# ---------------------------------------------------------------------------
# Three.js export
# ---------------------------------------------------------------------------


class TestThreeJSExport:
    """Tests for the to_three_js HUD export."""

    def test_to_three_js_returns_recent_events(self):
        bus = SimEventBus()
        for i in range(30):
            bus.emit(_make_event(tick=i, time=float(i), source_id=f"u_{i}"))
        data = bus.to_three_js(last_n=10)
        assert len(data) == 10
        # Should be the last 10
        assert data[0]["tick"] == 20

    def test_to_three_js_includes_position_when_present(self):
        bus = SimEventBus()
        bus.emit(_make_event(position=(10.5, 20.7)))
        data = bus.to_three_js(last_n=1)
        assert data[0]["x"] == 10.5
        assert data[0]["y"] == 20.7

    def test_to_three_js_excludes_position_when_none(self):
        bus = SimEventBus()
        bus.emit(_make_event(position=None))
        data = bus.to_three_js(last_n=1)
        assert "x" not in data[0]

    def test_to_three_js_empty_bus(self):
        bus = SimEventBus()
        data = bus.to_three_js(last_n=10)
        assert data == []


# ---------------------------------------------------------------------------
# EventFilter — stateless utilities
# ---------------------------------------------------------------------------


class TestEventFilter:
    """Tests for the stateless EventFilter helper class."""

    def test_filter_by_type(self):
        events = [
            _make_event(SimEventType.UNIT_SPAWNED),
            _make_event(SimEventType.SHOT_FIRED),
            _make_event(SimEventType.EXPLOSION),
            _make_event(SimEventType.UNIT_SPAWNED),
        ]
        filtered = EventFilter.filter_by_type(
            events, [SimEventType.UNIT_SPAWNED],
        )
        assert len(filtered) == 2

    def test_filter_by_area(self):
        events = [
            _make_event(position=(10.0, 10.0)),
            _make_event(position=(100.0, 100.0)),
            _make_event(position=(12.0, 10.0)),
            _make_event(position=None),  # no position, should be excluded
        ]
        filtered = EventFilter.filter_by_area(events, center=(10.0, 10.0), radius=5.0)
        assert len(filtered) == 2

    def test_filter_by_alliance(self):
        events = [
            _make_event(data={"alliance": "friendly"}),
            _make_event(data={"alliance": "hostile"}),
            _make_event(data={"alliance": "friendly"}),
            _make_event(data={}),
        ]
        filtered = EventFilter.filter_by_alliance(events, "friendly")
        assert len(filtered) == 2

    def test_filter_by_priority(self):
        events = [
            _make_event(priority=1),
            _make_event(priority=5),
            _make_event(priority=10),
            _make_event(priority=3),
        ]
        filtered = EventFilter.filter_by_priority(events, max_priority=3)
        assert len(filtered) == 2

    def test_filter_by_area_empty_list(self):
        filtered = EventFilter.filter_by_area([], center=(0, 0), radius=10.0)
        assert filtered == []

    def test_filter_by_type_with_set(self):
        """filter_by_type should accept a set as well as a list."""
        events = [
            _make_event(SimEventType.UNIT_SPAWNED),
            _make_event(SimEventType.SHOT_FIRED),
        ]
        filtered = EventFilter.filter_by_type(
            events, {SimEventType.UNIT_SPAWNED, SimEventType.SHOT_FIRED},
        )
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# SimEventType completeness
# ---------------------------------------------------------------------------


class TestSimEventType:
    """Verify the event type enum has broad coverage."""

    def test_has_combat_events(self):
        assert SimEventType.SHOT_FIRED
        assert SimEventType.EXPLOSION
        assert SimEventType.PROJECTILE_IMPACT

    def test_has_unit_lifecycle_events(self):
        assert SimEventType.UNIT_SPAWNED
        assert SimEventType.UNIT_KILLED
        assert SimEventType.UNIT_DAMAGED

    def test_has_game_flow_events(self):
        assert SimEventType.WAVE_STARTED
        assert SimEventType.WAVE_COMPLETED
        assert SimEventType.GAME_OVER

    def test_has_environment_events(self):
        assert SimEventType.WEATHER_CHANGED
        assert SimEventType.TIME_ADVANCED
