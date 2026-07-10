# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Reload event + ammo-snapshot coverage for WeaponSystem.

These pin the AMMO/RELOAD VISIBILITY surface the combat HUD and a real
turret's WeaponStatus telemetry both consume:

  - ``reload_started`` / ``reload_complete`` fire exactly once per cycle
  - the events carry the documented payload
  - no events when no ``event_bus`` is attached (LIB stays framework-free)
  - ``reload_state()`` / ``get_all_ammo_state()`` snapshots are correct
    at rest, mid-reload, and after refill
"""

from __future__ import annotations

from tritium_lib.sim_engine.combat.weapons import Weapon, WeaponSystem


class _RecordingBus:
    """Minimal event bus that records every publish() call."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))

    def of_type(self, event_type: str) -> list[dict]:
        return [d for t, d in self.events if t == event_type]


def _drain_to_empty(ws: WeaponSystem, tid: str) -> None:
    """Consume ammo until the weapon is empty."""
    while ws.consume_ammo(tid):
        pass


# ---------------------------------------------------------------------------
# reload_started / reload_complete events
# ---------------------------------------------------------------------------


def test_reload_started_fires_once_with_payload():
    bus = _RecordingBus()
    ws = WeaponSystem(event_bus=bus)
    ws.assign_weapon("u1", Weapon(name="tiny", ammo=1, max_ammo=10, damage=1.0))
    ws.consume_ammo("u1")  # -> 0

    # Several ticks that all stay inside the reload window.
    ws.tick(0.1)
    ws.tick(0.1)
    ws.tick(0.1)

    started = bus.of_type("reload_started")
    assert len(started) == 1, "reload_started must fire exactly once per cycle"
    payload = started[0]
    assert payload["target_id"] == "u1"
    assert payload["weapon"] == "tiny"
    assert payload["duration"] == ws._reload_duration


def test_reload_complete_fires_once_with_refill_amount():
    bus = _RecordingBus()
    ws = WeaponSystem(event_bus=bus)
    ws.assign_weapon("u1", Weapon(name="tiny", ammo=1, max_ammo=10, damage=1.0))
    ws.consume_ammo("u1")  # -> 0

    ws.tick(0.1)           # start reload
    assert ws.is_reloading("u1") is True
    ws.tick(ws._reload_duration)  # finish reload

    complete = bus.of_type("reload_complete")
    assert len(complete) == 1, "reload_complete must fire exactly once per cycle"
    payload = complete[0]
    assert payload["target_id"] == "u1"
    assert payload["weapon"] == "tiny"
    assert payload["ammo"] == 10  # refilled to max_ammo


def test_full_cycle_fires_each_event_exactly_once():
    bus = _RecordingBus()
    ws = WeaponSystem(event_bus=bus)
    ws.assign_weapon("u1", Weapon(name="tiny", ammo=1, max_ammo=5, damage=1.0))
    ws.consume_ammo("u1")

    # Grind many ticks across the whole reload.
    for _ in range(60):
        ws.tick(0.1)

    assert len(bus.of_type("reload_started")) == 1
    assert len(bus.of_type("reload_complete")) == 1
    assert ws.get_ammo("u1") == 5
    assert ws.is_reloading("u1") is False


def test_no_reload_events_without_event_bus():
    ws = WeaponSystem()  # no bus
    ws.assign_weapon("u1", Weapon(name="tiny", ammo=1, max_ammo=5, damage=1.0))
    ws.consume_ammo("u1")
    # Should reload silently — and simply not raise.
    ws.tick(0.1)
    ws.tick(ws._reload_duration)
    assert ws.get_ammo("u1") == 5
    assert ws.is_reloading("u1") is False


def test_reload_within_single_tick_fires_both_events_once():
    """A tick wider than the reload duration starts and finishes in one call."""
    bus = _RecordingBus()
    ws = WeaponSystem(event_bus=bus)
    ws.assign_weapon("u1", Weapon(name="tiny", ammo=1, max_ammo=3, damage=1.0))
    ws.consume_ammo("u1")

    ws.tick(ws._reload_duration + 1.0)

    assert len(bus.of_type("reload_started")) == 1
    assert len(bus.of_type("reload_complete")) == 1


# ---------------------------------------------------------------------------
# reload_state snapshot
# ---------------------------------------------------------------------------


def test_reload_state_unknown_unit_is_none():
    ws = WeaponSystem()
    assert ws.reload_state("nope") is None


def test_reload_state_known_unit_not_reloading():
    ws = WeaponSystem()
    ws.equip("u1", "turret")
    state = ws.reload_state("u1")
    assert state == {
        "reloading": False,
        "remaining_s": 0.0,
        "duration": ws._reload_duration,
    }


def test_reload_state_mid_reload():
    ws = WeaponSystem()
    ws.assign_weapon("u1", Weapon(name="tiny", ammo=1, max_ammo=10, damage=1.0))
    ws.consume_ammo("u1")
    ws.tick(0.1)  # start reload; timer decremented by 0.1

    state = ws.reload_state("u1")
    assert state is not None
    assert state["reloading"] is True
    # remaining is duration minus the elapsed tick, and strictly positive.
    assert 0.0 < state["remaining_s"] < ws._reload_duration
    assert state["duration"] == ws._reload_duration


# ---------------------------------------------------------------------------
# get_all_ammo_state bulk snapshot
# ---------------------------------------------------------------------------


def test_get_all_ammo_state_full_weapons():
    ws = WeaponSystem()
    ws.equip("a", "turret")   # 100/100
    ws.equip("b", "drone")    # 20/20
    ws.consume_ammo("b")      # -> 19

    state = ws.get_all_ammo_state()
    assert set(state) == {"a", "b"}
    assert state["a"] == {
        "ammo": 100, "max_ammo": 100, "reloading": False, "reload_remaining_s": 0.0,
    }
    assert state["b"]["ammo"] == 19
    assert state["b"]["max_ammo"] == 20
    assert state["b"]["reloading"] is False


def test_get_all_ammo_state_reflects_active_reload():
    ws = WeaponSystem()
    ws.assign_weapon("u1", Weapon(name="tiny", ammo=1, max_ammo=10, damage=1.0))
    ws.consume_ammo("u1")
    ws.tick(0.1)  # reload started

    entry = ws.get_all_ammo_state()["u1"]
    assert entry["reloading"] is True
    assert entry["ammo"] == 0
    assert entry["max_ammo"] == 10
    assert 0.0 < entry["reload_remaining_s"] < ws._reload_duration


def test_get_all_ammo_state_empty_when_no_units():
    ws = WeaponSystem()
    assert ws.get_all_ammo_state() == {}
