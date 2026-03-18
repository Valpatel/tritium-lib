# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.core — entity, inventory, movement,
state_machine, spatial.
"""

import pytest

from tritium_lib.sim_engine.core import (
    SimulationTarget,
    UnitIdentity,
    UnitInventory,
    InventoryItem,
    MovementController,
    StateMachine,
    State,
    Transition,
    SpatialGrid,
)
from tritium_lib.sim_engine.core.entity import (
    build_identity,
    generate_short_id,
    generate_mac_address,
)
from tritium_lib.sim_engine.core.inventory import build_loadout


# ---------------------------------------------------------------------------
# SimulationTarget
# ---------------------------------------------------------------------------

class TestSimulationTarget:
    """Test SimulationTarget creation and basic behavior."""

    def test_creation_with_defaults(self):
        t = SimulationTarget(
            target_id="test_rover_1",
            name="Rover Alpha",
            alliance="friendly",
            asset_type="rover",
            position=(100.0, 200.0),
        )
        assert t.target_id == "test_rover_1"
        assert t.status == "active"
        assert t.battery == 1.0
        assert t.health == 100.0
        assert t.identity is not None
        assert t.identity.short_id != ""

    def test_neutral_person_no_movement_controller(self):
        t = SimulationTarget(
            target_id="civ_1",
            name="Civilian",
            alliance="neutral",
            asset_type="person",
            position=(0.0, 0.0),
            speed=1.0,
        )
        # Neutral units don't get MovementController
        assert t.movement is None

    def test_friendly_combatant_gets_movement_controller(self):
        t = SimulationTarget(
            target_id="rover_1",
            name="Rover",
            alliance="friendly",
            asset_type="rover",
            position=(0.0, 0.0),
            speed=3.0,
        )
        assert t.movement is not None

    def test_apply_combat_profile(self):
        t = SimulationTarget(
            target_id="turret_1",
            name="Turret",
            alliance="friendly",
            asset_type="turret",
            position=(50.0, 50.0),
            speed=0.0,
        )
        t.apply_combat_profile()
        assert t.health == 200.0
        assert t.max_health == 200.0
        assert t.weapon_range == 80.0
        assert t.is_combatant is True

    def test_apply_damage(self):
        t = SimulationTarget(
            target_id="test_1",
            name="Test",
            alliance="hostile",
            asset_type="person",
            position=(0.0, 0.0),
        )
        t.health = 50.0
        assert t.apply_damage(30.0) is False
        assert t.health == 20.0
        assert t.apply_damage(25.0) is True
        assert t.status == "eliminated"

    def test_to_dict(self):
        t = SimulationTarget(
            target_id="d_1",
            name="Test Target",
            alliance="friendly",
            asset_type="rover",
            position=(10.0, 20.0),
        )
        d = t.to_dict()
        assert d["target_id"] == "d_1"
        assert d["position"] == {"x": 10.0, "y": 20.0}
        assert "identity" in d

    def test_tick_battery_drain(self):
        t = SimulationTarget(
            target_id="drone_1",
            name="Drone",
            alliance="friendly",
            asset_type="drone",
            position=(0.0, 0.0),
            speed=0.0,
            is_combatant=False,
        )
        t.battery = 0.06
        t.tick(100.0)  # Large dt to drain battery
        assert t.status == "low_battery"


# ---------------------------------------------------------------------------
# UnitIdentity
# ---------------------------------------------------------------------------

class TestUnitIdentity:
    """Test UnitIdentity deterministic generation."""

    def test_deterministic_generation(self):
        """Same target_id always produces the same identity."""
        id1 = build_identity("unit_abc", "person", "neutral")
        id2 = build_identity("unit_abc", "person", "neutral")
        assert id1.short_id == id2.short_id
        assert id1.first_name == id2.first_name
        assert id1.last_name == id2.last_name
        assert id1.bluetooth_mac == id2.bluetooth_mac

    def test_different_ids_differ(self):
        id1 = build_identity("unit_abc", "person", "neutral")
        id2 = build_identity("unit_xyz", "person", "neutral")
        assert id1.short_id != id2.short_id

    def test_person_has_devices(self):
        ident = build_identity("person_1", "person", "neutral")
        assert ident.bluetooth_mac != ""
        assert ident.wifi_mac != ""
        assert ident.cell_id != ""

    def test_vehicle_has_plate(self):
        ident = build_identity("car_1", "vehicle", "neutral")
        assert ident.license_plate != ""
        assert ident.vehicle_make != ""

    def test_robot_has_serial(self):
        ident = build_identity("rover_1", "rover", "friendly")
        assert ident.serial_number != ""
        assert ident.firmware_version != ""

    def test_short_id_format(self):
        sid = generate_short_id("test_123")
        assert len(sid) == 6
        assert sid == sid.upper()

    def test_mac_address_format(self):
        mac = generate_mac_address("test_123", "wifi")
        parts = mac.split(":")
        assert len(parts) == 6
        for p in parts:
            assert len(p) == 2


# ---------------------------------------------------------------------------
# InventoryItem with device fields
# ---------------------------------------------------------------------------

class TestInventoryItemDevice:
    """Test InventoryItem with BLE device fields."""

    def test_device_fields_defaults(self):
        item = InventoryItem(item_id="phone_1", item_type="device")
        assert item.ble_mac == ""
        assert item.wifi_mac == ""
        assert item.device_class == ""
        assert item.tx_power_dbm == -40
        assert item.always_on is True
        assert item.adv_interval_ms == 1000

    def test_device_with_ble_fields(self):
        item = InventoryItem(
            item_id="watch_1",
            item_type="device",
            name="Smart Watch",
            ble_mac="AA:BB:CC:DD:EE:FF",
            wifi_mac="",
            ble_service_uuid="0000180d-0000-1000-8000-00805f9b34fb",
            tx_power_dbm=-20,
            device_model="Galaxy Watch 6",
            device_class="smartwatch",
            always_on=True,
            adv_interval_ms=500,
        )
        assert item.device_class == "smartwatch"
        assert item.tx_power_dbm == -20
        d = item.to_dict()
        assert d["item_type"] == "device"
        assert d["ble_mac"] == "AA:BB:CC:DD:EE:FF"
        assert d["device_class"] == "smartwatch"
        assert d["adv_interval_ms"] == 500

    def test_device_fog_dict(self):
        item = InventoryItem(
            item_id="phone_1",
            item_type="device",
            ble_mac="AA:BB:CC:DD:EE:FF",
        )
        fog = item.to_fog_dict()
        assert "ble_mac" not in fog
        assert fog["status"] == "unknown"

    def test_get_devices(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="gun_1", item_type="weapon"))
        inv.add_item(InventoryItem(item_id="phone_1", item_type="device", device_class="phone"))
        inv.add_item(InventoryItem(item_id="watch_1", item_type="device", device_class="smartwatch"))
        devices = inv.get_devices()
        assert len(devices) == 2
        assert all(d.item_type == "device" for d in devices)


# ---------------------------------------------------------------------------
# MovementController
# ---------------------------------------------------------------------------

class TestMovementController:
    """Test MovementController tick behavior."""

    def test_initial_state(self):
        mc = MovementController(max_speed=5.0)
        assert mc.arrived is True
        assert mc.speed == 0.0

    def test_set_destination_and_tick(self):
        mc = MovementController(max_speed=5.0, x=0.0, y=0.0)
        mc.set_destination(10.0, 0.0)
        assert mc.arrived is False
        # Tick several times
        for _ in range(100):
            mc.tick(0.1)
        assert mc.arrived is True

    def test_acceleration(self):
        mc = MovementController(
            max_speed=10.0,
            acceleration=5.0,
            deceleration=5.0,
            x=0.0, y=0.0,
        )
        mc.set_destination(100.0, 0.0)
        mc.tick(0.1)
        # After first tick, speed should be > 0 (accelerating)
        assert mc.speed > 0.0

    def test_patrol_loop(self):
        mc = MovementController(max_speed=100.0, x=0.0, y=0.0)
        mc.set_path([(10.0, 0.0), (0.0, 0.0)], loop=True)
        # Tick many times — should not arrive (loops)
        for _ in range(200):
            mc.tick(0.1)
        assert mc.arrived is False

    def test_remaining_waypoints(self):
        mc = MovementController()
        assert mc.remaining_waypoints == 0
        mc.set_path([(1, 0), (2, 0), (3, 0)])
        assert mc.remaining_waypoints == 3

    def test_stop(self):
        mc = MovementController(x=0.0, y=0.0)
        mc.set_destination(100.0, 0.0)
        assert mc.arrived is False
        mc.stop()
        assert mc.arrived is True


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

class TestStateMachine:
    """Test StateMachine transitions."""

    def test_builder_api(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("alert"))
        sm.add_transition("idle", "alert", lambda ctx: ctx.get("enemy", False))
        sm.tick(0.1, {"enemy": False})
        assert sm.current_state == "idle"
        sm.tick(0.1, {"enemy": True})
        assert sm.current_state == "alert"

    def test_legacy_api(self):
        triggered = []
        sm = StateMachine(
            states=[State("a"), State("b")],
            transitions=[Transition("a", "b", lambda: True, on_transition=lambda: triggered.append(1))],
            initial_state="a",
        )
        sm.tick(0.1)
        assert sm.current_state == "b"
        assert len(triggered) == 1

    def test_force_state(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("combat"))
        sm.tick(0.1)
        sm.force_state("combat")
        assert sm.current_state == "combat"

    def test_time_in_state(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.tick(0.5)
        sm.tick(0.3)
        assert abs(sm.time_in_state - 0.8) < 0.001

    def test_history(self):
        sm = StateMachine("a")
        sm.add_state(State("a"))
        sm.add_state(State("b"))
        sm.add_transition("a", "b", lambda ctx: True)
        sm.tick(0.1)
        h = sm.history
        assert len(h) == 1
        assert h[0][1] == "a"
        assert h[0][2] == "b"

    def test_min_duration_blocks_transition(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle", min_duration=1.0))
        sm.add_state(State("alert"))
        sm.add_transition("idle", "alert", lambda ctx: True)
        sm.tick(0.1)  # time_in_state = 0.1, blocked
        assert sm.current_state == "idle"
        sm.tick(1.0)  # time_in_state = 1.1, unblocked
        assert sm.current_state == "alert"


# ---------------------------------------------------------------------------
# SpatialGrid
# ---------------------------------------------------------------------------

class TestSpatialGrid:
    """Test SpatialGrid queries."""

    def _make_target(self, tid: str, x: float, y: float) -> SimulationTarget:
        return SimulationTarget(
            target_id=tid,
            name=tid,
            alliance="friendly",
            asset_type="rover",
            position=(x, y),
            speed=0.0,
            is_combatant=False,
        )

    def test_query_radius(self):
        grid = SpatialGrid(cell_size=50.0)
        t1 = self._make_target("a", 10.0, 10.0)
        t2 = self._make_target("b", 15.0, 10.0)
        t3 = self._make_target("c", 200.0, 200.0)
        grid.rebuild([t1, t2, t3])
        near = grid.query_radius((10.0, 10.0), 20.0)
        ids = {t.target_id for t in near}
        assert "a" in ids
        assert "b" in ids
        assert "c" not in ids

    def test_query_rect(self):
        grid = SpatialGrid(cell_size=50.0)
        t1 = self._make_target("a", 10.0, 10.0)
        t2 = self._make_target("b", 100.0, 100.0)
        grid.rebuild([t1, t2])
        result = grid.query_rect((0.0, 0.0), (50.0, 50.0))
        ids = {t.target_id for t in result}
        assert "a" in ids
        assert "b" not in ids

    def test_empty_grid(self):
        grid = SpatialGrid()
        grid.rebuild([])
        assert grid.query_radius((0, 0), 100) == []
        assert grid.query_rect((0, 0), (100, 100)) == []


# ---------------------------------------------------------------------------
# Build loadout
# ---------------------------------------------------------------------------

class TestBuildLoadout:
    """Test inventory loadout generation."""

    def test_friendly_rover_loadout(self):
        inv = build_loadout("rover_1", "rover", "friendly")
        weapons = inv.get_weapons()
        assert len(weapons) >= 1

    def test_neutral_empty(self):
        inv = build_loadout("civ_1", "person", "neutral")
        assert len(inv.items) == 0

    def test_hostile_person_loadout(self):
        inv = build_loadout("bad_1", "person", "hostile")
        assert len(inv.items) >= 2  # pistol + vest at minimum
