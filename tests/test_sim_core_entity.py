# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.core.entity — SimulationTarget and identity generation."""

import math
import pytest

from tritium_lib.sim_engine.core.entity import (
    SimulationTarget,
    UnitIdentity,
    build_identity,
    generate_short_id,
    generate_mac_address,
    generate_license_plate,
    generate_cell_id,
    generate_address,
    generate_serial,
    generate_firmware_version,
    generate_person_name,
    generate_vehicle_info,
)


# ---------------------------------------------------------------------------
# Identity generation determinism
# ---------------------------------------------------------------------------

class TestIdentityGeneration:
    """Identity generators must be deterministic (same input -> same output)."""

    def test_short_id_deterministic(self):
        a = generate_short_id("test_unit_1")
        b = generate_short_id("test_unit_1")
        assert a == b
        assert len(a) == 6
        assert a == a.upper()

    def test_short_id_unique_per_target(self):
        a = generate_short_id("unit_a")
        b = generate_short_id("unit_b")
        assert a != b

    def test_mac_address_format(self):
        mac = generate_mac_address("test_unit_1")
        parts = mac.split(":")
        assert len(parts) == 6
        for part in parts:
            assert len(part) == 2
            int(part, 16)  # must be valid hex

    def test_mac_address_locally_administered(self):
        mac = generate_mac_address("test_unit_1")
        first_byte = int(mac.split(":")[0], 16)
        assert first_byte & 0x02 != 0, "Local bit should be set"
        assert first_byte & 0x01 == 0, "Multicast bit should be clear"

    def test_mac_address_deterministic(self):
        a = generate_mac_address("u1", "wifi")
        b = generate_mac_address("u1", "wifi")
        assert a == b

    def test_mac_address_different_devices(self):
        wifi = generate_mac_address("u1", "wifi")
        ble = generate_mac_address("u1", "ble")
        assert wifi != ble

    def test_license_plate_format(self):
        plate = generate_license_plate("vehicle_1")
        assert len(plate) == 7
        assert plate[0].isdigit()
        assert plate[1:4].isalpha()
        assert plate[4:7].isdigit()

    def test_license_plate_deterministic(self):
        a = generate_license_plate("v1")
        b = generate_license_plate("v1")
        assert a == b

    def test_cell_id_format(self):
        cid = generate_cell_id("person_1")
        assert cid.startswith("310-260-")
        msin = cid.split("-")[-1]
        assert len(msin) == 9
        assert msin.isdigit()

    def test_address_has_number_and_street(self):
        addr = generate_address("person_1")
        parts = addr.split(" ", 1)
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert len(parts[1]) > 0

    def test_serial_format(self):
        s = generate_serial("drone_1")
        assert s.startswith("TRT-")
        parts = s.split("-")
        assert len(parts) == 3
        assert parts[1].isdigit()
        assert len(parts[2]) == 5

    def test_person_name_tuple(self):
        first, last = generate_person_name("person_1")
        assert isinstance(first, str) and len(first) > 0
        assert isinstance(last, str) and len(last) > 0

    def test_vehicle_info_has_fields(self):
        info = generate_vehicle_info("vehicle_1")
        assert "make" in info
        assert "model" in info
        assert "color" in info
        assert "year" in info


# ---------------------------------------------------------------------------
# UnitIdentity via build_identity
# ---------------------------------------------------------------------------

class TestBuildIdentity:
    def test_person_identity(self):
        ident = build_identity("person_1", "person", "hostile")
        assert isinstance(ident, UnitIdentity)
        assert ident.bluetooth_mac  # persons should have BLE
        assert ident.wifi_mac
        assert ident.cell_id

    def test_vehicle_identity(self):
        ident = build_identity("vehicle_1", "vehicle", "neutral")
        assert isinstance(ident, UnitIdentity)
        assert ident.license_plate
        assert ident.vehicle_make
        assert ident.vehicle_model

    def test_drone_identity(self):
        ident = build_identity("drone_1", "drone", "friendly")
        assert isinstance(ident, UnitIdentity)
        assert ident.serial_number
        assert ident.firmware_version

    def test_identity_to_dict(self):
        ident = build_identity("person_1", "person", "hostile")
        d = ident.to_dict()
        assert isinstance(d, dict)
        assert "short_id" in d
        assert "bluetooth_mac" in d


# ---------------------------------------------------------------------------
# SimulationTarget construction and defaults
# ---------------------------------------------------------------------------

class TestSimulationTargetConstruction:
    def test_basic_construction(self):
        t = SimulationTarget(
            target_id="t1",
            name="Test Unit",
            alliance="friendly",
            asset_type="rover",
            position=(10.0, 20.0),
        )
        assert t.target_id == "t1"
        assert t.status == "active"
        assert t.health == 100.0
        assert t.battery == 1.0

    def test_auto_identity_generation(self):
        t = SimulationTarget(
            target_id="t1",
            name="Test",
            alliance="friendly",
            asset_type="rover",
            position=(0.0, 0.0),
        )
        assert t.identity is not None
        assert isinstance(t.identity, UnitIdentity)

    def test_auto_inventory_for_combatant(self):
        t = SimulationTarget(
            target_id="t1",
            name="Soldier",
            alliance="friendly",
            asset_type="person",
            position=(0.0, 0.0),
            is_combatant=True,
        )
        assert t.inventory is not None

    def test_no_inventory_for_noncombatant(self):
        t = SimulationTarget(
            target_id="t1",
            name="Civilian",
            alliance="neutral",
            asset_type="person",
            position=(0.0, 0.0),
            is_combatant=False,
        )
        assert t.inventory is None

    def test_waypoints_default_empty(self):
        t = SimulationTarget(
            target_id="t1",
            name="Test",
            alliance="friendly",
            asset_type="rover",
            position=(0.0, 0.0),
        )
        assert t.waypoints == []

    def test_heading_default_zero(self):
        t = SimulationTarget(
            target_id="t1",
            name="Test",
            alliance="friendly",
            asset_type="rover",
            position=(0.0, 0.0),
        )
        assert t.heading == 0.0


# ---------------------------------------------------------------------------
# SimulationTarget.apply_damage
# ---------------------------------------------------------------------------

class TestSimulationTargetDamage:
    def test_apply_damage_reduces_health(self):
        t = SimulationTarget(
            target_id="t1", name="T", alliance="hostile",
            asset_type="person", position=(0, 0), health=100.0,
        )
        killed = t.apply_damage(30.0)
        assert t.health == 70.0
        assert not killed

    def test_apply_damage_kills_at_zero(self):
        t = SimulationTarget(
            target_id="t1", name="T", alliance="hostile",
            asset_type="person", position=(0, 0), health=50.0,
        )
        killed = t.apply_damage(50.0)
        assert t.health <= 0
        assert killed

    def test_apply_damage_overkill(self):
        t = SimulationTarget(
            target_id="t1", name="T", alliance="hostile",
            asset_type="person", position=(0, 0), health=10.0,
        )
        killed = t.apply_damage(100.0)
        assert killed
        assert t.health <= 0


# ---------------------------------------------------------------------------
# SimulationTarget.tick (legacy movement)
# ---------------------------------------------------------------------------

class TestSimulationTargetTick:
    def test_stationary_no_movement(self):
        t = SimulationTarget(
            target_id="t1", name="T", alliance="friendly",
            asset_type="turret", position=(5.0, 5.0), speed=0.0,
        )
        t.tick(0.1)
        assert t.position == (5.0, 5.0)

    def test_waypoint_movement(self):
        t = SimulationTarget(
            target_id="t1", name="T", alliance="neutral",
            asset_type="person", position=(0.0, 0.0), speed=10.0,
            waypoints=[(100.0, 0.0)], is_combatant=False,
        )
        t.tick(1.0)
        # Should have moved toward the waypoint
        assert t.position[0] > 0.0

    def test_to_dict_returns_dict(self):
        t = SimulationTarget(
            target_id="t1", name="Test", alliance="friendly",
            asset_type="rover", position=(10.0, 20.0),
        )
        d = t.to_dict()
        assert isinstance(d, dict)
        assert d["target_id"] == "t1"
        assert d["name"] == "Test"
        assert "position" in d
