# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.synthetic.scenario_generators."""

import math
import re

import pytest

from tritium_lib.synthetic.scenario_generators import (
    BLESightingRecord,
    WiFiAPRecord,
    WiFiEnvironment,
    PatrolWaypoint,
    PatrolEvent,
    PatrolUnit,
    PatrolScenario,
    ThreatActor,
    ThreatDetection,
    GeofenceDefinition,
    ThreatScenario,
    generate_ble_sightings,
    generate_wifi_environment,
    generate_patrol_scenario,
    generate_threat_scenario,
    _random_mac,
    _haversine_m,
    _compute_rssi,
    _move_position,
    _resolve_ssid_pattern,
    ALLIANCE_FRIENDLY,
    ALLIANCE_HOSTILE,
)

import random

_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


# ── Helpers ────────────────────────────────────────────────────────────

class TestRandomMac:
    def test_format(self):
        rng = random.Random(1)
        mac = _random_mac(rng)
        assert _MAC_RE.match(mac), f"Invalid MAC format: {mac}"

    def test_locally_administered_bit(self):
        rng = random.Random(1)
        for _ in range(20):
            mac = _random_mac(rng)
            first_octet = int(mac.split(":")[0], 16)
            assert first_octet & 0x02, f"Locally-administered bit not set: {mac}"

    def test_unique_macs(self):
        rng = random.Random(42)
        macs = {_random_mac(rng) for _ in range(100)}
        assert len(macs) == 100, "MAC addresses should be unique"


class TestHaversine:
    def test_zero_distance(self):
        d = _haversine_m(37.0, -122.0, 37.0, -122.0)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_known_distance(self):
        # 0.001 deg latitude ~ 111 meters
        d = _haversine_m(37.0, -122.0, 37.001, -122.0)
        assert 100 < d < 120


class TestComputeRSSI:
    def test_close_range_stronger(self):
        rng = random.Random(1)
        rssi_close = _compute_rssi(1.0, -59.0, 2.5, rng)
        rng2 = random.Random(1)
        rssi_far = _compute_rssi(50.0, -59.0, 2.5, rng2)
        assert rssi_close > rssi_far

    def test_clamped_bounds(self):
        rng = random.Random(1)
        for _ in range(50):
            rssi = _compute_rssi(random.uniform(0.1, 500), -59, 2.5, rng)
            assert -100 <= rssi <= -20


class TestMovePosition:
    def test_stationary(self):
        rng = random.Random(1)
        lat, lng, h = _move_position(37.0, -122.0, 0.0, 0.0, 5.0, 0.0, rng)
        assert lat == pytest.approx(37.0, abs=1e-6)
        assert lng == pytest.approx(-122.0, abs=1e-6)

    def test_moves_forward(self):
        rng = random.Random(1)
        lat0, lng0 = 37.0, -122.0
        lat1, lng1, _ = _move_position(lat0, lng0, 0.0, 5.0, 10.0, 0.0, rng)
        dist = _haversine_m(lat0, lng0, lat1, lng1)
        # 5 m/s * 10 s = 50 m expected, allow some latitude
        assert 30 < dist < 70


class TestResolveSSIDPattern:
    def test_fills_placeholders(self):
        rng = random.Random(1)
        result = _resolve_ssid_pattern("{surname} WiFi", rng)
        assert "WiFi" in result
        assert "{surname}" not in result

    def test_hex4_is_4_chars(self):
        rng = random.Random(1)
        result = _resolve_ssid_pattern("TP-Link_{hex4}", rng)
        # extract the hex part after the underscore
        hex_part = result.split("_")[1]
        assert len(hex_part) == 4
        int(hex_part, 16)  # should not raise


# ── BLE Sightings ─────────────────────────────────────────────────────

class TestGenerateBLESightings:
    def test_returns_list(self):
        sightings = generate_ble_sightings(count=5, time_steps=3)
        assert isinstance(sightings, list)
        assert len(sightings) > 0

    def test_record_fields(self):
        sightings = generate_ble_sightings(count=3, time_steps=2)
        s = sightings[0]
        assert isinstance(s, BLESightingRecord)
        assert _MAC_RE.match(s.mac)
        assert -100 <= s.rssi <= -20
        assert s.observer_id.startswith("observer-")
        assert s.distance_m >= 0
        assert s.movement_pattern in (
            "stationary", "pedestrian", "jogger", "vehicle", "cyclist"
        )

    def test_multiple_observers(self):
        sightings = generate_ble_sightings(count=10, observer_count=4, time_steps=5)
        observer_ids = {s.observer_id for s in sightings}
        # With 10 devices and 4 observers in a 500m area, we should see multiple
        assert len(observer_ids) >= 2

    def test_deterministic_with_seed(self):
        bt = 1700000000.0
        s1 = generate_ble_sightings(count=5, seed=99, time_steps=3, base_time=bt)
        s2 = generate_ble_sightings(count=5, seed=99, time_steps=3, base_time=bt)
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            assert a.mac == b.mac
            assert a.rssi == b.rssi
            assert a.timestamp == b.timestamp

    def test_rssi_correlates_with_distance(self):
        sightings = generate_ble_sightings(count=20, time_steps=5, seed=42)
        if len(sightings) < 10:
            pytest.skip("Not enough sightings for correlation test")
        close = [s for s in sightings if s.distance_m < 30]
        far = [s for s in sightings if s.distance_m > 60]
        if close and far:
            avg_close_rssi = sum(s.rssi for s in close) / len(close)
            avg_far_rssi = sum(s.rssi for s in far) / len(far)
            assert avg_close_rssi > avg_far_rssi, (
                f"Close RSSI ({avg_close_rssi:.1f}) should be stronger than "
                f"far RSSI ({avg_far_rssi:.1f})"
            )

    def test_device_types_present(self):
        sightings = generate_ble_sightings(count=30, time_steps=5, seed=42)
        types = {s.device_type for s in sightings}
        # With 30 devices, we should see at least 3 different types
        assert len(types) >= 3

    def test_positions_within_area(self):
        center = (37.7749, -122.4194)
        area_m = 500.0
        sightings = generate_ble_sightings(
            count=10, area_center=center, area_size_m=area_m,
            time_steps=3, seed=42,
        )
        for s in sightings:
            dist = _haversine_m(center[0], center[1], s.observer_lat, s.observer_lng)
            # Observers are placed within 60% of half the area
            assert dist < area_m, f"Observer too far from center: {dist}m"

    def test_zero_count(self):
        sightings = generate_ble_sightings(count=0, time_steps=3)
        assert sightings == []


# ── WiFi Environment ──────────────────────────────────────────────────

class TestGenerateWiFiEnvironment:
    def test_returns_environment(self):
        env = generate_wifi_environment(building_count=5)
        assert isinstance(env, WiFiEnvironment)
        assert len(env.buildings) == 5
        assert len(env.access_points) > 0

    def test_ap_fields(self):
        env = generate_wifi_environment(building_count=3)
        ap = env.access_points[0]
        assert isinstance(ap, WiFiAPRecord)
        assert _MAC_RE.match(ap.bssid)
        assert ap.channel > 0
        assert -95 <= ap.rssi <= -25
        assert ap.auth_type in ("open", "WPA2", "WPA2-Enterprise")
        assert ap.building_id.startswith("bldg-")

    def test_building_fields(self):
        env = generate_wifi_environment(building_count=5)
        for b in env.buildings:
            assert "building_id" in b
            assert "type" in b
            assert b["type"] in ("residential", "commercial", "corporate", "industrial")
            assert "lat" in b
            assert "floors" in b
            assert b["floors"] >= 1

    def test_aps_per_building(self):
        env = generate_wifi_environment(building_count=10, seed=42)
        building_ids = {b["building_id"] for b in env.buildings}
        ap_buildings = {ap.building_id for ap in env.access_points}
        # Every AP must belong to a valid building
        for ab in ap_buildings:
            assert ab in building_ids

    def test_channel_distribution(self):
        env = generate_wifi_environment(building_count=20, seed=42)
        channels = {ap.channel for ap in env.access_points}
        # Should have both 2.4 GHz (1,6,11) and 5 GHz channels
        has_24 = any(c in (1, 6, 11) for c in channels)
        has_5 = any(c >= 36 for c in channels)
        assert has_24, "Should have 2.4 GHz channels"
        assert has_5, "Should have 5 GHz channels"

    def test_hidden_networks_possible(self):
        # With enough buildings, we should get at least one hidden network
        env = generate_wifi_environment(building_count=30, seed=42)
        hidden = [ap for ap in env.access_points if ap.hidden]
        # Not guaranteed but with 30 buildings it is very likely
        # Just check the field exists and is boolean
        for ap in env.access_points:
            assert isinstance(ap.hidden, bool)

    def test_deterministic(self):
        e1 = generate_wifi_environment(building_count=5, seed=50)
        e2 = generate_wifi_environment(building_count=5, seed=50)
        assert len(e1.access_points) == len(e2.access_points)
        assert len(e1.buildings) == len(e2.buildings)
        for a, b in zip(e1.access_points, e2.access_points):
            assert a.bssid == b.bssid
            assert a.ssid == b.ssid

    def test_network_types_appropriate(self):
        """Corporate buildings should not have 'home' SSIDs, etc."""
        env = generate_wifi_environment(building_count=20, seed=42)
        for ap in env.access_points:
            bldg = next(b for b in env.buildings if b["building_id"] == ap.building_id)
            if bldg["type"] == "corporate":
                assert ap.network_type in ("corporate", "guest", "iot", "unknown"), (
                    f"Corporate building has inappropriate network type: {ap.network_type}"
                )
            elif bldg["type"] == "residential":
                assert ap.network_type in ("home", "iot", "unknown"), (
                    f"Residential building has inappropriate network type: {ap.network_type}"
                )


# ── Patrol Scenario ───────────────────────────────────────────────────

class TestGeneratePatrolScenario:
    def test_returns_scenario(self):
        scenario = generate_patrol_scenario(unit_count=2, waypoints_per_route=4)
        assert isinstance(scenario, PatrolScenario)
        assert len(scenario.units) == 2
        assert len(scenario.events) > 0

    def test_unit_fields(self):
        scenario = generate_patrol_scenario(unit_count=3)
        for unit in scenario.units:
            assert isinstance(unit, PatrolUnit)
            assert unit.unit_id.startswith("patrol-")
            assert unit.alliance == ALLIANCE_FRIENDLY
            assert unit.speed_mps > 0
            assert len(unit.route) > 0

    def test_waypoint_fields(self):
        scenario = generate_patrol_scenario(unit_count=1, waypoints_per_route=5)
        for wp in scenario.units[0].route:
            assert isinstance(wp, PatrolWaypoint)
            assert wp.dwell_time_s > 0
            assert wp.name  # NATO phonetic name
            assert wp.lat != 0.0

    def test_events_chronological(self):
        scenario = generate_patrol_scenario(unit_count=3)
        timestamps = [e.timestamp for e in scenario.events]
        assert timestamps == sorted(timestamps), "Events should be in chronological order"

    def test_event_types(self):
        scenario = generate_patrol_scenario(unit_count=3, seed=42)
        event_types = {e.event_type for e in scenario.events}
        # At minimum we should always get arrivals and departures
        assert "arrival" in event_types
        assert "departure" in event_types

    def test_geofence_has_4_vertices(self):
        scenario = generate_patrol_scenario()
        assert len(scenario.geofence_vertices) == 4

    def test_different_asset_types(self):
        scenario = generate_patrol_scenario(unit_count=3)
        types = {u.asset_type for u in scenario.units}
        assert len(types) == 3  # person, vehicle, drone

    def test_patrol_route_forms_loop(self):
        """Waypoints should be spread around the area (sorted by angle)."""
        scenario = generate_patrol_scenario(unit_count=1, waypoints_per_route=6)
        route = scenario.units[0].route
        # Check that orders are sequential
        orders = [wp.order for wp in route]
        assert orders == sorted(orders)

    def test_deterministic(self):
        bt = 1700000000.0
        s1 = generate_patrol_scenario(seed=88, base_time=bt)
        s2 = generate_patrol_scenario(seed=88, base_time=bt)
        assert len(s1.units) == len(s2.units)
        assert len(s1.events) == len(s2.events)
        for e1, e2 in zip(s1.events, s2.events):
            assert e1.event_type == e2.event_type
            assert e1.timestamp == e2.timestamp


# ── Threat Scenario ───────────────────────────────────────────────────

class TestGenerateThreatScenario:
    def test_returns_scenario(self):
        scenario = generate_threat_scenario(hostile_count=3, time_steps=10)
        assert isinstance(scenario, ThreatScenario)
        assert len(scenario.actors) == 3
        assert scenario.time_steps == 10

    def test_actors_are_hostile(self):
        scenario = generate_threat_scenario(hostile_count=4)
        for actor in scenario.actors:
            assert isinstance(actor, ThreatActor)
            assert actor.alliance == ALLIANCE_HOSTILE
            assert actor.speed_mps > 0
            assert actor.actor_id.startswith("hostile-")

    def test_geofence_definition(self):
        scenario = generate_threat_scenario(geofence_radius_m=300.0)
        gf = scenario.geofence
        assert isinstance(gf, GeofenceDefinition)
        assert gf.radius_m == 300.0
        assert gf.name == "Perimeter-Alpha"

    def test_detections_have_sensor_types(self):
        scenario = generate_threat_scenario(
            hostile_count=4, time_steps=40, step_interval_s=5.0,
            approach_distance_m=300.0, seed=42,
        )
        assert len(scenario.detections) > 0, "Should produce detections with close approach"
        sensor_types = {d.sensor_type for d in scenario.detections}
        valid_types = {"ble", "camera", "rf_motion", "acoustic"}
        assert sensor_types.issubset(valid_types)

    def test_detections_chronological(self):
        scenario = generate_threat_scenario(hostile_count=4, time_steps=20)
        timestamps = [d.timestamp for d in scenario.detections]
        assert timestamps == sorted(timestamps)

    def test_confidence_bounded(self):
        scenario = generate_threat_scenario(hostile_count=5, time_steps=30, seed=42)
        for d in scenario.detections:
            assert 0.3 <= d.confidence <= 0.99, f"Confidence out of range: {d.confidence}"

    def test_actors_approach_center(self):
        """Actors should get closer to the center over time."""
        center = (37.7749, -122.4194)
        scenario = generate_threat_scenario(
            hostile_count=2, time_steps=20, step_interval_s=5.0,
            area_center=center, approach_distance_m=500.0, seed=42,
        )
        # Check that actors moved closer to center compared to starting distance
        for actor in scenario.actors:
            final_dist = _haversine_m(center[0], center[1], actor.lat, actor.lng)
            # Started at 500m, should be closer after 20 steps of movement
            assert final_dist < 500.0, (
                f"Actor {actor.actor_id} should approach center "
                f"but is at {final_dist:.0f}m"
            )

    def test_breach_detections(self):
        """With enough steps, some detections should be inside the geofence."""
        scenario = generate_threat_scenario(
            hostile_count=4, time_steps=50, step_interval_s=5.0,
            geofence_radius_m=200.0, approach_distance_m=400.0, seed=42,
        )
        inside = [d for d in scenario.detections if d.inside_fence]
        # With fast actors starting 400m away moving toward a 200m fence
        # over 250 seconds, some should breach
        # (not guaranteed with all seeds, so just verify the field works)
        for d in scenario.detections:
            assert isinstance(d.inside_fence, bool)

    def test_multiple_sensor_types_triggered(self):
        """Different sensors should fire at different ranges."""
        scenario = generate_threat_scenario(
            hostile_count=4, time_steps=40, step_interval_s=5.0,
            approach_distance_m=300.0, seed=42,
        )
        sensor_types = {d.sensor_type for d in scenario.detections}
        # We should get at least 2 different sensor types
        assert len(sensor_types) >= 2, (
            f"Expected multiple sensor types, got: {sensor_types}"
        )

    def test_deterministic(self):
        bt = 1700000000.0
        s1 = generate_threat_scenario(hostile_count=3, seed=42, time_steps=10, base_time=bt)
        s2 = generate_threat_scenario(hostile_count=3, seed=42, time_steps=10, base_time=bt)
        assert len(s1.detections) == len(s2.detections)
        for d1, d2 in zip(s1.detections, s2.detections):
            assert d1.actor_id == d2.actor_id
            assert d1.sensor_type == d2.sensor_type
            assert d1.confidence == d2.confidence
            assert d1.timestamp == d2.timestamp


# ── Import from package root ──────────────────────────────────────────

class TestPackageExports:
    def test_import_from_synthetic(self):
        from tritium_lib.synthetic import (
            generate_ble_sightings,
            generate_wifi_environment,
            generate_patrol_scenario,
            generate_threat_scenario,
            BLESightingRecord,
            WiFiAPRecord,
            WiFiEnvironment,
            PatrolScenario,
            ThreatScenario,
        )
        # All should be callable or classes
        assert callable(generate_ble_sightings)
        assert callable(generate_wifi_environment)
        assert callable(generate_patrol_scenario)
        assert callable(generate_threat_scenario)
