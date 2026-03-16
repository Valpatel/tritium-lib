# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the GTA-style city life simulation."""

import pytest

from tritium_lib.game_ai.city_sim import (
    Building,
    BuildingType,
    DailySchedule,
    DEFAULT_MIX,
    NeighborhoodSim,
    Resident,
    ResidentRole,
    ScheduleEntry,
    SimVehicle,
)


# ---------------------------------------------------------------------------
# DailySchedule tests
# ---------------------------------------------------------------------------

class TestDailySchedule:
    def test_office_worker_schedule(self):
        sched = DailySchedule.office_worker()
        assert len(sched.entries) > 0
        # At 3am should be sleeping
        entry = sched.activity_at(3.0)
        assert entry.activity == "sleeping"
        # At 9am should be working
        entry = sched.activity_at(9.0)
        assert entry.activity == "working"
        # At 12:30 should be at lunch
        entry = sched.activity_at(12.5)
        assert entry.activity == "lunch"

    def test_school_kid_schedule(self):
        sched = DailySchedule.school_kid()
        # At 10am should be at school
        entry = sched.activity_at(10.0)
        assert entry.activity == "at_school"
        # At 16:00 should be playing
        entry = sched.activity_at(16.0)
        assert entry.activity == "playing"

    def test_retired_schedule(self):
        sched = DailySchedule.retired()
        # At 6:30 should be on morning walk
        entry = sched.activity_at(6.75)
        assert entry.activity == "walking"
        # At 13:30 should be napping
        entry = sched.activity_at(13.5)
        assert entry.activity == "napping"

    def test_delivery_driver_schedule(self):
        sched = DailySchedule.delivery_driver()
        # At 10:30 should be delivering
        entry = sched.activity_at(10.5)
        assert entry.activity == "delivering"

    def test_work_from_home_schedule(self):
        sched = DailySchedule.work_from_home()
        # At 10:00 should be working (from home)
        entry = sched.activity_at(10.0)
        assert entry.activity == "working"
        assert entry.location_type == "home"

    def test_service_worker_schedule(self):
        sched = DailySchedule.service_worker()
        # At 8:00 should be working
        entry = sched.activity_at(8.0)
        assert entry.activity == "working"

    def test_empty_schedule_defaults_to_sleeping(self):
        sched = DailySchedule()
        entry = sched.activity_at(12.0)
        assert entry.activity == "sleeping"

    def test_schedule_wraps_around(self):
        sched = DailySchedule.office_worker()
        # Hour 25 wraps to 1 -> sleeping
        entry = sched.activity_at(25.0)
        assert entry.activity == "sleeping"


# ---------------------------------------------------------------------------
# NeighborhoodSim population tests
# ---------------------------------------------------------------------------

class TestNeighborhoodPopulate:
    def test_populates_correct_total(self):
        sim = NeighborhoodSim(num_residents=50, seed=42)
        sim.populate()
        # Should have roughly 50 residents (rounding may cause small deviation)
        assert 45 <= len(sim.residents) <= 60

    def test_populates_with_correct_mix(self):
        sim = NeighborhoodSim(num_residents=100, seed=42)
        sim.populate()
        role_counts = {}
        for r in sim.residents:
            role_counts[r.role] = role_counts.get(r.role, 0) + 1

        # Check relative proportions match default mix
        total = len(sim.residents)
        office_pct = role_counts.get(ResidentRole.OFFICE_WORKER, 0) / total
        kid_pct = role_counts.get(ResidentRole.SCHOOL_KID, 0) / total
        retired_pct = role_counts.get(ResidentRole.RETIRED, 0) / total

        assert 0.30 <= office_pct <= 0.50, f"Office workers: {office_pct:.0%}"
        assert 0.15 <= kid_pct <= 0.30, f"School kids: {kid_pct:.0%}"
        assert 0.10 <= retired_pct <= 0.25, f"Retired: {retired_pct:.0%}"

    def test_custom_mix(self):
        custom_mix = {
            ResidentRole.RETIRED: 1.0,
        }
        sim = NeighborhoodSim(num_residents=20, seed=42)
        sim.populate(mix=custom_mix)
        for r in sim.residents:
            assert r.role == ResidentRole.RETIRED

    def test_buildings_generated(self):
        sim = NeighborhoodSim(num_residents=50, seed=42)
        sim.populate()
        assert len(sim.buildings) > 0
        types = {b.building_type for b in sim.buildings}
        assert BuildingType.HOME in types
        assert BuildingType.OFFICE in types
        assert BuildingType.SCHOOL in types

    def test_vehicles_for_drivers(self):
        sim = NeighborhoodSim(num_residents=50, seed=42)
        sim.populate()
        drivers = [r for r in sim.residents if r.vehicle is not None]
        non_drivers = [r for r in sim.residents if r.vehicle is None]
        assert len(drivers) > 0
        assert len(non_drivers) > 0
        # Only vehicle roles should have cars
        for r in drivers:
            assert r.role in (
                ResidentRole.OFFICE_WORKER,
                ResidentRole.DELIVERY_DRIVER,
                ResidentRole.SERVICE_WORKER,
            )
        # Kids and retired should not have cars
        for r in non_drivers:
            assert r.role not in (
                ResidentRole.OFFICE_WORKER,
                ResidentRole.DELIVERY_DRIVER,
                ResidentRole.SERVICE_WORKER,
            )

    def test_residents_start_at_home(self):
        sim = NeighborhoodSim(num_residents=30, seed=42)
        sim.populate()
        for r in sim.residents:
            assert r.position == r.home_location
            assert r.current_activity == "sleeping"


# ---------------------------------------------------------------------------
# Simulation tick tests
# ---------------------------------------------------------------------------

class TestSimTick:
    def test_tick_changes_activities_morning(self):
        sim = NeighborhoodSim(num_residents=50, seed=42)
        sim.populate()
        # Tick to 8am — many should be commuting or working
        for _ in range(100):
            sim.tick(dt=1.0, current_time=8.0)
        stats = sim.get_statistics()
        activities = stats["activities"]
        # At 8am we expect commuting, working, at_school
        active = sum(
            activities.get(a, 0)
            for a in ("commuting", "working", "at_school", "delivering",
                      "waking_up", "walking")
        )
        assert active > 0, f"Nobody active at 8am: {activities}"

    def test_tick_changes_activities_night(self):
        sim = NeighborhoodSim(num_residents=50, seed=42)
        sim.populate()
        # Tick to midnight — almost all should be sleeping
        for _ in range(50):
            sim.tick(dt=1.0, current_time=1.0)
        stats = sim.get_statistics()
        sleeping = stats["activities"].get("sleeping", 0)
        assert sleeping > len(sim.residents) * 0.8, \
            f"At 1am only {sleeping}/{len(sim.residents)} sleeping"

    def test_tick_at_2pm_most_working(self):
        sim = NeighborhoodSim(num_residents=60, seed=42)
        sim.populate()
        for _ in range(50):
            sim.tick(dt=1.0, current_time=14.0)
        stats = sim.get_statistics()
        activities = stats["activities"]
        working = activities.get("working", 0) + activities.get("at_school", 0)
        total = len(sim.residents)
        # Office workers + WFH + service workers + kids should be working/school
        # That's 40% + 10% + 10% + 20% = 80%
        assert working > total * 0.4, \
            f"At 2pm only {working}/{total} working/school: {activities}"

    def test_vehicles_park_at_destination(self):
        sim = NeighborhoodSim(num_residents=30, seed=42)
        sim.populate()
        # Run enough ticks that commuters reach work
        for _ in range(500):
            sim.tick(dt=1.0, current_time=8.5)
        # Some vehicles should be parked (not driving)
        parked = [v for v in sim.vehicles if not v.driving]
        assert len(parked) > 0, "No vehicles parked after commute"

    def test_tick_advances_positions(self):
        sim = NeighborhoodSim(num_residents=20, seed=42)
        sim.populate()
        # Record initial positions
        initial = {r.resident_id: r.position for r in sim.residents}
        # Tick during commute time
        for _ in range(200):
            sim.tick(dt=1.0, current_time=7.5)
        # Some residents should have moved
        moved = 0
        for r in sim.residents:
            if r.position != initial[r.resident_id]:
                moved += 1
        assert moved > 0, "Nobody moved after 200 ticks during commute hour"


# ---------------------------------------------------------------------------
# Export format tests
# ---------------------------------------------------------------------------

class TestExportFormat:
    def test_get_all_entities_format(self):
        sim = NeighborhoodSim(num_residents=10, seed=42)
        sim.populate()
        entities = sim.get_all_entities()
        assert len(entities) > 0

        for e in entities:
            assert "target_id" in e
            assert "source" in e
            assert e["source"] == "city_sim"
            assert "position_x" in e
            assert "position_y" in e
            assert "alliance" in e
            assert e["alliance"] == "neutral"
            assert "classification" in e
            assert e["classification"] in ("person", "vehicle")
            assert "heading" in e
            assert "speed" in e
            assert "state" in e
            assert "metadata" in e
            assert e["metadata"]["simulated"] is True

    def test_resident_target_id_format(self):
        sim = NeighborhoodSim(num_residents=5, seed=42)
        sim.populate()
        entities = sim.get_all_entities()
        resident_ids = [e["target_id"] for e in entities if e["classification"] == "person"]
        for tid in resident_ids:
            assert tid.startswith("res_")

    def test_vehicle_target_id_format(self):
        sim = NeighborhoodSim(num_residents=10, seed=42)
        sim.populate()
        entities = sim.get_all_entities()
        vehicle_ids = [e["target_id"] for e in entities if e["classification"] == "vehicle"]
        for tid in vehicle_ids:
            assert tid.startswith("veh_")

    def test_entities_include_vehicles(self):
        sim = NeighborhoodSim(num_residents=20, seed=42)
        sim.populate()
        entities = sim.get_all_entities()
        vehicle_entities = [e for e in entities if e["classification"] == "vehicle"]
        assert len(vehicle_entities) > 0

    def test_get_statistics(self):
        sim = NeighborhoodSim(num_residents=30, seed=42)
        sim.populate()
        stats = sim.get_statistics()
        assert "total_residents" in stats
        assert stats["total_residents"] > 0
        assert "total_vehicles" in stats
        assert "vehicles_driving" in stats
        assert "vehicles_parked" in stats
        assert "activities" in stats
        assert "roles" in stats
        assert "total_buildings" in stats


# ---------------------------------------------------------------------------
# SimVehicle tests
# ---------------------------------------------------------------------------

class TestSimVehicle:
    def test_vehicle_parks(self):
        v = SimVehicle(vehicle_id="v1", owner_id="r1", position=(0.0, 0.0))
        v.park((100.0, 100.0))
        assert not v.driving
        assert v.parked_at == (100.0, 100.0)
        assert v.position == (100.0, 100.0)
        assert v.speed == 0.0

    def test_vehicle_drives_path(self):
        v = SimVehicle(vehicle_id="v1", owner_id="r1", position=(0.0, 0.0))
        v.start_driving([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)], speed=10.0)
        assert v.driving
        assert v.parked_at is None
        # Tick until arrived
        for _ in range(200):
            v.tick(1.0)
        assert not v.driving
        assert v.parked_at is not None

    def test_vehicle_to_dict(self):
        v = SimVehicle(vehicle_id="abc123", owner_id="r1", position=(50.0, 75.0))
        d = v.to_dict()
        assert d["target_id"] == "veh_abc123"
        assert d["classification"] == "vehicle"
        assert d["source"] == "city_sim"
        assert d["position_x"] == 50.0
        assert d["position_y"] == 75.0


# ---------------------------------------------------------------------------
# ScheduleEntry tests
# ---------------------------------------------------------------------------

class TestScheduleEntry:
    def test_all_schedules_cover_24h(self):
        """Every schedule factory should have entries starting near 0."""
        for factory in [
            DailySchedule.office_worker,
            DailySchedule.school_kid,
            DailySchedule.retired,
            DailySchedule.delivery_driver,
            DailySchedule.work_from_home,
            DailySchedule.service_worker,
        ]:
            sched = factory()
            assert len(sched.entries) >= 5
            # First entry should start at or near 0
            assert sched.entries[0].hour <= 1.0

    def test_no_overlapping_entries(self):
        """Entries should be sorted by hour with no duplicates."""
        for factory in [
            DailySchedule.office_worker,
            DailySchedule.school_kid,
            DailySchedule.retired,
        ]:
            sched = factory()
            hours = [e.hour for e in sched.entries]
            assert hours == sorted(hours), f"Not sorted: {hours}"
            # No duplicate hours
            assert len(set(hours)) == len(hours)


# ---------------------------------------------------------------------------
# Integration test — full day simulation
# ---------------------------------------------------------------------------

class TestFullDaySim:
    def test_simulate_full_day(self):
        """Run a full 24-hour day and verify patterns."""
        sim = NeighborhoodSim(num_residents=40, seed=123)
        sim.populate()

        hour_stats: dict[int, dict] = {}
        for hour_tenth in range(240):  # every 6 min
            current_time = hour_tenth / 10.0
            sim.tick(dt=1.0, current_time=current_time)
            hour = int(current_time)
            if hour not in hour_stats:
                hour_stats[hour] = sim.get_statistics()

        # Verify basic day patterns
        # Night: mostly sleeping
        night = hour_stats.get(2, {}).get("activities", {})
        assert night.get("sleeping", 0) > 20

        # Morning: some commuting
        morning = hour_stats.get(8, {}).get("activities", {})
        assert sum(morning.values()) == len(sim.residents)

        # All entities should be exportable at any time
        entities = sim.get_all_entities()
        assert len(entities) > 0

    def test_reproducible_with_seed(self):
        """Same seed produces same initial state."""
        sim1 = NeighborhoodSim(num_residents=20, seed=999)
        sim1.populate()
        sim2 = NeighborhoodSim(num_residents=20, seed=999)
        sim2.populate()

        assert len(sim1.residents) == len(sim2.residents)
        assert len(sim1.vehicles) == len(sim2.vehicles)
        for r1, r2 in zip(sim1.residents, sim2.residents):
            assert r1.name == r2.name
            assert r1.role == r2.role
            assert r1.home_location == r2.home_location
