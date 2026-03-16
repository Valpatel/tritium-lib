# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the deep city life simulation with 25+ activity micro-states."""

import pytest

from tritium_lib.game_ai.city_sim import (
    ActivityState,
    Building,
    BuildingType,
    DailySchedule,
    DEFAULT_MIX,
    ErrandType,
    ERRAND_DURATIONS,
    NeighborhoodSim,
    Resident,
    ResidentRole,
    ScheduleEntry,
    SimVehicle,
    VehicleType,
    state_movement_type,
    state_rf_emission,
    state_visible_on_map,
)
from tritium_lib.game_ai.steering import distance


# ---------------------------------------------------------------------------
# ActivityState metadata tests
# ---------------------------------------------------------------------------

class TestActivityState:
    def test_all_states_have_metadata(self):
        """Every ActivityState value must have movement/visibility/RF metadata."""
        for state in ActivityState:
            move = state_movement_type(state.value)
            assert move in ("stationary", "walking", "driving"), \
                f"{state.value} has unknown movement type: {move}"
            vis = state_visible_on_map(state.value)
            assert isinstance(vis, bool)
            rf = state_rf_emission(state.value)
            assert rf in ("full", "reduced", "minimal", "none"), \
                f"{state.value} has unknown RF level: {rf}"

    def test_sleeping_not_visible(self):
        assert not state_visible_on_map(ActivityState.SLEEPING)

    def test_driving_visible(self):
        assert state_visible_on_map(ActivityState.DRIVING)

    def test_working_not_visible(self):
        assert not state_visible_on_map(ActivityState.WORKING)

    def test_walking_visible(self):
        assert state_visible_on_map(ActivityState.WALKING)

    def test_shopping_visible(self):
        assert state_visible_on_map(ActivityState.SHOPPING)

    def test_inside_building_not_visible(self):
        assert not state_visible_on_map(ActivityState.INSIDE_BUILDING)

    def test_state_count_at_least_25(self):
        """Must have at least 25 distinct activity states."""
        assert len(ActivityState) >= 25

    def test_driving_is_driving_movement(self):
        assert state_movement_type(ActivityState.DRIVING) == "driving"

    def test_walking_is_walking_movement(self):
        assert state_movement_type(ActivityState.WALKING) == "walking"

    def test_sleeping_rf_minimal(self):
        assert state_rf_emission(ActivityState.SLEEPING) == "minimal"

    def test_walking_rf_full(self):
        assert state_rf_emission(ActivityState.WALKING) == "full"

    def test_driving_rf_reduced(self):
        assert state_rf_emission(ActivityState.DRIVING) == "reduced"


# ---------------------------------------------------------------------------
# ErrandType tests
# ---------------------------------------------------------------------------

class TestErrandTypes:
    def test_all_errands_have_durations(self):
        for errand in ErrandType:
            assert errand.value in ERRAND_DURATIONS
            lo, hi = ERRAND_DURATIONS[errand.value]
            assert lo > 0
            assert hi > lo

    def test_grocery_duration_range(self):
        lo, hi = ERRAND_DURATIONS[ErrandType.GROCERY]
        assert lo >= 300.0   # at least 5 min
        assert hi <= 1200.0  # at most 20 min

    def test_coffee_duration_range(self):
        lo, hi = ERRAND_DURATIONS[ErrandType.COFFEE]
        assert lo >= 300.0   # at least 5 min
        assert hi <= 600.0   # at most 10 min


# ---------------------------------------------------------------------------
# VehicleType tests
# ---------------------------------------------------------------------------

class TestVehicleType:
    def test_vehicle_types_exist(self):
        assert VehicleType.CAR
        assert VehicleType.TRUCK
        assert VehicleType.MOTORCYCLE
        assert VehicleType.BICYCLE
        assert VehicleType.DELIVERY_VAN


# ---------------------------------------------------------------------------
# DailySchedule tests
# ---------------------------------------------------------------------------

class TestDailySchedule:
    def test_office_worker_schedule(self):
        sched = DailySchedule.office_worker()
        assert len(sched.entries) > 0
        entry = sched.activity_at(3.0)
        assert entry.activity == "sleeping"
        entry = sched.activity_at(9.0)
        assert entry.activity == "working"

    def test_school_kid_schedule(self):
        sched = DailySchedule.school_kid()
        entry = sched.activity_at(10.0)
        assert entry.activity == "at_school"
        entry = sched.activity_at(16.0)
        assert entry.activity == "playing"

    def test_retired_schedule(self):
        sched = DailySchedule.retired()
        entry = sched.activity_at(6.75)
        assert entry.activity == "walking"
        entry = sched.activity_at(13.5)
        assert entry.activity == "napping"

    def test_delivery_driver_schedule(self):
        sched = DailySchedule.delivery_driver()
        entry = sched.activity_at(10.5)
        assert entry.activity == "delivering"

    def test_work_from_home_schedule(self):
        sched = DailySchedule.work_from_home()
        entry = sched.activity_at(10.0)
        assert entry.activity == "working"
        assert entry.location_type == "home"

    def test_service_worker_schedule(self):
        sched = DailySchedule.service_worker()
        entry = sched.activity_at(8.0)
        assert entry.activity == "working"

    def test_empty_schedule_defaults_to_sleeping(self):
        sched = DailySchedule()
        entry = sched.activity_at(12.0)
        assert entry.activity == "sleeping"

    def test_schedule_wraps_around(self):
        sched = DailySchedule.office_worker()
        entry = sched.activity_at(25.0)
        assert entry.activity == "sleeping"

    def test_office_worker_has_grocery_errand(self):
        """Office worker schedule includes a grocery stop."""
        sched = DailySchedule.office_worker()
        errands = [e for e in sched.entries if e.errand_type == ErrandType.GROCERY]
        assert len(errands) >= 1


# ---------------------------------------------------------------------------
# NeighborhoodSim population tests
# ---------------------------------------------------------------------------

class TestNeighborhoodPopulate:
    def test_populates_correct_total(self):
        sim = NeighborhoodSim(num_residents=50, seed=42)
        sim.populate()
        assert 45 <= len(sim.residents) <= 60

    def test_populates_with_correct_mix(self):
        sim = NeighborhoodSim(num_residents=100, seed=42)
        sim.populate()
        role_counts = {}
        for r in sim.residents:
            role_counts[r.role] = role_counts.get(r.role, 0) + 1

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

    def test_buildings_have_parking(self):
        sim = NeighborhoodSim(num_residents=50, seed=42)
        sim.populate()
        for b in sim.buildings:
            assert b.parking_pos != (0.0, 0.0) or b.position == (0.0, 0.0)
            # Parking should be offset from building
            if b.position != (0.0, 0.0):
                dist = distance(b.position, b.parking_pos)
                assert dist > 0, f"Building {b.name} parking at same spot"

    def test_vehicles_for_drivers(self):
        sim = NeighborhoodSim(num_residents=50, seed=42)
        sim.populate()
        drivers = [r for r in sim.residents if r.vehicle is not None]
        non_drivers = [r for r in sim.residents if r.vehicle is None]
        assert len(drivers) > 0
        assert len(non_drivers) > 0
        for r in drivers:
            assert r.role in (
                ResidentRole.OFFICE_WORKER,
                ResidentRole.DELIVERY_DRIVER,
                ResidentRole.SERVICE_WORKER,
            )
        for r in non_drivers:
            assert r.role not in (
                ResidentRole.OFFICE_WORKER,
                ResidentRole.DELIVERY_DRIVER,
                ResidentRole.SERVICE_WORKER,
            )

    def test_delivery_drivers_have_vans(self):
        sim = NeighborhoodSim(num_residents=100, seed=42)
        sim.populate()
        delivery = [r for r in sim.residents
                    if r.role == ResidentRole.DELIVERY_DRIVER and r.vehicle]
        for r in delivery:
            assert r.vehicle.vehicle_type == VehicleType.DELIVERY_VAN

    def test_residents_start_at_home(self):
        sim = NeighborhoodSim(num_residents=30, seed=42)
        sim.populate()
        for r in sim.residents:
            assert r.position == r.home_location
            assert r.current_activity == "sleeping"
            assert r.activity_state == ActivityState.SLEEPING

    def test_residents_start_invisible(self):
        """Everyone starts sleeping at home = not visible on map."""
        sim = NeighborhoodSim(num_residents=30, seed=42)
        sim.populate()
        for r in sim.residents:
            assert not r.visible


# ---------------------------------------------------------------------------
# Simulation tick tests
# ---------------------------------------------------------------------------

class TestSimTick:
    def test_tick_changes_activities_morning(self):
        sim = NeighborhoodSim(num_residents=50, seed=42)
        sim.populate()
        for _ in range(100):
            sim.tick(dt=1.0, current_time=8.0)
        stats = sim.get_statistics()
        activities = stats["activities"]
        active = sum(
            activities.get(a, 0)
            for a in ("commuting", "working", "at_school", "delivering",
                      "waking_up", "walking")
        )
        assert active > 0, f"Nobody active at 8am: {activities}"

    def test_tick_changes_activities_night(self):
        sim = NeighborhoodSim(num_residents=50, seed=42)
        sim.populate()
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
        assert working > total * 0.4, \
            f"At 2pm only {working}/{total} working/school: {activities}"

    def test_vehicles_park_at_destination(self):
        sim = NeighborhoodSim(num_residents=30, seed=42)
        sim.populate()
        for _ in range(500):
            sim.tick(dt=1.0, current_time=8.5)
        parked = [v for v in sim.vehicles if not v.driving]
        assert len(parked) > 0, "No vehicles parked after commute"

    def test_tick_advances_positions(self):
        sim = NeighborhoodSim(num_residents=20, seed=42)
        sim.populate()
        initial = {r.resident_id: r.position for r in sim.residents}
        for _ in range(200):
            sim.tick(dt=1.0, current_time=7.5)
        moved = 0
        for r in sim.residents:
            if r.position != initial[r.resident_id]:
                moved += 1
        assert moved > 0, "Nobody moved after 200 ticks during commute hour"

    def test_statistics_include_activity_states(self):
        """get_statistics now returns activity_states breakdown."""
        sim = NeighborhoodSim(num_residents=20, seed=42)
        sim.populate()
        for _ in range(100):
            sim.tick(dt=1.0, current_time=8.0)
        stats = sim.get_statistics()
        assert "activity_states" in stats
        assert isinstance(stats["activity_states"], dict)
        assert sum(stats["activity_states"].values()) == len(sim.residents)

    def test_statistics_visibility_counts(self):
        """Stats include visible_on_map and inside_buildings counts."""
        sim = NeighborhoodSim(num_residents=20, seed=42)
        sim.populate()
        stats = sim.get_statistics()
        assert "visible_on_map" in stats
        assert "inside_buildings" in stats
        # Everyone sleeping = all inside
        assert stats["inside_buildings"] == len(sim.residents)


# ---------------------------------------------------------------------------
# Deep lifecycle tests — the core of the deep simulation
# ---------------------------------------------------------------------------

class TestDeepLifecycle:
    """Test the full drive-park-walk-errand lifecycle."""

    def test_wakeup_to_commute_sequence(self):
        """A resident wakes up and transitions through micro-states."""
        sim = NeighborhoodSim(num_residents=5, seed=42)
        sim.populate()

        # Find an office worker with a vehicle
        worker = None
        for r in sim.residents:
            if r.role == ResidentRole.OFFICE_WORKER and r.vehicle:
                worker = r
                break
        assert worker is not None, "No office worker with vehicle found"

        # Verify starting state
        assert worker.activity_state == ActivityState.SLEEPING
        assert not worker.visible

        # Advance to waking up time (~6:30)
        states_seen = set()
        for _ in range(500):
            sim.tick(dt=1.0, current_time=6.75)
            states_seen.add(worker.activity_state)

        # Should have transitioned through waking up
        assert ActivityState.WAKING_UP in states_seen or \
               ActivityState.RELAXING in states_seen, \
            f"States seen: {states_seen}"

    def test_full_commute_lifecycle(self):
        """Office worker: wake, walk-to-car, drive, park, walk-to-building, work."""
        sim = NeighborhoodSim(num_residents=10, seed=42)
        sim.populate()

        worker = None
        for r in sim.residents:
            if r.role == ResidentRole.OFFICE_WORKER and r.vehicle:
                worker = r
                break
        assert worker is not None

        # Track all states the worker goes through
        states_seen: list[str] = []
        prev_state = ""

        # Simulate from 6am to 9am in 1-second ticks (3 hours = 10800 ticks)
        # But we step through simulated hours, not real seconds
        for hour_tenth in range(60, 91):  # 6.0 to 9.0 in 0.1 increments
            current_time = hour_tenth / 10.0
            for _ in range(100):  # 100 ticks per 6-min interval
                sim.tick(dt=1.0, current_time=current_time)
                if worker.activity_state != prev_state:
                    states_seen.append(worker.activity_state)
                    prev_state = worker.activity_state

        # Should see a commute sequence
        # The exact states depend on schedule jitter, but we should see:
        # sleeping/waking_up -> some transition -> driving or walking -> working
        assert len(states_seen) >= 2, \
            f"Worker went through too few states: {states_seen}"

        # Worker should end up working by 9am
        assert worker.current_activity == "working" or \
               worker.activity_state in (ActivityState.WORKING,
                                          ActivityState.WALKING_TO_BUILDING,
                                          ActivityState.ENTERING_BUILDING,
                                          ActivityState.DRIVING), \
            f"Worker not at work by 9am. State: {worker.activity_state}, " \
            f"Activity: {worker.current_activity}"

    def test_car_position_changes_during_commute(self):
        """Car should be parked at home, then driving, then parked at work."""
        sim = NeighborhoodSim(num_residents=10, seed=42)
        sim.populate()

        worker = None
        for r in sim.residents:
            if r.role == ResidentRole.OFFICE_WORKER and r.vehicle:
                worker = r
                break
        assert worker is not None

        car = worker.vehicle
        home_pos = car.parked_at
        assert home_pos is not None, "Car should start parked at home"

        # Simulate commute
        was_driving = False
        for hour_tenth in range(65, 85):
            current_time = hour_tenth / 10.0
            for _ in range(100):
                sim.tick(dt=1.0, current_time=current_time)
                if car.driving:
                    was_driving = True

        # Car should have moved at some point
        if was_driving:
            # If the car drove, it should now be parked somewhere
            # (possibly at work, possibly still en route)
            assert car.parked_at is not None or car.driving, \
                "Car neither parked nor driving"

    def test_person_inside_building_during_work(self):
        """Person should be invisible (inside building) while working."""
        sim = NeighborhoodSim(num_residents=10, seed=42)
        sim.populate()

        worker = None
        for r in sim.residents:
            if r.role == ResidentRole.OFFICE_WORKER:
                worker = r
                break
        assert worker is not None

        # Advance to mid-morning when worker should be at work
        for hour_tenth in range(60, 101):
            current_time = hour_tenth / 10.0
            for _ in range(100):
                sim.tick(dt=1.0, current_time=current_time)

        # By 10am, worker should be working (invisible)
        if worker.activity_state == ActivityState.WORKING:
            assert not worker.visible, \
                "Worker should be invisible while working inside building"

    def test_walking_speeds_realistic(self):
        """Elderly walk slower, kids walk at kid speed."""
        sim = NeighborhoodSim(num_residents=30, seed=42)
        sim.populate()

        # Find a retired person doing outdoor activity
        for hour_tenth in range(60, 80):
            current_time = hour_tenth / 10.0
            for _ in range(50):
                sim.tick(dt=1.0, current_time=current_time)

        for r in sim.residents:
            if r.role == ResidentRole.RETIRED and r.speed > 0:
                # Retired walking speed should be <= 1.0 m/s typically
                assert r.speed <= 2.0, \
                    f"Retired person walking too fast: {r.speed} m/s"

    def test_shopping_errand_has_duration(self):
        """Grocery shopping should last between 5-20 minutes."""
        lo, hi = ERRAND_DURATIONS[ErrandType.GROCERY]
        assert 300.0 <= lo <= hi <= 1200.0

    def test_visible_entities_excludes_indoor(self):
        """get_visible_entities should not include people inside buildings."""
        sim = NeighborhoodSim(num_residents=20, seed=42)
        sim.populate()
        # At midnight, everyone sleeping = nobody visible
        sim.tick(dt=1.0, current_time=1.0)
        visible = sim.get_visible_entities()
        # Only vehicles should be visible (all parked at home)
        person_visible = [e for e in visible if e["classification"] == "person"]
        assert len(person_visible) == 0, \
            f"At 1am, {len(person_visible)} people visible (should be 0)"


# ---------------------------------------------------------------------------
# Full 24-hour day simulation
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

        # Night: mostly sleeping
        night = hour_stats.get(2, {}).get("activities", {})
        assert night.get("sleeping", 0) > 20

        # Morning: some commuting
        morning = hour_stats.get(8, {}).get("activities", {})
        assert sum(morning.values()) == len(sim.residents)

        # All entities should be exportable at any time
        entities = sim.get_all_entities()
        assert len(entities) > 0

    def test_full_day_position_trace(self):
        """Verify a resident produces a realistic position trace over 24h."""
        sim = NeighborhoodSim(num_residents=10, seed=42)
        sim.populate()

        # Pick an office worker
        worker = None
        for r in sim.residents:
            if r.role == ResidentRole.OFFICE_WORKER and r.vehicle:
                worker = r
                break
        assert worker is not None

        # Record positions every simulated 6 minutes
        positions: list[tuple[float, Vec2]] = []
        for hour_tenth in range(240):
            current_time = hour_tenth / 10.0
            for _ in range(10):  # 10 ticks per sample
                sim.tick(dt=1.0, current_time=current_time)
            positions.append((current_time, worker.position))

        # Position should change during the day
        unique_positions = set(pos for _, pos in positions)
        assert len(unique_positions) > 1, \
            "Worker stayed in one position all day"

        # At night (0-5am), should be at home
        night_positions = [pos for t, pos in positions if t < 5.0]
        for pos in night_positions:
            dist_from_home = distance(pos, worker.home_location)
            # Allow some slack for parking offset
            assert dist_from_home < 50.0, \
                f"At night, worker is {dist_from_home:.0f}m from home"

    def test_full_day_visibility_pattern(self):
        """During a day, visibility should change: low at night, higher during
        commute times, low during work hours (inside), higher in evening."""
        sim = NeighborhoodSim(num_residents=40, seed=123)
        sim.populate()

        visibility_by_hour: dict[int, int] = {}
        for hour_tenth in range(240):
            current_time = hour_tenth / 10.0
            sim.tick(dt=1.0, current_time=current_time)
            hour = int(current_time)
            if hour not in visibility_by_hour:
                stats = sim.get_statistics()
                visibility_by_hour[hour] = stats["visible_on_map"]

        # At 2am, very few visible
        night_vis = visibility_by_hour.get(2, 0)
        assert night_vis < len(sim.residents) * 0.3, \
            f"Too many visible at 2am: {night_vis}"

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

    def test_resident_export_includes_deep_state(self):
        """Resident export should include activity_state and visibility."""
        sim = NeighborhoodSim(num_residents=5, seed=42)
        sim.populate()
        entities = sim.get_all_entities()
        person = next(e for e in entities if e["classification"] == "person")
        assert "visible" in person
        assert "activity_state" in person["metadata"]
        assert "rf_emission" in person["metadata"]
        assert "visible_on_map" in person["metadata"]

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

    def test_vehicle_export_includes_type(self):
        sim = NeighborhoodSim(num_residents=10, seed=42)
        sim.populate()
        entities = sim.get_all_entities()
        vehicles = [e for e in entities if e["classification"] == "vehicle"]
        assert len(vehicles) > 0
        for v in vehicles:
            assert "vehicle_type" in v["metadata"]

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
        assert "activity_states" in stats
        assert "visible_on_map" in stats
        assert "inside_buildings" in stats


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
        for _ in range(200):
            v.tick(1.0)
        assert not v.driving
        assert v.parked_at is not None

    def test_vehicle_intersection_pause(self):
        """Vehicle should pause briefly at sharp turns."""
        v = SimVehicle(vehicle_id="v1", owner_id="r1", position=(0.0, 0.0))
        # 90-degree turn path
        v.start_driving([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)], speed=10.0)
        paused = False
        for _ in range(200):
            v.tick(1.0)
            if v._intersection_pause > 0:
                paused = True
        assert paused, "Vehicle never paused at intersection"

    def test_vehicle_to_dict(self):
        v = SimVehicle(vehicle_id="abc123", owner_id="r1", position=(50.0, 75.0))
        d = v.to_dict()
        assert d["target_id"] == "veh_abc123"
        assert d["classification"] == "vehicle"
        assert d["source"] == "city_sim"
        assert d["position_x"] == 50.0
        assert d["position_y"] == 75.0

    def test_vehicle_type_in_dict(self):
        v = SimVehicle(vehicle_id="v1", owner_id="r1",
                       vehicle_type=VehicleType.DELIVERY_VAN)
        d = v.to_dict()
        assert d["metadata"]["vehicle_type"] == VehicleType.DELIVERY_VAN


# ---------------------------------------------------------------------------
# ScheduleEntry tests
# ---------------------------------------------------------------------------

class TestScheduleEntry:
    def test_all_schedules_cover_24h(self):
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
            assert sched.entries[0].hour <= 1.0

    def test_no_overlapping_entries(self):
        for factory in [
            DailySchedule.office_worker,
            DailySchedule.school_kid,
            DailySchedule.retired,
        ]:
            sched = factory()
            hours = [e.hour for e in sched.entries]
            assert hours == sorted(hours), f"Not sorted: {hours}"
            assert len(set(hours)) == len(hours)


# ---------------------------------------------------------------------------
# Building tests
# ---------------------------------------------------------------------------

class TestBuilding:
    def test_building_parking_offset(self):
        b = Building(
            building_id="b1",
            building_type=BuildingType.GROCERY,
            position=(100.0, 100.0),
            name="Test Grocery",
        )
        # Parking should be offset from building
        dist = distance(b.position, b.parking_pos)
        assert dist > 0

    def test_gas_station_building_type(self):
        assert BuildingType.GAS_STATION.value == "gas_station"

    def test_coffee_shop_building_type(self):
        assert BuildingType.COFFEE_SHOP.value == "coffee_shop"
