# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.scenarios — scenario generator, player, and data models."""

from __future__ import annotations

import pytest

from tritium_lib.scenarios import (
    AlertLevel,
    EntityAlliance,
    EntityType,
    EventKind,
    ExpectedAlert,
    ExpectedDetection,
    GeoZone,
    PlayerState,
    Scenario,
    ScenarioEntity,
    ScenarioEvent,
    ScenarioGenerator,
    ScenarioPlayer,
    TEMPLATE_NAMES,
    ZoneType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator() -> ScenarioGenerator:
    return ScenarioGenerator(seed=42)


@pytest.fixture
def airport_scenario(generator: ScenarioGenerator) -> Scenario:
    return generator.create("airport_surveillance")


@pytest.fixture
def border_scenario(generator: ScenarioGenerator) -> Scenario:
    return generator.create("border_crossing")


@pytest.fixture
def urban_scenario(generator: ScenarioGenerator) -> Scenario:
    return generator.create("urban_patrol")


@pytest.fixture
def maritime_scenario(generator: ScenarioGenerator) -> Scenario:
    return generator.create("maritime_port")


@pytest.fixture
def campus_scenario(generator: ScenarioGenerator) -> Scenario:
    return generator.create("campus_security")


# ---------------------------------------------------------------------------
# Test: Template names and generator basics
# ---------------------------------------------------------------------------

class TestGeneratorBasics:
    def test_template_names_exist(self):
        """All five built-in scenario templates are present."""
        assert "airport_surveillance" in TEMPLATE_NAMES
        assert "border_crossing" in TEMPLATE_NAMES
        assert "urban_patrol" in TEMPLATE_NAMES
        assert "maritime_port" in TEMPLATE_NAMES
        assert "campus_security" in TEMPLATE_NAMES

    def test_template_count(self):
        """Exactly five templates are defined."""
        assert len(TEMPLATE_NAMES) == 5

    def test_available_templates_matches(self, generator: ScenarioGenerator):
        """Generator's available_templates matches TEMPLATE_NAMES."""
        assert generator.available_templates == TEMPLATE_NAMES

    def test_unknown_template_raises(self, generator: ScenarioGenerator):
        """Requesting an unknown template raises ValueError."""
        with pytest.raises(ValueError, match="Unknown template"):
            generator.create("nonexistent_scenario")

    def test_create_all(self, generator: ScenarioGenerator):
        """create_all generates one scenario per template."""
        scenarios = generator.create_all()
        assert len(scenarios) == 5
        names = {s.template for s in scenarios}
        assert names == set(TEMPLATE_NAMES)


# ---------------------------------------------------------------------------
# Test: Scenario structure
# ---------------------------------------------------------------------------

class TestScenarioStructure:
    def test_scenario_has_name(self, airport_scenario: Scenario):
        assert airport_scenario.name != ""
        assert airport_scenario.description != ""

    def test_scenario_has_zones(self, airport_scenario: Scenario):
        """Airport scenario defines geographic zones."""
        assert len(airport_scenario.zones) > 0
        zone_types = {z.zone_type for z in airport_scenario.zones}
        assert ZoneType.TERMINAL in zone_types
        assert ZoneType.RUNWAY in zone_types
        assert ZoneType.PARKING in zone_types

    def test_scenario_has_entities(self, airport_scenario: Scenario):
        """Scenario has entities from normal patterns + threats."""
        assert len(airport_scenario.entities) > 0
        # Should have passengers, staff, security, vehicles, and threats
        types = {e.entity_type for e in airport_scenario.entities}
        assert EntityType.PERSON in types

    def test_scenario_has_events(self, airport_scenario: Scenario):
        """Scenario has timeline events."""
        assert len(airport_scenario.events) > 0
        kinds = {e.kind for e in airport_scenario.events}
        assert EventKind.SPAWN in kinds
        assert EventKind.THREAT_INJECT in kinds

    def test_scenario_has_expected_alerts(self, airport_scenario: Scenario):
        """Airport scenario defines expected alerts from threats."""
        assert len(airport_scenario.expected_alerts) > 0

    def test_scenario_has_expected_detections(self, airport_scenario: Scenario):
        """Scenario defines expected detections for sensor-visible entities."""
        assert len(airport_scenario.expected_detections) > 0
        sensor_types = {d.sensor_type for d in airport_scenario.expected_detections}
        # At least camera and BLE detections expected
        assert "camera" in sensor_types
        assert "ble" in sensor_types

    def test_scenario_has_geographic_center(self, airport_scenario: Scenario):
        """Scenario has a valid geographic center."""
        assert airport_scenario.center_lat != 0.0
        assert airport_scenario.center_lng != 0.0

    def test_scenario_has_duration(self, airport_scenario: Scenario):
        assert airport_scenario.duration_s > 0

    def test_scenario_has_tags(self, airport_scenario: Scenario):
        assert len(airport_scenario.tags) > 0
        assert "airport" in airport_scenario.tags


# ---------------------------------------------------------------------------
# Test: Entity classification
# ---------------------------------------------------------------------------

class TestEntityClassification:
    def test_hostile_entities(self, airport_scenario: Scenario):
        """Threat actors are classified as hostile."""
        hostiles = airport_scenario.hostile_entities()
        assert len(hostiles) > 0
        for h in hostiles:
            assert h.alliance == EntityAlliance.HOSTILE

    def test_friendly_entities(self, airport_scenario: Scenario):
        """Staff and security are classified as friendly."""
        friendlies = airport_scenario.friendly_entities()
        assert len(friendlies) > 0
        for f in friendlies:
            assert f.alliance == EntityAlliance.FRIENDLY

    def test_neutral_entities(self, airport_scenario: Scenario):
        """Passengers and vehicles are classified as neutral."""
        neutrals = airport_scenario.neutral_entities()
        assert len(neutrals) > 0
        for n in neutrals:
            assert n.alliance == EntityAlliance.NEUTRAL

    def test_entity_by_id(self, airport_scenario: Scenario):
        """Can look up entities by ID."""
        entity = airport_scenario.entities[0]
        found = airport_scenario.entity_by_id(entity.entity_id)
        assert found is entity

    def test_entity_by_id_not_found(self, airport_scenario: Scenario):
        assert airport_scenario.entity_by_id("nonexistent") is None

    def test_hostile_entities_have_sensor_signatures(self, airport_scenario: Scenario):
        """Threat actors have BLE and camera visibility."""
        for h in airport_scenario.hostile_entities():
            assert h.camera_visible is True
            assert h.ble_visible is True
            assert h.mac_address != ""


# ---------------------------------------------------------------------------
# Test: Zone operations
# ---------------------------------------------------------------------------

class TestZoneOperations:
    def test_zone_by_id(self, airport_scenario: Scenario):
        zone = airport_scenario.zone_by_id("terminal-main")
        assert zone is not None
        assert zone.name == "Main Terminal"

    def test_zone_by_id_not_found(self, airport_scenario: Scenario):
        assert airport_scenario.zone_by_id("nonexistent") is None

    def test_zone_contains(self):
        """GeoZone.contains correctly tests point inclusion."""
        zone = GeoZone(
            zone_id="test",
            name="Test Zone",
            zone_type=ZoneType.BUILDING,
            center_lat=37.0,
            center_lng=-122.0,
            radius_m=100.0,
        )
        # Center should be inside
        assert zone.contains(37.0, -122.0) is True
        # Point far away should be outside
        assert zone.contains(38.0, -122.0) is False

    def test_zone_radius_positive(self, airport_scenario: Scenario):
        for zone in airport_scenario.zones:
            assert zone.radius_m > 0


# ---------------------------------------------------------------------------
# Test: Events and timeline
# ---------------------------------------------------------------------------

class TestEventsTimeline:
    def test_sorted_events(self, airport_scenario: Scenario):
        """sorted_events returns events in chronological order."""
        sorted_ev = airport_scenario.sorted_events()
        for i in range(1, len(sorted_ev)):
            assert sorted_ev[i].time_offset_s >= sorted_ev[i - 1].time_offset_s

    def test_events_for_entity(self, airport_scenario: Scenario):
        """events_for_entity filters correctly."""
        # Get a threat entity
        hostiles = airport_scenario.hostile_entities()
        assert len(hostiles) > 0
        eid = hostiles[0].entity_id
        events = airport_scenario.events_for_entity(eid)
        assert len(events) > 0
        for e in events:
            assert e.entity_id == eid

    def test_threat_inject_events_have_positive_time(self, airport_scenario: Scenario):
        """Threat injections happen after t=0."""
        for e in airport_scenario.events:
            if e.kind == EventKind.THREAT_INJECT:
                assert e.time_offset_s > 0

    def test_spawn_events_at_time_zero(self, airport_scenario: Scenario):
        """Normal entity spawns happen at t=0."""
        spawn_events = [e for e in airport_scenario.events if e.kind == EventKind.SPAWN]
        assert len(spawn_events) > 0
        for e in spawn_events:
            assert e.time_offset_s == 0.0

    def test_computed_duration(self, airport_scenario: Scenario):
        """computed_duration returns the explicit duration."""
        assert airport_scenario.computed_duration() == airport_scenario.duration_s

    def test_computed_duration_from_events(self):
        """When duration_s=0, falls back to max event time."""
        s = Scenario(duration_s=0.0)
        s.events.append(ScenarioEvent(time_offset_s=50.0))
        s.events.append(ScenarioEvent(time_offset_s=100.0))
        assert s.computed_duration() == 100.0

    def test_computed_duration_empty(self):
        """Empty scenario with no duration returns 0."""
        s = Scenario(duration_s=0.0)
        assert s.computed_duration() == 0.0


# ---------------------------------------------------------------------------
# Test: Scenario serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict(self, airport_scenario: Scenario):
        """Scenario can be serialized to dict."""
        d = airport_scenario.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == airport_scenario.name
        assert d["template"] == "airport_surveillance"
        assert len(d["zones"]) == len(airport_scenario.zones)
        assert len(d["entities"]) == len(airport_scenario.entities)
        assert len(d["events"]) == len(airport_scenario.events)

    def test_to_dict_is_json_serializable(self, airport_scenario: Scenario):
        """Serialized dict can be JSON-encoded."""
        import json
        d = airport_scenario.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert len(serialized) > 100


# ---------------------------------------------------------------------------
# Test: Cross-template consistency
# ---------------------------------------------------------------------------

class TestCrossTemplate:
    @pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
    def test_all_templates_generate(self, generator: ScenarioGenerator, template_name: str):
        """Every template generates a valid scenario."""
        s = generator.create(template_name)
        assert s.name != ""
        assert len(s.zones) > 0
        assert len(s.entities) > 0
        assert len(s.events) > 0
        assert s.duration_s > 0
        assert s.center_lat != 0.0
        assert s.center_lng != 0.0

    @pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
    def test_all_templates_have_threats(self, generator: ScenarioGenerator, template_name: str):
        """Every template includes at least one threat."""
        s = generator.create(template_name)
        hostiles = s.hostile_entities()
        assert len(hostiles) >= 1

    @pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
    def test_all_templates_have_friendlies(self, generator: ScenarioGenerator, template_name: str):
        """Every template includes friendly entities (security/staff)."""
        s = generator.create(template_name)
        friendlies = s.friendly_entities()
        assert len(friendlies) >= 1

    @pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
    def test_all_templates_have_expected_alerts(self, generator: ScenarioGenerator, template_name: str):
        """Every template defines expected alerts."""
        s = generator.create(template_name)
        assert len(s.expected_alerts) >= 1


# ---------------------------------------------------------------------------
# Test: Generator overrides
# ---------------------------------------------------------------------------

class TestGeneratorOverrides:
    def test_override_seed(self, generator: ScenarioGenerator):
        """Different seeds produce different scenarios."""
        s1 = generator.create("airport_surveillance", seed=1)
        s2 = generator.create("airport_surveillance", seed=2)
        # Entity MACs should differ with different seeds
        macs1 = {e.mac_address for e in s1.entities if e.mac_address}
        macs2 = {e.mac_address for e in s2.entities if e.mac_address}
        assert macs1 != macs2

    def test_same_seed_reproducible(self):
        """Same seed produces identical scenarios."""
        g1 = ScenarioGenerator(seed=99)
        g2 = ScenarioGenerator(seed=99)
        s1 = g1.create("urban_patrol")
        s2 = g2.create("urban_patrol")
        assert len(s1.entities) == len(s2.entities)
        assert len(s1.events) == len(s2.events)
        assert s1.entities[0].mac_address == s2.entities[0].mac_address

    def test_override_center(self, generator: ScenarioGenerator):
        """Can override geographic center."""
        custom_center = (51.5074, -0.1278)  # London
        s = generator.create("airport_surveillance", center=custom_center)
        assert abs(s.center_lat - 51.5074) < 0.001
        assert abs(s.center_lng - (-0.1278)) < 0.001

    def test_override_duration(self, generator: ScenarioGenerator):
        """Can override scenario duration."""
        s = generator.create("airport_surveillance", duration_s=300.0)
        assert s.duration_s == 300.0

    def test_override_threat_count(self, generator: ScenarioGenerator):
        """Can limit the number of threats injected."""
        s0 = generator.create("border_crossing", threat_count=0)
        assert len(s0.hostile_entities()) == 0

        s1 = generator.create("border_crossing", threat_count=1)
        assert len(s1.hostile_entities()) == 1


# ---------------------------------------------------------------------------
# Test: Scenario Player
# ---------------------------------------------------------------------------

class TestScenarioPlayer:
    def test_player_creation(self, airport_scenario: Scenario):
        player = ScenarioPlayer(airport_scenario, step_interval_s=10.0)
        assert player.scenario is airport_scenario
        assert player.is_complete is False
        assert player.state.current_time_s == 0.0

    def test_player_initial_state(self, airport_scenario: Scenario):
        """Non-hostile entities are active from the start."""
        player = ScenarioPlayer(airport_scenario)
        # Non-hostiles should be positioned
        non_hostile_count = len([e for e in airport_scenario.entities
                                 if e.alliance != EntityAlliance.HOSTILE])
        assert len(player.state.active_entity_ids) == non_hostile_count

    def test_step_once_advances_time(self, airport_scenario: Scenario):
        player = ScenarioPlayer(airport_scenario, step_interval_s=10.0)
        events = player.step_once()
        assert player.state.current_time_s == 10.0
        assert player.state.step_index == 1
        assert isinstance(events, list)

    def test_step_once_returns_spawn_events(self, airport_scenario: Scenario):
        """First step should capture spawn events (at t=0)."""
        player = ScenarioPlayer(airport_scenario, step_interval_s=10.0)
        events = player.step_once()
        spawn_events = [e for e in events if e.kind == EventKind.SPAWN]
        assert len(spawn_events) > 0

    def test_player_completes(self, airport_scenario: Scenario):
        """Player completes after reaching scenario duration."""
        # Use a large step to finish quickly
        player = ScenarioPlayer(airport_scenario, step_interval_s=1000.0)
        player.step_once()
        assert player.is_complete is True

    def test_step_once_after_completion(self, airport_scenario: Scenario):
        """step_once returns empty list after completion."""
        player = ScenarioPlayer(airport_scenario, step_interval_s=1000.0)
        player.step_once()
        assert player.is_complete is True
        events = player.step_once()
        assert events == []

    def test_player_activates_threats(self, airport_scenario: Scenario):
        """Threats become active when their injection time is reached."""
        player = ScenarioPlayer(airport_scenario, step_interval_s=50.0)
        initial_active = len(player.state.active_entity_ids)

        # Step through until past the first threat injection
        for _ in range(20):
            player.step_once()

        # Should have more active entities now (threats activated)
        assert len(player.state.active_entity_ids) > initial_active

    def test_player_reset(self, airport_scenario: Scenario):
        """Reset returns player to initial state."""
        player = ScenarioPlayer(airport_scenario, step_interval_s=10.0)
        player.step_once()
        player.step_once()
        assert player.state.step_index == 2

        player.reset()
        assert player.state.step_index == 0
        assert player.state.current_time_s == 0.0
        assert player.is_complete is False

    def test_player_moves_entities(self, airport_scenario: Scenario):
        """Entities change position over time."""
        player = ScenarioPlayer(airport_scenario, step_interval_s=30.0)
        # Get initial position of first entity
        first_entity = airport_scenario.entities[0]
        initial_pos = player.state.entity_positions.get(first_entity.entity_id)
        assert initial_pos is not None

        # Advance several steps
        for _ in range(5):
            player.step_once()

        new_pos = player.state.entity_positions.get(first_entity.entity_id)
        assert new_pos is not None
        # Position should have changed if entity has waypoints and speed
        if first_entity.waypoints and first_entity.speed_mps > 0:
            assert new_pos != initial_pos

    def test_player_triggers_detections(self, airport_scenario: Scenario):
        """Expected detections are triggered as time progresses."""
        player = ScenarioPlayer(airport_scenario, step_interval_s=10.0)
        for _ in range(10):
            player.step_once()
        assert len(player.state.triggered_detections) > 0

    def test_step_iterator(self, airport_scenario: Scenario):
        """step() iterator yields events until completion."""
        player = ScenarioPlayer(airport_scenario, step_interval_s=100.0)
        steps = list(player.step())
        assert len(steps) > 0
        assert player.is_complete is True


# ---------------------------------------------------------------------------
# Test: Data model edge cases
# ---------------------------------------------------------------------------

class TestDataModels:
    def test_empty_scenario(self):
        """An empty scenario has sensible defaults."""
        s = Scenario()
        assert s.name == ""
        assert len(s.zones) == 0
        assert len(s.entities) == 0
        assert len(s.events) == 0
        # Default duration is 600s; computed_duration returns it
        assert s.computed_duration() == 600.0

    def test_empty_scenario_zero_duration(self):
        """Scenario with duration_s=0 and no events returns 0."""
        s = Scenario(duration_s=0.0)
        assert s.computed_duration() == 0.0

    def test_geozone_defaults(self):
        zone = GeoZone(zone_id="z1", name="Zone 1",
                       zone_type=ZoneType.BUILDING,
                       center_lat=0.0, center_lng=0.0)
        assert zone.radius_m == 50.0

    def test_scenario_entity_defaults(self):
        e = ScenarioEntity(entity_id="e1", name="E1",
                           entity_type=EntityType.PERSON)
        assert e.alliance == EntityAlliance.UNKNOWN
        assert e.speed_mps == 0.0
        assert e.ble_visible is False
        assert e.camera_visible is True

    def test_expected_alert_defaults(self):
        a = ExpectedAlert()
        assert a.level == AlertLevel.WARNING
        assert a.alert_id != ""

    def test_player_state_defaults(self):
        ps = PlayerState()
        assert ps.current_time_s == 0.0
        assert ps.step_index == 0
        assert ps.completed is False

    def test_enum_values(self):
        """Enum string values are correct."""
        assert EntityAlliance.FRIENDLY == "friendly"
        assert EntityAlliance.HOSTILE == "hostile"
        assert EntityType.VESSEL == "vessel"
        assert ZoneType.DOCK == "dock"
        assert EventKind.THREAT_INJECT == "threat_inject"
        assert AlertLevel.CRITICAL == "critical"


# ---------------------------------------------------------------------------
# Test: Border crossing specifics
# ---------------------------------------------------------------------------

class TestBorderCrossing:
    def test_has_checkpoint_zones(self, border_scenario: Scenario):
        zone_types = {z.zone_type for z in border_scenario.zones}
        assert ZoneType.CHECKPOINT in zone_types

    def test_has_approach_roads(self, border_scenario: Scenario):
        road_zones = [z for z in border_scenario.zones if z.zone_type == ZoneType.ROAD]
        assert len(road_zones) >= 2  # north and south

    def test_has_vehicle_entities(self, border_scenario: Scenario):
        vehicle_types = {e.entity_type for e in border_scenario.entities}
        assert EntityType.VEHICLE in vehicle_types


# ---------------------------------------------------------------------------
# Test: Maritime port specifics
# ---------------------------------------------------------------------------

class TestMaritimePort:
    def test_has_dock_zones(self, maritime_scenario: Scenario):
        dock_zones = [z for z in maritime_scenario.zones if z.zone_type == ZoneType.DOCK]
        assert len(dock_zones) >= 3

    def test_has_vessel_entities(self, maritime_scenario: Scenario):
        vessels = [e for e in maritime_scenario.entities if e.entity_type == EntityType.VESSEL]
        assert len(vessels) > 0

    def test_has_waterway_zones(self, maritime_scenario: Scenario):
        zone_types = {z.zone_type for z in maritime_scenario.zones}
        assert ZoneType.WATERWAY in zone_types

    def test_has_anchorage_zones(self, maritime_scenario: Scenario):
        zone_types = {z.zone_type for z in maritime_scenario.zones}
        assert ZoneType.ANCHORAGE in zone_types
