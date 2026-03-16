# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for TacticalScenario models."""

from tritium_lib.models.scenario import (
    ActorAlliance,
    ActorType,
    ScenarioActor,
    ScenarioEvent,
    ScenarioEventType,
    ScenarioObjective,
    ScenarioStatus,
    TacticalScenario,
)


def test_scenario_creation():
    """TacticalScenario can be created with defaults."""
    s = TacticalScenario(title="Test Scenario")
    assert s.title == "Test Scenario"
    assert s.status == ScenarioStatus.DRAFT
    assert len(s.actors) == 0
    assert len(s.events) == 0
    assert len(s.objectives) == 0
    assert s.scenario_id  # Should have a UUID


def test_scenario_with_actors():
    """Scenario can have actors added."""
    s = TacticalScenario(title="Multi-Actor Test")
    s.actors.append(ScenarioActor(
        name="Intruder",
        actor_type=ActorType.PERSON,
        alliance=ActorAlliance.HOSTILE,
        start_lat=30.0,
        start_lng=-97.0,
        mac_address="AA:BB:CC:DD:EE:FF",
    ))
    s.actors.append(ScenarioActor(
        name="Guard",
        actor_type=ActorType.ROBOT,
        alliance=ActorAlliance.FRIENDLY,
    ))
    assert len(s.actors) == 2
    assert s.actors[0].alliance == ActorAlliance.HOSTILE
    assert s.actors[1].actor_type == ActorType.ROBOT


def test_scenario_events():
    """Events can be created with time offsets."""
    e1 = ScenarioEvent(
        event_type=ScenarioEventType.SPAWN,
        time_offset_s=0.0,
        description="Target appears at entrance",
    )
    e2 = ScenarioEvent(
        event_type=ScenarioEventType.DETECT,
        time_offset_s=5.0,
        description="BLE scanner detects phone",
    )
    e3 = ScenarioEvent(
        event_type=ScenarioEventType.GEOFENCE_ENTER,
        time_offset_s=15.0,
    )
    s = TacticalScenario(title="Event Test", events=[e1, e2, e3])
    sorted_events = s.sorted_events()
    assert sorted_events[0].time_offset_s == 0.0
    assert sorted_events[2].time_offset_s == 15.0


def test_computed_duration():
    """Duration derived from events when not set explicitly."""
    s = TacticalScenario(title="Duration Test")
    s.events.append(ScenarioEvent(time_offset_s=0.0))
    s.events.append(ScenarioEvent(time_offset_s=30.0))
    s.events.append(ScenarioEvent(time_offset_s=120.0))
    assert s.computed_duration() == 120.0

    # Explicit duration overrides
    s.duration_s = 300.0
    assert s.computed_duration() == 300.0


def test_completion_pct():
    """Completion percentage calculated correctly."""
    s = TacticalScenario(title="Completion Test")
    s.objectives.append(ScenarioObjective(description="Obj 1", completed=True))
    s.objectives.append(ScenarioObjective(description="Obj 2", completed=False))
    s.objectives.append(ScenarioObjective(description="Obj 3", completed=True))
    assert s.completion_pct() == 66.7

    # All done
    s.objectives[1].completed = True
    assert s.completion_pct() == 100.0


def test_completion_pct_no_objectives():
    """100% completion when no objectives defined."""
    s = TacticalScenario(title="No Objectives")
    assert s.completion_pct() == 100.0


def test_actor_by_id():
    """Find actor by ID."""
    a = ScenarioActor(actor_id="test-123", name="Test Actor")
    s = TacticalScenario(title="Find Test", actors=[a])
    assert s.actor_by_id("test-123") is a
    assert s.actor_by_id("nonexistent") is None


def test_events_for_actor():
    """Get events involving a specific actor."""
    s = TacticalScenario(title="Events For Actor")
    s.events.append(ScenarioEvent(actor_id="a1", event_type=ScenarioEventType.SPAWN))
    s.events.append(ScenarioEvent(actor_id="a2", event_type=ScenarioEventType.SPAWN))
    s.events.append(ScenarioEvent(actor_id="a1", target_actor_id="a2", event_type=ScenarioEventType.CORRELATE))
    # a1 is involved in 2 events (as actor and as actor in correlate)
    assert len(s.events_for_actor("a1")) == 2
    # a2 is involved in 2 events (spawn and as target in correlate)
    assert len(s.events_for_actor("a2")) == 2


def test_to_dict():
    """Serialization to dictionary."""
    s = TacticalScenario(title="Serial Test", description="Test serialization")
    s.actors.append(ScenarioActor(name="Actor 1"))
    s.events.append(ScenarioEvent(event_type=ScenarioEventType.SPAWN))
    s.objectives.append(ScenarioObjective(description="Complete task"))

    d = s.to_dict()
    assert d["title"] == "Serial Test"
    assert len(d["actors"]) == 1
    assert len(d["events"]) == 1
    assert len(d["objectives"]) == 1
    assert isinstance(d["created_at"], str)  # ISO format


def test_scenario_status_values():
    """All status values are valid."""
    assert ScenarioStatus.DRAFT == "draft"
    assert ScenarioStatus.RUNNING == "running"
    assert ScenarioStatus.COMPLETED == "completed"


def test_event_types():
    """All event types are accessible."""
    assert ScenarioEventType.SPAWN == "spawn"
    assert ScenarioEventType.DETECT == "detect"
    assert ScenarioEventType.GEOFENCE_ENTER == "geofence_enter"
    assert ScenarioEventType.CORRELATE == "correlate"
    assert ScenarioEventType.ENRICH == "enrich"


def test_actor_types():
    """All actor types are accessible."""
    assert ActorType.PERSON == "person"
    assert ActorType.DRONE == "drone"
    assert ActorType.SENSOR_NODE == "sensor_node"
