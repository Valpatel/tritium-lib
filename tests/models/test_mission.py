# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for Mission management models."""

import pytest
from tritium_lib.models.mission import (
    GeofenceZone,
    Mission,
    MissionObjective,
    MissionStatus,
    MissionType,
)


class TestMissionObjective:
    def test_default_fields(self):
        obj = MissionObjective(description="Secure perimeter")
        assert obj.description == "Secure perimeter"
        assert obj.completed is False
        assert obj.priority == 1
        assert obj.completed_at is None
        assert len(obj.objective_id) == 8

    def test_unique_ids(self):
        a = MissionObjective()
        b = MissionObjective()
        assert a.objective_id != b.objective_id


class TestGeofenceZone:
    def test_polygon_zone(self):
        zone = GeofenceZone(
            zone_id="z1",
            name="Alpha",
            vertices=[(40.0, -74.0), (40.1, -74.0), (40.1, -73.9)],
        )
        assert not zone.is_circle
        assert len(zone.vertices) == 3

    def test_circle_zone(self):
        zone = GeofenceZone(
            zone_id="z2",
            name="Bravo",
            center_lat=40.0,
            center_lng=-74.0,
            radius_m=500.0,
        )
        assert zone.is_circle


class TestMission:
    def test_default_creation(self):
        m = Mission(title="Patrol Alpha")
        assert m.title == "Patrol Alpha"
        assert m.status == MissionStatus.DRAFT
        assert m.type == MissionType.CUSTOM
        assert m.assigned_assets == []
        assert m.objectives == []
        assert m.geofence_zone is None
        assert m.priority == 3

    def test_lifecycle_start(self):
        m = Mission(title="Test")
        assert m.started is None
        m.start()
        assert m.status == MissionStatus.ACTIVE
        assert m.started is not None

    def test_lifecycle_pause_resume(self):
        m = Mission(title="Test")
        m.start()
        m.pause()
        assert m.status == MissionStatus.PAUSED
        m.start()
        assert m.status == MissionStatus.ACTIVE

    def test_lifecycle_complete(self):
        m = Mission(title="Test")
        m.start()
        m.complete()
        assert m.status == MissionStatus.COMPLETED
        assert m.completed is not None
        assert m.is_terminal

    def test_lifecycle_abort(self):
        m = Mission(title="Test", description="Original")
        m.start()
        m.abort("Enemy contact")
        assert m.status == MissionStatus.ABORTED
        assert m.is_terminal
        assert "ABORTED: Enemy contact" in m.description

    def test_cannot_complete_draft(self):
        m = Mission(title="Test")
        m.complete()
        assert m.status == MissionStatus.DRAFT

    def test_objectives_progress(self):
        m = Mission(
            title="Test",
            objectives=[
                MissionObjective(objective_id="a", description="First"),
                MissionObjective(objective_id="b", description="Second"),
                MissionObjective(objective_id="c", description="Third"),
            ],
        )
        assert m.progress == 0.0
        m.complete_objective("a")
        assert abs(m.progress - 1 / 3) < 0.01
        m.complete_objective("b")
        m.complete_objective("c")
        assert m.progress == 1.0

    def test_complete_objective_not_found(self):
        m = Mission(title="Test")
        assert m.complete_objective("nonexistent") is False

    def test_progress_empty(self):
        m = Mission(title="Test")
        assert m.progress == 0.0

    def test_to_dict(self):
        m = Mission(
            title="Patrol Route",
            type=MissionType.PATROL,
            assigned_assets=["drone_01", "rover_02"],
            objectives=[MissionObjective(description="Check gate")],
            geofence_zone=GeofenceZone(
                zone_id="z1",
                name="North",
                center_lat=40.0,
                center_lng=-74.0,
                radius_m=200.0,
            ),
            tags=["night", "priority"],
        )
        d = m.to_dict()
        assert d["title"] == "Patrol Route"
        assert d["type"] == "patrol"
        assert d["status"] == "draft"
        assert len(d["assigned_assets"]) == 2
        assert len(d["objectives"]) == 1
        assert d["geofence_zone"]["radius_m"] == 200.0
        assert d["progress"] == 0.0
        assert d["tags"] == ["night", "priority"]

    def test_mission_types(self):
        for mt in MissionType:
            m = Mission(title="Test", type=mt)
            assert m.type == mt
            assert m.to_dict()["type"] == mt.value

    def test_unique_mission_ids(self):
        a = Mission(title="A")
        b = Mission(title="B")
        assert a.mission_id != b.mission_id
