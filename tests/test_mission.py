# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.mission — mission planning and coordination."""

import time
import pytest

from tritium_lib.mission import (
    Mission,
    MissionBrief,
    MissionConstraint,
    MissionObjective,
    MissionPlanner,
    MissionPriority,
    MissionSchedule,
    MissionState,
    MissionStatus,
    MissionType,
    ObjectiveStatus,
    ResourceAllocation,
)


# ---------------------------------------------------------------------------
# MissionType enum
# ---------------------------------------------------------------------------

class TestMissionType:
    def test_all_types_exist(self):
        assert MissionType.SURVEILLANCE == "surveillance"
        assert MissionType.TRACKING == "tracking"
        assert MissionType.PERIMETER == "perimeter"
        assert MissionType.INVESTIGATION == "investigation"
        assert MissionType.PATROL == "patrol"

    def test_type_from_string(self):
        assert MissionType("surveillance") == MissionType.SURVEILLANCE
        assert MissionType("patrol") == MissionType.PATROL


# ---------------------------------------------------------------------------
# MissionState enum and transitions
# ---------------------------------------------------------------------------

class TestMissionState:
    def test_all_states_exist(self):
        assert MissionState.PLANNING == "planning"
        assert MissionState.BRIEFED == "briefed"
        assert MissionState.ACTIVE == "active"
        assert MissionState.PAUSED == "paused"
        assert MissionState.COMPLETED == "completed"
        assert MissionState.ABORTED == "aborted"

    def test_valid_transitions_planning(self):
        m = Mission(name="test")
        assert m.state == MissionState.PLANNING
        assert m.can_transition_to(MissionState.BRIEFED)
        assert m.can_transition_to(MissionState.ABORTED)
        assert not m.can_transition_to(MissionState.ACTIVE)
        assert not m.can_transition_to(MissionState.COMPLETED)

    def test_valid_transitions_briefed(self):
        m = Mission(name="test", state=MissionState.BRIEFED)
        assert m.can_transition_to(MissionState.ACTIVE)
        assert m.can_transition_to(MissionState.PLANNING)
        assert m.can_transition_to(MissionState.ABORTED)
        assert not m.can_transition_to(MissionState.COMPLETED)

    def test_valid_transitions_active(self):
        m = Mission(name="test", state=MissionState.ACTIVE)
        assert m.can_transition_to(MissionState.PAUSED)
        assert m.can_transition_to(MissionState.COMPLETED)
        assert m.can_transition_to(MissionState.ABORTED)
        assert not m.can_transition_to(MissionState.PLANNING)

    def test_terminal_states_have_no_transitions(self):
        m_completed = Mission(name="test", state=MissionState.COMPLETED)
        m_aborted = Mission(name="test", state=MissionState.ABORTED)
        for state in MissionState:
            assert not m_completed.can_transition_to(state)
            assert not m_aborted.can_transition_to(state)


# ---------------------------------------------------------------------------
# MissionPriority
# ---------------------------------------------------------------------------

class TestMissionPriority:
    def test_all_priorities(self):
        assert MissionPriority.LOW == "low"
        assert MissionPriority.MEDIUM == "medium"
        assert MissionPriority.HIGH == "high"
        assert MissionPriority.CRITICAL == "critical"


# ---------------------------------------------------------------------------
# MissionObjective
# ---------------------------------------------------------------------------

class TestMissionObjective:
    def test_basic_creation(self):
        obj = MissionObjective(description="Monitor north entrance")
        assert obj.description == "Monitor north entrance"
        assert obj.area_id == ""
        assert obj.target_ids == []
        assert obj.priority == MissionPriority.MEDIUM

    def test_full_creation(self):
        obj = MissionObjective(
            description="Track suspect vehicle",
            area_id="zone-parking",
            target_ids=["ble_aabbccdd", "det_car_1"],
            checkpoint_coords=[(40.0, -74.0), (40.1, -74.1)],
            priority=MissionPriority.HIGH,
            required_sensors=["camera", "ble"],
            success_criteria="Vehicle identified and logged",
        )
        assert obj.area_id == "zone-parking"
        assert len(obj.target_ids) == 2
        assert len(obj.checkpoint_coords) == 2
        assert obj.required_sensors == ["camera", "ble"]

    def test_to_dict_roundtrip(self):
        obj = MissionObjective(
            description="Patrol perimeter",
            area_id="zone-fence",
            priority=MissionPriority.HIGH,
            required_sensors=["camera"],
            success_criteria="All checkpoints visited",
        )
        d = obj.to_dict()
        assert d["description"] == "Patrol perimeter"
        assert d["priority"] == "high"

        restored = MissionObjective.from_dict(d)
        assert restored.description == obj.description
        assert restored.area_id == obj.area_id
        assert restored.priority == obj.priority


# ---------------------------------------------------------------------------
# ObjectiveStatus
# ---------------------------------------------------------------------------

class TestObjectiveStatus:
    def test_default_values(self):
        os = ObjectiveStatus(objective_index=0)
        assert os.status == "pending"
        assert os.progress_pct == 0.0
        assert os.detections == 0

    def test_to_dict(self):
        os = ObjectiveStatus(
            objective_index=1,
            status="active",
            progress_pct=45.0,
            detections=12,
        )
        d = os.to_dict()
        assert d["objective_index"] == 1
        assert d["status"] == "active"
        assert d["progress_pct"] == 45.0
        assert d["detections"] == 12


# ---------------------------------------------------------------------------
# ResourceAllocation
# ---------------------------------------------------------------------------

class TestResourceAllocation:
    def test_auto_id_and_timestamp(self):
        alloc = ResourceAllocation(
            resource_id="cam-01",
            resource_type="camera",
        )
        assert alloc.allocation_id  # non-empty
        assert alloc.assigned_at > 0
        assert alloc.is_active

    def test_release(self):
        alloc = ResourceAllocation(resource_id="drone-01")
        assert alloc.is_active
        alloc.released_at = time.time()
        assert not alloc.is_active

    def test_to_dict_roundtrip(self):
        alloc = ResourceAllocation(
            resource_id="ble-scanner-02",
            resource_type="ble_scanner",
            objective_index=2,
            role="backup",
        )
        d = alloc.to_dict()
        assert d["resource_id"] == "ble-scanner-02"
        assert d["role"] == "backup"
        assert d["is_active"] is True

        restored = ResourceAllocation.from_dict(d)
        assert restored.resource_id == alloc.resource_id
        assert restored.resource_type == alloc.resource_type
        assert restored.objective_index == 2


# ---------------------------------------------------------------------------
# MissionSchedule
# ---------------------------------------------------------------------------

class TestMissionSchedule:
    def test_default_no_window(self):
        sched = MissionSchedule()
        assert not sched.has_time_window
        assert sched.duration_seconds == 0.0
        assert sched.is_within_window()

    def test_time_window(self):
        now = time.time()
        sched = MissionSchedule(
            planned_start=now - 100,
            planned_end=now + 100,
        )
        assert sched.has_time_window
        assert sched.duration_seconds == pytest.approx(200.0, abs=1.0)
        assert sched.is_within_window(now)
        assert not sched.is_within_window(now - 200)  # before start
        assert not sched.is_within_window(now + 200)  # after end

    def test_max_duration(self):
        sched = MissionSchedule(max_duration_hours=8.0)
        assert sched.duration_seconds == pytest.approx(28800.0)

    def test_recurring(self):
        sched = MissionSchedule(recurring=True, recurrence_interval_hours=4.0)
        assert sched.recurring
        assert sched.recurrence_interval_hours == 4.0

    def test_to_dict_roundtrip(self):
        sched = MissionSchedule(
            planned_start=1000.0,
            planned_end=2000.0,
            shift_duration_hours=6.0,
        )
        d = sched.to_dict()
        assert d["planned_start"] == 1000.0
        assert d["shift_duration_hours"] == 6.0
        assert d["has_time_window"] is True

        restored = MissionSchedule.from_dict(d)
        assert restored.planned_start == 1000.0
        assert restored.planned_end == 2000.0


# ---------------------------------------------------------------------------
# MissionConstraint
# ---------------------------------------------------------------------------

class TestMissionConstraint:
    def test_creation(self):
        con = MissionConstraint(
            constraint_type="roe",
            description="No active engagement without authorization",
            severity="mandatory",
        )
        assert con.constraint_type == "roe"
        assert con.severity == "mandatory"

    def test_to_dict_roundtrip(self):
        con = MissionConstraint(
            constraint_type="weather",
            description="No drone ops above 25kt wind",
            severity="hard_stop",
            parameters={"max_wind_kt": 25},
        )
        d = con.to_dict()
        restored = MissionConstraint.from_dict(d)
        assert restored.constraint_type == "weather"
        assert restored.parameters["max_wind_kt"] == 25


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------

class TestMission:
    def test_auto_id_and_timestamp(self):
        m = Mission(name="Overwatch Alpha")
        assert m.mission_id.startswith("msn_")
        assert m.created_at > 0
        assert m.state == MissionState.PLANNING
        assert not m.is_terminal

    def test_terminal_states(self):
        m_completed = Mission(name="done", state=MissionState.COMPLETED)
        m_aborted = Mission(name="cancelled", state=MissionState.ABORTED)
        assert m_completed.is_terminal
        assert m_aborted.is_terminal

    def test_active_resources(self):
        m = Mission(name="test")
        alloc1 = ResourceAllocation(resource_id="cam-01")
        alloc2 = ResourceAllocation(resource_id="cam-02")
        alloc2.released_at = time.time()
        m.resources = [alloc1, alloc2]
        assert len(m.active_resources) == 1
        assert m.active_resources[0].resource_id == "cam-01"

    def test_elapsed_seconds(self):
        m = Mission(name="test")
        assert m.elapsed_seconds == 0.0

        m.activated_at = time.time() - 60
        elapsed = m.elapsed_seconds
        assert elapsed >= 59.0  # at least ~60s

        m.completed_at = m.activated_at + 120
        assert m.elapsed_seconds == pytest.approx(120.0, abs=0.1)

    def test_to_dict_roundtrip(self):
        m = Mission(
            name="Perimeter Watch",
            mission_type=MissionType.PERIMETER,
            priority=MissionPriority.HIGH,
            description="Maintain perimeter security",
            objectives=[
                MissionObjective(description="Monitor gate A"),
            ],
            tags=["night-ops", "high-value"],
        )
        d = m.to_dict()
        assert d["name"] == "Perimeter Watch"
        assert d["mission_type"] == "perimeter"
        assert len(d["objectives"]) == 1
        assert d["is_terminal"] is False

        restored = Mission.from_dict(d)
        assert restored.name == "Perimeter Watch"
        assert restored.mission_type == MissionType.PERIMETER
        assert restored.priority == MissionPriority.HIGH
        assert len(restored.objectives) == 1


# ---------------------------------------------------------------------------
# MissionBrief
# ---------------------------------------------------------------------------

class TestMissionBrief:
    def test_brief_creation(self):
        brief = MissionBrief(
            mission_id="msn_abc123",
            mission_name="Alpha Watch",
            mission_type="surveillance",
            priority="high",
            summary="A surveillance mission.",
            objectives_text="  1. Monitor north entrance",
            resources_text="  - cam-01 (camera)",
            schedule_text="  Start: 2026-03-25 08:00",
            constraints_text="  No constraints.",
            generated_at=time.time(),
        )
        assert brief.mission_name == "Alpha Watch"

    def test_full_text(self):
        brief = MissionBrief(
            mission_id="msn_test",
            mission_name="Test Mission",
            mission_type="patrol",
            priority="medium",
            summary="Patrol mission.",
            objectives_text="  1. Patrol route A",
            resources_text="  - unit-01",
            schedule_text="  No schedule constraints.",
            constraints_text="  No constraints.",
            generated_at=time.time(),
        )
        text = brief.full_text
        assert "MISSION BRIEF: Test Mission" in text
        assert "PATROL" in text
        assert "OBJECTIVES" in text
        assert "Patrol route A" in text

    def test_to_dict(self):
        brief = MissionBrief(
            mission_id="msn_test",
            mission_name="Brief Test",
            mission_type="tracking",
            priority="critical",
            summary="Test.",
            objectives_text="obj",
            resources_text="res",
            schedule_text="sched",
            constraints_text="con",
            generated_at=1234567890.0,
        )
        d = brief.to_dict()
        assert d["mission_id"] == "msn_test"
        assert d["generated_at"] == 1234567890.0


# ---------------------------------------------------------------------------
# MissionStatus
# ---------------------------------------------------------------------------

class TestMissionStatus:
    def test_default_timestamp(self):
        status = MissionStatus(mission_id="msn_test", state="active")
        assert status.timestamp > 0

    def test_to_dict(self):
        status = MissionStatus(
            mission_id="msn_test",
            state="active",
            elapsed_seconds=300.0,
            active_resources=3,
            total_resources=5,
            total_detections=42,
            total_alerts=7,
            overall_progress_pct=65.0,
            health="green",
        )
        d = status.to_dict()
        assert d["mission_id"] == "msn_test"
        assert d["total_detections"] == 42
        assert d["health"] == "green"


# ---------------------------------------------------------------------------
# MissionPlanner — creation and CRUD
# ---------------------------------------------------------------------------

class TestMissionPlannerCreate:
    def test_create_basic_mission(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Test Mission")
        assert m.name == "Test Mission"
        assert m.state == MissionState.PLANNING
        assert m.mission_type == MissionType.SURVEILLANCE

    def test_create_with_all_params(self):
        planner = MissionPlanner()
        m = planner.create_mission(
            name="Full Mission",
            mission_type=MissionType.TRACKING,
            priority=MissionPriority.CRITICAL,
            description="Track a high-value target",
            objectives=[
                MissionObjective(description="Maintain visual"),
                MissionObjective(description="Record movements"),
            ],
            schedule=MissionSchedule(max_duration_hours=12.0),
            constraints=[
                MissionConstraint(
                    constraint_type="roe",
                    description="Observe only",
                ),
            ],
            created_by="operator",
            incident_id="inc_12345",
            tags=["hvt", "covert"],
            metadata={"case_number": "C-2026-001"},
        )
        assert m.mission_type == MissionType.TRACKING
        assert m.priority == MissionPriority.CRITICAL
        assert len(m.objectives) == 2
        assert len(m.constraints) == 1
        assert m.incident_id == "inc_12345"
        assert "hvt" in m.tags

    def test_create_with_string_enums(self):
        planner = MissionPlanner()
        m = planner.create_mission(
            name="String Enum Test",
            mission_type="patrol",
            priority="high",
        )
        assert m.mission_type == MissionType.PATROL
        assert m.priority == MissionPriority.HIGH

    def test_get_mission(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Findable")
        found = planner.get_mission(m.mission_id)
        assert found is not None
        assert found.name == "Findable"

    def test_get_mission_not_found(self):
        planner = MissionPlanner()
        assert planner.get_mission("nonexistent") is None

    def test_get_missions_filtering(self):
        planner = MissionPlanner()
        planner.create_mission(name="A", mission_type=MissionType.SURVEILLANCE, priority=MissionPriority.LOW)
        planner.create_mission(name="B", mission_type=MissionType.TRACKING, priority=MissionPriority.HIGH, tags=["covert"])
        planner.create_mission(name="C", mission_type=MissionType.SURVEILLANCE, priority=MissionPriority.HIGH)

        all_missions = planner.get_missions()
        assert len(all_missions) == 3

        surv = planner.get_missions(mission_type=MissionType.SURVEILLANCE)
        assert len(surv) == 2

        high_pri = planner.get_missions(priority=MissionPriority.HIGH)
        assert len(high_pri) == 2

        covert = planner.get_missions(tag="covert")
        assert len(covert) == 1
        assert covert[0].name == "B"

    def test_update_mission(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Original", priority=MissionPriority.LOW)
        updated = planner.update_mission(
            m.mission_id,
            name="Updated",
            priority=MissionPriority.HIGH,
            description="Now important",
        )
        assert updated is not None
        assert updated.name == "Updated"
        assert updated.priority == MissionPriority.HIGH
        assert updated.description == "Now important"

    def test_update_active_mission_rejected(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Active")
        planner.brief(m.mission_id)
        planner.activate(m.mission_id)
        result = planner.update_mission(m.mission_id, name="New Name")
        assert result is None  # cannot update active mission

    def test_delete_planning_mission(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Deletable")
        assert planner.delete_mission(m.mission_id)
        assert planner.get_mission(m.mission_id) is None

    def test_delete_active_mission_rejected(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Active")
        planner.brief(m.mission_id)
        planner.activate(m.mission_id)
        assert not planner.delete_mission(m.mission_id)


# ---------------------------------------------------------------------------
# MissionPlanner — lifecycle
# ---------------------------------------------------------------------------

class TestMissionPlannerLifecycle:
    def test_full_lifecycle(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Lifecycle Test")
        assert m.state == MissionState.PLANNING

        # PLANNING -> BRIEFED
        result = planner.brief(m.mission_id)
        assert result is not None
        assert result.state == MissionState.BRIEFED

        # BRIEFED -> ACTIVE
        result = planner.activate(m.mission_id)
        assert result is not None
        assert result.state == MissionState.ACTIVE
        assert result.activated_at > 0

        # ACTIVE -> PAUSED
        result = planner.pause(m.mission_id, reason="Shift change")
        assert result is not None
        assert result.state == MissionState.PAUSED

        # PAUSED -> ACTIVE
        result = planner.activate(m.mission_id)
        assert result is not None
        assert result.state == MissionState.ACTIVE

        # ACTIVE -> COMPLETED
        result = planner.complete(m.mission_id, summary="Mission successful")
        assert result is not None
        assert result.state == MissionState.COMPLETED
        assert result.completed_at > 0
        assert result.metadata.get("completion_summary") == "Mission successful"

    def test_abort_from_active(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Abort Test")
        planner.brief(m.mission_id)
        planner.activate(m.mission_id)

        result = planner.abort(m.mission_id, reason="Emergency evacuation")
        assert result is not None
        assert result.state == MissionState.ABORTED
        assert result.metadata.get("abort_reason") == "Emergency evacuation"

    def test_abort_from_planning(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Cancel")
        result = planner.abort(m.mission_id)
        assert result is not None
        assert result.state == MissionState.ABORTED

    def test_invalid_transition_rejected(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Invalid")
        # Cannot go directly from PLANNING to ACTIVE
        result = planner.activate(m.mission_id)
        assert result is None
        # State unchanged
        assert planner.get_mission(m.mission_id).state == MissionState.PLANNING

    def test_replan_from_briefed(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Replan")
        planner.brief(m.mission_id)
        result = planner.replan(m.mission_id)
        assert result is not None
        assert result.state == MissionState.PLANNING

    def test_activate_sets_objective_statuses(self):
        planner = MissionPlanner()
        m = planner.create_mission(
            name="Obj Status Test",
            objectives=[
                MissionObjective(description="Obj 1"),
                MissionObjective(description="Obj 2"),
            ],
        )
        planner.brief(m.mission_id)
        planner.activate(m.mission_id)

        status = planner.get_status(m.mission_id)
        assert status is not None
        assert len(status.objective_statuses) == 2
        for os in status.objective_statuses:
            assert os.status == "active"
            assert os.started_at > 0


# ---------------------------------------------------------------------------
# MissionPlanner — objectives
# ---------------------------------------------------------------------------

class TestMissionPlannerObjectives:
    def test_add_objective(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Obj Test")
        result = planner.add_objective(
            m.mission_id,
            MissionObjective(description="New objective"),
        )
        assert result is True
        mission = planner.get_mission(m.mission_id)
        assert len(mission.objectives) == 1

    def test_add_objective_to_active_rejected(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Active Obj")
        planner.brief(m.mission_id)
        planner.activate(m.mission_id)
        result = planner.add_objective(
            m.mission_id,
            MissionObjective(description="Too late"),
        )
        assert result is False

    def test_update_objective_status(self):
        planner = MissionPlanner()
        m = planner.create_mission(
            name="Status Test",
            objectives=[MissionObjective(description="Monitor")],
        )

        os = planner.update_objective_status(
            m.mission_id,
            objective_index=0,
            status="active",
            progress_pct=50.0,
            notes="Halfway done",
            detections_delta=5,
            alerts_delta=2,
        )
        assert os is not None
        assert os.status == "active"
        assert os.progress_pct == 50.0
        assert os.notes == "Halfway done"
        assert os.detections == 5
        assert os.alerts_fired == 2
        assert os.started_at > 0

    def test_update_objective_status_clamps_progress(self):
        planner = MissionPlanner()
        m = planner.create_mission(
            name="Clamp Test",
            objectives=[MissionObjective(description="Test")],
        )
        os = planner.update_objective_status(
            m.mission_id, 0, progress_pct=150.0,
        )
        assert os.progress_pct == 100.0

        os = planner.update_objective_status(
            m.mission_id, 0, progress_pct=-10.0,
        )
        assert os.progress_pct == 0.0

    def test_update_objective_invalid_index(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Invalid Idx")
        result = planner.update_objective_status(m.mission_id, 99)
        assert result is None


# ---------------------------------------------------------------------------
# MissionPlanner — resource allocation
# ---------------------------------------------------------------------------

class TestMissionPlannerResources:
    def test_allocate_resource(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Resource Test")
        alloc = planner.allocate_resource(
            m.mission_id,
            resource_id="cam-01",
            resource_type="camera",
            objective_index=0,
            role="primary",
        )
        assert alloc is not None
        assert alloc.resource_id == "cam-01"
        assert alloc.is_active

    def test_allocate_to_terminal_rejected(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Terminal")
        planner.abort(m.mission_id)
        alloc = planner.allocate_resource(m.mission_id, "cam-01")
        assert alloc is None

    def test_release_resource(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Release Test")
        alloc = planner.allocate_resource(m.mission_id, "drone-01")
        assert alloc.is_active

        released = planner.release_resource(m.mission_id, alloc.allocation_id)
        assert released is True

        allocs = planner.get_resource_allocations(m.mission_id, active_only=True)
        assert len(allocs) == 0

    def test_get_resource_allocations(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Alloc List")
        planner.allocate_resource(m.mission_id, "cam-01", "camera")
        alloc2 = planner.allocate_resource(m.mission_id, "cam-02", "camera")
        planner.release_resource(m.mission_id, alloc2.allocation_id)

        all_allocs = planner.get_resource_allocations(m.mission_id)
        assert len(all_allocs) == 2

        active = planner.get_resource_allocations(m.mission_id, active_only=True)
        assert len(active) == 1
        assert active[0].resource_id == "cam-01"


# ---------------------------------------------------------------------------
# MissionPlanner — constraints
# ---------------------------------------------------------------------------

class TestMissionPlannerConstraints:
    def test_add_constraint(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Constraint Test")
        result = planner.add_constraint(
            m.mission_id,
            MissionConstraint(
                constraint_type="weather",
                description="No ops in fog",
                severity="hard_stop",
            ),
        )
        assert result is True
        mission = planner.get_mission(m.mission_id)
        assert len(mission.constraints) == 1
        assert mission.constraints[0].severity == "hard_stop"


# ---------------------------------------------------------------------------
# MissionPlanner — brief generation
# ---------------------------------------------------------------------------

class TestMissionPlannerBrief:
    def test_generate_brief(self):
        planner = MissionPlanner()
        m = planner.create_mission(
            name="Brief Test",
            mission_type=MissionType.PERIMETER,
            priority=MissionPriority.HIGH,
            description="Secure north boundary.",
            objectives=[
                MissionObjective(
                    description="Monitor gate A",
                    area_id="zone-gate-a",
                    priority=MissionPriority.HIGH,
                    success_criteria="All entries logged",
                ),
            ],
            constraints=[
                MissionConstraint(
                    constraint_type="roe",
                    description="Report only, do not engage",
                    severity="mandatory",
                ),
            ],
            schedule=MissionSchedule(max_duration_hours=8.0),
        )
        planner.allocate_resource(
            m.mission_id, "cam-01", "camera", objective_index=0,
        )

        brief = planner.generate_brief(m.mission_id)
        assert brief is not None
        assert brief.mission_name == "Brief Test"
        assert brief.mission_type == "perimeter"
        assert brief.priority == "high"
        assert "Monitor gate A" in brief.objectives_text
        assert "cam-01" in brief.resources_text
        assert "8.0h" in brief.schedule_text
        assert "roe" in brief.constraints_text
        assert brief.generated_at > 0

    def test_brief_full_text_format(self):
        planner = MissionPlanner()
        m = planner.create_mission(name="Full Text Brief")
        brief = planner.generate_brief(m.mission_id)
        text = brief.full_text
        assert "MISSION BRIEF: Full Text Brief" in text
        assert "SUMMARY" in text
        assert "OBJECTIVES" in text
        assert "RESOURCES" in text
        assert "SCHEDULE" in text
        assert "CONSTRAINTS" in text

    def test_brief_not_found(self):
        planner = MissionPlanner()
        assert planner.generate_brief("nonexistent") is None


# ---------------------------------------------------------------------------
# MissionPlanner — status
# ---------------------------------------------------------------------------

class TestMissionPlannerStatus:
    def test_get_status(self):
        planner = MissionPlanner()
        m = planner.create_mission(
            name="Status Test",
            objectives=[
                MissionObjective(description="Obj 1"),
                MissionObjective(description="Obj 2"),
            ],
        )
        planner.allocate_resource(m.mission_id, "cam-01")
        planner.brief(m.mission_id)
        planner.activate(m.mission_id)

        # Update some progress
        planner.update_objective_status(
            m.mission_id, 0, progress_pct=80.0, detections_delta=10,
        )
        planner.update_objective_status(
            m.mission_id, 1, progress_pct=40.0, alerts_delta=3,
        )

        status = planner.get_status(m.mission_id)
        assert status is not None
        assert status.state == "active"
        assert status.active_resources == 1
        assert status.total_resources == 1
        assert status.total_detections == 10
        assert status.total_alerts == 3
        assert status.overall_progress_pct == pytest.approx(60.0)
        assert status.health == "green"

    def test_status_health_red_on_failure(self):
        planner = MissionPlanner()
        m = planner.create_mission(
            name="Failed Obj",
            objectives=[MissionObjective(description="Will fail")],
        )
        planner.brief(m.mission_id)
        planner.activate(m.mission_id)
        planner.update_objective_status(m.mission_id, 0, status="failed")
        status = planner.get_status(m.mission_id)
        assert status.health == "red"

    def test_status_health_red_no_resources(self):
        planner = MissionPlanner()
        m = planner.create_mission(
            name="No Resources",
            objectives=[MissionObjective(description="Needs sensors")],
        )
        planner.brief(m.mission_id)
        planner.activate(m.mission_id)
        status = planner.get_status(m.mission_id)
        assert status.health == "red"

    def test_status_not_found(self):
        planner = MissionPlanner()
        assert planner.get_status("nonexistent") is None


# ---------------------------------------------------------------------------
# MissionPlanner — statistics
# ---------------------------------------------------------------------------

class TestMissionPlannerStats:
    def test_stats(self):
        planner = MissionPlanner()
        planner.create_mission(name="A", mission_type=MissionType.SURVEILLANCE)
        planner.create_mission(name="B", mission_type=MissionType.TRACKING)
        m_c = planner.create_mission(name="C", mission_type=MissionType.SURVEILLANCE)
        planner.abort(m_c.mission_id)

        stats = planner.get_stats()
        assert stats["total_missions"] == 3
        assert stats["total_created"] == 3
        assert stats["total_aborted"] == 1
        assert stats["by_type"]["surveillance"] == 2
        assert stats["by_type"]["tracking"] == 1
        assert stats["by_state"]["planning"] == 2
        assert stats["by_state"]["aborted"] == 1


# ---------------------------------------------------------------------------
# MissionPlanner — enforce limit
# ---------------------------------------------------------------------------

class TestMissionPlannerLimit:
    def test_enforce_max_missions(self):
        planner = MissionPlanner(max_missions=3)
        missions = []
        for i in range(5):
            m = planner.create_mission(name=f"Mission {i}")
            if i < 3:
                # Complete older missions so they can be evicted
                planner.abort(m.mission_id)
            missions.append(m)

        # Should have pruned oldest terminal missions to stay at limit
        stats = planner.get_stats()
        assert stats["total_missions"] <= 3


# ---------------------------------------------------------------------------
# MissionPlanner — EventBus integration
# ---------------------------------------------------------------------------

class TestMissionPlannerEventBus:
    def test_events_published(self):
        """Verify events are published when an event_bus is provided."""
        events_received = []

        class MockEventBus:
            def publish(self, topic, data=None, source=None):
                events_received.append((topic, data))

        bus = MockEventBus()
        planner = MissionPlanner(event_bus=bus)
        m = planner.create_mission(name="Event Test")
        planner.brief(m.mission_id)
        planner.activate(m.mission_id)
        planner.complete(m.mission_id)

        topics = [e[0] for e in events_received]
        assert "mission.created" in topics
        assert "mission.state.briefed" in topics
        assert "mission.state.active" in topics
        assert "mission.state.completed" in topics

    def test_no_error_without_event_bus(self):
        """Planner works fine without an event bus."""
        planner = MissionPlanner()
        m = planner.create_mission(name="No Bus")
        planner.brief(m.mission_id)
        planner.activate(m.mission_id)
        planner.complete(m.mission_id)
        assert planner.get_mission(m.mission_id).state == MissionState.COMPLETED
