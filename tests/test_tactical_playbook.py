# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tactical.playbook — tactical playbook system."""

import time
import math
import pytest

from tritium_lib.tactical.playbook import (
    ActionType,
    Playbook,
    PlaybookAction,
    PlaybookResult,
    PlaybookRunner,
    StepResult,
    BUILTIN_PLAYBOOKS,
    load_builtin_playbooks,
    _circle_polygon,
)
from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.tracking.geofence import GeofenceEngine, GeoZone
from tritium_lib.events.bus import EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(
    with_tracker=True,
    with_geofence=True,
    with_bus=True,
):
    """Build a PlaybookRunner with optional components wired up."""
    bus = EventBus() if with_bus else None
    tracker = TargetTracker(event_bus=bus) if with_tracker else None
    geofence = GeofenceEngine(event_bus=bus) if with_geofence else None
    if tracker and geofence:
        tracker.set_geofence_engine(geofence)
    return PlaybookRunner(
        tracker=tracker,
        geofence=geofence,
        event_bus=bus,
    ), tracker, geofence, bus


def _add_target(tracker, target_id="tgt_1", alliance="unknown", position=(10.0, 20.0)):
    """Insert a target directly into the tracker."""
    tracker.update_from_simulation({
        "target_id": target_id,
        "name": f"Test {target_id}",
        "alliance": alliance,
        "asset_type": "person",
        "position": {"x": position[0], "y": position[1]},
        "heading": 0.0,
        "speed": 1.0,
        "battery": 1.0,
        "status": "active",
    })


# ---------------------------------------------------------------------------
# 1. ActionType enum
# ---------------------------------------------------------------------------

class TestActionType:
    def test_all_action_types_exist(self):
        expected = {
            "classify", "monitor", "alert", "dispatch", "record",
            "predict", "geofence", "sweep_zone", "check_density", "wait",
        }
        actual = {at.value for at in ActionType}
        assert expected == actual

    def test_action_type_is_string_enum(self):
        assert ActionType.CLASSIFY == "classify"
        assert isinstance(ActionType.ALERT, str)


# ---------------------------------------------------------------------------
# 2. PlaybookAction dataclass
# ---------------------------------------------------------------------------

class TestPlaybookAction:
    def test_basic_creation(self):
        action = PlaybookAction(
            action_type=ActionType.CLASSIFY,
            name="Test classify",
        )
        assert action.action_type == ActionType.CLASSIFY
        assert action.name == "Test classify"
        assert action.params == {}
        assert action.condition is None
        assert action.on_failure == "continue"

    def test_to_dict(self):
        action = PlaybookAction(
            action_type=ActionType.ALERT,
            name="Fire alert",
            params={"severity": "critical"},
        )
        d = action.to_dict()
        assert d["action_type"] == "alert"
        assert d["name"] == "Fire alert"
        assert d["params"]["severity"] == "critical"
        assert d["has_condition"] is False

    def test_with_condition(self):
        action = PlaybookAction(
            action_type=ActionType.ALERT,
            name="Conditional",
            condition=lambda ctx: ctx.get("fire", False),
        )
        d = action.to_dict()
        assert d["has_condition"] is True


# ---------------------------------------------------------------------------
# 3. Playbook dataclass
# ---------------------------------------------------------------------------

class TestPlaybook:
    def test_basic_creation(self):
        pb = Playbook(
            playbook_id="test",
            name="Test Playbook",
            description="A test",
            actions=[
                PlaybookAction(action_type=ActionType.RECORD, name="step1"),
            ],
            tags=["test"],
            priority=7,
        )
        assert pb.playbook_id == "test"
        assert len(pb.actions) == 1
        assert pb.priority == 7

    def test_to_dict(self):
        pb = Playbook(
            playbook_id="test",
            name="Test",
            description="Desc",
            actions=[
                PlaybookAction(action_type=ActionType.WAIT, name="wait"),
            ],
            tags=["a", "b"],
        )
        d = pb.to_dict()
        assert d["playbook_id"] == "test"
        assert len(d["actions"]) == 1
        assert d["tags"] == ["a", "b"]


# ---------------------------------------------------------------------------
# 4. StepResult dataclass
# ---------------------------------------------------------------------------

class TestStepResult:
    def test_basic(self):
        sr = StepResult(
            action_name="step1",
            action_type="classify",
            success=True,
        )
        assert sr.success is True
        assert sr.skipped is False
        assert sr.error == ""

    def test_to_dict(self):
        sr = StepResult(
            action_name="step1",
            action_type="alert",
            success=False,
            error="boom",
        )
        d = sr.to_dict()
        assert d["success"] is False
        assert d["error"] == "boom"


# ---------------------------------------------------------------------------
# 5. PlaybookResult properties
# ---------------------------------------------------------------------------

class TestPlaybookResult:
    def test_empty_result(self):
        pr = PlaybookResult(
            result_id="r1",
            playbook_id="test",
            playbook_name="Test",
            success=True,
        )
        assert pr.steps_succeeded == 0
        assert pr.steps_failed == 0
        assert pr.steps_skipped == 0
        assert pr.duration == 0.0

    def test_step_counts(self):
        pr = PlaybookResult(
            result_id="r2",
            playbook_id="test",
            playbook_name="Test",
            success=True,
            started_at=100.0,
            completed_at=102.5,
            steps=[
                StepResult(action_name="a", action_type="record", success=True),
                StepResult(action_name="b", action_type="alert", success=False),
                StepResult(action_name="c", action_type="wait", success=True, skipped=True),
            ],
        )
        assert pr.steps_succeeded == 1
        assert pr.steps_failed == 1
        assert pr.steps_skipped == 1
        assert pr.duration == pytest.approx(2.5)

    def test_to_dict(self):
        pr = PlaybookResult(
            result_id="r3",
            playbook_id="test",
            playbook_name="Test",
            success=False,
            started_at=1.0,
            completed_at=3.0,
            aborted=True,
            abort_reason="step failed",
        )
        d = pr.to_dict()
        assert d["aborted"] is True
        assert d["abort_reason"] == "step failed"
        assert d["duration"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 6. Built-in playbooks
# ---------------------------------------------------------------------------

class TestBuiltinPlaybooks:
    def test_all_five_builtins_exist(self):
        expected = {"unknown_entry", "pursuit", "gathering", "sweep", "perimeter"}
        assert set(BUILTIN_PLAYBOOKS.keys()) == expected

    def test_load_builtin_playbooks(self):
        playbooks = load_builtin_playbooks()
        assert len(playbooks) == 5
        ids = {pb.playbook_id for pb in playbooks}
        assert "unknown_entry" in ids
        assert "perimeter" in ids

    def test_each_builtin_has_actions(self):
        for pid, factory in BUILTIN_PLAYBOOKS.items():
            pb = factory()
            assert len(pb.actions) >= 2, f"{pid} should have at least 2 actions"
            assert pb.name, f"{pid} should have a name"
            assert pb.description, f"{pid} should have a description"
            assert pb.tags, f"{pid} should have tags"

    def test_builtin_serialization(self):
        for factory in BUILTIN_PLAYBOOKS.values():
            pb = factory()
            d = pb.to_dict()
            assert "playbook_id" in d
            assert "actions" in d
            assert len(d["actions"]) == len(pb.actions)


# ---------------------------------------------------------------------------
# 7. PlaybookRunner — registration
# ---------------------------------------------------------------------------

class TestPlaybookRunnerRegistration:
    def test_register_and_get(self):
        runner, _, _, _ = _make_runner()
        pb = Playbook(playbook_id="test", name="T", description="D")
        runner.register_playbook(pb)
        assert runner.get_playbook("test") is pb

    def test_unregister(self):
        runner, _, _, _ = _make_runner()
        pb = Playbook(playbook_id="test", name="T", description="D")
        runner.register_playbook(pb)
        assert runner.unregister_playbook("test") is True
        assert runner.get_playbook("test") is None

    def test_unregister_missing(self):
        runner, _, _, _ = _make_runner()
        assert runner.unregister_playbook("nope") is False

    def test_list_playbooks_sorted_by_priority(self):
        runner, _, _, _ = _make_runner()
        runner.register_playbook(Playbook(playbook_id="lo", name="Low", description="", priority=1))
        runner.register_playbook(Playbook(playbook_id="hi", name="High", description="", priority=10))
        runner.register_playbook(Playbook(playbook_id="mid", name="Mid", description="", priority=5))
        pbs = runner.list_playbooks()
        assert [pb.playbook_id for pb in pbs] == ["hi", "mid", "lo"]


# ---------------------------------------------------------------------------
# 8. PlaybookRunner — execute missing playbook
# ---------------------------------------------------------------------------

class TestPlaybookRunnerExecuteMissing:
    def test_execute_unregistered_playbook(self):
        runner, _, _, _ = _make_runner()
        result = runner.execute("nonexistent")
        assert result.success is False
        assert result.aborted is True
        assert "not registered" in result.abort_reason


# ---------------------------------------------------------------------------
# 9. PlaybookRunner — execute simple playbook
# ---------------------------------------------------------------------------

class TestPlaybookRunnerExecuteSimple:
    def test_execute_record_and_wait(self):
        runner, tracker, _, _ = _make_runner()
        _add_target(tracker, "tgt_1")

        pb = Playbook(
            playbook_id="simple",
            name="Simple",
            description="Test",
            actions=[
                PlaybookAction(action_type=ActionType.RECORD, name="record", params={"note": "hi", "target_id": "tgt_1"}),
                PlaybookAction(action_type=ActionType.WAIT, name="pause", params={"duration": 5}),
            ],
        )
        result = runner.execute_playbook(pb, context={"target_id": "tgt_1"})
        assert result.success is True
        assert len(result.steps) == 2
        assert result.steps[0].action_type == "record"
        assert result.steps[1].data["waited"] is True

    def test_context_flows_between_steps(self):
        runner, tracker, _, _ = _make_runner()
        _add_target(tracker, "tgt_1", alliance="hostile")

        pb = Playbook(
            playbook_id="flow",
            name="Flow",
            description="",
            actions=[
                PlaybookAction(action_type=ActionType.CLASSIFY, name="classify"),
                PlaybookAction(action_type=ActionType.RECORD, name="record", params={"note": "after classify"}),
            ],
        )
        result = runner.execute_playbook(pb, context={"target_id": "tgt_1"})
        assert result.success is True
        # Context should have classification set by classify step
        assert "classification" in result.context


# ---------------------------------------------------------------------------
# 10. Conditional actions
# ---------------------------------------------------------------------------

class TestConditionalActions:
    def test_condition_false_skips_action(self):
        runner, tracker, _, _ = _make_runner()
        _add_target(tracker, "tgt_1", alliance="friendly")

        pb = Playbook(
            playbook_id="cond",
            name="Conditional",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.ALERT,
                    name="conditional alert",
                    params={"message": "threat"},
                    condition=lambda ctx: ctx.get("is_threat", False),
                ),
            ],
        )
        result = runner.execute_playbook(pb, context={"target_id": "tgt_1"})
        assert result.success is True
        assert result.steps[0].skipped is True

    def test_condition_true_executes_action(self):
        runner, _, _, _ = _make_runner()

        pb = Playbook(
            playbook_id="cond2",
            name="Cond2",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.WAIT,
                    name="guarded wait",
                    params={"duration": 1},
                    condition=lambda ctx: ctx.get("go", False),
                ),
            ],
        )
        result = runner.execute_playbook(pb, context={"go": True})
        assert result.success is True
        assert result.steps[0].skipped is False
        assert result.steps[0].data["waited"] is True


# ---------------------------------------------------------------------------
# 11. Abort on failure
# ---------------------------------------------------------------------------

class TestAbortOnFailure:
    def test_abort_stops_execution(self):
        runner, _, _, _ = _make_runner()

        pb = Playbook(
            playbook_id="abort_test",
            name="Abort",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.CLASSIFY,
                    name="will fail",
                    params={},
                    on_failure="abort",
                ),
                PlaybookAction(
                    action_type=ActionType.WAIT,
                    name="should not run",
                    params={"duration": 1},
                ),
            ],
        )
        # No target_id in context -> classify will raise ValueError
        result = runner.execute_playbook(pb, context={})
        assert result.aborted is True
        assert result.success is False
        # Second step should not be present
        assert len(result.steps) == 1

    def test_continue_on_failure(self):
        runner, _, _, _ = _make_runner()

        pb = Playbook(
            playbook_id="continue_test",
            name="Continue",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.CLASSIFY,
                    name="will fail",
                    params={},
                    on_failure="continue",
                ),
                PlaybookAction(
                    action_type=ActionType.WAIT,
                    name="should still run",
                    params={"duration": 1},
                ),
            ],
        )
        result = runner.execute_playbook(pb, context={})
        assert result.aborted is False
        assert len(result.steps) == 2
        assert result.steps[0].success is False
        assert result.steps[1].success is True

    def test_skip_rest_on_failure(self):
        runner, _, _, _ = _make_runner()

        pb = Playbook(
            playbook_id="skip_test",
            name="SkipRest",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.CLASSIFY,
                    name="will fail",
                    params={},
                    on_failure="skip_rest",
                ),
                PlaybookAction(
                    action_type=ActionType.WAIT,
                    name="should not run",
                    params={"duration": 1},
                ),
            ],
        )
        result = runner.execute_playbook(pb, context={})
        assert result.aborted is False
        assert len(result.steps) == 1


# ---------------------------------------------------------------------------
# 12. Classify action
# ---------------------------------------------------------------------------

class TestClassifyAction:
    def test_classify_existing_target(self):
        runner, tracker, _, _ = _make_runner()
        _add_target(tracker, "tgt_1", alliance="hostile")

        pb = Playbook(
            playbook_id="cls",
            name="Classify",
            description="",
            actions=[PlaybookAction(action_type=ActionType.CLASSIFY, name="cls")],
        )
        result = runner.execute_playbook(pb, context={"target_id": "tgt_1"})
        assert result.success is True
        assert result.steps[0].data["target_id"] == "tgt_1"
        assert "classification" in result.context

    def test_classify_missing_target(self):
        runner, tracker, _, _ = _make_runner()

        pb = Playbook(
            playbook_id="cls2",
            name="Classify Missing",
            description="",
            actions=[PlaybookAction(action_type=ActionType.CLASSIFY, name="cls")],
        )
        result = runner.execute_playbook(pb, context={"target_id": "nonexistent"})
        assert result.success is True
        assert result.context["classification"] == "unknown"


# ---------------------------------------------------------------------------
# 13. Monitor action
# ---------------------------------------------------------------------------

class TestMonitorAction:
    def test_monitor_with_target(self):
        runner, tracker, _, _ = _make_runner()
        _add_target(tracker, "tgt_1", position=(5.0, 10.0))

        pb = Playbook(
            playbook_id="mon",
            name="Monitor",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.MONITOR,
                    name="monitor",
                    params={"duration": 120},
                ),
            ],
        )
        result = runner.execute_playbook(pb, context={"target_id": "tgt_1"})
        assert result.success is True
        assert result.context["monitor_active"] is True
        assert result.steps[0].data["monitoring"] is True

    def test_monitor_without_target(self):
        runner, _, _, _ = _make_runner()
        pb = Playbook(
            playbook_id="mon2",
            name="Monitor Empty",
            description="",
            actions=[
                PlaybookAction(action_type=ActionType.MONITOR, name="mon"),
            ],
        )
        result = runner.execute_playbook(pb, context={})
        assert result.success is True


# ---------------------------------------------------------------------------
# 14. Alert action
# ---------------------------------------------------------------------------

class TestAlertAction:
    def test_alert_fires(self):
        runner, _, _, bus = _make_runner()
        events_received = []
        bus.subscribe("playbook:alert", lambda e: events_received.append(e))

        pb = Playbook(
            playbook_id="alrt",
            name="Alert",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.ALERT,
                    name="fire",
                    params={"severity": "critical", "message": "Test alert"},
                ),
            ],
        )
        result = runner.execute_playbook(pb, context={"target_id": "tgt_1"})
        assert result.success is True
        assert result.context["alert_fired"] is True
        assert len(events_received) == 1

    def test_alert_template_substitution(self):
        runner, _, _, _ = _make_runner()
        pb = Playbook(
            playbook_id="tmpl",
            name="Template",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.ALERT,
                    name="templated",
                    params={"message": "Target {target_id} is {classification}"},
                ),
            ],
        )
        result = runner.execute_playbook(
            pb,
            context={"target_id": "tgt_1", "classification": "hostile"},
        )
        assert "tgt_1" in result.steps[0].data["message"]
        assert "hostile" in result.steps[0].data["message"]


# ---------------------------------------------------------------------------
# 15. Dispatch action
# ---------------------------------------------------------------------------

class TestDispatchAction:
    def test_dispatch_to_target_position(self):
        runner, tracker, _, _ = _make_runner()
        _add_target(tracker, "tgt_1", position=(50.0, 60.0))

        pb = Playbook(
            playbook_id="disp",
            name="Dispatch",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.DISPATCH,
                    name="send drone",
                    params={"asset_id": "drone_1"},
                ),
            ],
        )
        result = runner.execute_playbook(pb, context={"target_id": "tgt_1"})
        assert result.success is True
        assert result.context["dispatched"] is True
        assert result.steps[0].data["position"] is not None


# ---------------------------------------------------------------------------
# 16. Sweep zone action
# ---------------------------------------------------------------------------

class TestSweepZoneAction:
    def test_sweep_finds_targets(self):
        runner, tracker, _, _ = _make_runner()
        _add_target(tracker, "tgt_1", position=(10.0, 10.0))
        _add_target(tracker, "tgt_2", position=(12.0, 12.0))
        _add_target(tracker, "tgt_far", position=(500.0, 500.0))

        pb = Playbook(
            playbook_id="swp",
            name="Sweep",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.SWEEP_ZONE,
                    name="sweep",
                    params={"center": (10.0, 10.0), "radius": 20.0},
                ),
            ],
        )
        result = runner.execute_playbook(pb)
        assert result.success is True
        sweep = result.steps[0].data
        assert sweep["targets_found"] == 2
        assert sweep["targets_found"] < 3  # far target excluded

    def test_sweep_empty_area(self):
        runner, tracker, _, _ = _make_runner()

        pb = Playbook(
            playbook_id="swp2",
            name="Sweep Empty",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.SWEEP_ZONE,
                    name="sweep",
                    params={"center": (999.0, 999.0), "radius": 5.0},
                ),
            ],
        )
        result = runner.execute_playbook(pb)
        assert result.success is True
        assert result.steps[0].data["targets_found"] == 0


# ---------------------------------------------------------------------------
# 17. Check density action
# ---------------------------------------------------------------------------

class TestCheckDensityAction:
    def test_density_below_threshold(self):
        runner, tracker, _, _ = _make_runner()
        _add_target(tracker, "tgt_1", position=(0.0, 0.0))

        pb = Playbook(
            playbook_id="dens",
            name="Density",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.CHECK_DENSITY,
                    name="check",
                    params={"center": (0.0, 0.0), "radius": 50.0, "threshold": 5},
                ),
            ],
        )
        result = runner.execute_playbook(pb)
        assert result.success is True
        assert result.context["density_exceeds"] is False

    def test_density_exceeds_threshold(self):
        runner, tracker, _, _ = _make_runner()
        for i in range(6):
            _add_target(tracker, f"tgt_{i}", position=(float(i), float(i)))

        pb = Playbook(
            playbook_id="dens2",
            name="Density Over",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.CHECK_DENSITY,
                    name="check",
                    params={"center": (2.5, 2.5), "radius": 50.0, "threshold": 5},
                ),
            ],
        )
        result = runner.execute_playbook(pb)
        assert result.success is True
        assert result.context["density_exceeds"] is True
        assert result.context["density_count"] >= 5


# ---------------------------------------------------------------------------
# 18. Geofence action — create and check
# ---------------------------------------------------------------------------

class TestGeofenceAction:
    def test_geofence_create_from_center(self):
        runner, tracker, geofence, _ = _make_runner()

        pb = Playbook(
            playbook_id="gf_create",
            name="Create Zone",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.GEOFENCE,
                    name="create zone",
                    params={
                        "operation": "create",
                        "zone_name": "Test Zone",
                        "zone_type": "monitored",
                        "radius": 10.0,
                    },
                ),
            ],
        )
        result = runner.execute_playbook(pb, context={"target_position": (0.0, 0.0)})
        assert result.success is True
        assert result.steps[0].data["created"] is True
        assert len(geofence.list_zones()) == 1

    def test_geofence_check(self):
        runner, tracker, geofence, _ = _make_runner()
        _add_target(tracker, "tgt_1", position=(0.0, 0.0))

        zone = GeoZone(
            zone_id="z1",
            name="Zone 1",
            polygon=[(-50, -50), (50, -50), (50, 50), (-50, 50)],
        )
        geofence.add_zone(zone)

        pb = Playbook(
            playbook_id="gf_check",
            name="Check Zone",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.GEOFENCE,
                    name="check",
                    params={"operation": "check"},
                ),
            ],
        )
        result = runner.execute_playbook(pb, context={"target_id": "tgt_1"})
        assert result.success is True
        assert "z1" in result.context.get("in_zones", [])


# ---------------------------------------------------------------------------
# 19. Predict action
# ---------------------------------------------------------------------------

class TestPredictAction:
    def test_predict_no_history(self):
        runner, tracker, _, _ = _make_runner()
        _add_target(tracker, "tgt_1")

        pb = Playbook(
            playbook_id="pred",
            name="Predict",
            description="",
            actions=[
                PlaybookAction(action_type=ActionType.PREDICT, name="predict"),
            ],
        )
        result = runner.execute_playbook(pb, context={"target_id": "tgt_1"})
        assert result.success is True
        # With only one history point, prediction will have no results
        assert result.context.get("predicted") is False


# ---------------------------------------------------------------------------
# 20. Full unknown_entry playbook execution
# ---------------------------------------------------------------------------

class TestUnknownEntryPlaybook:
    def test_unknown_entry_friendly_no_alert(self):
        runner, tracker, _, bus = _make_runner()
        _add_target(tracker, "tgt_friendly", alliance="friendly")

        for pb in load_builtin_playbooks():
            runner.register_playbook(pb)

        alerts = []
        bus.subscribe("playbook:alert", lambda e: alerts.append(e))

        result = runner.execute("unknown_entry", context={"target_id": "tgt_friendly"})
        assert result.success is True
        # Alert should be skipped for friendly target (classification won't be hostile)
        alert_steps = [s for s in result.steps if s.action_type == "alert"]
        for s in alert_steps:
            assert s.skipped is True
        assert len(alerts) == 0

    def test_unknown_entry_hostile_fires_alert(self):
        runner, tracker, _, bus = _make_runner()
        _add_target(tracker, "tgt_hostile", alliance="hostile")

        for pb in load_builtin_playbooks():
            runner.register_playbook(pb)

        alerts = []
        bus.subscribe("playbook:alert", lambda e: alerts.append(e))

        result = runner.execute("unknown_entry", context={"target_id": "tgt_hostile"})
        assert result.success is True
        # Alert should fire for hostile target
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# 21. Full gathering playbook
# ---------------------------------------------------------------------------

class TestGatheringPlaybook:
    def test_gathering_below_threshold(self):
        runner, tracker, _, _ = _make_runner()
        _add_target(tracker, "tgt_1", position=(0.0, 0.0))

        for pb in load_builtin_playbooks():
            runner.register_playbook(pb)

        result = runner.execute("gathering", context={"target_position": (0.0, 0.0)})
        assert result.success is True
        # Alert and sweep should be skipped (below threshold)
        alert_step = [s for s in result.steps if s.action_type == "alert"]
        assert all(s.skipped for s in alert_step)


# ---------------------------------------------------------------------------
# 22. Full perimeter playbook
# ---------------------------------------------------------------------------

class TestPerimeterPlaybook:
    def test_perimeter_creates_zone(self):
        runner, tracker, geofence, _ = _make_runner()

        for pb in load_builtin_playbooks():
            runner.register_playbook(pb)

        result = runner.execute("perimeter", context={"target_position": (10.0, 10.0)})
        assert result.success is True
        # Should have created a geofence zone
        zones = geofence.list_zones()
        assert len(zones) >= 1


# ---------------------------------------------------------------------------
# 23. History tracking
# ---------------------------------------------------------------------------

class TestPlaybookHistory:
    def test_history_records_executions(self):
        runner, _, _, _ = _make_runner()
        pb = Playbook(
            playbook_id="hist",
            name="History",
            description="",
            actions=[PlaybookAction(action_type=ActionType.WAIT, name="w", params={"duration": 0})],
        )
        runner.register_playbook(pb)

        runner.execute("hist")
        runner.execute("hist")
        runner.execute("hist")

        history = runner.get_history()
        assert len(history) == 3
        # Newest first
        assert history[0].result_id != history[1].result_id

    def test_clear_history(self):
        runner, _, _, _ = _make_runner()
        pb = Playbook(
            playbook_id="h",
            name="H",
            description="",
            actions=[PlaybookAction(action_type=ActionType.WAIT, name="w", params={"duration": 0})],
        )
        runner.register_playbook(pb)
        runner.execute("h")
        runner.execute("h")

        count = runner.clear_history()
        assert count == 2
        assert len(runner.get_history()) == 0


# ---------------------------------------------------------------------------
# 24. Circle polygon helper
# ---------------------------------------------------------------------------

class TestCirclePolygon:
    def test_octagon(self):
        poly = _circle_polygon(0, 0, 10, sides=8)
        assert len(poly) == 8
        for x, y in poly:
            dist = math.sqrt(x * x + y * y)
            assert abs(dist - 10.0) < 0.01

    def test_custom_center(self):
        poly = _circle_polygon(100, 200, 5, sides=4)
        assert len(poly) == 4
        for x, y in poly:
            dist = math.sqrt((x - 100) ** 2 + (y - 200) ** 2)
            assert abs(dist - 5.0) < 0.01


# ---------------------------------------------------------------------------
# 25. EventBus integration
# ---------------------------------------------------------------------------

class TestEventBusIntegration:
    def test_playbook_started_event(self):
        runner, _, _, bus = _make_runner()
        events = []
        bus.subscribe("playbook:started", lambda e: events.append(e))

        pb = Playbook(
            playbook_id="ev",
            name="Event Test",
            description="",
            actions=[PlaybookAction(action_type=ActionType.WAIT, name="w", params={"duration": 0})],
        )
        runner.execute_playbook(pb)
        assert len(events) == 1

    def test_playbook_completed_event(self):
        runner, _, _, bus = _make_runner()
        events = []
        bus.subscribe("playbook:completed", lambda e: events.append(e))

        pb = Playbook(
            playbook_id="ev2",
            name="Event Test 2",
            description="",
            actions=[PlaybookAction(action_type=ActionType.WAIT, name="w", params={"duration": 0})],
        )
        runner.execute_playbook(pb)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# 26. Runner with no tracker/geofence (graceful degradation)
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_runner_no_tracker(self):
        runner = PlaybookRunner(tracker=None, geofence=None, event_bus=None)
        pb = Playbook(
            playbook_id="bare",
            name="Bare",
            description="",
            actions=[
                PlaybookAction(action_type=ActionType.WAIT, name="w", params={"duration": 0}),
                PlaybookAction(action_type=ActionType.RECORD, name="r", params={"note": "test"}),
            ],
        )
        result = runner.execute_playbook(pb)
        assert result.success is True

    def test_sweep_no_tracker(self):
        runner = PlaybookRunner(tracker=None, geofence=None)
        pb = Playbook(
            playbook_id="sw",
            name="Sweep",
            description="",
            actions=[
                PlaybookAction(
                    action_type=ActionType.SWEEP_ZONE,
                    name="sweep",
                    params={"center": (0, 0), "radius": 10},
                ),
            ],
        )
        result = runner.execute_playbook(pb)
        assert result.success is True
        assert result.steps[0].data["targets_found"] == 0
