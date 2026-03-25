# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.incident — incident management module."""

import time

import pytest

from tritium_lib.incident import (
    AssignedResource,
    Incident,
    IncidentManager,
    IncidentSeverity,
    IncidentState,
    Resolution,
    Timeline,
    TimelineEntry,
)


# ---------------------------------------------------------------------------
# TimelineEntry
# ---------------------------------------------------------------------------

class TestTimelineEntry:
    def test_create_entry(self):
        entry = TimelineEntry(
            entry_id="e1",
            timestamp=1000.0,
            description="Something happened",
            author="operator",
            entry_type="note",
        )
        assert entry.entry_id == "e1"
        assert entry.description == "Something happened"
        assert entry.author == "operator"
        assert entry.entry_type == "note"

    def test_to_dict_from_dict_roundtrip(self):
        entry = TimelineEntry(
            entry_id="e2",
            timestamp=2000.0,
            description="State changed",
            author="system",
            entry_type="state_change",
            metadata={"old": "detected", "new": "investigating"},
        )
        d = entry.to_dict()
        restored = TimelineEntry.from_dict(d)
        assert restored.entry_id == entry.entry_id
        assert restored.description == entry.description
        assert restored.metadata == entry.metadata


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

class TestTimeline:
    def test_add_and_count(self):
        tl = Timeline()
        assert tl.count == 0
        tl.add("First event", author="test")
        tl.add("Second event", author="test")
        assert tl.count == 2

    def test_chronological_order(self):
        tl = Timeline()
        tl.add("Second", timestamp=200.0)
        tl.add("First", timestamp=100.0)
        tl.add("Third", timestamp=300.0)
        entries = tl.get_entries()
        assert entries[0].description == "First"
        assert entries[1].description == "Second"
        assert entries[2].description == "Third"

    def test_latest(self):
        tl = Timeline()
        assert tl.latest is None
        tl.add("One", timestamp=100.0)
        tl.add("Two", timestamp=200.0)
        assert tl.latest is not None
        assert tl.latest.description == "Two"

    def test_filter_by_type(self):
        tl = Timeline()
        tl.add("Note 1", entry_type="note")
        tl.add("State change", entry_type="state_change")
        tl.add("Note 2", entry_type="note")
        notes = tl.get_entries(entry_type="note")
        assert len(notes) == 2
        assert all(e.entry_type == "note" for e in notes)

    def test_filter_since(self):
        tl = Timeline()
        tl.add("Old", timestamp=100.0)
        tl.add("New", timestamp=200.0)
        entries = tl.get_entries(since=150.0)
        assert len(entries) == 1
        assert entries[0].description == "New"

    def test_limit(self):
        tl = Timeline()
        for i in range(10):
            tl.add(f"Entry {i}", timestamp=float(i))
        entries = tl.get_entries(limit=3)
        assert len(entries) == 3

    def test_to_list_from_list_roundtrip(self):
        tl = Timeline()
        tl.add("A", timestamp=100.0, entry_type="note")
        tl.add("B", timestamp=200.0, entry_type="state_change")
        data = tl.to_list()
        restored = Timeline.from_list(data)
        assert restored.count == 2
        entries = restored.get_entries()
        assert entries[0].description == "A"
        assert entries[1].description == "B"


# ---------------------------------------------------------------------------
# AssignedResource
# ---------------------------------------------------------------------------

class TestAssignedResource:
    def test_create_resource(self):
        r = AssignedResource(
            resource_id="drone-01",
            resource_type="drone",
            role="surveillance",
            assigned_at=1000.0,
        )
        assert r.resource_id == "drone-01"
        assert r.is_active is True

    def test_release(self):
        r = AssignedResource(resource_id="unit-alpha", assigned_at=1000.0)
        assert r.is_active is True
        r.release(timestamp=2000.0)
        assert r.is_active is False
        assert r.released_at == 2000.0

    def test_to_dict_from_dict_roundtrip(self):
        r = AssignedResource(
            resource_id="sensor-05",
            resource_type="sensor",
            role="investigation",
            assigned_at=500.0,
            notes="Deployed to east perimeter",
        )
        d = r.to_dict()
        assert d["is_active"] is True
        restored = AssignedResource.from_dict(d)
        assert restored.resource_id == r.resource_id
        assert restored.notes == r.notes


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestResolution:
    def test_create_resolution(self):
        res = Resolution(
            summary="All clear",
            resolution_type="false_alarm",
            resolved_by="operator",
            resolved_at=3000.0,
            lessons_learned="Better sensor calibration needed",
            follow_up_actions=["Recalibrate sensor A", "Update SOP"],
        )
        assert res.summary == "All clear"
        assert len(res.follow_up_actions) == 2

    def test_to_dict_from_dict_roundtrip(self):
        res = Resolution(
            summary="Contained",
            resolution_type="contained",
            resolved_by="system",
            resolved_at=4000.0,
            lessons_learned="Good response",
            follow_up_actions=["Review"],
        )
        d = res.to_dict()
        restored = Resolution.from_dict(d)
        assert restored.summary == res.summary
        assert restored.follow_up_actions == res.follow_up_actions


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------

class TestIncident:
    def test_create_incident(self):
        inc = Incident(
            incident_id="inc_001",
            title="Hostile in Zone A",
            severity=IncidentSeverity.HIGH,
        )
        assert inc.incident_id == "inc_001"
        assert inc.state == IncidentState.DETECTED
        assert inc.is_open is True

    def test_is_open(self):
        inc = Incident(incident_id="inc_002", title="Test")
        assert inc.is_open is True
        inc.state = IncidentState.RESOLVED
        assert inc.is_open is False
        inc.state = IncidentState.CLOSED
        assert inc.is_open is False

    def test_can_transition_to(self):
        inc = Incident(incident_id="inc_003", title="Test")
        assert inc.state == IncidentState.DETECTED
        assert inc.can_transition_to(IncidentState.INVESTIGATING) is True
        assert inc.can_transition_to(IncidentState.RESPONDING) is True
        assert inc.can_transition_to(IncidentState.RESOLVED) is True
        # CLOSED is terminal, no transitions out
        inc.state = IncidentState.CLOSED
        assert inc.can_transition_to(IncidentState.DETECTED) is False
        assert inc.can_transition_to(IncidentState.INVESTIGATING) is False

    def test_reopen_from_resolved(self):
        inc = Incident(incident_id="inc_004", title="Test")
        inc.state = IncidentState.RESOLVED
        assert inc.can_transition_to(IncidentState.INVESTIGATING) is True

    def test_active_resources(self):
        inc = Incident(incident_id="inc_005", title="Test")
        r1 = AssignedResource(resource_id="drone-01", assigned_at=100.0)
        r2 = AssignedResource(resource_id="drone-02", assigned_at=100.0, released_at=200.0)
        inc.resources = [r1, r2]
        assert len(inc.active_resources) == 1
        assert inc.active_resources[0].resource_id == "drone-01"

    def test_to_dict_from_dict_roundtrip(self):
        inc = Incident(
            incident_id="inc_006",
            title="Test Roundtrip",
            state=IncidentState.INVESTIGATING,
            severity=IncidentSeverity.CRITICAL,
            source="alerting",
            created_at=1000.0,
            updated_at=2000.0,
            target_ids=["ble_aabb"],
            zone_id="zone-a",
            alert_ids=["alert-001"],
            tags=["auto"],
            description="Detailed description",
        )
        inc.timeline.add("Created", timestamp=1000.0)
        inc.resources.append(
            AssignedResource(resource_id="drone-01", assigned_at=1500.0)
        )
        inc.resolution = Resolution(summary="Done", resolved_at=2000.0)

        d = inc.to_dict()
        assert d["state"] == "investigating"
        assert d["is_open"] is True

        restored = Incident.from_dict(d)
        assert restored.incident_id == inc.incident_id
        assert restored.state == IncidentState.INVESTIGATING
        assert restored.severity == IncidentSeverity.CRITICAL
        assert restored.timeline.count == 1
        assert len(restored.resources) == 1
        assert restored.resolution is not None
        assert restored.resolution.summary == "Done"


# ---------------------------------------------------------------------------
# IncidentManager — core lifecycle
# ---------------------------------------------------------------------------

class TestIncidentManager:
    def test_create_incident(self):
        mgr = IncidentManager()
        inc = mgr.create(
            title="Perimeter breach",
            severity=IncidentSeverity.HIGH,
            source="alerting",
            target_ids=["ble_aabb"],
            zone_id="perimeter",
        )
        assert inc.incident_id.startswith("inc_")
        assert inc.state == IncidentState.DETECTED
        assert inc.severity == IncidentSeverity.HIGH
        assert "ble_aabb" in inc.target_ids
        # Should have an initial timeline entry
        assert inc.timeline.count >= 1

    def test_get_incident(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        retrieved = mgr.get(inc.incident_id)
        assert retrieved is not None
        assert retrieved.incident_id == inc.incident_id

    def test_get_nonexistent(self):
        mgr = IncidentManager()
        assert mgr.get("nonexistent") is None

    def test_investigate(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        result = mgr.investigate(inc.incident_id, reason="Checking it out")
        assert result is not None
        assert result.state == IncidentState.INVESTIGATING

    def test_respond(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        mgr.investigate(inc.incident_id)
        result = mgr.respond(inc.incident_id, reason="Sending team")
        assert result is not None
        assert result.state == IncidentState.RESPONDING

    def test_invalid_transition(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        mgr.resolve(inc.incident_id, summary="Done")
        mgr.close(inc.incident_id)
        # CLOSED is terminal
        result = mgr.investigate(inc.incident_id)
        assert result is None

    def test_resolve_incident(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        result = mgr.resolve(
            inc.incident_id,
            summary="All clear",
            resolution_type="false_alarm",
            resolved_by="operator",
            lessons_learned="Better detection needed",
            follow_up_actions=["Update rules"],
        )
        assert result is not None
        assert result.state == IncidentState.RESOLVED
        assert result.is_open is False
        assert result.resolution is not None
        assert result.resolution.summary == "All clear"
        assert result.resolution.resolution_type == "false_alarm"
        assert result.resolution.lessons_learned == "Better detection needed"

    def test_resolve_releases_resources(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        mgr.assign_resource(inc.incident_id, "drone-01", role="surveillance")
        mgr.assign_resource(inc.incident_id, "unit-alpha", role="response")

        # Both active
        updated = mgr.get(inc.incident_id)
        assert len(updated.active_resources) == 2

        mgr.resolve(inc.incident_id, summary="Done")
        resolved = mgr.get(inc.incident_id)
        assert len(resolved.active_resources) == 0

    def test_close_after_resolve(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        mgr.resolve(inc.incident_id, summary="Done")
        result = mgr.close(inc.incident_id)
        assert result is not None
        assert result.state == IncidentState.CLOSED

    def test_reopen_resolved(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        mgr.resolve(inc.incident_id, summary="Done")
        result = mgr.reopen(inc.incident_id, reason="New evidence")
        assert result is not None
        assert result.state == IncidentState.INVESTIGATING
        assert result.is_open is True

    def test_cannot_resolve_closed(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        mgr.resolve(inc.incident_id, summary="Done")
        mgr.close(inc.incident_id)
        result = mgr.resolve(inc.incident_id, summary="Try again")
        assert result is None


# ---------------------------------------------------------------------------
# IncidentManager — escalation
# ---------------------------------------------------------------------------

class TestIncidentEscalation:
    def test_escalate_up(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test", severity=IncidentSeverity.LOW)
        result = mgr.escalate(inc.incident_id, IncidentSeverity.HIGH, reason="More targets")
        assert result is not None
        assert result.severity == IncidentSeverity.HIGH

    def test_escalate_ignores_downgrade(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test", severity=IncidentSeverity.HIGH)
        result = mgr.escalate(inc.incident_id, IncidentSeverity.LOW)
        assert result is not None
        assert result.severity == IncidentSeverity.HIGH  # unchanged

    def test_escalate_same_severity(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test", severity=IncidentSeverity.MEDIUM)
        result = mgr.escalate(inc.incident_id, IncidentSeverity.MEDIUM)
        assert result is not None
        assert result.severity == IncidentSeverity.MEDIUM  # unchanged

    def test_escalate_closed_fails(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test", severity=IncidentSeverity.LOW)
        mgr.resolve(inc.incident_id, summary="Done")
        result = mgr.escalate(inc.incident_id, IncidentSeverity.CRITICAL)
        assert result is None

    def test_escalation_adds_timeline_entry(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test", severity=IncidentSeverity.LOW)
        mgr.escalate(inc.incident_id, IncidentSeverity.CRITICAL, reason="Major threat")
        entries = mgr.get_timeline(inc.incident_id)
        escalation_entries = [e for e in entries if e.entry_type == "escalation"]
        assert len(escalation_entries) == 1
        assert "critical" in escalation_entries[0].description


# ---------------------------------------------------------------------------
# IncidentManager — timeline
# ---------------------------------------------------------------------------

class TestIncidentTimeline:
    def test_add_timeline_entry(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        entry = mgr.add_timeline_entry(
            inc.incident_id,
            description="Drone dispatched",
            author="operator",
            entry_type="action",
        )
        assert entry is not None
        assert entry.description == "Drone dispatched"

    def test_add_timeline_entry_nonexistent(self):
        mgr = IncidentManager()
        result = mgr.add_timeline_entry("fake", description="Nothing")
        assert result is None

    def test_get_timeline(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        mgr.add_timeline_entry(inc.incident_id, description="Step 1")
        mgr.add_timeline_entry(inc.incident_id, description="Step 2")
        tl = mgr.get_timeline(inc.incident_id)
        # Initial creation entry + 2 added
        assert len(tl) >= 3


# ---------------------------------------------------------------------------
# IncidentManager — resources
# ---------------------------------------------------------------------------

class TestIncidentResources:
    def test_assign_resource(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        r = mgr.assign_resource(
            inc.incident_id,
            "drone-01",
            resource_type="drone",
            role="surveillance",
            notes="Covering east side",
        )
        assert r is not None
        assert r.resource_id == "drone-01"
        assert r.is_active is True

    def test_assign_duplicate_returns_existing(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        r1 = mgr.assign_resource(inc.incident_id, "drone-01")
        r2 = mgr.assign_resource(inc.incident_id, "drone-01")
        assert r1 is r2  # same object

    def test_release_resource(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        mgr.assign_resource(inc.incident_id, "drone-01")
        released = mgr.release_resource(inc.incident_id, "drone-01")
        assert released is True
        # Verify it is no longer active
        updated = mgr.get(inc.incident_id)
        assert len(updated.active_resources) == 0

    def test_release_nonexistent_resource(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        released = mgr.release_resource(inc.incident_id, "nonexistent")
        assert released is False


# ---------------------------------------------------------------------------
# IncidentManager — alert integration
# ---------------------------------------------------------------------------

class TestAlertIntegration:
    def test_add_alert(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        assert mgr.add_alert(inc.incident_id, "alert-001") is True
        updated = mgr.get(inc.incident_id)
        assert "alert-001" in updated.alert_ids

    def test_add_alert_duplicate_is_idempotent(self):
        mgr = IncidentManager()
        inc = mgr.create(title="Test")
        mgr.add_alert(inc.incident_id, "alert-001")
        mgr.add_alert(inc.incident_id, "alert-001")
        updated = mgr.get(inc.incident_id)
        assert updated.alert_ids.count("alert-001") == 1

    def test_create_from_alert_record(self):
        mgr = IncidentManager()

        # Simulate an AlertRecord (duck-typed)
        class FakeAlert:
            record_id = "rec-001"
            rule_name = "Hostile detected"
            severity = "critical"
            message = "Hostile in Zone Alpha"
            target_id = "ble_aabb"
            zone_id = "zone-alpha"

        inc = mgr.create_from_alert(FakeAlert())
        assert inc.severity == IncidentSeverity.CRITICAL
        assert "ble_aabb" in inc.target_ids
        assert "rec-001" in inc.alert_ids
        assert "from-alert" in inc.tags

    def test_connect_alert_engine(self):
        """Test that connecting AlertEngine wires up escalation handler."""
        from tritium_lib.alerting import AlertEngine, AlertRecord, DispatchAction

        mgr = IncidentManager()
        engine = AlertEngine(load_defaults=False)
        mgr.connect_alert_engine(engine)

        # Verify handler was registered
        assert DispatchAction.ESCALATE in engine._action_handlers

        # Simulate an escalation alert record
        record = AlertRecord(
            record_id="rec-esc-01",
            rule_id="test-rule",
            rule_name="Escalation Test",
            trigger="threat_detected",
            severity="critical",
            action="escalate",
            message="Major threat in Zone B",
            event_data={"target_id": "ble_ccdd"},
            target_id="ble_ccdd",
            zone_id="zone-b",
            timestamp=time.time(),
        )
        engine._action_handlers[DispatchAction.ESCALATE](record)

        # Should have created an incident
        incidents = mgr.get_all()
        assert len(incidents) == 1
        inc = incidents[0]
        assert inc.severity == IncidentSeverity.CRITICAL
        assert "ble_ccdd" in inc.target_ids
        assert "auto-created" in inc.tags


# ---------------------------------------------------------------------------
# IncidentManager — querying
# ---------------------------------------------------------------------------

class TestIncidentQuerying:
    def _create_several(self, mgr: IncidentManager) -> list[Incident]:
        i1 = mgr.create(title="Low zone-a", severity=IncidentSeverity.LOW, zone_id="zone-a", tags=["patrol"])
        i2 = mgr.create(title="High zone-b", severity=IncidentSeverity.HIGH, zone_id="zone-b", target_ids=["ble_aa"])
        i3 = mgr.create(title="Critical zone-a", severity=IncidentSeverity.CRITICAL, zone_id="zone-a", source="alerting")
        mgr.resolve(i1.incident_id, summary="Done")
        return [i1, i2, i3]

    def test_get_all(self):
        mgr = IncidentManager()
        self._create_several(mgr)
        all_inc = mgr.get_all()
        assert len(all_inc) == 3

    def test_get_open(self):
        mgr = IncidentManager()
        self._create_several(mgr)
        open_inc = mgr.get_open()
        assert len(open_inc) == 2

    def test_filter_by_state(self):
        mgr = IncidentManager()
        self._create_several(mgr)
        resolved = mgr.get_all(state=IncidentState.RESOLVED)
        assert len(resolved) == 1

    def test_filter_by_severity(self):
        mgr = IncidentManager()
        self._create_several(mgr)
        critical = mgr.get_all(severity=IncidentSeverity.CRITICAL)
        assert len(critical) == 1

    def test_filter_by_zone(self):
        mgr = IncidentManager()
        self._create_several(mgr)
        zone_a = mgr.get_all(zone_id="zone-a")
        assert len(zone_a) == 2

    def test_filter_by_target(self):
        mgr = IncidentManager()
        self._create_several(mgr)
        by_target = mgr.get_by_target("ble_aa")
        assert len(by_target) == 1

    def test_filter_by_tag(self):
        mgr = IncidentManager()
        self._create_several(mgr)
        tagged = mgr.get_all(tag="patrol")
        assert len(tagged) == 1

    def test_filter_by_source(self):
        mgr = IncidentManager()
        self._create_several(mgr)
        from_alerting = mgr.get_all(source="alerting")
        assert len(from_alerting) == 1

    def test_limit(self):
        mgr = IncidentManager()
        self._create_several(mgr)
        limited = mgr.get_all(limit=1)
        assert len(limited) == 1


# ---------------------------------------------------------------------------
# IncidentManager — merge + stats
# ---------------------------------------------------------------------------

class TestIncidentMergeAndStats:
    def test_merge_incidents(self):
        mgr = IncidentManager()
        i1 = mgr.create(title="Primary", target_ids=["t1"], alert_ids=["a1"])
        i2 = mgr.create(title="Secondary", target_ids=["t2", "t1"], alert_ids=["a2"], tags=["sensor"])

        result = mgr.merge(i1.incident_id, i2.incident_id)
        assert result is not None
        assert "t2" in result.target_ids
        assert "a2" in result.alert_ids
        assert "sensor" in result.tags

        # Secondary should be resolved
        secondary = mgr.get(i2.incident_id)
        assert secondary.state == IncidentState.RESOLVED
        assert secondary.resolution is not None

    def test_merge_nonexistent(self):
        mgr = IncidentManager()
        i1 = mgr.create(title="Primary")
        assert mgr.merge(i1.incident_id, "fake") is None
        assert mgr.merge("fake", i1.incident_id) is None

    def test_stats(self):
        mgr = IncidentManager()
        mgr.create(title="A", severity=IncidentSeverity.LOW)
        inc_b = mgr.create(title="B", severity=IncidentSeverity.HIGH)
        mgr.create(title="C", severity=IncidentSeverity.CRITICAL)
        mgr.resolve(inc_b.incident_id, summary="Done")
        mgr.escalate(
            mgr.get_all(severity=IncidentSeverity.LOW)[0].incident_id,
            IncidentSeverity.MEDIUM,
        )

        stats = mgr.get_stats()
        assert stats["total_incidents"] == 3
        assert stats["total_created"] == 3
        assert stats["total_resolved"] == 1
        assert stats["total_escalations"] == 1
        assert stats["open_count"] == 2

    def test_reset(self):
        mgr = IncidentManager()
        mgr.create(title="Test")
        mgr.reset()
        stats = mgr.get_stats()
        assert stats["total_incidents"] == 0
        assert stats["total_created"] == 0


# ---------------------------------------------------------------------------
# IncidentManager — callbacks
# ---------------------------------------------------------------------------

class TestIncidentCallbacks:
    def test_on_created_callback(self):
        created = []
        mgr = IncidentManager(on_incident_created=lambda inc: created.append(inc))
        mgr.create(title="Test")
        assert len(created) == 1

    def test_on_resolved_callback(self):
        resolved = []
        mgr = IncidentManager(on_incident_resolved=lambda inc: resolved.append(inc))
        inc = mgr.create(title="Test")
        mgr.resolve(inc.incident_id, summary="Done")
        assert len(resolved) == 1


# ---------------------------------------------------------------------------
# IncidentManager — eviction
# ---------------------------------------------------------------------------

class TestIncidentEviction:
    def test_evicts_oldest_closed(self):
        mgr = IncidentManager(max_incidents=3)
        i1 = mgr.create(title="First")
        mgr.resolve(i1.incident_id, summary="Done")
        i2 = mgr.create(title="Second")
        mgr.resolve(i2.incident_id, summary="Done")
        i3 = mgr.create(title="Third")
        # At capacity (3). Creating 4th should evict oldest closed.
        i4 = mgr.create(title="Fourth")

        stats = mgr.get_stats()
        # Should have evicted the oldest resolved incident
        assert stats["total_incidents"] == 3
        assert mgr.get(i1.incident_id) is None  # evicted


# ---------------------------------------------------------------------------
# IncidentManager — event bus integration
# ---------------------------------------------------------------------------

class TestEventBusIntegration:
    def test_publishes_events(self):
        published = []

        class FakeBus:
            def publish(self, topic, data=None, source=""):
                published.append({"topic": topic, "data": data})

        mgr = IncidentManager(event_bus=FakeBus())
        inc = mgr.create(title="Test")
        mgr.investigate(inc.incident_id)
        mgr.resolve(inc.incident_id, summary="Done")

        topics = [p["topic"] for p in published]
        assert "incident.created" in topics
        assert "incident.state_changed" in topics
        assert "incident.resolved" in topics
