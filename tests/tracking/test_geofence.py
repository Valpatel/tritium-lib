# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.tracking.geofence."""

import pytest

from tritium_lib.tracking.geofence import (
    GeofenceEngine,
    GeoZone,
    GeoEvent,
)


class TestPointInPolygon:
    """Tests for geofence point-in-polygon detection."""

    def _make_square_zone(self, zone_id="zone1", name="Test Zone"):
        """Create a square zone from (0,0) to (10,10)."""
        return GeoZone(
            zone_id=zone_id,
            name=name,
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        )

    def test_point_inside_zone(self):
        """A point inside the polygon should trigger an enter event."""
        engine = GeofenceEngine()
        zone = self._make_square_zone()
        engine.add_zone(zone)

        events = engine.check("target_1", (5.0, 5.0))
        assert len(events) == 1
        assert events[0].event_type == "enter"
        assert events[0].zone_id == "zone1"
        assert events[0].target_id == "target_1"

    def test_point_outside_zone(self):
        """A point outside the polygon should not trigger events."""
        engine = GeofenceEngine()
        zone = self._make_square_zone()
        engine.add_zone(zone)

        events = engine.check("target_1", (15.0, 15.0))
        assert len(events) == 0

    def test_enter_then_inside(self):
        """Second check inside same zone should produce 'inside' not 'enter'."""
        engine = GeofenceEngine()
        zone = self._make_square_zone()
        engine.add_zone(zone)

        events1 = engine.check("target_1", (5.0, 5.0))
        assert events1[0].event_type == "enter"

        events2 = engine.check("target_1", (6.0, 6.0))
        assert len(events2) == 1
        assert events2[0].event_type == "inside"

    def test_exit_detection(self):
        """Moving from inside to outside should trigger exit event."""
        engine = GeofenceEngine()
        zone = self._make_square_zone()
        engine.add_zone(zone)

        # Enter
        engine.check("target_1", (5.0, 5.0))
        # Exit
        events = engine.check("target_1", (15.0, 15.0))
        assert len(events) == 1
        assert events[0].event_type == "exit"
        assert events[0].zone_id == "zone1"


class TestZoneAlerts:
    """Tests for zone alert generation."""

    def test_alert_on_enter_published(self):
        """Enter event should be published to event bus when alert_on_enter=True."""
        published = []

        class MockBus:
            def publish(self, topic, data):
                published.append((topic, data))

        engine = GeofenceEngine(event_bus=MockBus())
        zone = GeoZone(
            zone_id="restricted",
            name="Restricted Area",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            zone_type="restricted",
            alert_on_enter=True,
        )
        engine.add_zone(zone)

        engine.check("intruder_1", (5.0, 5.0))

        assert len(published) == 1
        assert published[0][0] == "geofence:enter"
        assert published[0][1]["zone_name"] == "Restricted Area"
        assert published[0][1]["target_id"] == "intruder_1"

    def test_alert_on_exit_published(self):
        """Exit event should be published to event bus when alert_on_exit=True."""
        published = []

        class MockBus:
            def publish(self, topic, data):
                published.append((topic, data))

        engine = GeofenceEngine(event_bus=MockBus())
        zone = GeoZone(
            zone_id="monitored",
            name="Monitored Area",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            alert_on_exit=True,
        )
        engine.add_zone(zone)

        engine.check("target_1", (5.0, 5.0))  # enter
        engine.check("target_1", (20.0, 20.0))  # exit

        assert len(published) == 2  # enter + exit
        assert published[1][0] == "geofence:exit"

    def test_disabled_zone_ignored(self):
        """Disabled zones should not detect targets."""
        engine = GeofenceEngine()
        zone = GeoZone(
            zone_id="disabled",
            name="Disabled Zone",
            polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            enabled=False,
        )
        engine.add_zone(zone)

        events = engine.check("target_1", (5.0, 5.0))
        assert len(events) == 0

    def test_zone_crud(self):
        """Test add, get, list, and remove zone operations."""
        engine = GeofenceEngine()
        zone = GeoZone(
            zone_id="z1",
            name="Zone 1",
            polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
        )
        engine.add_zone(zone)
        assert engine.get_zone("z1") is not None
        assert len(engine.list_zones()) == 1

        assert engine.remove_zone("z1") is True
        assert engine.get_zone("z1") is None
        assert engine.remove_zone("z1") is False

    def test_multiple_zones(self):
        """Target can be in multiple zones simultaneously."""
        engine = GeofenceEngine()
        engine.add_zone(GeoZone(
            zone_id="z1",
            name="Zone 1",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        ))
        engine.add_zone(GeoZone(
            zone_id="z2",
            name="Zone 2",
            polygon=[(5, 5), (15, 5), (15, 15), (5, 15)],
        ))

        # Point (7, 7) is inside both zones
        events = engine.check("target_1", (7.0, 7.0))
        assert len(events) == 2
        assert all(e.event_type == "enter" for e in events)

    def test_get_zone_occupants(self):
        """Should return target IDs currently inside a zone."""
        engine = GeofenceEngine()
        engine.add_zone(GeoZone(
            zone_id="z1",
            name="Zone 1",
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        ))

        engine.check("t1", (5.0, 5.0))
        engine.check("t2", (3.0, 3.0))
        engine.check("t3", (20.0, 20.0))

        occupants = engine.get_zone_occupants("z1")
        assert set(occupants) == {"t1", "t2"}


class TestZoneBrief:
    """Tests for the cognition-grounding zone_brief() inventory."""

    def _square(self, zid, name, ztype="monitored"):
        return GeoZone(
            zone_id=zid,
            name=name,
            polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            zone_type=ztype,
        )

    def _far_square(self, zid, name, ztype="monitored"):
        return GeoZone(
            zone_id=zid,
            name=name,
            polygon=[(100, 100), (110, 100), (110, 110), (100, 110)],
            zone_type=ztype,
        )

    def test_empty_when_no_zones(self):
        """No zones defined -> empty string (no noise in grounding)."""
        engine = GeofenceEngine()
        assert engine.zone_brief() == ""

    def test_lists_zone_and_occupancy(self):
        """Brief reports zone count, occupancy, and per-zone inside counts."""
        engine = GeofenceEngine()
        engine.add_zone(self._square("z1", "North Gate"))
        engine.check("t1", (5.0, 5.0))
        engine.check("t2", (3.0, 3.0))

        brief = engine.zone_brief()
        assert "ZONES: 1 defined" in brief
        assert "1 occupied" in brief
        assert "2 target(s) inside" in brief
        assert "NORTH GATE".title().upper() in brief.upper()
        assert "2 inside" in brief

    def test_empty_zone_marked_empty(self):
        """A zone with no occupants is explicitly labelled empty."""
        engine = GeofenceEngine()
        engine.add_zone(self._square("z1", "Plaza"))
        brief = engine.zone_brief()
        assert "0 occupied" in brief
        assert "empty" in brief

    def test_alliance_breakdown_with_resolver(self):
        """With a resolver, occupants are broken down by alliance."""
        engine = GeofenceEngine()
        engine.add_zone(self._square("z1", "Plaza"))
        engine.check("hostile_1", (5.0, 5.0))
        engine.check("civ_1", (4.0, 4.0))

        roles = {"hostile_1": "hostile", "civ_1": "neutral"}
        brief = engine.zone_brief(
            occupant_resolver=lambda tid: {"alliance": roles.get(tid, "unknown")}
        )
        assert "1 hostile" in brief
        assert "1 neutral" in brief

    def test_hostile_in_restricted_zone_is_breach(self):
        """A hostile inside a RESTRICTED zone is flagged as a BREACH."""
        engine = GeofenceEngine()
        engine.add_zone(self._square("z1", "Vault", ztype="restricted"))
        engine.check("hostile_1", (5.0, 5.0))

        brief = engine.zone_brief(
            occupant_resolver=lambda tid: {"alliance": "hostile"}
        )
        assert "BREACH" in brief
        assert "RESTRICTED 'Vault'" in brief

    def test_neutral_in_restricted_is_not_breach(self):
        """A neutral in a restricted zone is reported but not a breach."""
        engine = GeofenceEngine()
        engine.add_zone(self._square("z1", "Vault", ztype="restricted"))
        engine.check("civ_1", (5.0, 5.0))

        brief = engine.zone_brief(
            occupant_resolver=lambda tid: {"alliance": "neutral"}
        )
        assert "BREACH" not in brief

    def test_max_zones_cap(self):
        """Lists at most max_zones, noting how many more exist."""
        engine = GeofenceEngine()
        for i in range(9):
            engine.add_zone(self._square(f"z{i}", f"Zone {i}"))
        brief = engine.zone_brief(max_zones=6)
        assert "ZONES: 9 defined" in brief
        assert "more zone(s)" in brief

    def test_occupied_zones_ranked_first(self):
        """An occupied zone sorts above an empty one within the cap."""
        engine = GeofenceEngine()
        # Many empty zones + one occupied far zone; the occupied one must show.
        for i in range(6):
            engine.add_zone(self._square(f"empty{i}", f"Empty {i}"))
        engine.add_zone(self._far_square("hot", "Hot Zone"))
        engine.check("t1", (105.0, 105.0))

        brief = engine.zone_brief(max_zones=6)
        assert "HOT ZONE" in brief.upper()
        assert "1 inside" in brief
