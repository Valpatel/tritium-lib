# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for GPX 1.1 XML generation models."""

from datetime import datetime, timezone

import pytest

from tritium_lib.models.gpx import (
    GPXDocument,
    GPXRoute,
    GPXTrack,
    GPXWaypoint,
)


class TestGPXWaypoint:
    def test_basic_waypoint(self):
        wpt = GPXWaypoint(lat=33.123456, lon=-117.654321)
        el = wpt.to_element()
        assert el.tag == "wpt"
        assert el.get("lat") == "33.12345600"
        assert el.get("lon") == "-117.65432100"

    def test_waypoint_with_all_fields(self):
        t = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)
        wpt = GPXWaypoint(
            lat=33.0, lon=-117.0, ele=100.5,
            time=t, name="Alpha", desc="Test point", sym="Pin",
        )
        el = wpt.to_element()
        assert el.find("ele").text == "100.50"
        assert el.find("time").text == t.isoformat()
        assert el.find("name").text == "Alpha"
        assert el.find("desc").text == "Test point"
        assert el.find("sym").text == "Pin"

    def test_to_dict_roundtrip(self):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        wpt = GPXWaypoint(lat=1.0, lon=2.0, ele=3.0, time=t, name="X")
        d = wpt.to_dict()
        wpt2 = GPXWaypoint.from_dict(d)
        assert wpt2.lat == wpt.lat
        assert wpt2.lon == wpt.lon
        assert wpt2.ele == wpt.ele
        assert wpt2.name == wpt.name

    def test_custom_tag(self):
        wpt = GPXWaypoint(lat=0.0, lon=0.0)
        el = wpt.to_element("trkpt")
        assert el.tag == "trkpt"


class TestGPXTrack:
    def test_empty_track(self):
        trk = GPXTrack(name="Empty")
        el = trk.to_element()
        assert el.find("name").text == "Empty"
        assert el.find("trkseg") is not None

    def test_add_points(self):
        trk = GPXTrack(name="Patrol")
        trk.add_point(33.0, -117.0)
        trk.add_point(33.001, -117.001, ele=50.0)
        assert len(trk.points) == 2
        el = trk.to_element()
        seg = el.find("trkseg")
        trkpts = seg.findall("trkpt")
        assert len(trkpts) == 2

    def test_to_dict_roundtrip(self):
        trk = GPXTrack(name="Test", desc="A track")
        trk.add_point(1.0, 2.0)
        trk.add_point(3.0, 4.0)
        d = trk.to_dict()
        trk2 = GPXTrack.from_dict(d)
        assert trk2.name == "Test"
        assert len(trk2.points) == 2


class TestGPXRoute:
    def test_empty_route(self):
        rte = GPXRoute(name="Nav")
        el = rte.to_element()
        assert el.find("name").text == "Nav"

    def test_add_points(self):
        rte = GPXRoute(name="Perimeter")
        rte.add_point(33.0, -117.0, name="Start")
        rte.add_point(33.01, -117.01, name="End")
        assert len(rte.points) == 2
        el = rte.to_element()
        rtepts = el.findall("rtept")
        assert len(rtepts) == 2

    def test_to_dict_roundtrip(self):
        rte = GPXRoute(name="R1")
        rte.add_point(10.0, 20.0)
        d = rte.to_dict()
        rte2 = GPXRoute.from_dict(d)
        assert rte2.name == "R1"
        assert len(rte2.points) == 1


class TestGPXDocument:
    def test_empty_document(self):
        doc = GPXDocument(name="Test")
        xml = doc.to_xml()
        assert '<?xml version="1.0"' in xml
        assert 'version="1.1"' in xml
        assert 'creator="Tritium"' in xml
        assert "<name>Test</name>" in xml

    def test_document_with_waypoints(self):
        doc = GPXDocument()
        doc.add_waypoint(33.0, -117.0, name="HQ")
        xml = doc.to_xml()
        assert "<wpt" in xml
        assert "<name>HQ</name>" in xml

    def test_document_with_track(self):
        doc = GPXDocument(name="Target Trail")
        trk = doc.add_track(name="ble_AA:BB:CC:DD:EE:FF")
        trk.add_point(33.0, -117.0)
        trk.add_point(33.001, -117.001)
        xml = doc.to_xml()
        assert "<trk>" in xml
        assert "<trkseg>" in xml
        assert "<trkpt" in xml

    def test_document_with_route(self):
        doc = GPXDocument()
        rte = doc.add_route(name="Patrol")
        rte.add_point(33.0, -117.0)
        xml = doc.to_xml()
        assert "<rte>" in xml
        assert "<rtept" in xml

    def test_to_bytes(self):
        doc = GPXDocument()
        b = doc.to_bytes()
        assert isinstance(b, bytes)
        assert b.startswith(b"<?xml")

    def test_no_declaration(self):
        doc = GPXDocument()
        xml = doc.to_xml(xml_declaration=False)
        assert not xml.startswith("<?xml")
        assert xml.startswith("<gpx")

    def test_full_integration(self):
        """Build a realistic target trail export."""
        doc = GPXDocument(
            creator="Tritium Command Center",
            name="Target ble_AA:BB trail export",
            desc="Movement trail for BLE target",
        )
        doc.add_waypoint(33.0, -117.0, name="First seen", sym="Flag")
        doc.add_waypoint(33.01, -117.01, name="Last seen", sym="Flag")

        trk = doc.add_track(name="Movement trail")
        t = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc)
        trk.add_point(33.0, -117.0, time=t)
        trk.add_point(33.005, -117.005, time=t)
        trk.add_point(33.01, -117.01, time=t)

        xml = doc.to_xml()
        assert "Tritium Command Center" in xml
        assert "<trk>" in xml
        assert len(doc.waypoints) == 2
        assert len(doc.tracks) == 1
        assert len(doc.tracks[0].points) == 3
