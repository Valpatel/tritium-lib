# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for KML 2.2 XML generation models."""

from datetime import datetime, timezone

import pytest

from tritium_lib.models.kml import (
    KMLDocument,
    KMLPoint,
    KMLTrack,
)


class TestKMLPoint:
    def test_basic_point(self):
        pt = KMLPoint(lat=33.123456, lon=-117.654321)
        coord = pt.coord_string()
        assert "-117.65432100,33.12345600,0" == coord

    def test_point_with_altitude(self):
        pt = KMLPoint(lat=33.0, lon=-117.0, alt=100.5)
        coord = pt.coord_string()
        assert "-117.00000000,33.00000000,100.50" == coord

    def test_to_dict_roundtrip(self):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        pt = KMLPoint(lat=1.0, lon=2.0, alt=3.0, time=t, name="X")
        d = pt.to_dict()
        pt2 = KMLPoint.from_dict(d)
        assert pt2.lat == pt.lat
        assert pt2.lon == pt.lon
        assert pt2.alt == pt.alt
        assert pt2.name == pt.name

    def test_from_dict_string_time(self):
        d = {"lat": 10.0, "lon": 20.0, "time": "2026-03-14T12:00:00+00:00"}
        pt = KMLPoint.from_dict(d)
        assert pt.time is not None
        assert pt.time.year == 2026


class TestKMLTrack:
    def test_empty_track(self):
        trk = KMLTrack(name="Empty")
        el = trk.to_element()
        assert el.tag == "Placemark"
        assert el.find("name").text == "Empty"

    def test_add_points(self):
        trk = KMLTrack(name="Patrol")
        trk.add_point(33.0, -117.0)
        trk.add_point(33.001, -117.001, alt=50.0)
        assert len(trk.points) == 2
        el = trk.to_element()
        linestring = el.find("LineString")
        assert linestring is not None
        coords_text = linestring.find("coordinates").text
        assert "," in coords_text

    def test_timespan_from_points(self):
        t1 = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 14, 11, 0, 0, tzinfo=timezone.utc)
        trk = KMLTrack(name="Timed")
        trk.add_point(33.0, -117.0, time=t1)
        trk.add_point(33.01, -117.01, time=t2)
        el = trk.to_element()
        ts = el.find("TimeSpan")
        assert ts is not None
        assert ts.find("begin") is not None
        assert ts.find("end") is not None

    def test_style_element(self):
        trk = KMLTrack(name="Styled", line_color="ff0000ff", line_width=5.0)
        el = trk.to_element()
        style = el.find("Style")
        assert style is not None
        ls = style.find("LineStyle")
        assert ls.find("color").text == "ff0000ff"
        assert ls.find("width").text == "5.0"

    def test_to_dict_roundtrip(self):
        trk = KMLTrack(name="Test", desc="A track")
        trk.add_point(1.0, 2.0)
        trk.add_point(3.0, 4.0)
        d = trk.to_dict()
        trk2 = KMLTrack.from_dict(d)
        assert trk2.name == "Test"
        assert len(trk2.points) == 2


class TestKMLDocument:
    def test_empty_document(self):
        doc = KMLDocument(name="Test")
        xml = doc.to_xml()
        assert '<?xml version="1.0"' in xml
        assert "kml" in xml
        assert "<name>Test</name>" in xml

    def test_document_with_placemarks(self):
        doc = KMLDocument()
        doc.add_placemark(33.0, -117.0, name="HQ")
        xml = doc.to_xml()
        assert "<Placemark>" in xml or "<Placemark" in xml
        assert "<name>HQ</name>" in xml

    def test_document_with_track(self):
        doc = KMLDocument(name="Target Trail")
        trk = doc.add_track(name="ble_AA:BB:CC:DD:EE:FF")
        trk.add_point(33.0, -117.0)
        trk.add_point(33.001, -117.001)
        xml = doc.to_xml()
        assert "LineString" in xml
        assert "coordinates" in xml

    def test_to_bytes(self):
        doc = KMLDocument()
        b = doc.to_bytes()
        assert isinstance(b, bytes)
        assert b.startswith(b"<?xml")

    def test_no_declaration(self):
        doc = KMLDocument()
        xml = doc.to_xml(xml_declaration=False)
        assert not xml.startswith("<?xml")
        assert xml.startswith("<kml")

    def test_full_integration(self):
        """Build a realistic target trail KML export."""
        doc = KMLDocument(
            name="Target ble_AA:BB trail export",
            desc="Movement trail for BLE target",
        )
        doc.add_placemark(33.0, -117.0, name="First seen")
        doc.add_placemark(33.01, -117.01, name="Last seen")

        trk = doc.add_track(name="Movement trail")
        t = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc)
        trk.add_point(33.0, -117.0, time=t)
        trk.add_point(33.005, -117.005, time=t)
        trk.add_point(33.01, -117.01, time=t)

        xml = doc.to_xml()
        assert "Document" in xml
        assert "LineString" in xml
        assert len(doc.placemarks) == 2
        assert len(doc.tracks) == 1
        assert len(doc.tracks[0].points) == 3

    def test_valid_xml_parse(self):
        """Output should be parseable by xml.etree."""
        import xml.etree.ElementTree as ET
        doc = KMLDocument(name="Parseable")
        doc.add_placemark(33.0, -117.0, name="Point")
        trk = doc.add_track(name="Track")
        trk.add_point(33.0, -117.0)
        trk.add_point(33.01, -117.01)
        xml_str = doc.to_xml()
        root = ET.fromstring(xml_str)
        assert "kml" in root.tag.lower()
