"""Tests for tritium_lib.models.cot — CoT event models and XML codec."""

from datetime import datetime, timedelta, timezone

from tritium_lib.models.cot import (
    CotEvent,
    CotPoint,
    CotDetail,
    CotContact,
    cot_to_xml,
    xml_to_cot,
    COT_FRIENDLY_GROUND_UNIT,
    COT_FRIENDLY_UAV,
    COT_FRIENDLY_GROUND_SENSOR,
    COT_HOSTILE_GROUND_UNIT,
    COT_NEUTRAL_GROUND,
    COT_UNKNOWN_GROUND,
    COT_MAP_MARKER,
)


class TestCotPoint:
    def test_defaults(self):
        p = CotPoint()
        assert p.lat == 0.0
        assert p.lon == 0.0
        assert p.ce == 9999999.0

    def test_create(self):
        p = CotPoint(lat=37.7749, lon=-122.4194, hae=15.0, ce=10.0, le=10.0)
        assert p.lat == 37.7749
        assert p.hae == 15.0


class TestCotContact:
    def test_create(self):
        c = CotContact(callsign="Alpha-1", endpoint="*:-1:stcp")
        assert c.callsign == "Alpha-1"
        assert c.endpoint == "*:-1:stcp"


class TestCotDetail:
    def test_defaults(self):
        d = CotDetail()
        assert d.contact is None
        assert d.remarks == ""
        assert d.extra == {}

    def test_with_contact(self):
        d = CotDetail(
            contact=CotContact(callsign="Bravo-2"),
            group_name="Cyan",
            group_role="Sensor",
            remarks="On patrol",
        )
        assert d.contact.callsign == "Bravo-2"
        assert d.group_name == "Cyan"

    def test_extra_fields(self):
        d = CotDetail(extra={"tritium_edge": {"device_id": "esp32-001", "role": "sensor"}})
        assert d.extra["tritium_edge"]["device_id"] == "esp32-001"


class TestCotEvent:
    def test_defaults(self):
        e = CotEvent()
        assert e.type == COT_FRIENDLY_GROUND_UNIT
        assert e.how == "m-g"
        assert e.version == "2.0"
        assert e.uid  # should have auto-generated UUID

    def test_create_full(self):
        now = datetime.now(timezone.utc)
        stale = now + timedelta(seconds=600)
        e = CotEvent(
            uid="test-unit-1",
            type=COT_FRIENDLY_UAV,
            how="m-g",
            time=now,
            start=now,
            stale=stale,
            point=CotPoint(lat=37.7749, lon=-122.4194, hae=100.0, ce=5.0, le=5.0),
            detail=CotDetail(
                contact=CotContact(callsign="Drone-1"),
                group_name="Cyan",
                group_role="Recon",
            ),
        )
        assert e.uid == "test-unit-1"
        assert e.type == "a-f-A-M-F-Q"
        assert e.point.lat == 37.7749

    def test_alliance_friendly(self):
        e = CotEvent(type="a-f-G-U-C")
        assert e.alliance == "friendly"

    def test_alliance_hostile(self):
        e = CotEvent(type="a-h-G-U-C")
        assert e.alliance == "hostile"

    def test_alliance_neutral(self):
        e = CotEvent(type="a-n-G")
        assert e.alliance == "neutral"

    def test_alliance_unknown(self):
        e = CotEvent(type="a-u-G")
        assert e.alliance == "unknown"

    def test_is_stale(self):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        e = CotEvent(stale=past)
        assert e.is_stale is True

    def test_not_stale(self):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        e = CotEvent(stale=future)
        assert e.is_stale is False

    def test_json_roundtrip(self):
        e = CotEvent(
            uid="test-1",
            type=COT_FRIENDLY_GROUND_SENSOR,
            point=CotPoint(lat=10.0, lon=20.0),
            detail=CotDetail(contact=CotContact(callsign="Test")),
        )
        e2 = CotEvent.model_validate_json(e.model_dump_json())
        assert e2.uid == e.uid
        assert e2.point.lat == 10.0
        assert e2.detail.contact.callsign == "Test"


class TestCotTypeConstants:
    def test_type_strings(self):
        assert COT_FRIENDLY_GROUND_UNIT == "a-f-G-U-C"
        assert COT_FRIENDLY_UAV == "a-f-A-M-F-Q"
        assert COT_FRIENDLY_GROUND_SENSOR == "a-f-G-E-S"
        assert COT_HOSTILE_GROUND_UNIT == "a-h-G-U-C"
        assert COT_NEUTRAL_GROUND == "a-n-G"
        assert COT_UNKNOWN_GROUND == "a-u-G"
        assert COT_MAP_MARKER == "b-m-p-s-m"


class TestCotToXml:
    def test_basic(self):
        e = CotEvent(
            uid="test-1",
            type=COT_FRIENDLY_GROUND_UNIT,
            point=CotPoint(lat=37.7749, lon=-122.4194),
        )
        xml = cot_to_xml(e)
        assert "test-1" in xml
        assert "37.7749" in xml
        assert "-122.4194" in xml
        assert 'version="2.0"' in xml

    def test_with_contact(self):
        e = CotEvent(
            uid="test-2",
            detail=CotDetail(contact=CotContact(callsign="Alpha-1")),
        )
        xml = cot_to_xml(e)
        assert "Alpha-1" in xml
        assert "<contact" in xml

    def test_with_group(self):
        e = CotEvent(
            uid="test-3",
            detail=CotDetail(group_name="Cyan", group_role="Sensor"),
        )
        xml = cot_to_xml(e)
        assert "Cyan" in xml
        assert "Sensor" in xml

    def test_with_remarks(self):
        e = CotEvent(
            uid="test-4",
            detail=CotDetail(remarks="On station"),
        )
        xml = cot_to_xml(e)
        assert "On station" in xml

    def test_with_extra(self):
        e = CotEvent(
            uid="test-5",
            detail=CotDetail(extra={"tritium_edge": {"device_id": "esp32-001"}}),
        )
        xml = cot_to_xml(e)
        assert "tritium_edge" in xml
        assert "esp32-001" in xml

    def test_xml_declaration(self):
        e = CotEvent(uid="test-6")
        xml = cot_to_xml(e)
        assert xml.startswith("<?xml")


class TestXmlToCot:
    def test_roundtrip(self):
        original = CotEvent(
            uid="roundtrip-1",
            type=COT_FRIENDLY_UAV,
            how="m-g",
            point=CotPoint(lat=37.7749, lon=-122.4194, hae=100.0, ce=5.0, le=5.0),
            detail=CotDetail(
                contact=CotContact(callsign="Drone-1"),
                group_name="Cyan",
                group_role="Recon",
                remarks="Overhead at 100m",
            ),
        )
        xml = cot_to_xml(original)
        parsed = xml_to_cot(xml)

        assert parsed is not None
        assert parsed.uid == "roundtrip-1"
        assert parsed.type == COT_FRIENDLY_UAV
        assert abs(parsed.point.lat - 37.7749) < 0.001
        assert abs(parsed.point.lon - (-122.4194)) < 0.001
        assert abs(parsed.point.hae - 100.0) < 0.1
        assert parsed.detail.contact.callsign == "Drone-1"
        assert parsed.detail.group_name == "Cyan"
        assert parsed.detail.remarks == "Overhead at 100m"

    def test_roundtrip_extra(self):
        original = CotEvent(
            uid="extra-1",
            detail=CotDetail(
                extra={"custom_tag": {"key1": "val1", "key2": "val2"}},
            ),
        )
        xml = cot_to_xml(original)
        parsed = xml_to_cot(xml)
        assert parsed is not None
        assert "custom_tag" in parsed.detail.extra
        assert parsed.detail.extra["custom_tag"]["key1"] == "val1"

    def test_invalid_xml(self):
        assert xml_to_cot("not xml at all") is None

    def test_non_event_xml(self):
        assert xml_to_cot("<root/>") is None

    def test_minimal_event(self):
        xml = (
            '<?xml version="1.0" ?>'
            '<event version="2.0" uid="min-1" type="a-f-G" how="m-g" '
            'time="2026-03-07T12:00:00.000000Z" '
            'start="2026-03-07T12:00:00.000000Z" '
            'stale="2026-03-07T12:05:00.000000Z">'
            '<point lat="0" lon="0" hae="0" ce="10" le="10"/>'
            '<detail/>'
            '</event>'
        )
        parsed = xml_to_cot(xml)
        assert parsed is not None
        assert parsed.uid == "min-1"
        assert parsed.type == "a-f-G"

    def test_contact_endpoint_preserved(self):
        original = CotEvent(
            uid="ep-1",
            detail=CotDetail(
                contact=CotContact(callsign="Node-1", endpoint="*:-1:stcp"),
            ),
        )
        xml = cot_to_xml(original)
        parsed = xml_to_cot(xml)
        assert parsed.detail.contact.endpoint == "*:-1:stcp"
