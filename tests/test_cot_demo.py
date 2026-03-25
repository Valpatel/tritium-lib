# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the CoT interoperability demo.

Validates the full CoT codec pipeline:
  - Target to CoT XML generation
  - CoT XML parsing and ingestion
  - Roundtrip fidelity (Target -> XML -> Target)
  - Edge device CoT generation
  - Alliance/type mapping correctness
  - REST endpoint behavior (via TestClient)
  - Sample CoT generation utility
  - Ingest log tracking
  - Error handling for malformed XML
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from tritium_lib.cot.demos.cot_demo import (
    CotDemoState,
    app,
    generate_sample_cot_event,
    roundtrip_demo,
)
from tritium_lib.models.cot import xml_to_cot, cot_to_xml, CotEvent, CotPoint, CotContact, CotDetail


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_state():
    """Fresh CotDemoState for each test."""
    return CotDemoState()


@pytest.fixture
def client():
    """FastAPI TestClient for endpoint tests."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: Target to CoT XML generation
# ---------------------------------------------------------------------------

class TestTargetToCotGeneration:
    def test_generates_xml_for_all_targets(self, demo_state):
        events = demo_state.targets_to_cot_events()
        assert len(events) == len(demo_state.targets)
        for xml_str in events:
            assert "<event" in xml_str
            assert "version=" in xml_str

    def test_generated_xml_is_parseable(self, demo_state):
        events = demo_state.targets_to_cot_events()
        for xml_str in events:
            root = ET.fromstring(xml_str)
            assert root.tag == "event"
            assert root.get("type") is not None
            point = root.find("point")
            assert point is not None
            assert point.get("lat") is not None

    def test_single_target_cot(self, demo_state):
        xml_str = demo_state.target_to_cot_xml("rover-alpha")
        assert xml_str is not None
        assert "rover-alpha" in xml_str
        root = ET.fromstring(xml_str)
        assert root.tag == "event"

    def test_missing_target_returns_none(self, demo_state):
        assert demo_state.target_to_cot_xml("nonexistent") is None

    def test_xml_generation_counter(self, demo_state):
        assert demo_state.generated_xml_count == 0
        demo_state.targets_to_cot_events()
        assert demo_state.generated_xml_count == len(demo_state.targets)


# ---------------------------------------------------------------------------
# Test 2: Alliance mapping in CoT type strings
# ---------------------------------------------------------------------------

class TestAllianceMapping:
    def test_friendly_target_has_f_prefix(self, demo_state):
        xml_str = demo_state.target_to_cot_xml("rover-alpha")
        root = ET.fromstring(xml_str)
        cot_type = root.get("type")
        assert cot_type.startswith("a-f-"), f"Expected a-f- prefix, got {cot_type}"

    def test_hostile_target_has_h_prefix(self, demo_state):
        xml_str = demo_state.target_to_cot_xml("det_vehicle_001")
        root = ET.fromstring(xml_str)
        cot_type = root.get("type")
        assert cot_type.startswith("a-h-"), f"Expected a-h- prefix, got {cot_type}"

    def test_unknown_target_has_u_prefix(self, demo_state):
        xml_str = demo_state.target_to_cot_xml("det_person_001")
        root = ET.fromstring(xml_str)
        cot_type = root.get("type")
        assert cot_type.startswith("a-u-"), f"Expected a-u- prefix, got {cot_type}"


# ---------------------------------------------------------------------------
# Test 3: Edge device CoT generation
# ---------------------------------------------------------------------------

class TestEdgeDeviceCot:
    def test_edge_devices_generate_cot(self, demo_state):
        events = demo_state.edge_devices_to_cot()
        assert len(events) == 2

    def test_edge_cot_has_xml_declaration(self, demo_state):
        events = demo_state.edge_devices_to_cot()
        for xml_str in events:
            assert "<?xml" in xml_str

    def test_edge_cot_has_tritium_detail(self, demo_state):
        events = demo_state.edge_devices_to_cot()
        for xml_str in events:
            root = ET.fromstring(xml_str)
            detail = root.find("detail")
            assert detail is not None
            edge = detail.find("tritium_edge")
            assert edge is not None
            assert edge.get("device_id") is not None


# ---------------------------------------------------------------------------
# Test 4: CoT XML ingestion
# ---------------------------------------------------------------------------

class TestCotIngestion:
    def test_ingest_valid_cot(self, demo_state):
        xml_str = generate_sample_cot_event(
            uid="external-unit-01",
            callsign="Bravo Actual",
            lat=37.7750,
            lon=-122.4190,
            alliance="friendly",
        )
        result = demo_state.ingest_cot_xml(xml_str)
        assert result["status"] == "ingested"
        assert "external-unit-01" in demo_state.targets

    def test_ingest_hostile_cot(self, demo_state):
        xml_str = generate_sample_cot_event(
            uid="hostile-sniper",
            callsign="Hostile Sniper",
            lat=37.7740,
            lon=-122.4200,
            alliance="hostile",
            asset_type="person",
        )
        result = demo_state.ingest_cot_xml(xml_str)
        assert result["status"] == "ingested"
        target = demo_state.targets["hostile-sniper"]
        assert target["source"] == "tak"

    def test_ingest_invalid_xml(self, demo_state):
        result = demo_state.ingest_cot_xml("not xml at all")
        assert "error" in result

    def test_ingest_non_event_xml(self, demo_state):
        result = demo_state.ingest_cot_xml("<root><child/></root>")
        assert "error" in result

    def test_ingest_updates_counter(self, demo_state):
        initial = demo_state.ingested_count
        xml_str = generate_sample_cot_event(uid="counter-test")
        demo_state.ingest_cot_xml(xml_str)
        assert demo_state.ingested_count == initial + 1

    def test_ingest_log_populated(self, demo_state):
        xml_str = generate_sample_cot_event(uid="log-test", callsign="LogTest")
        demo_state.ingest_cot_xml(xml_str)
        assert len(demo_state.ingest_log) == 1
        assert demo_state.ingest_log[0]["uid"] == "log-test"
        assert demo_state.ingest_log[0]["callsign"] == "LogTest"


# ---------------------------------------------------------------------------
# Test 5: Full roundtrip fidelity
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_roundtrip_demo_all_pass(self):
        result = roundtrip_demo()
        assert result["all_passed"], (
            f"Roundtrip failures: "
            f"{[r for r in result['results'] if not r['roundtrip_ok']]}"
        )

    def test_roundtrip_preserves_uid(self):
        xml_str = generate_sample_cot_event(uid="rt-test-001")
        parsed = xml_to_cot(xml_str)
        assert parsed is not None
        assert parsed.uid == "rt-test-001"

    def test_roundtrip_preserves_position(self):
        xml_str = generate_sample_cot_event(lat=40.7128, lon=-74.0060)
        parsed = xml_to_cot(xml_str)
        assert parsed is not None
        assert abs(parsed.point.lat - 40.7128) < 0.001
        assert abs(parsed.point.lon - (-74.0060)) < 0.001

    def test_roundtrip_preserves_callsign(self):
        xml_str = generate_sample_cot_event(callsign="Charlie-6")
        parsed = xml_to_cot(xml_str)
        assert parsed is not None
        assert parsed.detail.contact.callsign == "Charlie-6"


# ---------------------------------------------------------------------------
# Test 6: CotEvent model-level roundtrip
# ---------------------------------------------------------------------------

class TestCotEventRoundtrip:
    def test_model_to_xml_and_back(self):
        event = CotEvent(
            uid="model-test-001",
            type="a-f-G-U-C",
            how="h-g-i-g-o",
            point=CotPoint(lat=38.8977, lon=-77.0365, hae=15.0),
            detail=CotDetail(
                contact=CotContact(callsign="Pentagon-HQ"),
                group_name="Cyan",
                group_role="HQ",
                remarks="Test event from model",
            ),
        )
        xml_str = cot_to_xml(event)
        parsed = xml_to_cot(xml_str)
        assert parsed is not None
        assert parsed.uid == "model-test-001"
        assert parsed.type == "a-f-G-U-C"
        assert parsed.detail.contact.callsign == "Pentagon-HQ"
        assert abs(parsed.point.lat - 38.8977) < 0.001
        assert parsed.detail.remarks == "Test event from model"


# ---------------------------------------------------------------------------
# Test 7: Asset type inference from CoT type
# ---------------------------------------------------------------------------

class TestAssetTypeInference:
    @pytest.mark.parametrize("cot_type,expected", [
        ("a-f-G-U-C", "person"),
        ("a-h-G-E-V", "vehicle"),
        ("a-f-A-M-F-Q", "drone"),
        ("a-f-G-E-S", "sensor"),
        ("a-f-G-E-S-C", "camera"),
        ("a-u-G", "unknown"),
    ])
    def test_infer_asset_type(self, cot_type, expected):
        result = CotDemoState._infer_asset_type(cot_type)
        assert result == expected, f"CoT type {cot_type} -> {result}, expected {expected}"


# ---------------------------------------------------------------------------
# Test 8: REST endpoints via TestClient
# ---------------------------------------------------------------------------

class TestRestEndpoints:
    def test_get_targets_json(self, client):
        resp = client.get("/api/targets")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_single_target(self, client):
        resp = client.get("/api/targets/rover-alpha")
        assert resp.status_code == 200
        data = resp.json()
        assert data["target_id"] == "rover-alpha"

    def test_get_missing_target_404(self, client):
        resp = client.get("/api/targets/nonexistent")
        assert resp.status_code == 404

    def test_get_cot_events_xml(self, client):
        resp = client.get("/cot/events")
        assert resp.status_code == 200
        assert "event" in resp.text
        assert resp.headers["content-type"].startswith("application/xml")

    def test_get_cot_event_single(self, client):
        resp = client.get("/cot/events/rover-alpha")
        assert resp.status_code == 200
        assert "rover-alpha" in resp.text

    def test_get_cot_event_missing_404(self, client):
        resp = client.get("/cot/events/nonexistent")
        assert resp.status_code == 404

    def test_post_ingest_cot(self, client):
        xml_str = generate_sample_cot_event(
            uid="ingested-via-api",
            callsign="API Test",
        )
        resp = client.post(
            "/cot/ingest",
            content=xml_str,
            headers={"Content-Type": "application/xml"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ingested"

    def test_post_ingest_empty_body(self, client):
        resp = client.post(
            "/cot/ingest",
            content="",
            headers={"Content-Type": "application/xml"},
        )
        assert resp.status_code == 400

    def test_post_ingest_invalid_xml(self, client):
        resp = client.post(
            "/cot/ingest",
            content="not xml",
            headers={"Content-Type": "application/xml"},
        )
        assert resp.status_code == 400

    def test_get_stats(self, client):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_targets" in data
        assert "generated_xml_count" in data

    def test_get_ingest_log(self, client):
        resp = client.get("/api/ingest-log")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_dashboard_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "TRITIUM" in resp.text
        assert "CoT" in resp.text


# ---------------------------------------------------------------------------
# Test 9: Target position updates
# ---------------------------------------------------------------------------

class TestTargetUpdates:
    def test_update_moves_targets(self, demo_state):
        # Grab a moving target's initial position
        initial_lat = demo_state.targets["rover-alpha"]["lat"]
        result = demo_state.update_targets()
        assert result["tick"] == 1
        assert result["moved"] > 0
        new_lat = demo_state.targets["rover-alpha"]["lat"]
        # Position should have changed (rover has speed > 0)
        assert new_lat != initial_lat

    def test_stationary_targets_dont_move(self, demo_state):
        initial_lat = demo_state.targets["sensor-north-01"]["lat"]
        demo_state.update_targets()
        assert demo_state.targets["sensor-north-01"]["lat"] == initial_lat


# ---------------------------------------------------------------------------
# Test 10: Sample CoT generation utility
# ---------------------------------------------------------------------------

class TestSampleGeneration:
    def test_generate_sample_produces_valid_xml(self):
        xml_str = generate_sample_cot_event()
        root = ET.fromstring(xml_str)
        assert root.tag == "event"
        assert root.get("version") == "2.0"

    def test_generate_sample_custom_params(self):
        xml_str = generate_sample_cot_event(
            uid="custom-001",
            callsign="Custom Unit",
            lat=51.5074,
            lon=-0.1278,
            alliance="neutral",
            asset_type="vehicle",
        )
        parsed = xml_to_cot(xml_str)
        assert parsed is not None
        assert parsed.uid == "custom-001"
        assert abs(parsed.point.lat - 51.5074) < 0.001

    def test_generate_different_alliances(self):
        for alliance in ["friendly", "hostile", "neutral", "unknown"]:
            xml_str = generate_sample_cot_event(alliance=alliance)
            parsed = xml_to_cot(xml_str)
            assert parsed is not None
            assert parsed.alliance == alliance


# ---------------------------------------------------------------------------
# Test 11: Edge codec roundtrip via ingest
# ---------------------------------------------------------------------------

class TestEdgeCodecIngest:
    def test_edge_cot_can_be_ingested(self, demo_state):
        """Edge device CoT XML should be ingestible back as a target."""
        from tritium_lib.cot.codec import device_to_cot
        xml_str = device_to_cot(
            "test-edge-node",
            lat=37.7749,
            lng=-122.4194,
            capabilities=["camera"],
            callsign="TestCam",
        )
        result = demo_state.ingest_cot_xml(xml_str)
        assert result["status"] == "ingested"
        # The UID from edge codec is "tritium-edge-{device_id}"
        assert "tritium-edge-test-edge-node" in demo_state.targets

    def test_sensor_cot_can_be_ingested(self, demo_state):
        """Sensor reading CoT XML should be ingestible."""
        from tritium_lib.cot.codec import sensor_to_cot
        xml_str = sensor_to_cot(
            "temp-sensor-01", "temperature", 22.5,
            lat=37.7749, lng=-122.4194, unit="C",
        )
        result = demo_state.ingest_cot_xml(xml_str)
        assert result["status"] == "ingested"


# ---------------------------------------------------------------------------
# Test 12: Batch export formats
# ---------------------------------------------------------------------------

class TestBatchExport:
    def test_targets_to_cot_xml_batch(self, demo_state):
        from tritium_lib.models.tak_export import targets_to_cot_xml
        target_dicts = list(demo_state.targets.values())
        xml_str = targets_to_cot_xml(target_dicts)
        assert xml_str  # non-empty
        # Should contain multiple event elements
        assert xml_str.count("<event") == len(target_dicts)

    def test_targets_to_cot_file_has_wrapper(self, demo_state):
        from tritium_lib.models.tak_export import targets_to_cot_file
        target_dicts = list(demo_state.targets.values())
        xml_str = targets_to_cot_file(target_dicts)
        assert "<?xml" in xml_str
        assert "<cot-events" in xml_str
        assert f'count="{len(target_dicts)}"' in xml_str
