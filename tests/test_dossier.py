# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for Target Dossier models."""

import time
import uuid

import pytest
from pydantic import ValidationError

from tritium_lib.models.dossier import (
    DossierEnrichment,
    DossierSignal,
    PositionRecord,
    TargetDossier,
)


# --- DossierSignal ---

class TestDossierSignal:
    def test_defaults(self):
        sig = DossierSignal(source="ble", signal_type="mac_sighting")
        assert sig.source == "ble"
        assert sig.signal_type == "mac_sighting"
        assert sig.confidence == 0.5
        assert sig.position is None
        assert sig.data == {}
        uuid.UUID(sig.signal_id)  # valid UUID

    def test_with_position(self):
        sig = DossierSignal(
            source="yolo",
            signal_type="visual_detection",
            data={"class": "person", "bbox": [10, 20, 100, 200]},
            position=(37.7749, -122.4194),
            confidence=0.92,
        )
        assert sig.position == (37.7749, -122.4194)
        assert sig.confidence == 0.92
        assert sig.data["class"] == "person"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            DossierSignal(source="x", signal_type="y", confidence=1.5)
        with pytest.raises(ValidationError):
            DossierSignal(source="x", signal_type="y", confidence=-0.1)

    def test_roundtrip_json(self):
        sig = DossierSignal(source="wifi", signal_type="probe_request", confidence=0.7)
        data = sig.model_dump()
        restored = DossierSignal(**data)
        assert restored == sig


# --- DossierEnrichment ---

class TestDossierEnrichment:
    def test_defaults(self):
        e = DossierEnrichment(provider="oui_lookup", enrichment_type="manufacturer")
        assert e.provider == "oui_lookup"
        assert e.data == {}
        assert e.timestamp > 0

    def test_with_data(self):
        e = DossierEnrichment(
            provider="wigle",
            enrichment_type="location_history",
            data={"ssid": "CoffeeShop", "lat": 37.77, "lon": -122.42},
        )
        assert e.data["ssid"] == "CoffeeShop"


# --- PositionRecord ---

class TestPositionRecord:
    def test_basic(self):
        p = PositionRecord(x=10.0, y=20.0)
        assert p.x == 10.0
        assert p.y == 20.0
        assert p.source == "unknown"
        assert 0.0 <= p.confidence <= 1.0

    def test_with_source(self):
        p = PositionRecord(x=1.0, y=2.0, source="trilateration", confidence=0.85)
        assert p.source == "trilateration"
        assert p.confidence == 0.85

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            PositionRecord(x=0, y=0, confidence=2.0)


# --- TargetDossier ---

class TestTargetDossier:
    def test_empty_dossier(self):
        d = TargetDossier()
        uuid.UUID(d.dossier_id)
        assert d.name == "Unknown"
        assert d.entity_type == "unknown"
        assert d.confidence == 0.0
        assert d.threat_level == "none"
        assert d.alliance == "unknown"
        assert d.signals == []
        assert d.identifiers == {}
        assert d.enrichments == []
        assert d.position_history == []
        assert d.notes == []
        assert d.tags == []

    def test_full_dossier(self):
        now = time.time()
        sig = DossierSignal(
            source="ble",
            signal_type="mac_sighting",
            data={"mac": "AA:BB:CC:DD:EE:FF", "rssi": -65},
            timestamp=now,
            confidence=0.8,
        )
        enrichment = DossierEnrichment(
            provider="oui_lookup",
            enrichment_type="manufacturer",
            data={"manufacturer": "Apple Inc."},
        )
        pos = PositionRecord(x=5.0, y=12.0, source="ble_trilat", confidence=0.7)

        d = TargetDossier(
            name="Matt's iPhone",
            entity_type="device",
            confidence=0.85,
            first_seen=now - 3600,
            last_seen=now,
            signals=[sig],
            identifiers={"ble_mac": "AA:BB:CC:DD:EE:FF"},
            enrichments=[enrichment],
            position_history=[pos],
            alliance="friendly",
            threat_level="none",
            notes=["Seen daily near office"],
            tags=["mobile", "apple"],
        )
        assert d.name == "Matt's iPhone"
        assert d.entity_type == "device"
        assert d.confidence == 0.85
        assert len(d.signals) == 1
        assert d.identifiers["ble_mac"] == "AA:BB:CC:DD:EE:FF"
        assert d.enrichments[0].provider == "oui_lookup"
        assert len(d.position_history) == 1
        assert d.alliance == "friendly"
        assert "mobile" in d.tags

    def test_entity_types(self):
        for etype in ("person", "vehicle", "device", "animal", "unknown"):
            d = TargetDossier(entity_type=etype)
            assert d.entity_type == etype

    def test_invalid_entity_type(self):
        with pytest.raises(ValidationError):
            TargetDossier(entity_type="robot")

    def test_threat_levels(self):
        for level in ("none", "low", "medium", "high", "critical"):
            d = TargetDossier(threat_level=level)
            assert d.threat_level == level

    def test_invalid_threat_level(self):
        with pytest.raises(ValidationError):
            TargetDossier(threat_level="extreme")

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            TargetDossier(confidence=1.5)
        with pytest.raises(ValidationError):
            TargetDossier(confidence=-0.1)

    def test_accumulate_signals(self):
        d = TargetDossier(name="Tracked Vehicle", entity_type="vehicle")
        for i in range(5):
            d.signals.append(
                DossierSignal(
                    source="yolo",
                    signal_type="visual_detection",
                    data={"frame": i},
                    confidence=0.6 + i * 0.05,
                )
            )
        assert len(d.signals) == 5
        assert d.signals[4].confidence == pytest.approx(0.8)

    def test_roundtrip_json(self):
        d = TargetDossier(
            name="Test Target",
            entity_type="person",
            confidence=0.75,
            signals=[DossierSignal(source="manual", signal_type="observation")],
            identifiers={"face_id": "abc123"},
            tags=["suspect"],
            threat_level="medium",
        )
        data = d.model_dump()
        restored = TargetDossier(**data)
        assert restored.name == d.name
        assert restored.dossier_id == d.dossier_id
        assert len(restored.signals) == 1
        assert restored.threat_level == "medium"

    def test_multiple_identifiers(self):
        d = TargetDossier(
            identifiers={
                "ble_mac": "AA:BB:CC:DD:EE:FF",
                "wifi_bssid": "11:22:33:44:55:66",
                "license_plate": "7ABC123",
            }
        )
        assert len(d.identifiers) == 3

    def test_import_from_package(self):
        """Verify models are exported from the package __init__."""
        from tritium_lib.models import (
            DossierSignal,
            DossierEnrichment,
            PositionRecord,
            TargetDossier,
        )
        assert TargetDossier is not None
