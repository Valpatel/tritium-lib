# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.data_exchange — export/import in JSON, CSV, GeoJSON."""

from __future__ import annotations

import csv
import io
import json
import time

import pytest

from tritium_lib.data_exchange import (
    ImportResult,
    TritiumExporter,
    TritiumImporter,
    _safe_float,
)
from tritium_lib.store.dossiers import DossierStore
from tritium_lib.store.event_store import EventStore
from tritium_lib.store.targets import TargetStore


# ---------------------------------------------------------------------------
# Fixtures — in-memory stores with seed data
# ---------------------------------------------------------------------------


@pytest.fixture
def target_store():
    ts = TargetStore(":memory:")
    ts.record_sighting(
        target_id="ble_AA:BB:CC:DD:EE:01",
        name="Phone Alpha",
        alliance="unknown",
        asset_type="phone",
        source="ble",
        position_x=10.5,
        position_y=20.3,
        position_confidence=0.8,
        metadata={"manufacturer": "Apple"},
        timestamp=1000.0,
    )
    ts.record_sighting(
        target_id="ble_AA:BB:CC:DD:EE:01",
        position_x=11.0,
        position_y=21.0,
        source="ble",
        timestamp=1010.0,
    )
    ts.record_sighting(
        target_id="det_person_1",
        name="Unknown Person",
        alliance="hostile",
        asset_type="person",
        source="yolo",
        position_x=50.0,
        position_y=60.0,
        position_confidence=0.6,
        timestamp=1005.0,
    )
    ts.record_sighting(
        target_id="wifi_00:11:22:33:44:55",
        name="Router",
        alliance="friendly",
        asset_type="ap",
        source="wifi",
        position_x=0.0,
        position_y=0.0,
        timestamp=990.0,
    )
    return ts


@pytest.fixture
def dossier_store():
    ds = DossierStore(":memory:")
    d1 = ds.create_dossier(
        "Subject Alpha",
        entity_type="person",
        identifiers={"mac": "AA:BB:CC:DD:EE:01"},
        alliance="unknown",
        threat_level="low",
        tags=["bluetooth", "recurring"],
        timestamp=1000.0,
    )
    ds.add_signal(
        d1, "ble", "advertisement",
        data={"rssi": -45},
        position_x=10.5, position_y=20.3,
        confidence=0.8, timestamp=1000.0,
    )
    ds.add_enrichment(
        d1, "oui_lookup", "manufacturer",
        data={"manufacturer": "Apple Inc."},
        timestamp=1001.0,
    )
    d2 = ds.create_dossier(
        "Vehicle Bravo",
        entity_type="vehicle",
        identifiers={"plate": "ABC1234"},
        alliance="hostile",
        threat_level="high",
        timestamp=1005.0,
    )
    return ds


@pytest.fixture
def event_store():
    es = EventStore(":memory:")
    es.record(
        "target_detected",
        severity="info",
        source="ble_scanner",
        target_id="ble_AA:BB:CC:DD:EE:01",
        summary="BLE device detected",
        data={"rssi": -45},
        position_lat=40.7128,
        position_lng=-74.0060,
        timestamp=1000.0,
    )
    es.record(
        "alert_raised",
        severity="warning",
        source="geofence",
        target_id="det_person_1",
        summary="Hostile entered restricted zone",
        timestamp=1005.0,
    )
    es.record(
        "command_sent",
        severity="info",
        source="operator",
        operator="admin",
        summary="Dispatched rover to investigate",
        timestamp=1010.0,
    )
    return es


@pytest.fixture
def zones():
    return [
        {
            "zone_id": "zone_hq",
            "name": "HQ Perimeter",
            "zone_type": "restricted",
            "vertices": [
                (40.7127, -74.0060),
                (40.7130, -74.0060),
                (40.7130, -74.0055),
                (40.7127, -74.0055),
            ],
        },
        {
            "zone_id": "zone_park",
            "name": "Park Watch",
            "zone_type": "monitored",
            "center_lat": 40.7580,
            "center_lng": -73.9855,
            "radius_m": 200.0,
        },
    ]


# ---------------------------------------------------------------------------
# Tests: JSON export
# ---------------------------------------------------------------------------


class TestJsonExport:
    def test_full_export_structure(self, target_store, dossier_store, event_store, zones):
        exporter = TritiumExporter(
            target_store=target_store,
            dossier_store=dossier_store,
            event_store=event_store,
            zones=zones,
        )
        raw = exporter.export_json()
        doc = json.loads(raw)

        assert doc["_magic"] == "tritium-data-exchange"
        assert doc["_version"] == "1.0.0"
        assert doc["_scope"] == "full"
        assert doc["_since"] is None
        assert isinstance(doc["_exported_at"], float)
        assert isinstance(doc["_export_id"], str)

        assert doc["_counts"]["targets"] == 3
        assert doc["_counts"]["dossiers"] == 2
        assert doc["_counts"]["events"] == 3
        assert doc["_counts"]["zones"] == 2

    def test_incremental_export_since(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        raw = exporter.export_json(since=1005.0)
        doc = json.loads(raw)

        assert doc["_scope"] == "incremental"
        assert doc["_since"] == 1005.0
        # Should only include targets updated at or after 1005.0
        target_ids = {t["target_id"] for t in doc["targets"]}
        assert "ble_AA:BB:CC:DD:EE:01" in target_ids  # last_seen=1010
        assert "det_person_1" in target_ids             # last_seen=1005
        # wifi target was seen at 990, should be excluded
        assert "wifi_00:11:22:33:44:55" not in target_ids

    def test_filtered_export_by_source(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        raw = exporter.export_json(source="ble")
        doc = json.loads(raw)

        assert len(doc["targets"]) == 1
        assert doc["targets"][0]["target_id"] == "ble_AA:BB:CC:DD:EE:01"

    def test_filtered_export_by_alliance(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        raw = exporter.export_json(alliance="hostile")
        doc = json.loads(raw)

        assert len(doc["targets"]) == 1
        assert doc["targets"][0]["target_id"] == "det_person_1"

    def test_export_includes_history(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        raw = exporter.export_json(include_history=True)
        doc = json.loads(raw)

        history = doc["target_history"]
        assert "ble_AA:BB:CC:DD:EE:01" in history
        # BLE target had 2 sightings with positions
        assert len(history["ble_AA:BB:CC:DD:EE:01"]) == 2

    def test_export_without_history(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        raw = exporter.export_json(include_history=False)
        doc = json.loads(raw)

        assert doc["target_history"] == {}

    def test_export_dossiers_with_signals(self, dossier_store):
        exporter = TritiumExporter(dossier_store=dossier_store)
        raw = exporter.export_json(include_signals=True)
        doc = json.loads(raw)

        dossiers = doc["dossiers"]
        assert len(dossiers) == 2
        # First dossier (Subject Alpha) should have signals and enrichments
        alpha = next(d for d in dossiers if d["name"] == "Subject Alpha")
        assert len(alpha["signals"]) == 1
        assert len(alpha["enrichments"]) == 1

    def test_export_dossiers_without_signals(self, dossier_store):
        exporter = TritiumExporter(dossier_store=dossier_store)
        raw = exporter.export_json(include_signals=False)
        doc = json.loads(raw)

        # Summaries only — no 'signals' key
        for d in doc["dossiers"]:
            assert "signals" not in d

    def test_export_with_no_stores(self):
        exporter = TritiumExporter()
        raw = exporter.export_json()
        doc = json.loads(raw)

        assert doc["_counts"]["targets"] == 0
        assert doc["_counts"]["dossiers"] == 0
        assert doc["_counts"]["events"] == 0
        assert doc["_counts"]["zones"] == 0

    def test_compact_json(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        raw = exporter.export_json(indent=None)
        # Compact JSON should have no newlines (within reason)
        assert "\n" not in raw


# ---------------------------------------------------------------------------
# Tests: CSV export
# ---------------------------------------------------------------------------


class TestCsvExport:
    def test_targets_csv(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        csv_str = exporter.export_targets_csv()

        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 3
        ids = {r["target_id"] for r in rows}
        assert "ble_AA:BB:CC:DD:EE:01" in ids
        assert "det_person_1" in ids
        assert "wifi_00:11:22:33:44:55" in ids

        # Check metadata column is JSON
        ble_row = next(r for r in rows if r["target_id"] == "ble_AA:BB:CC:DD:EE:01")
        meta = json.loads(ble_row["metadata"])
        assert meta["manufacturer"] == "Apple"

    def test_dossiers_csv(self, dossier_store):
        exporter = TritiumExporter(dossier_store=dossier_store)
        csv_str = exporter.export_dossiers_csv()

        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 2
        names = {r["name"] for r in rows}
        assert "Subject Alpha" in names
        assert "Vehicle Bravo" in names

        # Tags column should be JSON list
        alpha = next(r for r in rows if r["name"] == "Subject Alpha")
        tags = json.loads(alpha["tags"])
        assert "bluetooth" in tags

    def test_events_csv(self, event_store):
        exporter = TritiumExporter(event_store=event_store)
        csv_str = exporter.export_events_csv()

        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 3
        types = {r["event_type"] for r in rows}
        assert "target_detected" in types
        assert "alert_raised" in types

    def test_csv_filtered_by_source(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        csv_str = exporter.export_targets_csv(source="yolo")

        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["target_id"] == "det_person_1"


# ---------------------------------------------------------------------------
# Tests: GeoJSON export
# ---------------------------------------------------------------------------


class TestGeoJsonExport:
    def test_geojson_structure(self, target_store, zones):
        exporter = TritiumExporter(target_store=target_store, zones=zones)
        raw = exporter.export_geojson()
        doc = json.loads(raw)

        assert doc["type"] == "FeatureCollection"
        assert isinstance(doc["features"], list)
        # 3 targets (all have positions) + 2 zones = 5 features
        assert len(doc["features"]) >= 5

    def test_geojson_target_points(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        raw = exporter.export_geojson()
        doc = json.loads(raw)

        target_features = [
            f for f in doc["features"]
            if f["properties"].get("feature_type") == "target_position"
        ]
        assert len(target_features) == 3
        for f in target_features:
            assert f["geometry"]["type"] == "Point"
            assert len(f["geometry"]["coordinates"]) == 2

    def test_geojson_trajectories(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        raw = exporter.export_geojson(include_trajectories=True)
        doc = json.loads(raw)

        traj_features = [
            f for f in doc["features"]
            if f["properties"].get("feature_type") == "trajectory"
        ]
        # BLE target has 2 history points -> 1 trajectory line
        assert len(traj_features) >= 1
        for f in traj_features:
            assert f["geometry"]["type"] == "LineString"
            assert len(f["geometry"]["coordinates"]) >= 2

    def test_geojson_zone_polygon(self, zones):
        exporter = TritiumExporter(zones=zones)
        raw = exporter.export_geojson()
        doc = json.loads(raw)

        zone_features = [
            f for f in doc["features"]
            if f["properties"].get("feature_type") == "zone"
        ]
        assert len(zone_features) == 2
        for f in zone_features:
            assert f["geometry"]["type"] == "Polygon"
            ring = f["geometry"]["coordinates"][0]
            # Ring should be closed
            assert ring[0] == ring[-1]

    def test_geojson_circle_zone_approximation(self, zones):
        exporter = TritiumExporter(zones=zones)
        raw = exporter.export_geojson()
        doc = json.loads(raw)

        park_feature = next(
            f for f in doc["features"]
            if f["properties"].get("zone_id") == "zone_park"
        )
        ring = park_feature["geometry"]["coordinates"][0]
        # 32-point circle + closing point = 33 vertices
        assert len(ring) == 33

    def test_geojson_events_with_positions(self, event_store):
        exporter = TritiumExporter(event_store=event_store)
        raw = exporter.export_geojson()
        doc = json.loads(raw)

        event_features = [
            f for f in doc["features"]
            if f["properties"].get("feature_type") == "event"
        ]
        # Only 1 event has position_lat/lng
        assert len(event_features) == 1
        assert event_features[0]["properties"]["event_type"] == "target_detected"


# ---------------------------------------------------------------------------
# Tests: JSON import
# ---------------------------------------------------------------------------


class TestJsonImport:
    def test_roundtrip_targets(self, target_store):
        """Export targets, import into a fresh store, verify equivalence."""
        exporter = TritiumExporter(target_store=target_store)
        json_str = exporter.export_json()

        new_store = TargetStore(":memory:")
        importer = TritiumImporter(target_store=new_store)
        result = importer.import_json(json_str)

        assert result.success
        assert result.targets_imported == 3
        assert result.total_imported >= 3

        # Verify data in new store
        all_targets = new_store.get_all_targets()
        assert len(all_targets) == 3
        ble = new_store.get_target("ble_AA:BB:CC:DD:EE:01")
        assert ble is not None
        assert ble["name"] == "Phone Alpha"
        assert ble["source"] == "ble"

    def test_roundtrip_dossiers(self, dossier_store):
        """Export dossiers with signals, import into fresh store."""
        exporter = TritiumExporter(dossier_store=dossier_store)
        json_str = exporter.export_json(include_signals=True)

        new_store = DossierStore(":memory:")
        importer = TritiumImporter(dossier_store=new_store)
        result = importer.import_json(json_str)

        assert result.success
        assert result.dossiers_imported == 2

        all_dossiers = new_store.get_recent(limit=100)
        assert len(all_dossiers) == 2

    def test_roundtrip_events(self, event_store):
        """Export events, import into fresh store."""
        exporter = TritiumExporter(event_store=event_store)
        json_str = exporter.export_json()

        new_store = EventStore(":memory:")
        importer = TritiumImporter(event_store=new_store)
        result = importer.import_json(json_str)

        assert result.success
        assert result.events_imported == 3

    def test_import_invalid_json(self):
        importer = TritiumImporter()
        result = importer.import_json("this is not json{{{")

        assert not result.success
        assert len(result.errors) == 1
        assert "Invalid JSON" in result.errors[0]

    def test_import_missing_magic(self):
        importer = TritiumImporter()
        result = importer.import_json('{"targets": [], "dossiers": [], "events": []}')

        # Should still import but warn
        assert result.success
        assert any("_magic" in w for w in result.warnings)

    def test_import_skips_invalid_targets(self, target_store):
        doc = {
            "_magic": "tritium-data-exchange",
            "targets": [
                {"name": "no id"},  # missing target_id
                {"target_id": "good_one", "name": "Valid"},
            ],
            "target_history": {},
            "dossiers": [],
            "events": [],
            "zones": [],
        }
        importer = TritiumImporter(target_store=TargetStore(":memory:"))
        result = importer.import_json(json.dumps(doc))

        assert result.targets_imported == 1
        assert result.targets_skipped == 1

    def test_import_without_store_skips(self):
        """If no store is provided, records are counted as skipped."""
        doc = {
            "_magic": "tritium-data-exchange",
            "targets": [{"target_id": "t1"}],
            "target_history": {},
            "dossiers": [{"dossier_id": "d1", "name": "test"}],
            "events": [{"event_type": "test"}],
            "zones": [],
        }
        importer = TritiumImporter()  # no stores
        result = importer.import_json(json.dumps(doc))

        assert result.targets_skipped == 1
        assert result.dossiers_skipped == 1
        assert result.events_skipped == 1


# ---------------------------------------------------------------------------
# Tests: CSV import
# ---------------------------------------------------------------------------


class TestCsvImport:
    def test_csv_roundtrip_targets(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        csv_str = exporter.export_targets_csv()

        new_store = TargetStore(":memory:")
        importer = TritiumImporter(target_store=new_store)
        result = importer.import_csv(csv_str, section="targets")

        assert result.success
        assert result.targets_imported == 3

        all_targets = new_store.get_all_targets()
        assert len(all_targets) == 3

    def test_csv_roundtrip_events(self, event_store):
        exporter = TritiumExporter(event_store=event_store)
        csv_str = exporter.export_events_csv()

        new_store = EventStore(":memory:")
        importer = TritiumImporter(event_store=new_store)
        result = importer.import_csv(csv_str, section="events")

        assert result.success
        assert result.events_imported == 3

    def test_csv_import_unknown_section(self):
        importer = TritiumImporter()
        result = importer.import_csv("a,b\n1,2", section="nonsense")

        assert not result.success
        assert "Unknown section" in result.errors[0]


# ---------------------------------------------------------------------------
# Tests: ImportResult
# ---------------------------------------------------------------------------


class TestImportResult:
    def test_total_imported(self):
        r = ImportResult(
            targets_imported=3,
            dossiers_imported=2,
            events_imported=5,
            zones_imported=1,
        )
        assert r.total_imported == 11

    def test_total_skipped(self):
        r = ImportResult(
            targets_skipped=1,
            dossiers_skipped=2,
            events_skipped=0,
            zones_skipped=3,
        )
        assert r.total_skipped == 6


# ---------------------------------------------------------------------------
# Tests: export stats
# ---------------------------------------------------------------------------


class TestExportStats:
    def test_get_export_stats(self, target_store, dossier_store, event_store, zones):
        exporter = TritiumExporter(
            target_store=target_store,
            dossier_store=dossier_store,
            event_store=event_store,
            zones=zones,
        )
        stats = exporter.get_export_stats()
        assert stats["targets"] == 3
        assert stats["dossiers"] == 2
        assert stats["events"] == 3
        assert stats["zones"] == 2

    def test_stats_with_since(self, target_store):
        exporter = TritiumExporter(target_store=target_store)
        stats = exporter.get_export_stats(since=1005.0)
        assert stats["targets"] == 2  # BLE (1010) + person (1005)


# ---------------------------------------------------------------------------
# Tests: helper utilities
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_empty_string(self):
        assert _safe_float("") is None

    def test_safe_float_valid(self):
        assert _safe_float("3.14") == 3.14
        assert _safe_float(42) == 42.0

    def test_safe_float_invalid(self):
        assert _safe_float("not_a_number") is None
        assert _safe_float([1, 2]) is None
