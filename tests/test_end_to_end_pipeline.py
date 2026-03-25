# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""End-to-end pipeline integration test.

Exercises the ENTIRE Tritium pipeline from raw sensor data through to
exported situation reports and recorded sessions:

    Synthetic sensor data (BLE + WiFi + Camera)
      -> FusionEngine (target tracking + correlation)
        -> AnomalyEngine (behavioral anomaly detection)
          -> AlertEngine (rules-based alert generation)
            -> RulesEngine (IF-THEN automation)
              -> SitRepGenerator (situation report)
                -> TritiumExporter (data exchange)
                  -> Recorder (session recording)

Every stage is verified to produce output, and data flows correctly
between stages. No mocks — all REAL implementations.
"""

import json
import os
import tempfile
import time

import pytest

from tritium_lib.events.bus import EventBus
from tritium_lib.fusion import FusionEngine
from tritium_lib.intelligence.anomaly_engine import AnomalyEngine
from tritium_lib.alerting import (
    AlertEngine,
    AlertRecord,
    AlertRule,
    AlertTrigger,
    ConditionOperator,
    NotificationChannel,
    NotificationSeverity,
)
from tritium_lib.rules import (
    Action,
    ActionType,
    Condition,
    Rule,
    RuleEngine,
    RuleResult,
)
from tritium_lib.reporting import SitRepGenerator, SitRep
from tritium_lib.store.targets import TargetStore
from tritium_lib.store.dossiers import DossierStore as PersistentDossierStore
from tritium_lib.store.event_store import EventStore
from tritium_lib.data_exchange import TritiumExporter, TritiumImporter
from tritium_lib.recording import Recorder


# ---------------------------------------------------------------------------
# Synthetic sensor data generators
# ---------------------------------------------------------------------------

def _ble_sightings() -> list[dict]:
    """Generate synthetic BLE sightings representing phones/devices."""
    return [
        {
            "mac": "AA:BB:CC:DD:EE:01",
            "rssi": -45,
            "name": "iPhone-Matt",
            "device_type": "phone",
            "position": {"x": 10.0, "y": 20.0},
        },
        {
            "mac": "AA:BB:CC:DD:EE:02",
            "rssi": -60,
            "name": "Galaxy-Watch",
            "device_type": "wearable",
            "position": {"x": 12.0, "y": 22.0},
        },
        {
            "mac": "AA:BB:CC:DD:EE:03",
            "rssi": -70,
            "name": "Unknown-Device",
            "device_type": "unknown",
            "position": {"x": 50.0, "y": 50.0},
        },
        {
            "mac": "AA:BB:CC:DD:EE:04",
            "rssi": -55,
            "name": "AirTag-KeyChain",
            "device_type": "tracker",
            "position": {"x": 11.0, "y": 21.0},
        },
    ]


def _wifi_probes() -> list[dict]:
    """Generate synthetic WiFi probe requests."""
    return [
        {
            "mac": "AA:BB:CC:DD:EE:01",
            "ssid": "HomeWiFi",
            "rssi": -50,
            "position": {"x": 10.5, "y": 20.5},
        },
        {
            "mac": "AA:BB:CC:DD:EE:02",
            "ssid": "CoffeeShop_5G",
            "rssi": -65,
            "position": {"x": 12.5, "y": 22.5},
        },
        {
            "mac": "FF:EE:DD:CC:BB:AA",
            "ssid": "CorpNet",
            "rssi": -40,
        },
    ]


def _camera_detections() -> list[dict]:
    """Generate synthetic camera/YOLO detections."""
    return [
        {
            "class_name": "person",
            "confidence": 0.92,
            "center_x": 10.0,
            "center_y": 20.0,
        },
        {
            "class_name": "person",
            "confidence": 0.88,
            "center_x": 50.0,
            "center_y": 50.0,
        },
        {
            "class_name": "car",
            "confidence": 0.95,
            "center_x": 100.0,
            "center_y": 200.0,
        },
        {
            "class_name": "person",
            "confidence": 0.75,
            "center_x": 12.0,
            "center_y": 22.0,
        },
        {
            "class_name": "dog",
            "confidence": 0.60,
            "center_x": 30.0,
            "center_y": 40.0,
        },
    ]


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Full pipeline integration: sensor data -> sitrep -> export -> recording."""

    def setup_method(self):
        """Set up shared event bus and all pipeline components."""
        self.event_bus = EventBus()
        self.tmpdir = tempfile.mkdtemp(prefix="tritium_e2e_")

        # Collected events for verification
        self.events_received: list[dict] = []

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Stage 1: Ingest synthetic data through FusionEngine
    # ------------------------------------------------------------------

    def _stage_fusion(self) -> FusionEngine:
        """Ingest BLE + WiFi + camera data through FusionEngine."""
        engine = FusionEngine(event_bus=self.event_bus)

        # Ingest BLE sightings
        ble_ids = []
        for sighting in _ble_sightings():
            tid = engine.ingest_ble(sighting)
            assert tid is not None, f"BLE ingestion failed for {sighting['mac']}"
            ble_ids.append(tid)

        # Ingest WiFi probes
        wifi_ids = []
        for probe in _wifi_probes():
            tid = engine.ingest_wifi(probe)
            assert tid is not None, f"WiFi ingestion failed for {probe['mac']}"
            wifi_ids.append(tid)

        # Ingest camera detections
        cam_ids = []
        for detection in _camera_detections():
            tid = engine.ingest_camera(detection)
            if tid is not None:
                cam_ids.append(tid)

        # Verify: targets were created
        all_targets = engine.tracker.get_all()
        assert len(all_targets) > 0, "No targets created after ingestion"
        assert len(ble_ids) == 4, f"Expected 4 BLE IDs, got {len(ble_ids)}"
        assert len(cam_ids) >= 3, f"Expected at least 3 camera IDs, got {len(cam_ids)}"

        # Verify: sensor records were stored
        fused = engine.get_fused_targets()
        assert len(fused) > 0, "No fused targets returned"

        # Verify: heatmap has data
        assert engine.heatmap.event_count() > 0, "Heatmap has no events"

        # Run correlation to link BLE + WiFi for same MAC
        correlations = engine.run_correlation()
        # (May or may not produce correlations depending on distance thresholds)

        return engine

    # ------------------------------------------------------------------
    # Stage 2: Run through AnomalyEngine
    # ------------------------------------------------------------------

    def _stage_anomaly(self, fusion_engine: FusionEngine) -> tuple[AnomalyEngine, list]:
        """Feed fusion results through anomaly detection."""
        anomaly = AnomalyEngine(
            event_bus=self.event_bus,
            min_baseline_samples=3,
            speed_threshold_sigma=2.0,
            dwell_threshold_sigma=2.0,
            cooldown_seconds=0,
        )

        # Build baselines with "normal" observations for a test zone
        zone_id = "test_zone_alpha"
        for i in range(15):
            anomaly.observe(
                zone_id,
                target_id=f"baseline_target_{i}",
                speed=1.0 + (i % 3) * 0.1,
                dwell_seconds=60.0 + (i % 5) * 10,
                entity_count=float(3 + i % 2),
            )

        # Verify baseline was established
        baseline = anomaly.get_baseline(zone_id)
        assert baseline is not None, "Baseline was not created"
        assert baseline.observation_count >= 15, "Not enough baseline observations"

        # Now check fusion targets against anomaly engine with abnormal values
        fused_targets = fusion_engine.get_fused_targets()
        all_anomaly_alerts = []

        # Inject some targets with anomalous speed into the zone
        for ft in fused_targets[:2]:
            alerts = anomaly.check_target(
                zone_id,
                target_id=ft.target_id,
                speed=50.0,  # Way above baseline ~1.0 m/s
                dwell_seconds=5.0,
            )
            all_anomaly_alerts.extend(alerts)

        # Also check zone count anomaly
        count_alert = anomaly.check_zone_count(
            zone_id,
            current_count=50.0,  # Way above baseline ~3-4
        )
        if count_alert:
            all_anomaly_alerts.append(count_alert)

        # Also test the ingest_from_fusion integration path
        zone_assignments = {zone_id: {ft.target_id for ft in fused_targets[:3]}}
        fusion_alerts = anomaly.ingest_from_fusion(
            fused_targets[:3],
            zone_assignments=zone_assignments,
        )
        all_anomaly_alerts.extend(fusion_alerts)

        # Verify: anomalies were detected
        assert len(all_anomaly_alerts) > 0, "No anomaly alerts generated"

        # Verify: at least one speed anomaly was detected
        speed_alerts = [a for a in all_anomaly_alerts if a.alert_type == "speed"]
        assert len(speed_alerts) > 0, "No speed anomalies detected"

        # Verify: alert history is populated
        history = anomaly.get_alert_history(limit=100, include_suppressed=True)
        assert len(history) > 0, "Anomaly alert history is empty"

        # Verify: stats are populated
        stats = anomaly.get_stats()
        assert stats["total_observations"] > 0
        assert stats["total_alerts"] > 0

        return anomaly, all_anomaly_alerts

    # ------------------------------------------------------------------
    # Stage 3: Generate alerts via AlertEngine
    # ------------------------------------------------------------------

    def _stage_alert(self, anomaly_alerts: list) -> tuple[AlertEngine, list[AlertRecord]]:
        """Evaluate anomaly alerts through the AlertEngine."""
        alert_engine = AlertEngine(
            event_bus=self.event_bus,
            load_defaults=True,
        )

        all_alert_records = []

        # Feed anomaly alerts as events into the alert engine
        for anomaly_alert in anomaly_alerts:
            if anomaly_alert.suppressed:
                continue
            event_data = anomaly_alert.to_dict()
            records = alert_engine.evaluate_event(
                f"anomaly.alert.{anomaly_alert.alert_type}",
                event_data,
            )
            all_alert_records.extend(records)

        # Also evaluate a direct geofence event
        geofence_records = alert_engine.evaluate_event(
            "geofence:enter",
            {
                "target_id": "ble_aabbccddeeff",
                "zone_id": "perimeter_alpha",
                "zone_name": "Perimeter Alpha",
                "zone_type": "restricted",
            },
        )
        all_alert_records.extend(geofence_records)

        # Verify: alerts were generated
        assert len(all_alert_records) > 0, "No alert records generated"

        # Verify: each record has required fields
        for record in all_alert_records:
            assert record.record_id, "Alert record missing ID"
            assert record.rule_id, "Alert record missing rule_id"
            assert record.message, "Alert record missing message"
            assert record.severity, "Alert record missing severity"
            assert record.timestamp > 0, "Alert record has invalid timestamp"

        # Verify: alert history is populated
        history = alert_engine.get_history(limit=100)
        assert len(history) > 0, "Alert engine history is empty"

        # Verify: stats reflect activity
        stats = alert_engine.get_stats()
        assert stats["total_events_processed"] > 0
        assert stats["total_alerts_fired"] > 0

        return alert_engine, all_alert_records

    # ------------------------------------------------------------------
    # Stage 4: Run RulesEngine
    # ------------------------------------------------------------------

    def _stage_rules(self, fusion_engine: FusionEngine) -> tuple[RuleEngine, list[RuleResult]]:
        """Evaluate IF-THEN rules against the current tracking state."""
        rules_engine = RuleEngine()

        # Track executed actions for verification
        actions_executed = []

        def on_alert(action: Action, state: dict):
            actions_executed.append(("alert", action.params, time.time()))

        def on_report(action: Action, state: dict):
            actions_executed.append(("report", action.params, time.time()))

        def on_log(action: Action, state: dict):
            actions_executed.append(("log", action.params, time.time()))

        rules_engine.register_action_handler(ActionType.SEND_ALERT, on_alert)
        rules_engine.register_action_handler(ActionType.GENERATE_REPORT, on_report)
        rules_engine.register_action_handler(ActionType.LOG, on_log)

        # Add rules that match our test state
        rule_hostile = (
            Rule("detect_hostile", name="Hostile Detection Rule")
            .when(Condition("target_alliance_is", alliance="hostile"))
            .then(Action(ActionType.SEND_ALERT, message="Hostile detected!"))
            .then(Action(ActionType.GENERATE_REPORT))
            .with_priority(10)
        )

        rule_count = (
            Rule("zone_crowded", name="Zone Overcrowded Rule")
            .when(Condition("field_compare",
                            field_path="zones.test_zone.target_count",
                            operator="gt",
                            value=2))
            .then(Action(ActionType.LOG, message="Zone overcrowded"))
            .with_priority(5)
        )

        rule_always = (
            Rule("always_log", name="Always Log Rule")
            .then(Action(ActionType.LOG, message="Evaluation cycle completed"))
            .with_priority(0)
        )

        rules_engine.add_rule(rule_hostile)
        rules_engine.add_rule(rule_count)
        rules_engine.add_rule(rule_always)

        # Build state from fusion engine targets
        fused = fusion_engine.get_fused_targets()
        targets_state = {}
        for ft in fused:
            targets_state[ft.target_id] = {
                "target_id": ft.target_id,
                "alliance": ft.target.alliance or "unknown",
                "speed": ft.target.speed,
                "zone_id": "test_zone",
                "threat_level": ft.target.threat_score,
                "asset_type": ft.target.asset_type,
            }

        # Mark one target as hostile to trigger that rule
        if targets_state:
            first_key = next(iter(targets_state))
            targets_state[first_key]["alliance"] = "hostile"

        state = {
            "targets": targets_state,
            "zones": {
                "test_zone": {
                    "target_count": len(targets_state),
                },
            },
            "sensors": {},
        }

        results = rules_engine.evaluate(state)

        # Verify: at least the always-fire rule should have triggered
        assert len(results) > 0, "No rules fired during evaluation"

        # Verify: the hostile detection rule should have fired
        hostile_results = [r for r in results if r.rule_id == "detect_hostile"]
        assert len(hostile_results) > 0, "Hostile detection rule did not fire"

        # Verify: the always-log rule should have fired
        always_results = [r for r in results if r.rule_id == "always_log"]
        assert len(always_results) > 0, "Always-log rule did not fire"

        # Verify: actions were actually executed
        assert len(actions_executed) > 0, "No actions were executed"

        # Verify: action handlers received correct types
        alert_actions = [a for a in actions_executed if a[0] == "alert"]
        assert len(alert_actions) > 0, "Alert action handler was not called"

        # Verify: stats are populated
        stats = rules_engine.get_stats()
        assert stats["total_evaluations"] > 0
        assert stats["total_rules_fired"] > 0
        assert stats["total_actions_executed"] > 0

        # Verify: history is populated
        history = rules_engine.get_history()
        assert len(history) > 0, "Rules engine history is empty"

        return rules_engine, results

    # ------------------------------------------------------------------
    # Stage 5: Generate SitRep via ReportingEngine
    # ------------------------------------------------------------------

    def _stage_sitrep(
        self,
        fusion_engine: FusionEngine,
        anomaly_alerts: list,
        alert_records: list[AlertRecord],
    ) -> SitRep:
        """Generate a situation report from the current state."""
        # Create a temporary event store and populate it with our pipeline data
        event_store = EventStore(":memory:")

        # Record anomaly events
        for alert in anomaly_alerts:
            if hasattr(alert, "to_dict"):
                event_store.record(
                    event_type=f"anomaly_{alert.alert_type}",
                    severity="warning" if alert.severity in ("low", "medium") else "critical",
                    source="anomaly_engine",
                    target_id=alert.target_id,
                    summary=alert.detail,
                    data=alert.to_dict(),
                )

        # Record alert engine events
        for record in alert_records:
            event_store.record(
                event_type="alert_fired",
                severity=record.severity,
                source="alert_engine",
                target_id=record.target_id,
                summary=record.message,
                data=record.to_dict(),
            )

        # Record some additional tactical events
        event_store.record(
            event_type="geofence_enter",
            severity="info",
            source="geofence_engine",
            target_id="ble_aabbccddee01",
            summary="Target entered perimeter zone",
            data={"zone_id": "perimeter", "zone_name": "Perimeter Alpha"},
        )
        event_store.record(
            event_type="geofence_exit",
            severity="info",
            source="geofence_engine",
            target_id="ble_aabbccddee01",
            summary="Target exited perimeter zone",
            data={"zone_id": "perimeter", "zone_name": "Perimeter Alpha"},
        )

        # Generate the situation report
        generator = SitRepGenerator(
            tracker=fusion_engine.tracker,
            event_store=event_store,
            title="E2E Pipeline Test SitRep",
        )

        # Use a wide time range to capture all events
        now = time.time()
        report = generator.generate(
            event_time_range=(now - 3600, now + 3600),
            notes="Generated by end-to-end pipeline integration test",
        )

        # Verify: report was generated
        assert report is not None, "SitRep was not generated"
        assert isinstance(report, SitRep)

        # Verify: targets section is populated
        assert report.targets.total > 0, "SitRep has no targets"

        # Verify: threats section is populated (total_assessed > 0)
        assert report.threats.total_assessed > 0, "SitRep has no threat assessments"

        # Verify: anomalies section reflects our injected anomalies
        assert report.anomalies.total_anomalies > 0, "SitRep has no anomalies"

        # Verify: timeline has events
        assert report.timeline.total_events > 0, "SitRep timeline is empty"

        # Verify: output formats work
        text = report.to_text()
        assert len(text) > 100, "SitRep text output is too short"
        assert "TARGET SUMMARY" in text
        assert "THREAT ASSESSMENT" in text

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert "targets" in parsed
        assert "threats" in parsed
        assert "anomalies" in parsed

        html_str = report.to_html()
        assert "<html>" in html_str
        assert "Target Summary" in html_str

        # Verify: notes are included
        assert "end-to-end" in text.lower()

        return report

    # ------------------------------------------------------------------
    # Stage 6: Export via DataExchange
    # ------------------------------------------------------------------

    def _stage_export(
        self,
        fusion_engine: FusionEngine,
        anomaly_alerts: list,
        alert_records: list[AlertRecord],
    ) -> str:
        """Export all data via TritiumExporter and verify the output."""
        # Persist targets to a SQLite-backed TargetStore
        target_store = TargetStore(":memory:")
        for ft in fusion_engine.get_fused_targets():
            t = ft.target
            target_store.record_sighting(
                target_id=t.target_id,
                name=t.name,
                alliance=t.alliance or "unknown",
                asset_type=t.asset_type or "unknown",
                source=t.source or "unknown",
                position_x=t.position[0] if t.position else None,
                position_y=t.position[1] if t.position else None,
                position_confidence=t.effective_confidence,
            )

        # Persist events
        event_store = EventStore(":memory:")
        for alert in anomaly_alerts:
            if hasattr(alert, "to_dict"):
                event_store.record(
                    event_type=f"anomaly_{alert.alert_type}",
                    severity="warning",
                    source="anomaly_engine",
                    target_id=alert.target_id,
                    summary=alert.detail,
                    data=alert.to_dict(),
                )
        for record in alert_records:
            event_store.record(
                event_type="alert_fired",
                severity=record.severity,
                source="alert_engine",
                target_id=record.target_id,
                summary=record.message,
                data=record.to_dict(),
            )

        # Persist dossiers
        dossier_store = PersistentDossierStore(":memory:")
        dossier_id = dossier_store.create_dossier(
            name="Test Subject Alpha",
            entity_type="person",
            identifiers={"mac": "AA:BB:CC:DD:EE:01", "ble_id": "ble_aabbccddee01"},
            alliance="unknown",
            tags=["test", "e2e"],
        )

        # Add a signal to the dossier
        dossier_store.add_signal(
            dossier_id=dossier_id,
            source="ble",
            signal_type="ble_sighting",
            data={"mac": "AA:BB:CC:DD:EE:01", "rssi": -45},
            confidence=0.9,
        )

        exporter = TritiumExporter(
            target_store=target_store,
            dossier_store=dossier_store,
            event_store=event_store,
        )

        # JSON export
        json_export = exporter.export_json()
        assert json_export, "JSON export is empty"
        parsed = json.loads(json_export)
        assert parsed.get("_magic") == "tritium-data-exchange"
        assert len(parsed.get("targets", [])) > 0, "JSON export has no targets"
        assert len(parsed.get("events", [])) > 0, "JSON export has no events"
        assert len(parsed.get("dossiers", [])) > 0, "JSON export has no dossiers"

        # CSV export
        csv_targets = exporter.export_targets_csv()
        assert csv_targets, "CSV targets export is empty"
        assert "target_id" in csv_targets, "CSV targets missing header"
        lines = csv_targets.strip().split("\n")
        assert len(lines) > 1, "CSV targets has no data rows"

        csv_events = exporter.export_events_csv()
        assert csv_events, "CSV events export is empty"
        assert "event_id" in csv_events

        # GeoJSON export
        geojson_str = exporter.export_geojson()
        assert geojson_str, "GeoJSON export is empty"
        geojson = json.loads(geojson_str)
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) > 0, "GeoJSON has no features"

        # Export stats
        stats = exporter.get_export_stats()
        assert stats["targets"] > 0
        assert stats["events"] > 0

        # Import round-trip: verify the exported JSON can be imported
        target_store_2 = TargetStore(":memory:")
        dossier_store_2 = PersistentDossierStore(":memory:")
        event_store_2 = EventStore(":memory:")
        importer = TritiumImporter(
            target_store=target_store_2,
            dossier_store=dossier_store_2,
            event_store=event_store_2,
        )
        result = importer.import_json(json_export)
        assert result.success, f"Import failed: {result.errors}"
        assert result.targets_imported > 0, "No targets imported in round-trip"
        assert result.events_imported > 0, "No events imported in round-trip"

        return json_export

    # ------------------------------------------------------------------
    # Stage 7: Record via RecordingModule
    # ------------------------------------------------------------------

    def _stage_recording(
        self,
        fusion_engine: FusionEngine,
        anomaly_alerts: list,
        alert_records: list[AlertRecord],
        rule_results: list[RuleResult],
        json_export: str,
    ) -> dict:
        """Record the pipeline session to a JSONL file."""
        recording_path = os.path.join(self.tmpdir, "pipeline_session.jsonl")
        recorder = Recorder(
            recording_path,
            metadata={"test": "end_to_end_pipeline", "version": "1.0"},
        )

        recorder.start()

        # Record BLE sightings
        for sighting in _ble_sightings():
            recorder.record(
                "ble_sighting",
                source="e2e_test",
                data=sighting,
            )

        # Record WiFi probes
        for probe in _wifi_probes():
            recorder.record(
                "wifi_probe",
                source="e2e_test",
                data=probe,
            )

        # Record camera detections
        for detection in _camera_detections():
            recorder.record(
                "camera_detection",
                source="e2e_test",
                data=detection,
            )

        # Record fusion results
        for ft in fusion_engine.get_fused_targets():
            recorder.record(
                "fusion_result",
                source="fusion_engine",
                data=ft.to_dict(),
            )

        # Record anomaly alerts
        for alert in anomaly_alerts:
            recorder.record(
                "alert",
                source="anomaly_engine",
                data=alert.to_dict(),
            )

        # Record alert engine records
        for record in alert_records:
            recorder.record(
                "alert",
                source="alert_engine",
                data=record.to_dict(),
            )

        # Record rule results
        for result in rule_results:
            recorder.record(
                "alert",
                source="rules_engine",
                data=result.to_dict(),
            )

        # Record the export as a pipeline artifact
        recorder.record(
            "fusion_result",
            source="data_exchange",
            data={"export_size": len(json_export), "format": "json"},
        )

        summary = recorder.stop()

        # Verify: recording was created
        assert os.path.exists(recording_path), "Recording file was not created"

        # Verify: session summary has correct counts
        assert summary["event_count"] > 0, "Recording has no events"
        assert summary["session_id"], "Recording has no session ID"
        assert summary["duration"] >= 0, "Recording has negative duration"
        assert len(summary["sensor_types"]) > 0, "Recording has no sensor types"

        # Verify: JSONL file is parseable
        with open(recording_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 3, "Recording file too short (header + events + footer)"

        # Parse and verify header
        header = json.loads(lines[0])
        assert header["event_type"] == "_session_start"

        # Parse and verify footer
        footer = json.loads(lines[-1])
        assert footer["event_type"] == "_session_end"
        assert footer["data"]["event_count"] == summary["event_count"]

        # Parse and verify at least one data line
        data_line = json.loads(lines[1])
        assert "event_type" in data_line
        assert "ts" in data_line
        assert "data" in data_line

        # Verify: all sensor types were recorded
        recorded_types = set()
        for line in lines[1:-1]:
            evt = json.loads(line)
            recorded_types.add(evt["event_type"])
        assert "ble_sighting" in recorded_types
        assert "wifi_probe" in recorded_types
        assert "camera_detection" in recorded_types
        assert "fusion_result" in recorded_types
        assert "alert" in recorded_types

        return summary

    # ------------------------------------------------------------------
    # THE FULL PIPELINE TEST
    # ------------------------------------------------------------------

    def test_full_pipeline_sensor_to_sitrep(self):
        """Exercise the ENTIRE pipeline from sensor data to situation report.

        This is a single test that chains all 7 stages together, verifying
        that data flows correctly between each stage and every stage
        produces meaningful output.
        """
        # Stage 1: Ingest sensor data
        fusion_engine = self._stage_fusion()
        fused_targets = fusion_engine.get_fused_targets()
        assert len(fused_targets) >= 5, (
            f"Expected at least 5 fused targets, got {len(fused_targets)}"
        )

        # Stage 2: Anomaly detection
        anomaly_engine, anomaly_alerts = self._stage_anomaly(fusion_engine)
        assert len(anomaly_alerts) >= 1, (
            f"Expected at least 1 anomaly alert, got {len(anomaly_alerts)}"
        )

        # Stage 3: Alert generation
        alert_engine, alert_records = self._stage_alert(anomaly_alerts)
        assert len(alert_records) >= 1, (
            f"Expected at least 1 alert record, got {len(alert_records)}"
        )

        # Stage 4: Rules evaluation
        rules_engine, rule_results = self._stage_rules(fusion_engine)
        assert len(rule_results) >= 2, (
            f"Expected at least 2 rule results, got {len(rule_results)}"
        )

        # Stage 5: Situation report
        sitrep = self._stage_sitrep(
            fusion_engine, anomaly_alerts, alert_records
        )
        assert sitrep.targets.total > 0, "SitRep targets empty"
        assert sitrep.anomalies.total_anomalies > 0, "SitRep anomalies empty"

        # Stage 6: Data export
        json_export = self._stage_export(
            fusion_engine, anomaly_alerts, alert_records
        )
        assert len(json_export) > 500, "JSON export suspiciously small"

        # Stage 7: Session recording
        recording_summary = self._stage_recording(
            fusion_engine, anomaly_alerts, alert_records,
            rule_results, json_export,
        )
        assert recording_summary["event_count"] > 10, (
            f"Expected >10 recorded events, got {recording_summary['event_count']}"
        )

        # ------------------------------------------------------------------
        # Cross-stage integrity checks
        # ------------------------------------------------------------------

        # Verify: fusion target count matches what went into the sitrep
        assert sitrep.targets.total == len(fused_targets), (
            f"SitRep target count ({sitrep.targets.total}) does not match "
            f"fusion target count ({len(fused_targets)})"
        )

        # Verify: the JSON export contains at least as many targets as fusion produced
        export_parsed = json.loads(json_export)
        assert len(export_parsed["targets"]) == len(fused_targets), (
            f"Export target count ({len(export_parsed['targets'])}) does not match "
            f"fusion target count ({len(fused_targets)})"
        )

        # Verify: recording captured events from all pipeline stages
        assert "ble_sighting" in recording_summary["sensor_types"]
        assert "camera_detection" in recording_summary["sensor_types"]
        assert "fusion_result" in recording_summary["sensor_types"]
        assert "alert" in recording_summary["sensor_types"]

        # Verify: no pipeline stage had zero output
        stage_outputs = {
            "fusion_targets": len(fused_targets),
            "anomaly_alerts": len(anomaly_alerts),
            "alert_records": len(alert_records),
            "rule_results": len(rule_results),
            "sitrep_targets": sitrep.targets.total,
            "export_targets": len(export_parsed["targets"]),
            "recorded_events": recording_summary["event_count"],
        }
        for stage_name, count in stage_outputs.items():
            assert count > 0, f"Stage '{stage_name}' produced zero output"

    # ------------------------------------------------------------------
    # Individual stage tests (for targeted debugging)
    # ------------------------------------------------------------------

    def test_fusion_ingestion(self):
        """Verify FusionEngine correctly ingests all three sensor types."""
        engine = self._stage_fusion()
        targets = engine.tracker.get_all()

        # BLE targets should have ble_ prefix
        ble_targets = [t for t in targets if t.target_id.startswith("ble_")]
        assert len(ble_targets) >= 4, f"Expected 4+ BLE targets, got {len(ble_targets)}"

        # Camera targets should have det_ prefix
        cam_targets = [t for t in targets if t.target_id.startswith("det_")]
        assert len(cam_targets) >= 1, f"Expected 1+ camera targets, got {len(cam_targets)}"

        # Verify sensor records
        fused = engine.get_fused_targets()
        ble_fused = [f for f in fused if "ble" in f.source_types]
        assert len(ble_fused) > 0, "No fused targets have BLE source records"

    def test_anomaly_detection_produces_alerts(self):
        """Verify AnomalyEngine detects speed anomalies."""
        engine = self._stage_fusion()
        anomaly, alerts = self._stage_anomaly(engine)

        # Verify speed alert details
        speed_alerts = [a for a in alerts if a.alert_type == "speed"]
        for alert in speed_alerts:
            assert alert.observed_value > alert.baseline_mean
            assert alert.deviation_sigma > 0
            assert alert.score > 0
            assert alert.severity in ("low", "medium", "high", "critical")

    def test_alert_engine_evaluates_rules(self):
        """Verify AlertEngine fires rules for anomaly events."""
        engine = self._stage_fusion()
        _, anomaly_alerts = self._stage_anomaly(engine)
        alert_engine, records = self._stage_alert(anomaly_alerts)

        # Verify rule stats
        rule_stats = alert_engine.get_rule_stats()
        assert len(rule_stats) > 0, "No rules loaded in alert engine"

        # Check that at least one rule has fire_count > 0
        fired_rules = [r for r in rule_stats if r["fire_count"] > 0]
        assert len(fired_rules) > 0, "No rules have been fired"

    def test_rules_engine_conditional_evaluation(self):
        """Verify RulesEngine conditions evaluate correctly."""
        engine = self._stage_fusion()
        rules_engine, results = self._stage_rules(engine)

        # Verify result structure
        for result in results:
            assert result.result_id
            assert result.rule_id
            assert result.rule_name
            assert result.timestamp > 0
            assert isinstance(result.actions, list)

    def test_sitrep_all_formats(self):
        """Verify SitRep generates text, JSON, and HTML correctly."""
        engine = self._stage_fusion()
        _, alerts = self._stage_anomaly(engine)
        _, records = self._stage_alert(alerts)
        report = self._stage_sitrep(engine, alerts, records)

        # Text
        text = report.to_text()
        assert "THREAT ASSESSMENT" in text
        assert "ZONE ACTIVITY" in text
        assert "ANOMALIES" in text
        assert "END OF REPORT" in text

        # JSON round-trip
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["targets"]["total"] == report.targets.total
        assert parsed["threats"]["total_assessed"] == report.threats.total_assessed

        # HTML structure
        html_str = report.to_html()
        assert "<!DOCTYPE html>" in html_str
        assert "cyberpunk" not in html_str.lower() or "color" in html_str.lower()

    def test_data_exchange_roundtrip(self):
        """Verify TritiumExporter/Importer preserves data fidelity."""
        engine = self._stage_fusion()
        _, alerts = self._stage_anomaly(engine)
        _, records = self._stage_alert(alerts)

        export_json = self._stage_export(engine, alerts, records)
        original = json.loads(export_json)

        # Import into fresh stores
        ts2 = TargetStore(":memory:")
        ds2 = PersistentDossierStore(":memory:")
        es2 = EventStore(":memory:")
        importer = TritiumImporter(
            target_store=ts2,
            dossier_store=ds2,
            event_store=es2,
        )
        result = importer.import_json(export_json)

        assert result.success
        assert result.targets_imported == len(original["targets"])
        assert result.events_imported == len(original["events"])
        assert result.total_imported > 0
        assert len(result.errors) == 0, f"Import errors: {result.errors}"

    def test_recording_context_manager(self):
        """Verify Recorder works as a context manager."""
        path = os.path.join(self.tmpdir, "ctx_recording.jsonl")
        with Recorder(path) as rec:
            rec.record("ble_sighting", source="test", data={"mac": "AA:BB:CC:DD:EE:FF"})
            rec.record("camera_detection", source="test", data={"class": "person"})
            assert rec.event_count == 2

        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 4  # header + 2 events + footer
