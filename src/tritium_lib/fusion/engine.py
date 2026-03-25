# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""FusionEngine — multi-sensor identity resolution orchestrator.

Composes:
  - TargetTracker    (per-source target registry + stale pruning)
  - TargetCorrelator (multi-strategy identity resolution)
  - GeofenceEngine   (polygon zone monitoring)
  - HeatmapEngine    (spatial activity accumulator)
  - DossierStore     (persistent identity records)
  - FusionMetrics    (pipeline health)
  - NetworkAnalyzer  (WiFi probe graph)

into a single ingest-then-query API.

Thread-safe — all public methods acquire appropriate locks.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.tracking.correlator import TargetCorrelator, CorrelationRecord
from tritium_lib.tracking.geofence import GeofenceEngine, GeoZone, GeoEvent
from tritium_lib.tracking.heatmap import HeatmapEngine
from tritium_lib.tracking.dossier import DossierStore, TargetDossier
from tritium_lib.tracking.network_analysis import NetworkAnalyzer
from tritium_lib.intelligence.fusion_metrics import FusionMetrics

logger = logging.getLogger("fusion_engine")


# ---------------------------------------------------------------------------
# Data classes for query results
# ---------------------------------------------------------------------------

@dataclass
class SensorRecord:
    """A single sensor observation associated with a target."""

    source: str  # "ble", "wifi", "camera", "acoustic", "mesh", "adsb", "rf_motion"
    raw_data: dict
    timestamp: float = field(default_factory=time.time)


@dataclass
class FusedTarget:
    """A correlated target with all associated sensor data.

    This is the primary query result from the fusion engine.
    It wraps a TrackedTarget with its full sensor history,
    dossier link, and zone membership.
    """

    target: TrackedTarget
    sensor_records: list[SensorRecord] = field(default_factory=list)
    dossier: TargetDossier | None = None
    zones: set[str] = field(default_factory=set)
    correlations: list[CorrelationRecord] = field(default_factory=list)

    @property
    def target_id(self) -> str:
        return self.target.target_id

    @property
    def source_types(self) -> set[str]:
        """All sensor source types that have contributed to this target."""
        sources = set(self.target.confirming_sources)
        for rec in self.sensor_records:
            sources.add(rec.source)
        return sources

    @property
    def source_count(self) -> int:
        return len(self.source_types)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "name": self.target.name,
            "alliance": self.target.alliance,
            "asset_type": self.target.asset_type,
            "position": {"x": self.target.position[0], "y": self.target.position[1]},
            "heading": self.target.heading,
            "speed": self.target.speed,
            "effective_confidence": self.target.effective_confidence,
            "source": self.target.source,
            "source_types": sorted(self.source_types),
            "source_count": self.source_count,
            "signal_count": self.target.signal_count,
            "classification": self.target.classification,
            "classification_confidence": self.target.classification_confidence,
            "zones": sorted(self.zones),
            "dossier_uuid": self.dossier.uuid if self.dossier else None,
            "correlated_ids": list(self.target.correlated_ids),
            "correlation_confidence": self.target.correlation_confidence,
            "sensor_records": len(self.sensor_records),
            "last_seen": self.target.last_seen,
            "first_seen": self.target.first_seen,
        }


@dataclass
class FusionSnapshot:
    """Point-in-time snapshot of the full fusion state."""

    targets: list[FusedTarget]
    total_targets: int
    total_dossiers: int
    total_correlations: int
    total_zones: int
    metrics: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": [t.to_dict() for t in self.targets],
            "total_targets": self.total_targets,
            "total_dossiers": self.total_dossiers,
            "total_correlations": self.total_correlations,
            "total_zones": self.total_zones,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """Multi-sensor fusion orchestrator.

    Provides a unified ingest API for all sensor types and query methods
    that return correlated, enriched target views.

    Args:
        event_bus: Optional EventBus for publishing fusion events.
        correlation_interval: How often the correlator runs (seconds).
        correlation_threshold: Minimum confidence to trigger correlation.
        correlation_radius: Maximum distance for correlation candidates.
        heatmap_retention: How long heatmap events are retained (seconds).
        auto_correlate: If True, start the background correlator immediately.
    """

    def __init__(
        self,
        event_bus=None,
        *,
        correlation_interval: float = 5.0,
        correlation_threshold: float = 0.3,
        correlation_radius: float = 5.0,
        heatmap_retention: float = 86400.0,
        auto_correlate: bool = False,
    ) -> None:
        self._event_bus = event_bus
        self._lock = threading.Lock()

        # Core components
        self._dossier_store = DossierStore()
        self._tracker = TargetTracker(event_bus=event_bus)
        self._geofence = GeofenceEngine(event_bus=event_bus)
        self._heatmap = HeatmapEngine(retention_seconds=heatmap_retention)
        self._network_analyzer = NetworkAnalyzer()
        self._fusion_metrics = FusionMetrics()

        # Wire geofence into tracker
        self._tracker.set_geofence_engine(self._geofence)

        # Correlator
        self._correlator = TargetCorrelator(
            self._tracker,
            radius=correlation_radius,
            interval=correlation_interval,
            confidence_threshold=correlation_threshold,
            dossier_store=self._dossier_store,
        )

        # Sensor record store: target_id -> list of SensorRecords
        self._sensor_records: dict[str, list[SensorRecord]] = {}
        self._max_records_per_target = 500

        if auto_correlate:
            self._correlator.start()

    # ------------------------------------------------------------------
    # Component accessors (for advanced use / wiring)
    # ------------------------------------------------------------------

    @property
    def tracker(self) -> TargetTracker:
        """The underlying TargetTracker."""
        return self._tracker

    @property
    def correlator(self) -> TargetCorrelator:
        """The underlying TargetCorrelator."""
        return self._correlator

    @property
    def geofence(self) -> GeofenceEngine:
        """The underlying GeofenceEngine."""
        return self._geofence

    @property
    def heatmap(self) -> HeatmapEngine:
        """The underlying HeatmapEngine."""
        return self._heatmap

    @property
    def dossier_store(self) -> DossierStore:
        """The underlying DossierStore."""
        return self._dossier_store

    @property
    def network_analyzer(self) -> NetworkAnalyzer:
        """The underlying NetworkAnalyzer."""
        return self._network_analyzer

    @property
    def fusion_metrics(self) -> FusionMetrics:
        """The underlying FusionMetrics."""
        return self._fusion_metrics

    # ------------------------------------------------------------------
    # Sensor ingestion
    # ------------------------------------------------------------------

    def _record_sensor(self, target_id: str, source: str, raw_data: dict) -> None:
        """Store a raw sensor record for a target (internal)."""
        rec = SensorRecord(source=source, raw_data=dict(raw_data), timestamp=time.time())
        with self._lock:
            if target_id not in self._sensor_records:
                self._sensor_records[target_id] = []
            records = self._sensor_records[target_id]
            records.append(rec)
            # Prune old records
            if len(records) > self._max_records_per_target:
                self._sensor_records[target_id] = records[-self._max_records_per_target:]

    def _publish_event(self, topic: str, data: dict) -> None:
        """Publish to event bus if available."""
        if self._event_bus is not None:
            try:
                self._event_bus.publish(topic, data)
            except Exception:
                pass

    def ingest_ble(self, sighting: dict) -> str | None:
        """Ingest a BLE sighting into the fusion pipeline.

        Args:
            sighting: Dict with keys: mac (required), rssi, name,
                device_type, position, node_position, classification.

        Returns:
            target_id if ingested, None if rejected.
        """
        mac = sighting.get("mac", "")
        if not mac:
            return None

        self._tracker.update_from_ble(sighting)
        tid = f"ble_{mac.replace(':', '').lower()}"
        self._record_sensor(tid, "ble", sighting)

        # Record heatmap event if position is known
        pos = sighting.get("position") or sighting.get("node_position")
        if pos:
            x = float(pos.get("x", 0))
            y = float(pos.get("y", 0))
            if x != 0.0 or y != 0.0:
                self._heatmap.record_event("ble_activity", x, y)

        self._publish_event("fusion.sensor.ingested", {
            "source": "ble", "target_id": tid, "mac": mac,
        })

        logger.debug("Ingested BLE sighting: %s", tid)
        return tid

    def ingest_wifi(self, probe: dict) -> str | None:
        """Ingest a WiFi probe request into the fusion pipeline.

        WiFi probes do not directly create targets in the tracker (they
        lack position data).  Instead they feed the NetworkAnalyzer for
        device correlation and may enrich existing BLE targets.

        Args:
            probe: Dict with keys: mac (required), ssid, rssi, position.

        Returns:
            target_id if a tracker target was updated, None otherwise.
        """
        mac = probe.get("mac", "")
        if not mac:
            return None

        ssid = probe.get("ssid", "")
        rssi = probe.get("rssi", -80)

        # Feed network analyzer
        self._network_analyzer.record_probe(mac, ssid, rssi=rssi)

        # Check if there is already a BLE target for this MAC — enrich it
        tid = f"wifi_{mac.replace(':', '').lower()}"
        ble_tid = f"ble_{mac.replace(':', '').lower()}"

        # If probe has position data, update the tracker
        pos = probe.get("position")
        if pos:
            x = float(pos.get("x", 0))
            y = float(pos.get("y", 0))
            self._tracker.update_from_ble({
                "mac": mac,
                "name": probe.get("name", mac),
                "rssi": rssi,
                "device_type": "wifi_device",
                "position": pos,
            })
            self._heatmap.record_event("ble_activity", x, y)
            result_tid = ble_tid
        else:
            result_tid = tid

        self._record_sensor(result_tid, "wifi", probe)
        self._publish_event("fusion.sensor.ingested", {
            "source": "wifi", "target_id": result_tid, "mac": mac, "ssid": ssid,
        })

        logger.debug("Ingested WiFi probe: mac=%s ssid=%s", mac, ssid)
        return result_tid

    def ingest_camera(self, detection: dict) -> str | None:
        """Ingest a camera/YOLO detection into the fusion pipeline.

        Args:
            detection: Dict with keys: class_name, confidence, center_x, center_y.
                Optional: camera_lat, camera_lng for geo-referenced cameras.

        Returns:
            target_id if ingested, None if rejected (low confidence).
        """
        confidence = detection.get("confidence", 0)
        if confidence < 0.4:
            return None

        # Count targets before to determine new target ID
        before = set(t.target_id for t in self._tracker.get_all())
        self._tracker.update_from_detection(detection)
        after = set(t.target_id for t in self._tracker.get_all())

        # Find the affected target
        new_ids = after - before
        if new_ids:
            tid = new_ids.pop()
        else:
            # Updated an existing target — find it by proximity
            cx = detection.get("center_x", 0.0)
            cy = detection.get("center_y", 0.0)
            class_name = detection.get("class_name", "unknown")
            asset_type = "person" if class_name == "person" else (
                "vehicle" if class_name in ("car", "motorcycle", "bicycle") else class_name
            )
            tid = None
            for t in self._tracker.get_all():
                if t.source != "yolo" or t.asset_type != asset_type:
                    continue
                dx = t.position[0] - cx
                dy = t.position[1] - cy
                if dx * dx + dy * dy < 10.0:
                    tid = t.target_id
                    break
            if tid is None:
                return None

        self._record_sensor(tid, "camera", detection)

        cx = detection.get("center_x", 0.0)
        cy = detection.get("center_y", 0.0)
        if cx != 0.0 or cy != 0.0:
            self._heatmap.record_event("camera_activity", cx, cy)

        self._publish_event("fusion.sensor.ingested", {
            "source": "camera",
            "target_id": tid,
            "class_name": detection.get("class_name", "unknown"),
        })

        logger.debug("Ingested camera detection: %s", tid)
        return tid

    def ingest_acoustic(self, event: dict) -> str | None:
        """Ingest an acoustic event into the fusion pipeline.

        Acoustic events (gunshots, voices, vehicles) can create or
        update targets when they have position data.

        Args:
            event: Dict with keys: event_type (required), position (optional),
                confidence, sensor_id, bearing, description.

        Returns:
            target_id if a target was created/updated, None otherwise.
        """
        event_type = event.get("event_type", "")
        if not event_type:
            return None

        sensor_id = event.get("sensor_id", "unknown")
        pos = event.get("position")
        confidence = float(event.get("confidence", 0.5))

        # Map acoustic events to target asset types
        acoustic_to_asset = {
            "gunshot": "person",
            "voice": "person",
            "scream": "person",
            "footsteps": "person",
            "vehicle_engine": "vehicle",
            "vehicle_horn": "vehicle",
            "vehicle_crash": "vehicle",
            "dog_bark": "animal",
        }
        asset_type = acoustic_to_asset.get(event_type, "unknown")

        tid = None
        if pos:
            x = float(pos.get("x", 0))
            y = float(pos.get("y", 0))

            if x != 0.0 or y != 0.0:
                # Create as an RF motion-like target
                tid = f"acoustic_{sensor_id}_{event_type}"
                self._tracker.update_from_rf_motion({
                    "target_id": tid,
                    "position": pos,
                    "confidence": confidence,
                    "direction_hint": event.get("bearing", "unknown"),
                    "pair_id": f"{sensor_id}:{event_type}",
                })
                # Override asset_type from the default rf_motion
                target = self._tracker.get_target(tid)
                if target:
                    target.asset_type = asset_type
                    target.source = "acoustic"
                    target.classification = event_type
                    target.classification_confidence = confidence
                    target.confirming_sources.add("acoustic")

                self._heatmap.record_event("motion_activity", x, y)

        if tid is None:
            tid = f"acoustic_{sensor_id}_{event_type}"

        self._record_sensor(tid, "acoustic", event)
        self._publish_event("fusion.sensor.ingested", {
            "source": "acoustic",
            "target_id": tid,
            "event_type": event_type,
        })

        logger.debug("Ingested acoustic event: %s (%s)", event_type, tid)
        return tid

    def ingest_mesh(self, mesh_data: dict) -> str | None:
        """Ingest a Meshtastic mesh node into the fusion pipeline.

        Args:
            mesh_data: Dict with keys: target_id (required), name, lat, lng, alt,
                position, battery, alliance, asset_type.

        Returns:
            target_id if ingested, None if rejected.
        """
        tid = mesh_data.get("target_id", "")
        if not tid:
            return None

        self._tracker.update_from_mesh(mesh_data)
        self._record_sensor(tid, "mesh", mesh_data)

        pos = mesh_data.get("position")
        if pos:
            x = float(pos.get("x", 0))
            y = float(pos.get("y", 0))
            if x != 0.0 or y != 0.0:
                self._heatmap.record_event("ble_activity", x, y)

        self._publish_event("fusion.sensor.ingested", {
            "source": "mesh", "target_id": tid,
        })

        logger.debug("Ingested mesh node: %s", tid)
        return tid

    def ingest_adsb(self, adsb_data: dict) -> str | None:
        """Ingest an ADS-B aircraft detection.

        Args:
            adsb_data: Dict with keys: target_id (required), name, lat, lng,
                alt, heading, speed.

        Returns:
            target_id if ingested, None if rejected.
        """
        tid = adsb_data.get("target_id", "")
        if not tid:
            return None

        self._tracker.update_from_adsb(adsb_data)
        self._record_sensor(tid, "adsb", adsb_data)

        self._publish_event("fusion.sensor.ingested", {
            "source": "adsb", "target_id": tid,
        })

        logger.debug("Ingested ADS-B: %s", tid)
        return tid

    def ingest_rf_motion(self, motion: dict) -> str | None:
        """Ingest an RF motion detection event.

        Args:
            motion: Dict with keys: target_id (required), position, confidence,
                direction_hint, pair_id.

        Returns:
            target_id if ingested, None if rejected.
        """
        tid = motion.get("target_id", "")
        if not tid:
            return None

        self._tracker.update_from_rf_motion(motion)
        self._record_sensor(tid, "rf_motion", motion)

        pos = motion.get("position")
        if pos:
            if isinstance(pos, dict):
                x = float(pos.get("x", 0))
                y = float(pos.get("y", 0))
            else:
                x, y = float(pos[0]), float(pos[1])
            if x != 0.0 or y != 0.0:
                self._heatmap.record_event("motion_activity", x, y)

        self._publish_event("fusion.sensor.ingested", {
            "source": "rf_motion", "target_id": tid,
        })

        logger.debug("Ingested RF motion: %s", tid)
        return tid

    # ------------------------------------------------------------------
    # Correlation control
    # ------------------------------------------------------------------

    def run_correlation(self) -> list[CorrelationRecord]:
        """Run one manual correlation pass.

        Returns new correlations found in this pass. Also updates
        fusion metrics for each correlation.
        """
        new_corr = self._correlator.correlate()

        # Update fusion metrics
        for rec in new_corr:
            strategy_scores = [
                (s.strategy_name, s.score) for s in rec.strategy_scores
            ]
            self._fusion_metrics.record_fusion(
                source_a=rec.primary_id.split("_")[0] if "_" in rec.primary_id else "unknown",
                source_b=rec.secondary_id.split("_")[0] if "_" in rec.secondary_id else "unknown",
                confidence=rec.confidence,
                strategy_scores=strategy_scores,
                primary_id=rec.primary_id,
                secondary_id=rec.secondary_id,
            )

            # Merge sensor records from secondary into primary
            with self._lock:
                secondary_records = self._sensor_records.pop(rec.secondary_id, [])
                if rec.primary_id not in self._sensor_records:
                    self._sensor_records[rec.primary_id] = []
                self._sensor_records[rec.primary_id].extend(secondary_records)

            self._publish_event("fusion.target.correlated", {
                "primary_id": rec.primary_id,
                "secondary_id": rec.secondary_id,
                "confidence": rec.confidence,
                "dossier_uuid": rec.dossier_uuid,
            })

        return new_corr

    def start_correlator(self) -> None:
        """Start the background correlation loop."""
        self._correlator.start()

    def stop_correlator(self) -> None:
        """Stop the background correlation loop."""
        self._correlator.stop()

    # ------------------------------------------------------------------
    # Zone management
    # ------------------------------------------------------------------

    def add_zone(self, zone: GeoZone) -> GeoZone:
        """Add a geofence zone."""
        return self._geofence.add_zone(zone)

    def remove_zone(self, zone_id: str) -> bool:
        """Remove a geofence zone."""
        return self._geofence.remove_zone(zone_id)

    def get_zones(self) -> list[GeoZone]:
        """Return all geofence zones."""
        return self._geofence.list_zones()

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_fused_targets(self) -> list[FusedTarget]:
        """Return all targets with their full sensor context.

        Each target is enriched with:
          - All sensor records that contributed to it
          - Its dossier (if correlated)
          - Current zone membership
          - Correlation records
        """
        targets = self._tracker.get_all()
        correlations = self._correlator.get_correlations()

        result: list[FusedTarget] = []
        for t in targets:
            # Find sensor records
            with self._lock:
                records = list(self._sensor_records.get(t.target_id, []))

            # Find dossier
            dossier = self._dossier_store.find_by_signal(t.target_id)

            # Find zones
            zones = self._geofence.get_target_zones(t.target_id)

            # Find correlations involving this target
            target_corr = [
                c for c in correlations
                if c.primary_id == t.target_id or c.secondary_id == t.target_id
            ]

            result.append(FusedTarget(
                target=t,
                sensor_records=records,
                dossier=dossier,
                zones=zones,
                correlations=target_corr,
            ))

        return result

    def get_fused_target(self, target_id: str) -> FusedTarget | None:
        """Get a single fused target by ID."""
        t = self._tracker.get_target(target_id)
        if t is None:
            return None

        with self._lock:
            records = list(self._sensor_records.get(target_id, []))

        dossier = self._dossier_store.find_by_signal(target_id)
        zones = self._geofence.get_target_zones(target_id)
        correlations = [
            c for c in self._correlator.get_correlations()
            if c.primary_id == target_id or c.secondary_id == target_id
        ]

        return FusedTarget(
            target=t,
            sensor_records=records,
            dossier=dossier,
            zones=zones,
            correlations=correlations,
        )

    def get_target_dossier(self, target_id: str) -> dict[str, Any] | None:
        """Get a full target dossier with cross-sensor history.

        Returns a dict containing:
          - target: current TrackedTarget state
          - dossier: persistent identity record (if correlated)
          - sensor_history: all raw sensor records grouped by source
          - zones: current zone membership
          - wifi_profile: WiFi probe profile (if MAC-based)
          - correlations: all correlation records
          - timeline: timestamps of all observations
        """
        t = self._tracker.get_target(target_id)
        if t is None:
            # Maybe the target was consumed by correlation — check dossiers
            dossier = self._dossier_store.find_by_signal(target_id)
            if dossier is None:
                return None
            # Return dossier-only view
            return {
                "target": None,
                "dossier": dossier.to_dict(),
                "sensor_history": {},
                "zones": [],
                "wifi_profile": None,
                "correlations": [],
                "timeline": [],
                "status": "consumed_by_correlation",
            }

        with self._lock:
            records = list(self._sensor_records.get(target_id, []))

        # Group records by source
        by_source: dict[str, list[dict]] = {}
        timeline: list[float] = []
        for rec in records:
            by_source.setdefault(rec.source, []).append({
                "data": rec.raw_data,
                "timestamp": rec.timestamp,
            })
            timeline.append(rec.timestamp)

        dossier = self._dossier_store.find_by_signal(target_id)
        zones = self._geofence.get_target_zones(target_id)

        correlations = [
            {
                "primary_id": c.primary_id,
                "secondary_id": c.secondary_id,
                "confidence": c.confidence,
                "reason": c.reason,
                "dossier_uuid": c.dossier_uuid,
            }
            for c in self._correlator.get_correlations()
            if c.primary_id == target_id or c.secondary_id == target_id
        ]

        # WiFi probe profile
        wifi_profile = None
        mac = target_id.replace("ble_", "").replace("wifi_", "")
        # Reformat as MAC address
        if len(mac) == 12:
            formatted_mac = ":".join(mac[i:i + 2] for i in range(0, 12, 2)).upper()
            wifi_profile = self._network_analyzer.get_device_profile(formatted_mac)

        # Trail from tracker history
        trail = self._tracker.history.get_trail_dicts(target_id, max_points=50)

        timeline.sort()

        return {
            "target": t.to_dict(),
            "dossier": dossier.to_dict() if dossier else None,
            "sensor_history": by_source,
            "zones": sorted(zones),
            "wifi_profile": wifi_profile,
            "correlations": correlations,
            "timeline": timeline,
            "trail": trail,
            "status": "active",
        }

    def get_zone_activity(self, zone_id: str) -> dict[str, Any]:
        """Get activity report for a specific geofence zone.

        Returns:
            Dict with zone info, current occupants (as FusedTargets),
            recent events, and heatmap data.
        """
        zone = self._geofence.get_zone(zone_id)
        if zone is None:
            return {
                "zone": None,
                "occupants": [],
                "events": [],
                "error": f"Zone '{zone_id}' not found",
            }

        # Current occupants
        occupant_ids = self._geofence.get_zone_occupants(zone_id)
        occupants = []
        for oid in occupant_ids:
            ft = self.get_fused_target(oid)
            if ft is not None:
                occupants.append(ft.to_dict())

        # Recent geofence events for this zone
        events = self._geofence.get_events(limit=50, zone_id=zone_id)
        event_dicts = [e.to_dict() for e in events]

        return {
            "zone": zone.to_dict(),
            "occupant_count": len(occupants),
            "occupants": occupants,
            "events": event_dicts,
        }

    def get_snapshot(self) -> FusionSnapshot:
        """Get a full point-in-time snapshot of the fusion state."""
        fused = self.get_fused_targets()
        return FusionSnapshot(
            targets=fused,
            total_targets=len(fused),
            total_dossiers=self._dossier_store.count,
            total_correlations=len(self._correlator.get_correlations()),
            total_zones=len(self._geofence.list_zones()),
            metrics=self._fusion_metrics.get_status(),
        )

    def get_targets_by_source(self, source: str) -> list[FusedTarget]:
        """Get fused targets filtered by primary source type."""
        return [
            ft for ft in self.get_fused_targets()
            if ft.target.source == source
        ]

    def get_targets_in_zone(self, zone_id: str) -> list[FusedTarget]:
        """Get all fused targets currently inside a zone."""
        occupant_ids = set(self._geofence.get_zone_occupants(zone_id))
        return [
            ft for ft in self.get_fused_targets()
            if ft.target_id in occupant_ids
        ]

    def get_multi_source_targets(self, min_sources: int = 2) -> list[FusedTarget]:
        """Get targets confirmed by multiple sensor sources."""
        return [
            ft for ft in self.get_fused_targets()
            if ft.source_count >= min_sources
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop all background tasks and clean up."""
        self._correlator.stop()
        logger.info("FusionEngine shutdown complete")

    def clear(self) -> None:
        """Reset all state. Useful for testing."""
        self._correlator.stop()
        self._dossier_store.clear()
        self._heatmap.clear()
        with self._lock:
            self._sensor_records.clear()
        # Re-create tracker (simplest way to clear it)
        self._tracker = TargetTracker(event_bus=self._event_bus)
        self._tracker.set_geofence_engine(self._geofence)
        self._correlator = TargetCorrelator(
            self._tracker,
            dossier_store=self._dossier_store,
        )
