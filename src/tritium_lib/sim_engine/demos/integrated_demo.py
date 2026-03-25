# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Integrated city sim -> sensor fusion pipeline demo.

The crown jewel: proves the entire Tritium system works end-to-end.

Pipeline:
    City Sim (vehicles + NPCs with daily routines)
        -> RF Signature Generator (BLE ads, WiFi probes, TPMS)
            -> TargetTracker (unified registry)
                -> TargetCorrelator (multi-signal identity resolution)
                    -> GeofenceEngine (zone monitoring)
                    -> HeatmapEngine (activity hotspots)

Vehicles generate WiFi sightings as they drive routes.
NPCs generate BLE sightings as they follow daily routines.
TargetTracker correlates: "WiFi_xyz and BLE_abc are the same person"
GeofenceEngine detects: "Target entered zone Alpha at 14:30"
HeatmapEngine shows: "Area around intersection is high-activity"

Run standalone::

    python3 -m tritium_lib.sim_engine.demos.integrated_demo
    # Then open http://localhost:8099

Run headless::

    python3 -m tritium_lib.sim_engine.demos.integrated_demo --headless --ticks 200
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field

from tritium_lib.sim_engine.ai.city_sim import (
    NeighborhoodSim,
    state_rf_emission,
    state_visible_on_map,
)
from tritium_lib.sim_engine.ai.rf_signatures import (
    RFSignatureGenerator,
    PersonRFProfile,
    VehicleRFProfile,
)
from tritium_lib.tracking.target_tracker import TargetTracker, TrackedTarget
from tritium_lib.tracking.correlator import TargetCorrelator, CorrelationRecord
from tritium_lib.tracking.geofence import GeofenceEngine, GeoZone, GeoEvent
from tritium_lib.tracking.heatmap import HeatmapEngine
from tritium_lib.tracking.dossier import DossierStore

logger = logging.getLogger("integrated_demo")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORLD_SIZE = 500.0
NUM_RESIDENTS = 30
SIM_DT = 1.0  # seconds per physics tick
TIME_SCALE_HOURS_PER_SEC = 10.0 / 60.0  # 10 sim minutes per real second
START_HOUR = 7.0


# ---------------------------------------------------------------------------
# SensorBridge — converts sim entities into TargetTracker sightings
# ---------------------------------------------------------------------------

class SensorBridge:
    """Bridges city sim entities to the tracking pipeline.

    For each sim tick:
    - Visible NPCs with RF emission -> BLE sightings fed to TargetTracker
    - Driving vehicles -> WiFi sightings fed to TargetTracker
    - All position events -> HeatmapEngine
    - All position events -> GeofenceEngine (via tracker)

    Maintains RF profiles per entity so the same simulated person
    always emits the same MAC addresses (until rotation).
    """

    def __init__(
        self,
        tracker: TargetTracker,
        heatmap: HeatmapEngine,
        rng: random.Random | None = None,
    ) -> None:
        self.tracker = tracker
        self.heatmap = heatmap
        self._rng = rng or random.Random(42)
        self._person_profiles: dict[str, PersonRFProfile] = {}
        self._vehicle_profiles: dict[str, VehicleRFProfile] = {}
        self._stats = {
            "ble_sightings_total": 0,
            "wifi_sightings_total": 0,
            "heatmap_events_total": 0,
            "ticks_processed": 0,
            "entities_seen": 0,
        }

    def _get_person_profile(self, resident_id: str) -> PersonRFProfile:
        """Get or create a persistent RF profile for a resident."""
        if resident_id not in self._person_profiles:
            self._person_profiles[resident_id] = RFSignatureGenerator.random_person(
                rng=self._rng,
            )
        return self._person_profiles[resident_id]

    def _get_vehicle_profile(self, vehicle_id: str) -> VehicleRFProfile:
        """Get or create a persistent RF profile for a vehicle."""
        if vehicle_id not in self._vehicle_profiles:
            self._vehicle_profiles[vehicle_id] = RFSignatureGenerator.random_vehicle(
                rng=self._rng,
            )
        return self._vehicle_profiles[vehicle_id]

    def process_tick(self, sim: NeighborhoodSim) -> dict:
        """Process one sim tick: extract entities, generate sightings, feed tracker.

        Returns per-tick stats dict.
        """
        tick_stats = {
            "ble_sightings": 0,
            "wifi_sightings": 0,
            "heatmap_events": 0,
            "visible_residents": 0,
            "driving_vehicles": 0,
        }

        # Process residents -> BLE sightings
        for resident in sim.residents:
            if not resident.visible:
                continue

            rf_level = state_rf_emission(resident.activity_state)
            if rf_level == "none":
                continue

            tick_stats["visible_residents"] += 1
            pos = resident.position

            # Record heatmap event for all visible residents
            self.heatmap.record_event("ble_activity", pos[0], pos[1])
            tick_stats["heatmap_events"] += 1

            # Generate BLE sightings based on RF emission level
            if rf_level in ("full", "reduced"):
                profile = self._get_person_profile(resident.resident_id)
                ble_ads = profile.emit_ble_advertisements(pos)
                for ad in ble_ads:
                    # Convert RF signature format to TargetTracker BLE format
                    sighting = {
                        "mac": ad["mac"],
                        "rssi": ad["rssi"],
                        "name": ad.get("name", ""),
                        "device_type": ad.get("device_type", "ble_device"),
                        "position": {"x": ad["position_x"], "y": ad["position_y"]},
                        "classification": ad.get("device_type", "phone"),
                    }
                    self.tracker.update_from_ble(sighting)
                    tick_stats["ble_sightings"] += 1

                # WiFi probes (less frequent than BLE) — use YOLO detection
                # path so they create separate targets from different source.
                # In reality WiFi and BLE MACs differ; the correlator's job
                # is to fuse them via spatial + temporal proximity.
                if rf_level == "full" and self._rng.random() < 0.3:
                    wifi_probes = profile.emit_wifi_probes(pos)
                    for probe in wifi_probes:
                        # Feed WiFi probes through the YOLO/detection path
                        # so the correlator sees BLE + YOLO sources and can fuse
                        detection = {
                            "class_name": "person",
                            "confidence": 0.75,
                            "center_x": probe["position_x"] + self._rng.gauss(0, 1.5),
                            "center_y": probe["position_y"] + self._rng.gauss(0, 1.5),
                        }
                        self.tracker.update_from_detection(detection)
                        tick_stats["wifi_sightings"] += 1

        # Process vehicles -> WiFi / TPMS sightings
        for vehicle in sim.vehicles:
            if not vehicle.driving:
                continue

            tick_stats["driving_vehicles"] += 1
            pos = vehicle.position

            # Record heatmap for driving vehicles
            self.heatmap.record_event("motion_activity", pos[0], pos[1])
            tick_stats["heatmap_events"] += 1

            profile = self._get_vehicle_profile(vehicle.vehicle_id)

            # Dashcam WiFi hotspot — feed as YOLO vehicle detection
            # so it creates a different source from the keyfob BLE,
            # allowing correlator to fuse keyfob + camera sightings.
            if profile.has_dashcam_wifi:
                detection = {
                    "class_name": "car",
                    "confidence": 0.80,
                    "center_x": pos[0] + self._rng.gauss(0, 2.0),
                    "center_y": pos[1] + self._rng.gauss(0, 2.0),
                }
                self.tracker.update_from_detection(detection)
                tick_stats["wifi_sightings"] += 1

            # Keyfob BLE (while driving)
            if profile.has_keyfob and self._rng.random() < 0.5:
                sighting = {
                    "mac": profile.keyfob_mac,
                    "rssi": self._rng.randint(-65, -45),
                    "name": f"Keyfob {profile.make_model}",
                    "device_type": "keyfob",
                    "position": {"x": pos[0], "y": pos[1]},
                    "classification": "vehicle",
                }
                self.tracker.update_from_ble(sighting)
                tick_stats["ble_sightings"] += 1

        # Update cumulative stats
        self._stats["ble_sightings_total"] += tick_stats["ble_sightings"]
        self._stats["wifi_sightings_total"] += tick_stats["wifi_sightings"]
        self._stats["heatmap_events_total"] += tick_stats["heatmap_events"]
        self._stats["ticks_processed"] += 1
        self._stats["entities_seen"] += (
            tick_stats["visible_residents"] + tick_stats["driving_vehicles"]
        )

        return tick_stats

    @property
    def stats(self) -> dict:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# IntegratedPipeline — orchestrates the full sim -> tracking pipeline
# ---------------------------------------------------------------------------

class IntegratedPipeline:
    """Full city sim -> sensor fusion pipeline.

    Wires together:
        NeighborhoodSim -> SensorBridge -> TargetTracker
                                        -> TargetCorrelator
                                        -> GeofenceEngine
                                        -> HeatmapEngine
    """

    def __init__(
        self,
        num_residents: int = NUM_RESIDENTS,
        world_size: float = WORLD_SIZE,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.world_size = world_size
        self.num_residents = num_residents

        # City simulation
        self.sim = NeighborhoodSim(
            num_residents=num_residents,
            bounds=((0.0, 0.0), (world_size, world_size)),
            seed=seed,
        )

        # Tracking pipeline
        self.tracker = TargetTracker()
        self.dossier_store = DossierStore()
        self.correlator = TargetCorrelator(
            self.tracker,
            radius=15.0,  # wider radius for city-scale
            max_age=60.0,
            interval=5.0,
            confidence_threshold=0.25,
            dossier_store=self.dossier_store,
        )
        self.geofence = GeofenceEngine()
        self.heatmap = HeatmapEngine(retention_seconds=3600)

        # Wire geofence to tracker
        self.tracker.set_geofence_engine(self.geofence)

        # Sensor bridge
        self.bridge = SensorBridge(
            tracker=self.tracker,
            heatmap=self.heatmap,
            rng=random.Random(seed),
        )

        # State
        self.sim_hour = START_HOUR
        self.tick_count = 0
        self.real_start_time = 0.0
        self._geofence_events: list[dict] = []
        self._correlation_log: list[dict] = []
        self._running = False

    def setup(self) -> None:
        """Initialize the simulation and create geofence zones."""
        self.sim.populate()
        self._setup_geofence_zones()
        self.real_start_time = time.time()
        logger.info(
            "Pipeline initialized: %d residents, %d vehicles, %d buildings, %d zones",
            len(self.sim.residents),
            len(self.sim.vehicles),
            len(self.sim.buildings),
            len(self.geofence.list_zones()),
        )

    def _setup_geofence_zones(self) -> None:
        """Create monitoring zones at key locations."""
        ws = self.world_size

        # Zone Alpha: city center
        cx, cy = ws * 0.5, ws * 0.5
        r = ws * 0.15
        self.geofence.add_zone(GeoZone(
            zone_id="zone_alpha",
            name="Zone Alpha - City Center",
            polygon=[
                (cx - r, cy - r), (cx + r, cy - r),
                (cx + r, cy + r), (cx - r, cy + r),
            ],
            zone_type="monitored",
        ))

        # Zone Bravo: commercial district (upper right)
        self.geofence.add_zone(GeoZone(
            zone_id="zone_bravo",
            name="Zone Bravo - Commercial",
            polygon=[
                (ws * 0.6, ws * 0.6), (ws * 0.95, ws * 0.6),
                (ws * 0.95, ws * 0.95), (ws * 0.6, ws * 0.95),
            ],
            zone_type="monitored",
        ))

        # Zone Charlie: residential area (lower left)
        self.geofence.add_zone(GeoZone(
            zone_id="zone_charlie",
            name="Zone Charlie - Residential",
            polygon=[
                (ws * 0.05, ws * 0.05), (ws * 0.4, ws * 0.05),
                (ws * 0.4, ws * 0.4), (ws * 0.05, ws * 0.4),
            ],
            zone_type="restricted",
        ))

        # Zone Delta: park area
        self.geofence.add_zone(GeoZone(
            zone_id="zone_delta",
            name="Zone Delta - Park",
            polygon=[
                (ws * 0.1, ws * 0.6), (ws * 0.4, ws * 0.6),
                (ws * 0.4, ws * 0.9), (ws * 0.1, ws * 0.9),
            ],
            zone_type="monitored",
        ))

    def tick(self) -> dict:
        """Run one pipeline tick: sim -> sensors -> tracking -> intelligence.

        Returns tick stats dict.
        """
        # Advance simulation time
        self.sim_hour += SIM_DT * TIME_SCALE_HOURS_PER_SEC
        if self.sim_hour >= 24.0:
            self.sim_hour -= 24.0

        # Tick city simulation
        self.sim.tick(SIM_DT, self.sim_hour)

        # Bridge sim entities -> tracking pipeline
        tick_stats = self.bridge.process_tick(self.sim)

        # Run correlator (every 10 ticks to avoid overhead)
        new_correlations: list[CorrelationRecord] = []
        if self.tick_count % 10 == 0:
            new_correlations = self.correlator.correlate()
            for c in new_correlations:
                self._correlation_log.append({
                    "tick": self.tick_count,
                    "sim_hour": self.sim_hour,
                    "primary_id": c.primary_id,
                    "secondary_id": c.secondary_id,
                    "confidence": c.confidence,
                    "reason": c.reason,
                })

        # Collect geofence events
        geo_events = self.geofence.get_events(limit=50)
        enter_events = [e for e in geo_events if e.event_type == "enter"]

        self.tick_count += 1
        tick_stats["correlations"] = len(new_correlations)
        tick_stats["sim_hour"] = self.sim_hour
        tick_stats["tick"] = self.tick_count
        tick_stats["tracked_targets"] = len(self.tracker.get_all())
        tick_stats["geofence_enter_events"] = len(enter_events)

        return tick_stats

    def get_pipeline_stats(self) -> dict:
        """Full pipeline statistics."""
        targets = self.tracker.get_all()
        sim_stats = self.sim.get_statistics()
        correlations = self.correlator.get_correlations()
        geo_events = self.geofence.get_events(limit=1000)
        dossiers = self.dossier_store.get_all()

        # Count targets by source
        source_counts: dict[str, int] = {}
        multi_source_count = 0
        for t in targets:
            source_counts[t.source] = source_counts.get(t.source, 0) + 1
            if len(t.confirming_sources) > 1:
                multi_source_count += 1

        # Zone occupancy
        zone_occupancy: dict[str, int] = {}
        for zone in self.geofence.list_zones():
            occupants = self.geofence.get_zone_occupants(zone.zone_id)
            zone_occupancy[zone.name] = len(occupants)

        elapsed = time.time() - self.real_start_time if self.real_start_time else 0

        return {
            "pipeline": {
                "ticks": self.tick_count,
                "sim_hour": round(self.sim_hour, 2),
                "sim_hour_display": _hour_to_string(self.sim_hour),
                "elapsed_seconds": round(elapsed, 1),
                "ticks_per_second": round(self.tick_count / max(elapsed, 0.001), 1),
            },
            "simulation": {
                "residents": sim_stats["total_residents"],
                "vehicles": sim_stats["total_vehicles"],
                "buildings": sim_stats["total_buildings"],
                "vehicles_driving": sim_stats["vehicles_driving"],
                "visible_on_map": sim_stats["visible_on_map"],
                "inside_buildings": sim_stats["inside_buildings"],
                "activities": sim_stats.get("activities", {}),
            },
            "tracking": {
                "total_targets": len(targets),
                "by_source": source_counts,
                "multi_source_targets": multi_source_count,
                "total_correlations": len(correlations),
                "total_dossiers": len(dossiers),
            },
            "sensor_bridge": self.bridge.stats,
            "geofence": {
                "zones": len(self.geofence.list_zones()),
                "zone_occupancy": zone_occupancy,
                "total_events": len(geo_events),
                "enter_events": len([e for e in geo_events if e.event_type == "enter"]),
                "exit_events": len([e for e in geo_events if e.event_type == "exit"]),
            },
            "heatmap": {
                "ble_events": self.heatmap.event_count("ble_activity"),
                "motion_events": self.heatmap.event_count("motion_activity"),
                "total_events": self.heatmap.event_count("all"),
            },
        }

    def get_targets_list(self) -> list[dict]:
        """All tracked targets as serializable dicts."""
        targets = self.tracker.get_all()
        result = []
        for t in targets:
            result.append({
                "target_id": t.target_id,
                "name": t.name,
                "alliance": t.alliance,
                "asset_type": t.asset_type,
                "source": t.source,
                "position": {"x": t.position[0], "y": t.position[1]},
                "heading": t.heading,
                "speed": t.speed,
                "signal_count": t.signal_count,
                "confidence": round(t.effective_confidence, 3),
                "confirming_sources": list(t.confirming_sources),
                "correlated_ids": list(t.correlated_ids),
                "correlation_confidence": round(t.correlation_confidence, 3),
                "classification": t.classification,
                "status": t.status,
            })
        return result

    def get_correlations_list(self) -> list[dict]:
        """All correlation records."""
        return list(self._correlation_log)

    def get_geofence_events_list(self, limit: int = 100) -> list[dict]:
        """Recent geofence events."""
        events = self.geofence.get_events(limit=limit)
        return [e.to_dict() for e in events]

    def get_zones_list(self) -> list[dict]:
        """All geofence zones with occupancy."""
        zones = self.geofence.list_zones()
        result = []
        for z in zones:
            d = z.to_dict()
            d["occupants"] = self.geofence.get_zone_occupants(z.zone_id)
            d["occupant_count"] = len(d["occupants"])
            result.append(d)
        return result

    def get_heatmap_data(self, layer: str = "all", resolution: int = 30) -> dict:
        """Heatmap grid data."""
        return self.heatmap.get_heatmap(
            time_window_minutes=60,
            resolution=resolution,
            layer=layer,
        )

    def get_dossiers_list(self) -> list[dict]:
        """All target dossiers."""
        dossiers = self.dossier_store.get_all()
        return [d.to_dict() for d in dossiers]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hour_to_string(hour: float) -> str:
    """Convert fractional hour to HH:MM AM/PM string."""
    h = int(hour) % 24
    m = int((hour % 1.0) * 60)
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ampm}"


# ---------------------------------------------------------------------------
# Headless runner
# ---------------------------------------------------------------------------

def run_headless(ticks: int = 200, seed: int = 42) -> dict:
    """Run the pipeline headless and return final stats.

    Useful for testing and benchmarking.
    """
    pipeline = IntegratedPipeline(seed=seed)
    pipeline.setup()

    for i in range(ticks):
        tick_stats = pipeline.tick()
        if (i + 1) % 50 == 0:
            stats = pipeline.get_pipeline_stats()
            print(
                f"  Tick {i + 1}/{ticks} | "
                f"{stats['pipeline']['sim_hour_display']} | "
                f"Targets: {stats['tracking']['total_targets']} | "
                f"Correlations: {stats['tracking']['total_correlations']} | "
                f"BLE: {stats['sensor_bridge']['ble_sightings_total']} | "
                f"Heatmap: {stats['heatmap']['total_events']}"
            )

    final_stats = pipeline.get_pipeline_stats()

    print("\n" + "=" * 70)
    print("  TRITIUM INTEGRATED PIPELINE DEMO — Final Report")
    print("=" * 70)
    print(f"  Simulation Time:    {final_stats['pipeline']['sim_hour_display']}")
    print(f"  Ticks:              {final_stats['pipeline']['ticks']}")
    print(f"  Elapsed:            {final_stats['pipeline']['elapsed_seconds']}s")
    print(f"  Throughput:         {final_stats['pipeline']['ticks_per_second']} ticks/s")
    print()
    print(f"  Residents:          {final_stats['simulation']['residents']}")
    print(f"  Vehicles:           {final_stats['simulation']['vehicles']}")
    print(f"  Vehicles Driving:   {final_stats['simulation']['vehicles_driving']}")
    print(f"  Visible on Map:     {final_stats['simulation']['visible_on_map']}")
    print()
    print(f"  Tracked Targets:    {final_stats['tracking']['total_targets']}")
    print(f"  Multi-Source:       {final_stats['tracking']['multi_source_targets']}")
    print(f"  Correlations:       {final_stats['tracking']['total_correlations']}")
    print(f"  Dossiers:           {final_stats['tracking']['total_dossiers']}")
    print()
    print(f"  BLE Sightings:      {final_stats['sensor_bridge']['ble_sightings_total']}")
    print(f"  WiFi Sightings:     {final_stats['sensor_bridge']['wifi_sightings_total']}")
    print(f"  Heatmap Events:     {final_stats['heatmap']['total_events']}")
    print()
    print(f"  Geofence Zones:     {final_stats['geofence']['zones']}")
    print(f"  Zone Enter Events:  {final_stats['geofence']['enter_events']}")
    print(f"  Zone Exit Events:   {final_stats['geofence']['exit_events']}")
    for zname, count in final_stats["geofence"]["zone_occupancy"].items():
        print(f"    {zname}: {count} occupants")
    print("=" * 70)

    return final_stats


# ---------------------------------------------------------------------------
# FastAPI web server with HTML dashboard
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8099, seed: int = 42) -> None:
    """Run the integrated demo as a FastAPI web server with live dashboard."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        import uvicorn
    except ImportError:
        print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        print("Falling back to headless mode.")
        run_headless(seed=seed)
        return

    app = FastAPI(title="Tritium Integrated Pipeline Demo")
    pipeline = IntegratedPipeline(seed=seed)
    pipeline.setup()

    # Background tick loop
    def _tick_loop():
        while pipeline._running:
            pipeline.tick()
            time.sleep(0.05)  # ~20 ticks/sec

    pipeline._running = True
    tick_thread = threading.Thread(target=_tick_loop, daemon=True)
    tick_thread.start()

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return _dashboard_html()

    @app.get("/api/targets")
    async def api_targets():
        return JSONResponse(pipeline.get_targets_list())

    @app.get("/api/heatmap")
    async def api_heatmap(layer: str = "all", resolution: int = 30):
        return JSONResponse(pipeline.get_heatmap_data(layer=layer, resolution=resolution))

    @app.get("/api/zones")
    async def api_zones():
        return JSONResponse(pipeline.get_zones_list())

    @app.get("/api/geofence/events")
    async def api_geofence_events(limit: int = 100):
        return JSONResponse(pipeline.get_geofence_events_list(limit=limit))

    @app.get("/api/correlations")
    async def api_correlations():
        return JSONResponse(pipeline.get_correlations_list())

    @app.get("/api/dossiers")
    async def api_dossiers():
        return JSONResponse(pipeline.get_dossiers_list())

    @app.get("/api/pipeline/stats")
    async def api_stats():
        return JSONResponse(pipeline.get_pipeline_stats())

    @app.on_event("shutdown")
    async def shutdown():
        pipeline._running = False

    print(f"\n  Tritium Integrated Pipeline Demo")
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  API:       http://localhost:{port}/api/pipeline/stats")
    print()

    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

def _dashboard_html() -> str:
    """Generate the cyberpunk HTML dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tritium Integrated Pipeline</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
    background: #0a0a0f;
    color: #c0c0c0;
    font-family: 'Courier New', monospace;
    padding: 12px;
}
h1 {
    color: #00f0ff;
    text-align: center;
    font-size: 20px;
    margin-bottom: 12px;
    text-shadow: 0 0 10px #00f0ff44;
}
h2 {
    color: #05ffa1;
    font-size: 14px;
    margin-bottom: 8px;
    border-bottom: 1px solid #1a1a2e;
    padding-bottom: 4px;
}
.grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 12px;
}
.card {
    background: #111118;
    border: 1px solid #1a1a2e;
    border-radius: 6px;
    padding: 10px;
}
.metric {
    text-align: center;
}
.metric .value {
    font-size: 28px;
    font-weight: bold;
    color: #00f0ff;
}
.metric .label {
    font-size: 10px;
    color: #666;
    text-transform: uppercase;
}
.row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 12px;
}
.row3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-bottom: 12px;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
}
th {
    background: #0e0e18;
    color: #05ffa1;
    text-align: left;
    padding: 4px 6px;
    border-bottom: 1px solid #1a1a2e;
}
td {
    padding: 3px 6px;
    border-bottom: 1px solid #0e0e18;
}
tr:hover { background: #15152a; }
.tag {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: bold;
}
.tag-ble { background: #00f0ff22; color: #00f0ff; border: 1px solid #00f0ff44; }
.tag-wifi { background: #ff2a6d22; color: #ff2a6d; border: 1px solid #ff2a6d44; }
.tag-multi { background: #05ffa122; color: #05ffa1; border: 1px solid #05ffa144; }
.tag-enter { background: #fcee0a22; color: #fcee0a; border: 1px solid #fcee0a44; }
.tag-exit { background: #ff2a6d22; color: #ff2a6d; border: 1px solid #ff2a6d44; }
canvas {
    border: 1px solid #1a1a2e;
    border-radius: 4px;
    width: 100%;
    image-rendering: pixelated;
}
#simClock {
    color: #fcee0a;
    font-size: 16px;
    text-align: center;
    margin-bottom: 8px;
}
.zone-bar {
    height: 18px;
    background: #05ffa133;
    border-radius: 3px;
    margin: 2px 0;
    display: flex;
    align-items: center;
    padding: 0 6px;
    font-size: 10px;
}
.zone-bar .name { flex: 1; color: #05ffa1; }
.zone-bar .count { color: #fcee0a; font-weight: bold; }
#statusBar {
    text-align: center;
    font-size: 10px;
    color: #555;
    margin-top: 8px;
}
</style>
</head>
<body>
<h1>TRITIUM INTEGRATED PIPELINE</h1>
<div id="simClock">Loading...</div>

<!-- Top metrics -->
<div class="grid" id="metrics">
    <div class="card metric"><div class="value" id="mTargets">-</div><div class="label">Tracked Targets</div></div>
    <div class="card metric"><div class="value" id="mCorrelations">-</div><div class="label">Correlations</div></div>
    <div class="card metric"><div class="value" id="mBLE">-</div><div class="label">BLE Sightings</div></div>
    <div class="card metric"><div class="value" id="mHeatmap">-</div><div class="label">Heatmap Events</div></div>
</div>

<!-- Middle row: target map + zones -->
<div class="row">
    <div class="card">
        <h2>Target Map</h2>
        <canvas id="mapCanvas" width="400" height="400"></canvas>
    </div>
    <div class="card">
        <h2>Activity Heatmap</h2>
        <canvas id="heatCanvas" width="400" height="400"></canvas>
    </div>
</div>

<!-- Bottom row: tables -->
<div class="row3">
    <div class="card">
        <h2>Zone Occupancy</h2>
        <div id="zoneList"></div>
        <h2 style="margin-top:10px">Geofence Events</h2>
        <div style="max-height:200px;overflow-y:auto" id="geoEvents"></div>
    </div>
    <div class="card">
        <h2>Tracked Targets</h2>
        <div style="max-height:350px;overflow-y:auto" id="targetTable"></div>
    </div>
    <div class="card">
        <h2>Correlations</h2>
        <div style="max-height:200px;overflow-y:auto" id="corrTable"></div>
        <h2 style="margin-top:10px">Pipeline Stats</h2>
        <div id="pipeStats"></div>
    </div>
</div>

<div id="statusBar">Connecting...</div>

<script>
const API = '';
let lastStats = null;

async function fetchJSON(url) {
    const r = await fetch(API + url);
    return r.json();
}

function $(id) { return document.getElementById(id); }

async function refresh() {
    try {
        const [stats, targets, zones, geoEvts, corrs, heatmap] = await Promise.all([
            fetchJSON('/api/pipeline/stats'),
            fetchJSON('/api/targets'),
            fetchJSON('/api/zones'),
            fetchJSON('/api/geofence/events?limit=30'),
            fetchJSON('/api/correlations'),
            fetchJSON('/api/heatmap?resolution=40'),
        ]);
        lastStats = stats;

        // Clock
        $('simClock').textContent =
            stats.pipeline.sim_hour_display + ' | Tick ' + stats.pipeline.ticks +
            ' | ' + stats.pipeline.ticks_per_second + ' tps';

        // Metrics
        $('mTargets').textContent = stats.tracking.total_targets;
        $('mCorrelations').textContent = stats.tracking.total_correlations;
        $('mBLE').textContent = stats.sensor_bridge.ble_sightings_total;
        $('mHeatmap').textContent = stats.heatmap.total_events;

        // Target map
        drawTargetMap(targets, zones);

        // Heatmap
        drawHeatmap(heatmap);

        // Zones
        let zh = '';
        for (const z of zones) {
            const pct = Math.min(100, z.occupant_count * 5);
            zh += '<div class="zone-bar" style="background:linear-gradient(90deg,#05ffa133 ' +
                pct + '%,transparent ' + pct + '%)">' +
                '<span class="name">' + z.name + '</span>' +
                '<span class="count">' + z.occupant_count + '</span></div>';
        }
        $('zoneList').innerHTML = zh;

        // Geofence events
        let gh = '<table><tr><th>Type</th><th>Target</th><th>Zone</th></tr>';
        for (const e of geoEvts.slice(0, 20)) {
            const cls = e.event_type === 'enter' ? 'tag-enter' : 'tag-exit';
            gh += '<tr><td><span class="tag ' + cls + '">' + e.event_type +
                '</span></td><td>' + e.target_id.substring(0, 16) +
                '</td><td>' + e.zone_name.substring(0, 20) + '</td></tr>';
        }
        gh += '</table>';
        $('geoEvents').innerHTML = gh;

        // Targets table
        let th = '<table><tr><th>ID</th><th>Type</th><th>Source</th><th>Conf</th><th>Signals</th></tr>';
        const sorted = targets.sort((a, b) => b.signal_count - a.signal_count).slice(0, 30);
        for (const t of sorted) {
            const srcCls = t.confirming_sources.length > 1 ? 'tag-multi' :
                (t.source === 'ble' ? 'tag-ble' : 'tag-wifi');
            th += '<tr><td>' + t.target_id.substring(0, 18) +
                '</td><td>' + t.asset_type +
                '</td><td><span class="tag ' + srcCls + '">' +
                t.confirming_sources.join('+') +
                '</span></td><td>' + t.confidence +
                '</td><td>' + t.signal_count + '</td></tr>';
        }
        th += '</table>';
        $('targetTable').innerHTML = th;

        // Correlations
        let ch = '<table><tr><th>Primary</th><th>Secondary</th><th>Conf</th></tr>';
        for (const c of corrs.slice(-20).reverse()) {
            ch += '<tr><td>' + c.primary_id.substring(0, 16) +
                '</td><td>' + c.secondary_id.substring(0, 16) +
                '</td><td>' + c.confidence.toFixed(2) + '</td></tr>';
        }
        ch += '</table>';
        $('corrTable').innerHTML = ch;

        // Pipeline stats
        let ps = '<table>';
        ps += '<tr><td>Residents</td><td>' + stats.simulation.residents + '</td></tr>';
        ps += '<tr><td>Vehicles</td><td>' + stats.simulation.vehicles + '</td></tr>';
        ps += '<tr><td>Driving</td><td>' + stats.simulation.vehicles_driving + '</td></tr>';
        ps += '<tr><td>Visible</td><td>' + stats.simulation.visible_on_map + '</td></tr>';
        ps += '<tr><td>Multi-source</td><td>' + stats.tracking.multi_source_targets + '</td></tr>';
        ps += '<tr><td>Dossiers</td><td>' + stats.tracking.total_dossiers + '</td></tr>';
        ps += '<tr><td>WiFi sightings</td><td>' + stats.sensor_bridge.wifi_sightings_total + '</td></tr>';
        ps += '<tr><td>Zone enters</td><td>' + stats.geofence.enter_events + '</td></tr>';
        ps += '</table>';
        $('pipeStats').innerHTML = ps;

        $('statusBar').textContent = 'Last update: ' + new Date().toLocaleTimeString();
    } catch(e) {
        $('statusBar').textContent = 'Error: ' + e.message;
    }
}

function drawTargetMap(targets, zones) {
    const c = $('mapCanvas');
    const ctx = c.getContext('2d');
    const w = c.width, h = c.height;
    ctx.fillStyle = '#0a0a14';
    ctx.fillRect(0, 0, w, h);

    // Draw zones
    for (const z of zones) {
        ctx.beginPath();
        const poly = z.polygon;
        if (!poly || poly.length < 3) continue;
        const sx = poly[0][0] / 500 * w, sy = (1 - poly[0][1] / 500) * h;
        ctx.moveTo(sx, sy);
        for (let i = 1; i < poly.length; i++) {
            ctx.lineTo(poly[i][0] / 500 * w, (1 - poly[i][1] / 500) * h);
        }
        ctx.closePath();
        ctx.fillStyle = z.zone_type === 'restricted' ? '#ff2a6d11' : '#05ffa111';
        ctx.fill();
        ctx.strokeStyle = z.zone_type === 'restricted' ? '#ff2a6d44' : '#05ffa144';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Draw targets
    for (const t of targets) {
        const x = t.position.x / 500 * w;
        const y = (1 - t.position.y / 500) * h;
        const multi = t.confirming_sources.length > 1;

        ctx.beginPath();
        if (t.asset_type === 'vehicle' || t.asset_type === 'vehicle_wifi' || t.asset_type === 'keyfob') {
            ctx.rect(x - 3, y - 3, 6, 6);
            ctx.fillStyle = multi ? '#05ffa1' : '#fcee0a';
        } else {
            ctx.arc(x, y, multi ? 4 : 3, 0, Math.PI * 2);
            ctx.fillStyle = multi ? '#05ffa1' : '#00f0ff';
        }
        ctx.fill();

        if (multi) {
            ctx.strokeStyle = '#05ffa188';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(x, y, 7, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    // Legend
    ctx.fillStyle = '#555';
    ctx.font = '10px monospace';
    ctx.fillText('Cyan=BLE  Yellow=Vehicle  Green=Multi-source', 5, h - 5);
}

function drawHeatmap(data) {
    const c = $('heatCanvas');
    const ctx = c.getContext('2d');
    const w = c.width, h = c.height;
    ctx.fillStyle = '#0a0a14';
    ctx.fillRect(0, 0, w, h);

    if (!data.grid || data.max_value === 0) {
        ctx.fillStyle = '#333';
        ctx.font = '14px monospace';
        ctx.fillText('Collecting data...', w/2 - 60, h/2);
        return;
    }

    const grid = data.grid;
    const res = grid.length;
    const cellW = w / res, cellH = h / res;
    const maxVal = data.max_value;

    for (let r = 0; r < res; r++) {
        for (let co = 0; co < grid[r].length; co++) {
            const v = grid[r][co];
            if (v <= 0) continue;
            const intensity = Math.min(1, v / maxVal);
            const red = Math.floor(255 * intensity);
            const green = Math.floor(80 * (1 - intensity));
            const blue = Math.floor(200 * (1 - intensity * 0.5));
            ctx.fillStyle = 'rgb(' + red + ',' + green + ',' + blue + ')';
            ctx.globalAlpha = 0.3 + intensity * 0.7;
            ctx.fillRect(co * cellW, (res - 1 - r) * cellH, cellW + 1, cellH + 1);
        }
    }
    ctx.globalAlpha = 1;

    ctx.fillStyle = '#555';
    ctx.font = '10px monospace';
    ctx.fillText('Events: ' + data.event_count + '  Max: ' + data.max_value.toFixed(0), 5, h - 5);
}

// Refresh loop
setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tritium integrated city sim -> sensor fusion pipeline demo"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without web server, print stats to terminal",
    )
    parser.add_argument(
        "--ticks", type=int, default=200,
        help="Number of ticks for headless mode (default: 200)",
    )
    parser.add_argument(
        "--port", type=int, default=8099,
        help="Web server port (default: 8099)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.headless:
        run_headless(ticks=args.ticks, seed=args.seed)
    else:
        run_server(port=args.port, seed=args.seed)


if __name__ == "__main__":
    main()
