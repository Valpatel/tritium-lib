# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone tracking demo — FastAPI app exercising the full tracking pipeline.

Proves tritium-lib tracking works independently of tritium-sc.

Run with:
    PYTHONPATH=src python3 src/tritium_lib/tracking/demos/tracking_demo.py

Endpoints:
    GET /targets       — all tracked targets
    GET /target/{id}   — single target detail
    GET /heatmap       — heatmap grid data
    GET /geofence      — geofence zones and events
    GET /threats       — threat scores for all targets
    GET /correlations  — correlation records
    GET /dwells        — active dwell events
    GET /status        — pipeline status summary
    GET /              — HTML dashboard
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from tritium_lib.tracking import (
    TargetTracker,
    TargetCorrelator,
    GeofenceEngine,
    GeoZone,
    HeatmapEngine,
    ThreatScorer,
    TargetHistory,
)
from tritium_lib.tracking.dwell_tracker import DwellTracker

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

DEMO_PORT = 9091
SIGHTING_INTERVAL = 2.0  # seconds between synthetic sighting batches

_BLE_DEVICES = [
    {"mac": "AA:BB:CC:11:22:01", "name": "iPhone-Matt", "device_type": "phone"},
    {"mac": "AA:BB:CC:11:22:02", "name": "Galaxy-S24", "device_type": "phone"},
    {"mac": "DD:EE:FF:33:44:01", "name": "AirTag-Keys", "device_type": "tracker"},
    {"mac": "DD:EE:FF:33:44:02", "name": "", "device_type": "ble_device"},
    {"mac": "11:22:33:AA:BB:01", "name": "FitBit-Charge", "device_type": "wearable"},
]

_WIFI_DETECTIONS = [
    {"class_name": "person", "confidence": 0.82},
    {"class_name": "car", "confidence": 0.91},
    {"class_name": "person", "confidence": 0.76},
    {"class_name": "bicycle", "confidence": 0.68},
]


class SimpleEventBus:
    """Minimal event bus for the demo."""

    def __init__(self) -> None:
        self._log: list[tuple[str, dict]] = []

    def publish(self, topic: str, data: dict) -> None:
        self._log.append((topic, data))

    @property
    def events(self) -> list[tuple[str, dict]]:
        return list(self._log)

    @property
    def event_count(self) -> int:
        return len(self._log)


class TrackingPipeline:
    """Encapsulates the full tracking pipeline for the demo."""

    def __init__(self) -> None:
        self.bus = SimpleEventBus()
        self.rng = random.Random(42)
        self.tick = 0

        # Core components
        self.tracker = TargetTracker(event_bus=self.bus)
        self.geofence = GeofenceEngine(event_bus=self.bus)
        self.tracker.set_geofence_engine(self.geofence)
        self.heatmap = HeatmapEngine()
        self.correlator = TargetCorrelator(
            self.tracker,
            radius=15.0,
            confidence_threshold=0.25,
        )
        self.dwell_tracker = DwellTracker(
            event_bus=self.bus,
            target_tracker=self.tracker,
            threshold_s=10.0,   # lower threshold for demo
            radius_m=8.0,
        )

        def geofence_check(tid: str, pos: tuple[float, float]) -> bool:
            return len(self.geofence.check(tid, pos)) > 0

        self.threat_scorer = ThreatScorer(geofence_checker=geofence_check)
        self.correlation_log: list[dict] = []

        # Set up geofence zones
        self.geofence.add_zone(GeoZone(
            zone_id="restricted-hq",
            name="HQ Restricted Area",
            polygon=[(20, 20), (80, 20), (80, 80), (20, 80)],
            zone_type="restricted",
        ))
        self.geofence.add_zone(GeoZone(
            zone_id="parking-lot",
            name="Parking Lot",
            polygon=[(100, 0), (200, 0), (200, 60), (100, 60)],
            zone_type="monitored",
        ))
        self.geofence.add_zone(GeoZone(
            zone_id="perimeter-north",
            name="North Perimeter",
            polygon=[(-50, 80), (250, 80), (250, 120), (-50, 120)],
            zone_type="restricted",
        ))

    def generate_sightings(self) -> dict:
        """Generate one batch of synthetic sightings. Returns stats."""
        self.tick += 1
        stats = {"ble": 0, "yolo": 0, "correlations": 0, "threats": 0}

        # BLE sightings — devices wander around the map
        for dev in _BLE_DEVICES:
            # Each device has a home position and wanders
            idx = _BLE_DEVICES.index(dev)
            home_x = 30.0 + idx * 35.0
            home_y = 40.0 + idx * 15.0
            angle = self.tick * 0.3 + idx * 1.2
            radius = 10.0 + 5.0 * math.sin(self.tick * 0.1 + idx)
            x = home_x + radius * math.cos(angle)
            y = home_y + radius * math.sin(angle)
            rssi = self.rng.randint(-75, -30)

            self.tracker.update_from_ble({
                "mac": dev["mac"],
                "name": dev["name"],
                "rssi": rssi,
                "device_type": dev["device_type"],
                "node_position": {"x": x, "y": y},
            })
            self.heatmap.record_event("ble_activity", x, y)
            stats["ble"] += 1

        # YOLO camera detections — different source, some near BLE positions
        for det in _WIFI_DETECTIONS:
            # Place detections near some BLE positions to enable correlation
            base_idx = self.rng.randint(0, len(_BLE_DEVICES) - 1)
            home_x = 30.0 + base_idx * 35.0
            home_y = 40.0 + base_idx * 15.0
            # Add noise — some close enough to correlate, some not
            offset = self.rng.uniform(2.0, 20.0)
            angle = self.rng.uniform(0, 2 * math.pi)
            cx = home_x + offset * math.cos(angle)
            cy = home_y + offset * math.sin(angle)

            self.tracker.update_from_detection({
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "center_x": cx,
                "center_y": cy,
            })
            self.heatmap.record_event("camera_activity", cx, cy)
            stats["yolo"] += 1

        # Run correlator — fuse multi-source sightings
        new_correlations = self.correlator.correlate()
        for rec in new_correlations:
            self.correlation_log.append({
                "primary_id": rec.primary_id,
                "secondary_id": rec.secondary_id,
                "confidence": round(rec.confidence, 3),
                "reason": rec.reason,
                "dossier_uuid": rec.dossier_uuid,
            })
        stats["correlations"] = len(new_correlations)

        # Run threat scorer
        all_targets = self.tracker.get_all()
        target_dicts = []
        for t in all_targets:
            target_dicts.append({
                "target_id": t.target_id,
                "position": t.position,
                "heading": t.heading,
                "speed": t.speed,
                "source": t.source,
                "alliance": t.alliance,
            })
        scores = self.threat_scorer.evaluate(target_dicts)
        stats["threats"] = sum(1 for s in scores.values() if s > 0.1)

        return stats


# Singleton pipeline
pipeline = TrackingPipeline()
_bg_task: asyncio.Task | None = None


async def _sighting_loop() -> None:
    """Background loop generating synthetic sightings every N seconds."""
    while True:
        try:
            pipeline.generate_sightings()
        except Exception as e:
            print(f"Sighting generation error: {e}")
        await asyncio.sleep(SIGHTING_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background sighting generation on startup."""
    global _bg_task
    # Start dwell tracker thread
    pipeline.dwell_tracker.start()
    # Start async sighting loop
    _bg_task = asyncio.create_task(_sighting_loop())
    print(f"Tracking demo running on http://localhost:{DEMO_PORT}")
    print(f"  Generating sightings every {SIGHTING_INTERVAL}s")
    print(f"  {len(_BLE_DEVICES)} BLE devices, {len(_WIFI_DETECTIONS)} YOLO sources")
    yield
    _bg_task.cancel()
    pipeline.dwell_tracker.stop()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tritium Tracking Demo",
    description="Standalone tracking pipeline demo — no SC dependency",
    lifespan=lifespan,
)


def _target_to_dict(t) -> dict:
    """Convert a TrackedTarget to a JSON-safe dict without geo dependencies."""
    return {
        "target_id": t.target_id,
        "name": t.name,
        "alliance": t.alliance,
        "asset_type": t.asset_type,
        "position": {"x": t.position[0], "y": t.position[1]},
        "heading": round(t.heading, 1),
        "speed": round(t.speed, 2),
        "source": t.source,
        "signal_count": t.signal_count,
        "position_confidence": round(t.effective_confidence, 3),
        "threat_score": round(t.threat_score, 3),
        "confirming_sources": list(t.confirming_sources),
        "correlated_ids": list(t.correlated_ids),
        "correlation_confidence": round(t.correlation_confidence, 3),
        "classification": t.classification,
    }


@app.get("/targets")
async def get_targets():
    """Return all tracked targets."""
    targets = pipeline.tracker.get_all()
    return [_target_to_dict(t) for t in targets]


@app.get("/target/{target_id}")
async def get_target(target_id: str):
    """Return a single target by ID."""
    t = pipeline.tracker.get_target(target_id)
    if t is None:
        return JSONResponse({"error": "Target not found"}, status_code=404)
    profile = pipeline.threat_scorer.get_profile(target_id)
    trail = pipeline.tracker.history.get_trail(target_id, max_points=20)
    data = _target_to_dict(t)
    data["trail"] = [{"x": p[0], "y": p[1], "t": p[2]} for p in trail]
    data["threat_profile"] = profile
    data["zones"] = list(pipeline.geofence.get_target_zones(target_id))
    return data


@app.get("/heatmap")
async def get_heatmap(resolution: int = 20, minutes: int = 60, layer: str = "all"):
    """Return heatmap grid data."""
    return pipeline.heatmap.get_heatmap(
        time_window_minutes=minutes,
        resolution=resolution,
        layer=layer,
    )


@app.get("/geofence")
async def get_geofence():
    """Return geofence zones and recent events."""
    zones = pipeline.geofence.list_zones()
    events = pipeline.geofence.get_events(limit=50)
    return {
        "zones": [z.to_dict() for z in zones],
        "events": [e.to_dict() for e in events],
    }


@app.get("/threats")
async def get_threats():
    """Return threat scores and behavioral profiles for all targets."""
    profiles = pipeline.threat_scorer.get_all_profiles()
    return {
        "profiles": profiles,
        "status": pipeline.threat_scorer.get_status(),
    }


@app.get("/correlations")
async def get_correlations():
    """Return correlation records from the multi-strategy correlator."""
    # Also include dossier store contents
    dossiers = pipeline.correlator.dossier_store.get_all()
    return {
        "correlations": pipeline.correlation_log[-100:],
        "total": len(pipeline.correlation_log),
        "dossiers": [
            {
                "uuid": d.uuid,
                "signal_ids": d.signal_ids,
                "sources": d.sources,
                "confidence": round(d.confidence, 3),
                "correlation_count": d.correlation_count,
            }
            for d in dossiers
        ],
    }


@app.get("/dwells")
async def get_dwells():
    """Return active dwell events from the DwellTracker."""
    active = pipeline.dwell_tracker.active_dwells
    history = pipeline.dwell_tracker.history
    return {
        "active": [d.model_dump() for d in active],
        "history": [d.model_dump() for d in history[-20:]],
    }


@app.get("/status")
async def get_status():
    """Pipeline status summary."""
    targets = pipeline.tracker.get_all()
    return {
        "tick": pipeline.tick,
        "total_targets": len(targets),
        "ble_targets": sum(1 for t in targets if t.source == "ble"),
        "yolo_targets": sum(1 for t in targets if t.source == "yolo"),
        "hostiles": len(pipeline.tracker.get_hostiles()),
        "friendlies": len(pipeline.tracker.get_friendlies()),
        "geofence_zones": len(pipeline.geofence.list_zones()),
        "geofence_events": len(pipeline.geofence.get_events(limit=9999)),
        "heatmap_events": pipeline.heatmap.event_count(),
        "correlations": len(pipeline.correlation_log),
        "active_dwells": len(pipeline.dwell_tracker.active_dwells),
        "event_bus_events": pipeline.bus.event_count,
        "threat_status": pipeline.threat_scorer.get_status(),
    }


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tritium Tracking Demo</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0a; color: #c0c0c0; font-family: 'Courier New', monospace; }
h1 { color: #00f0ff; text-align: center; padding: 16px; font-size: 20px;
     text-shadow: 0 0 10px #00f0ff44; border-bottom: 1px solid #1a1a1a; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px; }
.panel { background: #111; border: 1px solid #1a1a1a; border-radius: 4px; padding: 12px; }
.panel h2 { color: #05ffa1; font-size: 13px; margin-bottom: 8px; }
.stat { display: flex; justify-content: space-between; padding: 2px 0;
        border-bottom: 1px solid #0a0a0a; font-size: 12px; }
.stat .val { color: #00f0ff; font-weight: bold; }
canvas { width: 100%; height: 300px; background: #080808; border: 1px solid #1a1a1a;
         border-radius: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 11px; }
th { color: #ff2a6d; text-align: left; padding: 4px; border-bottom: 1px solid #222; }
td { padding: 3px 4px; border-bottom: 1px solid #111; }
tr:hover { background: #1a1a1a; }
.hostile { color: #ff2a6d; }
.friendly { color: #05ffa1; }
.unknown { color: #fcee0a; }
.badge { display: inline-block; padding: 1px 6px; border-radius: 3px;
         font-size: 10px; font-weight: bold; }
.badge-ble { background: #00f0ff22; color: #00f0ff; border: 1px solid #00f0ff44; }
.badge-yolo { background: #ff2a6d22; color: #ff2a6d; border: 1px solid #ff2a6d44; }
.badge-sim { background: #05ffa122; color: #05ffa1; border: 1px solid #05ffa144; }
#heatmap-canvas { image-rendering: pixelated; }
.fullwidth { grid-column: 1 / -1; }
</style>
</head>
<body>
<h1>TRITIUM TRACKING PIPELINE DEMO</h1>
<div class="grid">
  <div class="panel">
    <h2>STATUS</h2>
    <div id="status">Loading...</div>
  </div>
  <div class="panel">
    <h2>HEATMAP</h2>
    <canvas id="heatmap-canvas" width="300" height="300"></canvas>
  </div>
  <div class="panel fullwidth">
    <h2>TARGETS</h2>
    <table>
      <thead><tr><th>ID</th><th>Name</th><th>Alliance</th><th>Source</th>
        <th>Position</th><th>Signals</th><th>Confidence</th><th>Threat</th></tr></thead>
      <tbody id="targets-body"></tbody>
    </table>
  </div>
  <div class="panel">
    <h2>CORRELATIONS</h2>
    <div id="correlations">None yet</div>
  </div>
  <div class="panel">
    <h2>GEOFENCE EVENTS</h2>
    <div id="geofence-events">None yet</div>
  </div>
</div>
<script>
const API = '';

async function fetchJSON(url) {
    const r = await fetch(url);
    return r.json();
}

function allianceClass(a) {
    if (a === 'hostile') return 'hostile';
    if (a === 'friendly') return 'friendly';
    return 'unknown';
}

function sourceBadge(s) {
    const cls = s === 'ble' ? 'badge-ble' : s === 'yolo' ? 'badge-yolo' : 'badge-sim';
    return `<span class="badge ${cls}">${s}</span>`;
}

async function updateStatus() {
    const s = await fetchJSON('/status');
    document.getElementById('status').innerHTML = [
        ['Tick', s.tick],
        ['Total Targets', s.total_targets],
        ['BLE Targets', s.ble_targets],
        ['YOLO Targets', s.yolo_targets],
        ['Hostiles', s.hostiles],
        ['Geofence Zones', s.geofence_zones],
        ['Geofence Events', s.geofence_events],
        ['Heatmap Events', s.heatmap_events],
        ['Correlations', s.correlations],
        ['Active Dwells', s.active_dwells],
        ['Bus Events', s.event_bus_events],
    ].map(([k,v]) => `<div class="stat"><span>${k}</span><span class="val">${v}</span></div>`).join('');
}

async function updateTargets() {
    const targets = await fetchJSON('/targets');
    const tbody = document.getElementById('targets-body');
    tbody.innerHTML = targets.slice(0, 30).map(t => `<tr>
        <td>${t.target_id.substring(0,20)}</td>
        <td>${t.name}</td>
        <td class="${allianceClass(t.alliance)}">${t.alliance}</td>
        <td>${sourceBadge(t.source)}</td>
        <td>${t.position.x.toFixed(1)}, ${t.position.y.toFixed(1)}</td>
        <td>${t.signal_count}</td>
        <td>${(t.position_confidence * 100).toFixed(0)}%</td>
        <td>${(t.threat_score * 100).toFixed(0)}%</td>
    </tr>`).join('');
}

async function updateHeatmap() {
    const data = await fetchJSON('/heatmap?resolution=30&minutes=60&layer=all');
    const canvas = document.getElementById('heatmap-canvas');
    const ctx = canvas.getContext('2d');
    const g = data.grid;
    const res = g.length;
    if (!res) return;
    canvas.width = res;
    canvas.height = res;
    const maxV = data.max_value || 1;
    for (let r = 0; r < res; r++) {
        for (let c = 0; c < res; c++) {
            const v = Math.min(1, g[r][c] / maxV);
            if (v < 0.01) {
                ctx.fillStyle = '#080808';
            } else {
                const r2 = Math.floor(v * 255);
                const g2 = Math.floor((1 - v) * 80);
                const b2 = Math.floor((1 - v) * 255);
                ctx.fillStyle = `rgb(${r2},${g2},${b2})`;
            }
            ctx.fillRect(c, r, 1, 1);
        }
    }
}

async function updateCorrelations() {
    const data = await fetchJSON('/correlations');
    const el = document.getElementById('correlations');
    if (!data.correlations.length) { el.textContent = 'No correlations yet'; return; }
    el.innerHTML = data.correlations.slice(-10).reverse().map(c =>
        `<div class="stat"><span>${c.primary_id.substring(0,16)} + ${c.secondary_id.substring(0,16)}</span><span class="val">${(c.confidence*100).toFixed(0)}%</span></div>`
    ).join('');
}

async function updateGeofence() {
    const data = await fetchJSON('/geofence');
    const el = document.getElementById('geofence-events');
    if (!data.events.length) { el.textContent = 'No events yet'; return; }
    el.innerHTML = data.events.slice(0, 10).map(e =>
        `<div class="stat"><span>${e.event_type.toUpperCase()} ${e.target_id.substring(0,16)} -> ${e.zone_name}</span><span class="val">${new Date(e.timestamp*1000).toLocaleTimeString()}</span></div>`
    ).join('');
}

async function refresh() {
    try {
        await Promise.all([updateStatus(), updateTargets(), updateHeatmap(),
                           updateCorrelations(), updateGeofence()]);
    } catch(e) { console.error('Refresh error:', e); }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the tracking demo dashboard."""
    return _DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEMO_PORT, log_level="warning")
