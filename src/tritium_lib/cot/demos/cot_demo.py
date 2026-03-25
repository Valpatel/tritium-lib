# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""CoT (Cursor on Target) interoperability demo — full TAK integration cycle.

Demonstrates the complete CoT codec pipeline:
  1. Generate TrackedTargets from multiple sensor sources
  2. Convert targets to MIL-STD-2045 CoT XML (outbound to TAK)
  3. Parse incoming CoT XML back into Tritium targets (inbound from TAK)
  4. Serve REST endpoints for live CoT event feed and ingestion

This demo proves Tritium can interoperate with military C2 systems
(ATAK, WinTAK, WebTAK) in both directions — publishing SA data
and consuming external CoT feeds.

Run with:
    PYTHONPATH=src python3 src/tritium_lib/cot/demos/cot_demo.py

Endpoints:
    GET  /                    — HTML dashboard
    GET  /cot/events          — All current targets as CoT XML events
    GET  /cot/events/{uid}    — Single target as CoT XML
    POST /cot/ingest          — Receive CoT XML, create/update targets
    GET  /api/targets         — All targets as JSON
    GET  /api/targets/{id}    — Single target as JSON
    GET  /api/stats           — Demo statistics
    GET  /api/ingest-log      — Log of ingested external CoT events
"""

from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from tritium_lib.cot.codec import device_to_cot, sensor_to_cot, parse_cot
from tritium_lib.models.cot import (
    CotEvent,
    CotPoint,
    CotContact,
    CotDetail,
    cot_to_xml,
    xml_to_cot,
)
from tritium_lib.models.tak_export import (
    CoTExportEvent,
    targets_to_cot_xml,
    targets_to_cot_file,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_PORT = 9094
TICK_INTERVAL = 3.0  # seconds between synthetic target updates

_DT_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Synthetic target scenarios — simulates a realistic operational picture
# ---------------------------------------------------------------------------

_FRIENDLY_ASSETS = [
    {
        "target_id": "rover-alpha",
        "name": "Rover Alpha",
        "alliance": "friendly",
        "asset_type": "rover",
        "lat": 37.7749,
        "lng": -122.4194,
        "speed": 2.5,
        "heading": 45.0,
        "source": "simulation",
    },
    {
        "target_id": "drone-eagle",
        "name": "Eagle Eye",
        "alliance": "friendly",
        "asset_type": "drone",
        "lat": 37.7760,
        "lng": -122.4180,
        "speed": 15.0,
        "heading": 90.0,
        "source": "simulation",
    },
    {
        "target_id": "sensor-north-01",
        "name": "North Perimeter Sensor",
        "alliance": "friendly",
        "asset_type": "sensor",
        "lat": 37.7780,
        "lng": -122.4200,
        "speed": 0.0,
        "heading": 0.0,
        "source": "mqtt",
    },
]

_DETECTED_TARGETS = [
    {
        "target_id": "det_person_001",
        "name": "Unknown Subject 1",
        "alliance": "unknown",
        "asset_type": "person",
        "lat": 37.7752,
        "lng": -122.4190,
        "speed": 1.2,
        "heading": 180.0,
        "source": "yolo",
    },
    {
        "target_id": "det_vehicle_001",
        "name": "Dark SUV",
        "alliance": "hostile",
        "asset_type": "vehicle",
        "lat": 37.7745,
        "lng": -122.4185,
        "speed": 8.0,
        "heading": 270.0,
        "source": "yolo",
    },
    {
        "target_id": "ble_aa:bb:cc:11:22:33",
        "name": "BLE Phone",
        "alliance": "unknown",
        "asset_type": "phone",
        "lat": 37.7755,
        "lng": -122.4192,
        "speed": 0.8,
        "heading": 90.0,
        "source": "ble",
    },
    {
        "target_id": "mesh_node_42",
        "name": "Mesh Relay 42",
        "alliance": "friendly",
        "asset_type": "mesh_radio",
        "lat": 37.7770,
        "lng": -122.4175,
        "speed": 0.0,
        "heading": 0.0,
        "source": "mesh",
    },
]

_EDGE_DEVICES = [
    {
        "device_id": "esp32-alpha-01",
        "lat": 37.7749,
        "lng": -122.4194,
        "capabilities": ["camera", "imu", "wifi"],
        "callsign": "Alpha-Cam",
    },
    {
        "device_id": "esp32-bravo-02",
        "lat": 37.7760,
        "lng": -122.4180,
        "capabilities": ["lora", "gps"],
        "callsign": "Bravo-Relay",
    },
]


# ---------------------------------------------------------------------------
# Demo state
# ---------------------------------------------------------------------------

class CotDemoState:
    """Manages all demo state: targets, ingested events, tick counter."""

    def __init__(self) -> None:
        self.rng = random.Random(42)
        self.tick = 0
        self.targets: dict[str, dict] = {}
        self.ingest_log: list[dict] = []
        self.generated_xml_count = 0
        self.ingested_count = 0
        self._init_targets()

    def _init_targets(self) -> None:
        """Seed initial targets."""
        for t in _FRIENDLY_ASSETS + _DETECTED_TARGETS:
            self.targets[t["target_id"]] = dict(t)

    def update_targets(self) -> dict:
        """Move targets around, simulating a live operational picture.

        Returns stats dict.
        """
        self.tick += 1
        moved = 0

        for tid, t in self.targets.items():
            speed = t.get("speed", 0.0)
            if speed <= 0:
                continue

            # Move target along heading with some drift
            heading_rad = math.radians(t.get("heading", 0.0))
            heading_rad += self.rng.uniform(-0.2, 0.2)

            # Approximate degrees per meter at this latitude
            lat_deg_per_m = 1.0 / 111_111.0
            lng_deg_per_m = 1.0 / (111_111.0 * math.cos(math.radians(t["lat"])))

            dx = speed * math.sin(heading_rad) * TICK_INTERVAL
            dy = speed * math.cos(heading_rad) * TICK_INTERVAL

            t["lat"] += dy * lat_deg_per_m
            t["lng"] += dx * lng_deg_per_m
            t["heading"] = math.degrees(heading_rad) % 360
            moved += 1

        return {"tick": self.tick, "moved": moved, "total": len(self.targets)}

    def targets_to_cot_events(self) -> list[str]:
        """Convert all current targets to CoT XML strings."""
        events = []
        for t in self.targets.values():
            evt = CoTExportEvent.from_target_dict(t)
            events.append(evt.to_xml())
            self.generated_xml_count += 1
        return events

    def target_to_cot_xml(self, target_id: str) -> str | None:
        """Convert a single target to CoT XML."""
        t = self.targets.get(target_id)
        if t is None:
            return None
        evt = CoTExportEvent.from_target_dict(t)
        self.generated_xml_count += 1
        return evt.to_xml()

    def edge_devices_to_cot(self) -> list[str]:
        """Generate CoT XML for edge devices using the low-level codec."""
        events = []
        for dev in _EDGE_DEVICES:
            xml_str = device_to_cot(
                device_id=dev["device_id"],
                lat=dev["lat"],
                lng=dev["lng"],
                capabilities=dev.get("capabilities", []),
                callsign=dev.get("callsign", ""),
            )
            events.append(xml_str)
            self.generated_xml_count += 1
        return events

    def ingest_cot_xml(self, xml_str: str) -> dict:
        """Parse incoming CoT XML and create/update a target.

        Returns the parsed target dict or an error dict.
        """
        # Try the Pydantic model parser first (richer)
        cot_event = xml_to_cot(xml_str)
        if cot_event is not None:
            return self._ingest_cot_event(cot_event, xml_str)

        # Fall back to the codec parser (lighter)
        parsed = parse_cot(xml_str)
        if parsed is not None:
            return self._ingest_parsed_dict(parsed, xml_str)

        return {"error": "Invalid CoT XML — could not parse"}

    def _ingest_cot_event(self, event: CotEvent, raw_xml: str) -> dict:
        """Create/update a target from a parsed CotEvent model."""
        uid = event.uid
        callsign = ""
        if event.detail.contact:
            callsign = event.detail.contact.callsign

        target = {
            "target_id": uid,
            "name": callsign or uid,
            "alliance": event.alliance,
            "asset_type": self._infer_asset_type(event.type),
            "lat": event.point.lat,
            "lng": event.point.lon,
            "speed": 0.0,
            "heading": 0.0,
            "source": "tak",
            "cot_type": event.type,
            "cot_how": event.how,
            "stale": event.stale.strftime(_DT_FMT),
        }

        # Extract speed/course from detail extras
        if "track" in event.detail.extra:
            track = event.detail.extra["track"]
            target["speed"] = float(track.get("speed", 0))
            target["heading"] = float(track.get("course", 0))

        self.targets[uid] = target
        self.ingested_count += 1

        log_entry = {
            "uid": uid,
            "callsign": callsign,
            "type": event.type,
            "alliance": event.alliance,
            "lat": event.point.lat,
            "lon": event.point.lon,
            "ingested_at": _utcnow().strftime(_DT_FMT),
        }
        self.ingest_log.append(log_entry)

        return {"status": "ingested", "target": target, "log": log_entry}

    def _ingest_parsed_dict(self, parsed: dict, raw_xml: str) -> dict:
        """Create/update a target from a lightweight parsed dict."""
        uid = parsed.get("uid", str(uuid.uuid4()))
        callsign = parsed.get("callsign", "")

        # Infer alliance from CoT type
        cot_type = parsed.get("type", "")
        alliance = "unknown"
        if cot_type:
            parts = cot_type.split("-")
            if len(parts) >= 2:
                affil_map = {"f": "friendly", "h": "hostile", "n": "neutral"}
                alliance = affil_map.get(parts[1], "unknown")

        target = {
            "target_id": uid,
            "name": callsign or parsed.get("device_id", uid),
            "alliance": alliance,
            "asset_type": self._infer_asset_type(cot_type),
            "lat": parsed.get("lat", 0.0),
            "lng": parsed.get("lng", 0.0),
            "speed": 0.0,
            "heading": 0.0,
            "source": "tak",
            "cot_type": cot_type,
            "cot_how": parsed.get("how", ""),
        }

        self.targets[uid] = target
        self.ingested_count += 1

        log_entry = {
            "uid": uid,
            "callsign": callsign,
            "type": cot_type,
            "alliance": alliance,
            "lat": target["lat"],
            "lon": target["lng"],
            "ingested_at": _utcnow().strftime(_DT_FMT),
        }
        self.ingest_log.append(log_entry)

        return {"status": "ingested", "target": target, "log": log_entry}

    @staticmethod
    def _infer_asset_type(cot_type: str) -> str:
        """Infer Tritium asset_type from a CoT type string."""
        if not cot_type:
            return "unknown"

        # Reverse-map from CoT type suffix to asset type
        suffix_map = {
            "G-U-C-I": "person",
            "G-U-C": "person",
            "G-E-V": "vehicle",
            "A-M-F-Q": "drone",
            "G-E-W": "turret",
            "G-E-S-C": "camera",
            "G-E-S": "sensor",
            "G-E-C-I": "computer",
            "G-U": "animal",
        }
        # Strip the alliance prefix (a-X-)
        parts = cot_type.split("-", 2)
        if len(parts) >= 3:
            suffix = parts[2]
            for pattern, asset in suffix_map.items():
                if suffix == pattern:
                    return asset
        return "unknown"


# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

state = CotDemoState()
_bg_task: asyncio.Task | None = None


async def _tick_loop() -> None:
    """Background loop: update target positions periodically."""
    while True:
        try:
            state.update_targets()
        except Exception as e:
            print(f"[CoT Demo] Tick error: {e}")
        await asyncio.sleep(TICK_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background target updates on startup."""
    global _bg_task
    _bg_task = asyncio.create_task(_tick_loop())
    print(f"CoT Demo running on http://localhost:{DEMO_PORT}")
    print(f"  {len(state.targets)} initial targets")
    print(f"  {len(_EDGE_DEVICES)} edge devices")
    print(f"  Endpoints:")
    print(f"    GET  /cot/events       — all targets as CoT XML")
    print(f"    POST /cot/ingest       — receive CoT XML")
    print(f"    GET  /api/targets      — targets as JSON")
    print(f"    GET  /                 — dashboard")
    yield
    if _bg_task:
        _bg_task.cancel()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tritium CoT Interoperability Demo",
    description="TAK-compatible CoT XML generation and ingestion",
    lifespan=lifespan,
)


@app.get("/cot/events")
async def get_cot_events():
    """Return all current targets as CoT XML events.

    This is what a TAK server would receive from Tritium.
    Returns concatenated CoT XML events.
    """
    target_events = state.targets_to_cot_events()
    edge_events = state.edge_devices_to_cot()
    all_events = target_events + edge_events
    xml_body = "\n".join(all_events)
    return Response(content=xml_body, media_type="application/xml")


@app.get("/cot/events/{target_id}")
async def get_cot_event(target_id: str):
    """Return a single target as CoT XML."""
    xml_str = state.target_to_cot_xml(target_id)
    if xml_str is None:
        return JSONResponse(
            {"error": f"Target '{target_id}' not found"},
            status_code=404,
        )
    return Response(content=xml_str, media_type="application/xml")


@app.post("/cot/ingest")
async def ingest_cot(request: Request):
    """Receive CoT XML from an external TAK system.

    Accepts a single CoT event XML in the request body.
    Parses it and creates/updates a target in the demo.
    """
    body = await request.body()
    xml_str = body.decode("utf-8", errors="replace")

    if not xml_str.strip():
        return JSONResponse({"error": "Empty body"}, status_code=400)

    result = state.ingest_cot_xml(xml_str)
    if "error" in result:
        return JSONResponse(result, status_code=400)

    return result


@app.get("/api/targets")
async def get_targets():
    """Return all targets as JSON."""
    return list(state.targets.values())


@app.get("/api/targets/{target_id}")
async def get_target(target_id: str):
    """Return a single target as JSON."""
    t = state.targets.get(target_id)
    if t is None:
        return JSONResponse({"error": "Target not found"}, status_code=404)
    return t


@app.get("/api/stats")
async def get_stats():
    """Demo statistics."""
    alliances = {}
    sources = {}
    for t in state.targets.values():
        a = t.get("alliance", "unknown")
        alliances[a] = alliances.get(a, 0) + 1
        s = t.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1

    return {
        "tick": state.tick,
        "total_targets": len(state.targets),
        "edge_devices": len(_EDGE_DEVICES),
        "alliances": alliances,
        "sources": sources,
        "generated_xml_count": state.generated_xml_count,
        "ingested_count": state.ingested_count,
        "ingest_log_size": len(state.ingest_log),
    }


@app.get("/api/ingest-log")
async def get_ingest_log():
    """Return the log of ingested external CoT events."""
    return state.ingest_log[-50:]


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tritium CoT Interop Demo</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0a; color: #c0c0c0; font-family: 'Courier New', monospace; }
h1 { color: #00f0ff; text-align: center; padding: 16px; font-size: 18px;
     text-shadow: 0 0 10px #00f0ff44; border-bottom: 1px solid #1a1a1a; }
h1 small { color: #666; font-size: 11px; display: block; margin-top: 4px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px; }
.panel { background: #111; border: 1px solid #1a1a1a; border-radius: 4px; padding: 12px; }
.panel h2 { color: #05ffa1; font-size: 13px; margin-bottom: 8px; }
.stat { display: flex; justify-content: space-between; padding: 2px 0;
        border-bottom: 1px solid #0a0a0a; font-size: 12px; }
.stat .val { color: #00f0ff; font-weight: bold; }
table { width: 100%; border-collapse: collapse; font-size: 11px; }
th { color: #ff2a6d; text-align: left; padding: 4px; border-bottom: 1px solid #222; }
td { padding: 3px 4px; border-bottom: 1px solid #111; }
tr:hover { background: #1a1a1a; }
.hostile { color: #ff2a6d; }
.friendly { color: #05ffa1; }
.unknown { color: #fcee0a; }
.neutral { color: #ffffff; }
.badge { display: inline-block; padding: 1px 6px; border-radius: 3px;
         font-size: 10px; font-weight: bold; }
.badge-tak { background: #00f0ff22; color: #00f0ff; border: 1px solid #00f0ff44; }
.badge-sim { background: #05ffa122; color: #05ffa1; border: 1px solid #05ffa144; }
.badge-yolo { background: #ff2a6d22; color: #ff2a6d; border: 1px solid #ff2a6d44; }
.badge-ble { background: #fcee0a22; color: #fcee0a; border: 1px solid #fcee0a44; }
.badge-mqtt { background: #ffffff22; color: #ffffff; border: 1px solid #ffffff44; }
.badge-mesh { background: #a855f722; color: #a855f7; border: 1px solid #a855f744; }
.fullwidth { grid-column: 1 / -1; }
pre.xml { background: #080808; padding: 8px; border: 1px solid #1a1a1a;
          border-radius: 4px; font-size: 10px; overflow-x: auto;
          max-height: 300px; overflow-y: auto; color: #05ffa1; white-space: pre-wrap; }
.btn { background: #00f0ff22; color: #00f0ff; border: 1px solid #00f0ff;
       padding: 6px 12px; cursor: pointer; border-radius: 3px; font-family: inherit;
       font-size: 11px; margin: 4px 2px; }
.btn:hover { background: #00f0ff44; }
.btn-danger { color: #ff2a6d; border-color: #ff2a6d; background: #ff2a6d22; }
.btn-danger:hover { background: #ff2a6d44; }
textarea { background: #080808; color: #c0c0c0; border: 1px solid #1a1a1a;
           border-radius: 4px; padding: 8px; font-family: 'Courier New', monospace;
           font-size: 10px; width: 100%; min-height: 100px; resize: vertical; }
#ingest-result { margin-top: 8px; font-size: 11px; padding: 4px; }
.success { color: #05ffa1; }
.error { color: #ff2a6d; }
</style>
</head>
<body>
<h1>TRITIUM CoT INTEROPERABILITY DEMO
<small>MIL-STD-2045 Cursor on Target &mdash; TAK/ATAK Bridge</small></h1>
<div class="grid">
  <div class="panel">
    <h2>STATUS</h2>
    <div id="stats">Loading...</div>
  </div>
  <div class="panel">
    <h2>INGEST CoT XML</h2>
    <textarea id="ingest-xml" placeholder="Paste CoT XML here..."></textarea>
    <div>
      <button class="btn" onclick="ingestXml()">INGEST</button>
      <button class="btn" onclick="loadSampleCot()">LOAD SAMPLE</button>
    </div>
    <div id="ingest-result"></div>
  </div>
  <div class="panel fullwidth">
    <h2>TARGETS &mdash; OPERATING PICTURE</h2>
    <table>
      <thead><tr><th>UID</th><th>Name</th><th>Alliance</th><th>Type</th>
        <th>Source</th><th>Lat</th><th>Lng</th><th>Speed</th><th>CoT</th></tr></thead>
      <tbody id="targets-body"></tbody>
    </table>
  </div>
  <div class="panel">
    <h2>CoT XML OUTPUT (Live)</h2>
    <button class="btn" onclick="fetchCotXml()">REFRESH CoT</button>
    <pre class="xml" id="cot-xml">Click REFRESH to load...</pre>
  </div>
  <div class="panel">
    <h2>INGEST LOG</h2>
    <div id="ingest-log">No ingested events yet</div>
  </div>
</div>
<script>
function sourceBadge(s) {
    const cls = {tak:'badge-tak',simulation:'badge-sim',yolo:'badge-yolo',
                 ble:'badge-ble',mqtt:'badge-mqtt',mesh:'badge-mesh'}[s] || 'badge-tak';
    return '<span class="badge ' + cls + '">' + s + '</span>';
}
function allianceClass(a) {
    return {'hostile':'hostile','friendly':'friendly','neutral':'neutral'}[a] || 'unknown';
}

async function updateStats() {
    const s = await (await fetch('/api/stats')).json();
    const lines = [
        ['Tick', s.tick],
        ['Total Targets', s.total_targets],
        ['Edge Devices', s.edge_devices],
        ['CoT XML Generated', s.generated_xml_count],
        ['CoT Ingested', s.ingested_count],
    ];
    for (const [k, v] of Object.entries(s.alliances)) {
        lines.push(['Alliance: ' + k, v]);
    }
    for (const [k, v] of Object.entries(s.sources)) {
        lines.push(['Source: ' + k, v]);
    }
    document.getElementById('stats').innerHTML = lines.map(
        ([k,v]) => '<div class="stat"><span>'+k+'</span><span class="val">'+v+'</span></div>'
    ).join('');
}

async function updateTargets() {
    const targets = await (await fetch('/api/targets')).json();
    const tbody = document.getElementById('targets-body');
    tbody.innerHTML = targets.map(t => '<tr>' +
        '<td>' + (t.target_id || '').substring(0,24) + '</td>' +
        '<td>' + (t.name || '') + '</td>' +
        '<td class="' + allianceClass(t.alliance) + '">' + t.alliance + '</td>' +
        '<td>' + (t.asset_type || '') + '</td>' +
        '<td>' + sourceBadge(t.source || '') + '</td>' +
        '<td>' + (t.lat||0).toFixed(6) + '</td>' +
        '<td>' + (t.lng||0).toFixed(6) + '</td>' +
        '<td>' + (t.speed||0).toFixed(1) + '</td>' +
        '<td>' + (t.cot_type || '') + '</td>' +
    '</tr>').join('');
}

async function fetchCotXml() {
    const r = await fetch('/cot/events');
    const xml = await r.text();
    document.getElementById('cot-xml').textContent = xml;
}

async function ingestXml() {
    const xml = document.getElementById('ingest-xml').value.trim();
    if (!xml) { alert('Paste CoT XML first'); return; }
    const r = await fetch('/cot/ingest', {method:'POST', body:xml,
        headers:{'Content-Type':'application/xml'}});
    const data = await r.json();
    const el = document.getElementById('ingest-result');
    if (data.error) {
        el.innerHTML = '<span class="error">ERROR: ' + data.error + '</span>';
    } else {
        el.innerHTML = '<span class="success">Ingested: ' +
            (data.target?.name || data.target?.target_id) +
            ' (' + (data.target?.alliance || '') + ')</span>';
    }
}

function loadSampleCot() {
    const sample = '<event version="2.0" uid="ATAK-user-bravo" type="a-f-G-U-C"' +
        ' how="h-g-i-g-o" time="' + new Date().toISOString().replace(/\\.\\d+Z/, '.000000Z') + '"' +
        ' start="' + new Date().toISOString().replace(/\\.\\d+Z/, '.000000Z') + '"' +
        ' stale="' + new Date(Date.now()+300000).toISOString().replace(/\\.\\d+Z/, '.000000Z') + '">\\n' +
        '  <point lat="37.7755" lon="-122.4188" hae="10.0" ce="5.0" le="5.0"/>\\n' +
        '  <detail>\\n' +
        '    <contact callsign="BRAVO-ACTUAL"/>\\n' +
        '    <__group name="Cyan" role="Team Lead"/>\\n' +
        '    <track speed="1.5" course="135.0"/>\\n' +
        '    <remarks>ATAK operator on foot patrol</remarks>\\n' +
        '  </detail>\\n' +
        '</event>';
    document.getElementById('ingest-xml').value = sample.replace(/\\\\n/g, '\\n');
}

async function updateIngestLog() {
    const log = await (await fetch('/api/ingest-log')).json();
    const el = document.getElementById('ingest-log');
    if (!log.length) { el.textContent = 'No ingested events yet'; return; }
    el.innerHTML = log.slice(-10).reverse().map(e =>
        '<div class="stat"><span>' + e.uid.substring(0,20) +
        ' (' + e.alliance + ')</span><span class="val">' +
        e.callsign + '</span></div>'
    ).join('');
}

async function refresh() {
    try {
        await Promise.all([updateStats(), updateTargets(), updateIngestLog()]);
    } catch(e) { console.error('Refresh error:', e); }
}

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the CoT demo dashboard."""
    return _DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Convenience: direct codec functions for programmatic use
# ---------------------------------------------------------------------------

def generate_sample_cot_event(
    uid: str = "sample-target-001",
    callsign: str = "Sample Target",
    lat: float = 37.7749,
    lon: float = -122.4194,
    alliance: str = "friendly",
    asset_type: str = "person",
) -> str:
    """Generate a sample CoT XML event for testing.

    Useful for feeding into /cot/ingest or for unit tests.
    """
    target = {
        "target_id": uid,
        "name": callsign,
        "alliance": alliance,
        "asset_type": asset_type,
        "lat": lat,
        "lng": lon,
        "speed": 0.0,
        "heading": 0.0,
        "source": "manual",
    }
    evt = CoTExportEvent.from_target_dict(target)
    return evt.to_xml()


def roundtrip_demo() -> dict:
    """Run a full CoT roundtrip: Target -> XML -> Parse -> Target.

    Returns a dict summarizing the roundtrip for each test case.
    """
    test_cases = [
        {"uid": "atak-infantry-01", "callsign": "Alpha-1", "lat": 37.7749,
         "lon": -122.4194, "alliance": "friendly", "asset_type": "person"},
        {"uid": "hostile-vehicle-01", "callsign": "Bandit-SUV", "lat": 37.7752,
         "lon": -122.4190, "alliance": "hostile", "asset_type": "vehicle"},
        {"uid": "uav-eagle-eye", "callsign": "Eagle Eye", "lat": 37.7760,
         "lon": -122.4180, "alliance": "friendly", "asset_type": "drone"},
        {"uid": "unknown-ped-01", "callsign": "Unknown Subject", "lat": 37.7745,
         "lon": -122.4185, "alliance": "unknown", "asset_type": "person"},
    ]

    results = []
    for tc in test_cases:
        # Step 1: Generate CoT XML
        xml_str = generate_sample_cot_event(**tc)

        # Step 2: Parse back
        parsed = xml_to_cot(xml_str)

        # Step 3: Verify roundtrip fidelity
        result = {
            "input": tc,
            "xml_length": len(xml_str),
            "parsed_uid": parsed.uid if parsed else None,
            "parsed_callsign": (
                parsed.detail.contact.callsign
                if parsed and parsed.detail.contact else None
            ),
            "parsed_alliance": parsed.alliance if parsed else None,
            "parsed_lat": parsed.point.lat if parsed else None,
            "parsed_lon": parsed.point.lon if parsed else None,
            "roundtrip_ok": (
                parsed is not None
                and parsed.uid == tc["uid"]
                and abs(parsed.point.lat - tc["lat"]) < 0.001
                and abs(parsed.point.lon - tc["lon"]) < 0.001
            ),
        }
        results.append(result)

    return {
        "test_count": len(results),
        "all_passed": all(r["roundtrip_ok"] for r in results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--roundtrip" in sys.argv:
        # Run roundtrip demo without server
        print("=" * 60)
        print("CoT ROUNDTRIP DEMO")
        print("=" * 60)
        result = roundtrip_demo()
        for r in result["results"]:
            status = "PASS" if r["roundtrip_ok"] else "FAIL"
            print(f"  [{status}] {r['input']['uid']}: "
                  f"{r['input']['callsign']} ({r['input']['alliance']})")
            print(f"         XML={r['xml_length']} bytes, "
                  f"parsed_uid={r['parsed_uid']}")
        print(f"\nTotal: {result['test_count']} tests, "
              f"All passed: {result['all_passed']}")
        sys.exit(0 if result["all_passed"] else 1)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEMO_PORT, log_level="warning")
