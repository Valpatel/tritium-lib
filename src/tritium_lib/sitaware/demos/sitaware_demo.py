# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Situational Awareness Engine — THE showcase demo.

Runs the full SitAwareEngine with synthetic multi-sensor data showing the
complete pipeline: tracking -> fusion -> intelligence -> alerting -> reporting.

Run with:
    PYTHONPATH=src python3 src/tritium_lib/sitaware/demos/sitaware_demo.py

Endpoints:
    GET /              — HTML dashboard (cyberpunk themed)
    GET /picture       — Full operating picture JSON
    GET /targets       — All fused targets
    GET /alerts        — Recent alerts
    GET /updates       — Delta updates since timestamp
    GET /stats         — Engine statistics
    GET /health        — System health status
    GET /incidents     — Active incidents
    GET /missions      — Active missions
    GET /target/{id}   — Single target dossier
    POST /reset        — Reset the engine and restart simulation
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from tritium_lib.sitaware import SitAwareEngine, OperatingPicture, UpdateType
from tritium_lib.tracking.geofence import GeoZone
from tritium_lib.incident import IncidentSeverity
from tritium_lib.mission import MissionPlanner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEMO_PORT = 9095
TICK_INTERVAL = 1.5  # seconds between simulation ticks
SCENARIO_NAME = "Tritium SitAware Full-Stack Demo"

# Synthetic BLE devices
_BLE_DEVICES = [
    {"mac": "AA:BB:CC:11:22:01", "name": "iPhone-Alpha", "device_type": "phone"},
    {"mac": "AA:BB:CC:11:22:02", "name": "Galaxy-S24", "device_type": "phone"},
    {"mac": "DD:EE:FF:33:44:01", "name": "AirTag-Keys", "device_type": "tracker"},
    {"mac": "DD:EE:FF:33:44:02", "name": "FitBit-Charge5", "device_type": "wearable"},
    {"mac": "11:22:33:AA:BB:01", "name": "BLE-Beacon-01", "device_type": "beacon"},
]

# Synthetic WiFi probes
_WIFI_PROBES = [
    {"mac": "FF:00:11:22:33:01", "ssid": "CompanyWiFi", "rssi": -55},
    {"mac": "FF:00:11:22:33:02", "ssid": "GuestNetwork", "rssi": -68},
    {"mac": "AA:BB:CC:11:22:01", "ssid": "HomeWiFi", "rssi": -42},  # Same MAC as BLE phone
]

# Synthetic camera detections
_CAMERA_DETECTIONS = [
    {"class_name": "person", "confidence": 0.92},
    {"class_name": "car", "confidence": 0.88},
    {"class_name": "person", "confidence": 0.79},
    {"class_name": "motorcycle", "confidence": 0.71},
]

# Acoustic events
_ACOUSTIC_EVENTS = [
    {"event_type": "vehicle_engine", "sensor_id": "mic-north", "confidence": 0.85},
    {"event_type": "voice", "sensor_id": "mic-east", "confidence": 0.72},
    {"event_type": "footsteps", "sensor_id": "mic-south", "confidence": 0.65},
]


# ---------------------------------------------------------------------------
# Scenario simulation
# ---------------------------------------------------------------------------

class ScenarioSimulator:
    """Generates synthetic multi-sensor data with three scenario threads:

    1. Normal traffic   — BLE devices, WiFi probes, camera detections wandering
    2. Hostile approach  — a single hostile entity approaching the restricted zone
    3. Vehicle convoy    — three vehicles moving in formation through the area
    """

    def __init__(self, engine: SitAwareEngine, seed: int = 42) -> None:
        self.engine = engine
        self.rng = random.Random(seed)
        self.tick = 0
        self._running = False
        self._task: asyncio.Task | None = None

        # Hostile approach state
        self._hostile_x = -40.0
        self._hostile_y = 50.0

        # Convoy state (3 vehicles in formation)
        self._convoy_start_x = -60.0
        self._convoy_y = 30.0

        # Setup geofence zones
        self._setup_zones()

    def _setup_zones(self) -> None:
        """Create geofence zones for the scenario."""
        # Restricted HQ zone
        self.engine.fusion.add_zone(GeoZone(
            zone_id="restricted-hq",
            name="HQ Restricted Area",
            polygon=[(40, 40), (80, 40), (80, 80), (40, 80)],
            zone_type="restricted",
            alert_on_enter=True,
            alert_on_exit=True,
        ))

        # Monitored parking lot
        self.engine.fusion.add_zone(GeoZone(
            zone_id="parking-lot",
            name="Parking Lot",
            polygon=[(100, 0), (200, 0), (200, 50), (100, 50)],
            zone_type="monitored",
        ))

        # North perimeter
        self.engine.fusion.add_zone(GeoZone(
            zone_id="perimeter-north",
            name="North Perimeter",
            polygon=[(-60, 85), (220, 85), (220, 120), (-60, 120)],
            zone_type="restricted",
        ))

        # South checkpoint
        self.engine.fusion.add_zone(GeoZone(
            zone_id="checkpoint-south",
            name="South Checkpoint",
            polygon=[(-10, -10), (50, -10), (50, 10), (-10, 10)],
            zone_type="monitored",
        ))

    def _generate_normal_traffic(self) -> dict[str, int]:
        """Simulate normal BLE, WiFi, and camera traffic."""
        stats = {"ble": 0, "wifi": 0, "camera": 0, "acoustic": 0}

        # BLE devices wandering around
        for i, dev in enumerate(_BLE_DEVICES):
            home_x = 20.0 + i * 30.0
            home_y = 25.0 + i * 10.0
            angle = self.tick * 0.25 + i * 1.5
            radius = 8.0 + 4.0 * math.sin(self.tick * 0.08 + i)
            x = home_x + radius * math.cos(angle)
            y = home_y + radius * math.sin(angle)
            rssi = self.rng.randint(-75, -30)

            self.engine.fusion.ingest_ble({
                "mac": dev["mac"],
                "name": dev["name"],
                "rssi": rssi,
                "device_type": dev["device_type"],
                "position": {"x": x, "y": y},
                "node_position": {"x": x, "y": y},
            })
            stats["ble"] += 1

        # WiFi probes (every other tick)
        if self.tick % 2 == 0:
            for probe in _WIFI_PROBES:
                px = self.rng.uniform(0, 180)
                py = self.rng.uniform(0, 100)
                self.engine.fusion.ingest_wifi({
                    **probe,
                    "position": {"x": px, "y": py},
                })
                stats["wifi"] += 1

        # Camera detections (every 3rd tick)
        if self.tick % 3 == 0:
            for det in _CAMERA_DETECTIONS:
                cx = self.rng.uniform(10, 190)
                cy = self.rng.uniform(5, 95)
                self.engine.fusion.ingest_camera({
                    **det,
                    "center_x": cx,
                    "center_y": cy,
                })
                stats["camera"] += 1

        # Acoustic events (every 5th tick)
        if self.tick % 5 == 0:
            evt = self.rng.choice(_ACOUSTIC_EVENTS)
            ax = self.rng.uniform(0, 150)
            ay = self.rng.uniform(0, 80)
            self.engine.fusion.ingest_acoustic({
                **evt,
                "position": {"x": ax, "y": ay},
            })
            stats["acoustic"] += 1

        return stats

    def _generate_hostile_approach(self) -> dict[str, Any]:
        """Simulate a hostile target approaching from the west toward HQ."""
        # Hostile moves east toward the restricted zone center (60, 60)
        speed = 1.2 + 0.3 * math.sin(self.tick * 0.15)
        self._hostile_x += speed
        self._hostile_y += 0.3 * math.sin(self.tick * 0.2)

        # BLE signal from the hostile's phone
        self.engine.fusion.ingest_ble({
            "mac": "66:66:66:66:66:01",
            "name": "Unknown-Device",
            "rssi": self.rng.randint(-80, -50),
            "device_type": "phone",
            "position": {"x": self._hostile_x, "y": self._hostile_y},
            "node_position": {"x": self._hostile_x, "y": self._hostile_y},
            "classification": "hostile",
        })

        # Camera picks it up once it enters visual range (x > 0)
        if self._hostile_x > 0:
            self.engine.fusion.ingest_camera({
                "class_name": "person",
                "confidence": 0.87,
                "center_x": self._hostile_x,
                "center_y": self._hostile_y,
            })

        # Acoustic footstep detection when close (x > 20)
        if self._hostile_x > 20 and self.tick % 3 == 0:
            self.engine.fusion.ingest_acoustic({
                "event_type": "footsteps",
                "sensor_id": "mic-west",
                "confidence": 0.78,
                "position": {"x": self._hostile_x + 2, "y": self._hostile_y},
            })

        # Feed anomaly engine observations for the zone
        if self._hostile_x > 30:
            self.engine.anomaly.observe(
                "restricted-hq",
                target_id="ble_666666666601",
                speed=speed * 10,
                dwell_seconds=self.tick * TICK_INTERVAL,
            )

        # Reset hostile position after it crosses through
        if self._hostile_x > 120:
            self._hostile_x = -40.0
            self._hostile_y = 50.0

        return {
            "hostile_x": round(self._hostile_x, 1),
            "hostile_y": round(self._hostile_y, 1),
            "in_zone": 40 <= self._hostile_x <= 80 and 40 <= self._hostile_y <= 80,
        }

    def _generate_convoy(self) -> dict[str, Any]:
        """Simulate a 3-vehicle convoy moving through the area."""
        convoy_speed = 2.0
        self._convoy_start_x += convoy_speed
        convoy_data = []

        for v_idx in range(3):
            vx = self._convoy_start_x - v_idx * 12.0  # 12m spacing
            vy = self._convoy_y + 2.0 * math.sin(self.tick * 0.1 + v_idx)

            mac = f"CC:CC:CC:00:00:0{v_idx + 1}"
            self.engine.fusion.ingest_ble({
                "mac": mac,
                "name": f"Vehicle-{v_idx + 1}",
                "rssi": self.rng.randint(-60, -35),
                "device_type": "vehicle",
                "position": {"x": vx, "y": vy},
                "node_position": {"x": vx, "y": vy},
            })

            # Camera detects vehicles
            if 0 < vx < 200:
                self.engine.fusion.ingest_camera({
                    "class_name": "car",
                    "confidence": 0.91,
                    "center_x": vx,
                    "center_y": vy,
                })

            # Engine noise
            if self.tick % 4 == 0 and v_idx == 0:
                self.engine.fusion.ingest_acoustic({
                    "event_type": "vehicle_engine",
                    "sensor_id": "mic-road",
                    "confidence": 0.82,
                    "position": {"x": vx, "y": vy},
                })

            convoy_data.append({"x": round(vx, 1), "y": round(vy, 1)})

        # Reset convoy when lead vehicle passes through
        if self._convoy_start_x > 220:
            self._convoy_start_x = -60.0

        return {"vehicles": convoy_data, "speed_mps": convoy_speed}

    def _build_anomaly_baselines(self) -> None:
        """Feed enough observations to build anomaly baselines."""
        zones = ["restricted-hq", "parking-lot", "perimeter-north", "checkpoint-south"]
        for zone_id in zones:
            for i in range(20):
                self.engine.anomaly.observe(
                    zone_id,
                    target_id=f"baseline_{i}",
                    speed=self.rng.uniform(0.5, 3.0),
                    dwell_seconds=self.rng.uniform(10, 120),
                    entity_count=self.rng.uniform(1, 8),
                    hour_of_day=self.rng.randint(6, 22),
                )

    def _create_demo_incident(self) -> None:
        """Create a sample incident for the demo picture."""
        self.engine.incidents.create(
            title="Unauthorized access attempt at HQ perimeter",
            severity=IncidentSeverity.HIGH,
            source="sitaware_demo",
            target_ids=["ble_666666666601"],
            zone_id="restricted-hq",
            description="Hostile target detected approaching restricted zone.",
        )

    def _create_demo_mission(self) -> None:
        """Create a sample surveillance mission."""
        self.engine.missions.create_mission(
            name="Perimeter Surveillance Alpha",
            mission_type="surveillance",
            priority="high",
            description="Continuous monitoring of HQ restricted zone and north perimeter.",
            created_by="sitaware_demo",
        )

    def simulate_tick(self) -> dict[str, Any]:
        """Run one simulation tick. Returns tick stats."""
        self.tick += 1
        traffic_stats = self._generate_normal_traffic()
        hostile_status = self._generate_hostile_approach()
        convoy_status = self._generate_convoy()

        # Trigger alert evaluation via geofence checks on hostile
        if hostile_status["in_zone"]:
            self.engine.alerting.evaluate_event("geofence:enter", {
                "target_id": "ble_666666666601",
                "zone_id": "restricted-hq",
                "zone_type": "restricted",
                "zone_name": "HQ Restricted Area",
            })

        # Check anomaly for hostile if in zone
        if self._hostile_x > 30:
            self.engine.anomaly.check_target(
                "restricted-hq",
                target_id="ble_666666666601",
                speed=15.0,  # fast — anomalous
                dwell_seconds=self.tick * TICK_INTERVAL,
            )

        return {
            "tick": self.tick,
            "traffic": traffic_stats,
            "hostile": hostile_status,
            "convoy": convoy_status,
            "timestamp": time.time(),
        }

    async def run_loop(self) -> None:
        """Async simulation loop."""
        self._running = True
        # Build baselines first
        self._build_anomaly_baselines()
        # Create demo incident and mission
        self._create_demo_incident()
        self._create_demo_mission()

        while self._running:
            self.simulate_tick()
            await asyncio.sleep(TICK_INTERVAL)

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start the async simulation loop."""
        self._task = asyncio.ensure_future(self.run_loop())

    def stop(self) -> None:
        """Stop the simulation loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_engine: SitAwareEngine | None = None
_simulator: ScenarioSimulator | None = None
_update_log: list[dict] = []


def _get_engine() -> SitAwareEngine:
    global _engine
    if _engine is None:
        _engine = SitAwareEngine(alert_window=600.0, anomaly_window=600.0)
        _engine.start()
        # Subscribe to collect updates
        _engine.subscribe(_on_update)
    return _engine


def _on_update(update) -> None:
    """Collect updates for the /updates endpoint."""
    global _update_log
    _update_log.append(update.to_dict())
    # Keep only last 500 updates
    if len(_update_log) > 500:
        _update_log = _update_log[-500:]


def _get_simulator() -> ScenarioSimulator:
    global _simulator
    if _simulator is None:
        _simulator = ScenarioSimulator(_get_engine())
    return _simulator


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start simulation on startup, stop on shutdown."""
    engine = _get_engine()
    sim = _get_simulator()
    sim.start()
    yield
    sim.stop()
    engine.shutdown()


app = FastAPI(
    title="Tritium SitAware Demo",
    description="Full situational awareness engine demonstration",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/picture", response_class=JSONResponse)
async def get_picture():
    """Full operating picture — the unified view of everything."""
    engine = _get_engine()
    picture = engine.get_picture()
    return picture.to_dict()


@app.get("/targets", response_class=JSONResponse)
async def get_targets():
    """All fused targets with sensor context."""
    engine = _get_engine()
    targets = engine.fusion.get_fused_targets()
    return {
        "targets": [t.to_dict() for t in targets],
        "count": len(targets),
        "timestamp": time.time(),
    }


@app.get("/target/{target_id}", response_class=JSONResponse)
async def get_target(target_id: str):
    """Detailed view for a single target (dossier + alerts + anomalies)."""
    engine = _get_engine()
    result = engine.get_target_picture(target_id)
    if result is None:
        return JSONResponse(
            {"error": f"Target {target_id} not found"}, status_code=404,
        )
    return result


@app.get("/alerts", response_class=JSONResponse)
async def get_alerts():
    """Recent alert records."""
    engine = _get_engine()
    history = engine.alerting.get_history(limit=100)
    return {
        "alerts": [a.to_dict() for a in history],
        "count": len(history),
        "timestamp": time.time(),
    }


@app.get("/updates", response_class=JSONResponse)
async def get_updates(since: float = Query(0.0, description="Epoch timestamp")):
    """Delta updates since a given timestamp."""
    engine = _get_engine()
    updates = engine.get_updates_since(since)
    return {
        "updates": [u.to_dict() for u in updates],
        "count": len(updates),
        "since": since,
        "timestamp": time.time(),
    }


@app.get("/stats", response_class=JSONResponse)
async def get_stats():
    """Engine-wide statistics from all subsystems."""
    engine = _get_engine()
    return engine.get_stats()


@app.get("/health", response_class=JSONResponse)
async def get_health():
    """System health status."""
    engine = _get_engine()
    return engine.health.check_all().to_dict()


@app.get("/incidents", response_class=JSONResponse)
async def get_incidents():
    """Active incidents."""
    engine = _get_engine()
    incidents = engine.incidents.get_all()
    return {
        "incidents": [inc.to_dict() for inc in incidents],
        "count": len(incidents),
        "timestamp": time.time(),
    }


@app.get("/missions", response_class=JSONResponse)
async def get_missions():
    """Active missions."""
    engine = _get_engine()
    missions = engine.missions.get_missions()
    return {
        "missions": [m.to_dict() for m in missions],
        "count": len(missions),
        "timestamp": time.time(),
    }


@app.post("/reset", response_class=JSONResponse)
async def reset_demo():
    """Reset the engine and restart simulation."""
    global _engine, _simulator, _update_log
    if _simulator:
        _simulator.stop()
    if _engine:
        _engine.shutdown()
    _engine = None
    _simulator = None
    _update_log = []
    engine = _get_engine()
    sim = _get_simulator()
    sim.start()
    return {"status": "reset", "timestamp": time.time()}


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Cyberpunk-themed HTML dashboard showing the full operating picture."""
    return HTMLResponse(_build_dashboard_html())


def _build_dashboard_html() -> str:
    """Build the full HTML dashboard."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TRITIUM // Situational Awareness Engine</title>
<style>
{_DASHBOARD_CSS}
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="header-left">
      <h1>TRITIUM <span class="accent-dim">// SitAware Engine</span></h1>
      <div id="summary" class="summary">Initializing...</div>
    </div>
    <div class="header-right">
      <div id="threat-badge" class="threat-badge threat-green">GREEN</div>
      <div id="clock" class="clock"></div>
    </div>
  </header>

  <div class="stats-bar" id="stats-bar">
    <div class="stat"><span class="stat-val" id="stat-targets">0</span><span class="stat-label">TARGETS</span></div>
    <div class="stat"><span class="stat-val" id="stat-multi">0</span><span class="stat-label">MULTI-SRC</span></div>
    <div class="stat"><span class="stat-val" id="stat-alerts">0</span><span class="stat-label">ALERTS</span></div>
    <div class="stat"><span class="stat-val" id="stat-anomalies">0</span><span class="stat-label">ANOMALIES</span></div>
    <div class="stat"><span class="stat-val" id="stat-incidents">0</span><span class="stat-label">INCIDENTS</span></div>
    <div class="stat"><span class="stat-val" id="stat-missions">0</span><span class="stat-label">MISSIONS</span></div>
    <div class="stat"><span class="stat-val" id="stat-zones">0</span><span class="stat-label">ZONES</span></div>
  </div>

  <div class="grid">
    <!-- Target list -->
    <div class="panel" id="panel-targets">
      <h2>TRACKED TARGETS</h2>
      <div id="target-list" class="scroll-area"></div>
    </div>

    <!-- Zone map (canvas) -->
    <div class="panel" id="panel-map">
      <h2>ZONE MAP</h2>
      <canvas id="zone-canvas" width="600" height="400"></canvas>
    </div>

    <!-- Alert feed -->
    <div class="panel" id="panel-alerts">
      <h2>ALERT FEED</h2>
      <div id="alert-feed" class="scroll-area"></div>
    </div>

    <!-- System health -->
    <div class="panel" id="panel-health">
      <h2>SYSTEM HEALTH</h2>
      <div id="health-grid"></div>
    </div>

    <!-- Update stream -->
    <div class="panel" id="panel-updates">
      <h2>LIVE UPDATE STREAM</h2>
      <div id="update-stream" class="scroll-area"></div>
    </div>

    <!-- Incidents & Missions -->
    <div class="panel" id="panel-ops">
      <h2>OPERATIONS</h2>
      <div id="ops-content"></div>
    </div>
  </div>
</div>

<script>
{_DASHBOARD_JS}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Dashboard CSS
# ---------------------------------------------------------------------------

_DASHBOARD_CSS = """
*{margin:0;padding:0;box-sizing:border-box}

:root {
  --bg: #0a0a0a;
  --bg-card: #111111;
  --bg-card-hover: #161616;
  --accent: #00f0ff;
  --accent-dim: #00f0ff33;
  --accent-mid: #00f0ff66;
  --magenta: #ff2a6d;
  --green: #05ffa1;
  --yellow: #fcee0a;
  --orange: #ff8c00;
  --red: #ff3333;
  --text: #c0c0c0;
  --text-dim: #555555;
  --border: #1a1a1a;
  --font: 'Courier New', 'Fira Code', monospace;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 13px;
  line-height: 1.5;
  overflow-x: hidden;
}

/* Scanline overlay */
body::after {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  pointer-events: none;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,240,255,0.012) 2px,
    rgba(0,240,255,0.012) 4px
  );
  z-index: 9999;
}

#app { padding: 12px 16px; max-width: 1400px; margin: 0 auto; }

header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--accent-dim);
  margin-bottom: 10px;
}

h1 {
  color: var(--accent);
  font-size: 20px;
  text-shadow: 0 0 12px var(--accent-dim);
  letter-spacing: 2px;
}
h1 .accent-dim { color: var(--text-dim); font-size: 14px; }

.summary {
  color: var(--text-dim);
  font-size: 12px;
  margin-top: 4px;
}

.header-right { text-align: right; }

.threat-badge {
  display: inline-block;
  padding: 4px 16px;
  border-radius: 3px;
  font-weight: bold;
  font-size: 14px;
  letter-spacing: 2px;
  text-shadow: 0 0 8px currentColor;
}
.threat-green  { background: #05ffa122; color: var(--green);  border: 1px solid var(--green); }
.threat-yellow { background: #fcee0a22; color: var(--yellow); border: 1px solid var(--yellow); }
.threat-orange { background: #ff8c0022; color: var(--orange); border: 1px solid var(--orange); }
.threat-red    { background: #ff333322; color: var(--red);    border: 1px solid var(--red); animation: pulse 1s infinite; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.clock { color: var(--text-dim); font-size: 11px; margin-top: 4px; }

.stats-bar {
  display: flex;
  gap: 20px;
  padding: 8px 0;
  margin-bottom: 10px;
  border-bottom: 1px solid var(--border);
}
.stat { text-align: center; }
.stat-val {
  display: block;
  color: var(--accent);
  font-size: 22px;
  font-weight: bold;
  text-shadow: 0 0 8px var(--accent-dim);
}
.stat-label {
  display: block;
  color: var(--text-dim);
  font-size: 10px;
  letter-spacing: 1px;
}

.grid {
  display: grid;
  grid-template-columns: 1fr 1.5fr 1fr;
  grid-template-rows: auto auto;
  gap: 10px;
}

.panel {
  background: var(--bg-card);
  border: 1px solid var(--accent-dim);
  border-radius: 4px;
  padding: 10px;
  min-height: 200px;
}
.panel:hover { border-color: var(--accent-mid); }

.panel h2 {
  color: var(--accent);
  font-size: 11px;
  letter-spacing: 2px;
  margin-bottom: 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--border);
}

.scroll-area {
  max-height: 280px;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: var(--accent-dim) var(--bg);
}

/* Target cards */
.target-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 6px 8px;
  margin-bottom: 4px;
  font-size: 11px;
  transition: border-color 0.2s;
  cursor: pointer;
}
.target-card:hover { border-color: var(--accent); }
.target-id { color: var(--accent); font-weight: bold; }
.target-type { color: var(--magenta); }
.target-sources { color: var(--text-dim); font-size: 10px; }
.target-conf { float: right; }

/* Alert items */
.alert-item {
  background: var(--bg);
  border-left: 3px solid var(--yellow);
  padding: 4px 8px;
  margin-bottom: 4px;
  font-size: 11px;
}
.alert-item.sev-critical { border-left-color: var(--red); }
.alert-item.sev-warning  { border-left-color: var(--yellow); }
.alert-item.sev-error    { border-left-color: var(--orange); }
.alert-item.sev-info     { border-left-color: var(--accent); }

.alert-time { color: var(--text-dim); font-size: 10px; }
.alert-msg  { color: var(--text); }
.alert-sev  { font-weight: bold; text-transform: uppercase; font-size: 10px; }

/* Health items */
.health-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}
.health-name { color: var(--text); }
.health-status { font-weight: bold; }
.health-up       { color: var(--green); }
.health-degraded { color: var(--yellow); }
.health-down     { color: var(--red); }

/* Update stream */
.update-item {
  font-size: 10px;
  padding: 2px 0;
  border-bottom: 1px solid #0f0f0f;
  color: var(--text-dim);
}
.update-type { color: var(--magenta); font-weight: bold; }

/* Operations */
.ops-section { margin-bottom: 8px; }
.ops-section h3 {
  color: var(--magenta);
  font-size: 10px;
  letter-spacing: 1px;
  margin-bottom: 4px;
}
.ops-item {
  background: var(--bg);
  border: 1px solid var(--border);
  padding: 4px 8px;
  margin-bottom: 3px;
  font-size: 11px;
  border-radius: 2px;
}
.ops-sev-high     { border-left: 3px solid var(--orange); }
.ops-sev-critical { border-left: 3px solid var(--red); }
.ops-sev-medium   { border-left: 3px solid var(--yellow); }

/* Canvas */
#zone-canvas {
  width: 100%;
  height: 280px;
  background: #050505;
  border: 1px solid var(--border);
  border-radius: 3px;
}
"""

# ---------------------------------------------------------------------------
# Dashboard JavaScript
# ---------------------------------------------------------------------------

_DASHBOARD_JS = """
// Auto-refresh every 2 seconds
const REFRESH_MS = 2000;
let lastUpdateTs = 0;

async function fetchJSON(url) {
    try {
        const r = await fetch(url);
        return await r.json();
    } catch(e) {
        console.error('Fetch error:', url, e);
        return null;
    }
}

function formatTime(epoch) {
    const d = new Date(epoch * 1000);
    return d.toLocaleTimeString();
}

function updateClock() {
    document.getElementById('clock').textContent = new Date().toLocaleTimeString();
}

// ---------- Picture rendering ----------

function renderPicture(pic) {
    if (!pic) return;

    // Summary
    document.getElementById('summary').textContent = pic.summary || 'No data';

    // Threat badge
    const badge = document.getElementById('threat-badge');
    badge.textContent = (pic.threat_level || 'green').toUpperCase();
    badge.className = 'threat-badge threat-' + (pic.threat_level || 'green');

    // Stats bar
    document.getElementById('stat-targets').textContent = pic.target_count || 0;
    document.getElementById('stat-multi').textContent = pic.multi_source_targets || 0;
    document.getElementById('stat-alerts').textContent = pic.active_alert_count || 0;
    document.getElementById('stat-anomalies').textContent = pic.active_anomaly_count || 0;
    document.getElementById('stat-incidents').textContent = pic.incident_count || 0;
    document.getElementById('stat-missions').textContent = pic.mission_count || 0;
    document.getElementById('stat-zones').textContent = pic.zone_count || 0;

    // Targets
    renderTargets(pic.targets || []);

    // Alerts
    renderAlerts(pic.alerts || []);

    // Health
    renderHealth(pic.health || {});

    // Zone map
    renderZoneMap(pic.targets || [], pic.zones || []);

    // Operations (incidents + missions)
    renderOps(pic.incidents || [], pic.missions || []);
}

function renderTargets(targets) {
    const el = document.getElementById('target-list');
    if (targets.length === 0) {
        el.innerHTML = '<div style="color:var(--text-dim);padding:8px;">No targets tracked</div>';
        return;
    }
    el.innerHTML = targets.map(t => {
        const sources = (t.source_types || []).join(', ');
        const conf = Math.round((t.effective_confidence || 0) * 100);
        const cls = t.classification || t.asset_type || 'unknown';
        return `<div class="target-card" onclick="window.open('/target/${t.target_id}','_blank')">
            <span class="target-id">${t.target_id}</span>
            <span class="target-conf">${conf}%</span><br>
            <span class="target-type">${cls}</span>
            ${t.name ? ' &mdash; ' + t.name : ''}
            <div class="target-sources">Sources: ${sources} | Signals: ${t.signal_count || 0}</div>
        </div>`;
    }).join('');
}

function renderAlerts(alerts) {
    const el = document.getElementById('alert-feed');
    if (alerts.length === 0) {
        el.innerHTML = '<div style="color:var(--text-dim);padding:8px;">No alerts</div>';
        return;
    }
    el.innerHTML = alerts.slice(0, 30).map(a => {
        const sev = a.severity || 'info';
        return `<div class="alert-item sev-${sev}">
            <span class="alert-sev" style="color:${sevColor(sev)}">${sev}</span>
            <span class="alert-time">${formatTime(a.timestamp)}</span><br>
            <span class="alert-msg">${a.message || a.rule_name || 'Alert'}</span>
        </div>`;
    }).join('');
}

function sevColor(sev) {
    const m = {critical:'#ff3333',error:'#ff8c00',warning:'#fcee0a',info:'#00f0ff',high:'#ff8c00'};
    return m[sev] || '#c0c0c0';
}

function renderHealth(health) {
    const el = document.getElementById('health-grid');
    const components = health.components || [];
    if (components.length === 0 && health.overall) {
        el.innerHTML = `<div class="health-item">
            <span class="health-name">Overall</span>
            <span class="health-status health-${health.overall}">${health.overall.toUpperCase()}</span>
        </div>`;
        return;
    }
    el.innerHTML = components.map(c => {
        const st = c.status || 'unknown';
        const cls = st === 'up' ? 'health-up' : (st === 'degraded' ? 'health-degraded' : 'health-down');
        return `<div class="health-item">
            <span class="health-name">${c.name}</span>
            <span class="health-status ${cls}">${st.toUpperCase()}</span>
        </div>`;
    }).join('');
}

function renderOps(incidents, missions) {
    const el = document.getElementById('ops-content');
    let html = '';

    html += '<div class="ops-section"><h3>INCIDENTS</h3>';
    if (incidents.length === 0) {
        html += '<div style="color:var(--text-dim);font-size:11px;">No active incidents</div>';
    } else {
        html += incidents.map(inc => {
            const sev = inc.severity || 'medium';
            return `<div class="ops-item ops-sev-${sev}">
                <strong>${inc.title}</strong><br>
                <span style="color:var(--text-dim);font-size:10px;">
                    State: ${inc.state} | Severity: ${sev} | Targets: ${(inc.target_ids||[]).length}
                </span>
            </div>`;
        }).join('');
    }
    html += '</div>';

    html += '<div class="ops-section"><h3>MISSIONS</h3>';
    if (missions.length === 0) {
        html += '<div style="color:var(--text-dim);font-size:11px;">No active missions</div>';
    } else {
        html += missions.map(m => {
            return `<div class="ops-item">
                <strong>${m.name}</strong><br>
                <span style="color:var(--text-dim);font-size:10px;">
                    Type: ${m.mission_type} | State: ${m.state} | Priority: ${m.priority}
                </span>
            </div>`;
        }).join('');
    }
    html += '</div>';

    el.innerHTML = html;
}

// ---------- Zone Map (Canvas) ----------

function renderZoneMap(targets, zones) {
    const canvas = document.getElementById('zone-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.offsetWidth;
    const H = canvas.height = canvas.offsetHeight || 280;

    // World bounds: x=-60..220, y=-20..130
    const wx0 = -60, wx1 = 220, wy0 = -20, wy1 = 130;
    function tx(x) { return ((x - wx0) / (wx1 - wx0)) * W; }
    function ty(y) { return H - ((y - wy0) / (wy1 - wy0)) * H; }

    // Clear
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 0.5;
    for (let gx = -40; gx <= 220; gx += 20) {
        ctx.beginPath(); ctx.moveTo(tx(gx), 0); ctx.lineTo(tx(gx), H); ctx.stroke();
    }
    for (let gy = -20; gy <= 130; gy += 20) {
        ctx.beginPath(); ctx.moveTo(0, ty(gy)); ctx.lineTo(W, ty(gy)); ctx.stroke();
    }

    // Zones
    const zoneColors = {restricted:'#ff2a6d44', monitored:'#00f0ff22', safe:'#05ffa122'};
    const zoneBorders = {restricted:'#ff2a6d', monitored:'#00f0ff', safe:'#05ffa1'};
    zones.forEach(z => {
        if (!z.polygon || z.polygon.length < 3) return;
        ctx.beginPath();
        ctx.moveTo(tx(z.polygon[0][0]), ty(z.polygon[0][1]));
        for (let i = 1; i < z.polygon.length; i++) {
            ctx.lineTo(tx(z.polygon[i][0]), ty(z.polygon[i][1]));
        }
        ctx.closePath();
        ctx.fillStyle = zoneColors[z.zone_type] || '#ffffff11';
        ctx.fill();
        ctx.strokeStyle = zoneBorders[z.zone_type] || '#444';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Zone label
        const cx = z.polygon.reduce((s,p)=>s+p[0],0) / z.polygon.length;
        const cy = z.polygon.reduce((s,p)=>s+p[1],0) / z.polygon.length;
        ctx.fillStyle = '#ffffff44';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(z.name || z.zone_id, tx(cx), ty(cy));
    });

    // Targets
    targets.forEach(t => {
        if (!t.position) return;
        const px = tx(t.position.x);
        const py = ty(t.position.y);

        // Determine color based on classification/alliance
        let color = '#00f0ff';
        const cls = (t.classification || t.asset_type || '').toLowerCase();
        if (cls === 'hostile' || t.alliance === 'hostile') color = '#ff2a6d';
        else if (cls === 'vehicle' || cls === 'car') color = '#fcee0a';
        else if (cls === 'person') color = '#05ffa1';

        // Glow
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;

        // Dot
        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Source count ring for multi-source targets
        if (t.source_count >= 2) {
            ctx.beginPath();
            ctx.arc(px, py, 7, 0, Math.PI * 2);
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        ctx.shadowBlur = 0;

        // Label
        ctx.fillStyle = color + 'aa';
        ctx.font = '9px monospace';
        ctx.textAlign = 'left';
        const label = t.name || t.target_id.substring(0, 12);
        ctx.fillText(label, px + 7, py + 3);
    });
}

// ---------- Update stream ----------

async function fetchUpdates() {
    const data = await fetchJSON('/updates?since=' + lastUpdateTs);
    if (!data || !data.updates) return;
    const el = document.getElementById('update-stream');
    data.updates.forEach(u => {
        lastUpdateTs = Math.max(lastUpdateTs, u.timestamp || 0);
        const div = document.createElement('div');
        div.className = 'update-item';
        div.innerHTML = `<span class="update-type">${u.update_type}</span> ${u.target_id || ''} <span style="color:#333">${formatTime(u.timestamp)}</span>`;
        el.prepend(div);
    });
    // Trim
    while (el.children.length > 100) el.removeChild(el.lastChild);
}

// ---------- Main loop ----------

async function refresh() {
    const pic = await fetchJSON('/picture');
    renderPicture(pic);
    await fetchUpdates();
    updateClock();
}

// Initial load + interval
refresh();
setInterval(refresh, REFRESH_MS);
updateClock();
setInterval(updateClock, 1000);
"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the demo server."""
    import uvicorn
    print(f"\n{'='*60}")
    print(f"  TRITIUM // Situational Awareness Engine Demo")
    print(f"  Dashboard: http://localhost:{DEMO_PORT}")
    print(f"  API:       http://localhost:{DEMO_PORT}/picture")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=DEMO_PORT, log_level="info")


if __name__ == "__main__":
    main()
