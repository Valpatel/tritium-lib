# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone notification system demo — unified alert pipeline.

Wires together TargetTracker, GeofenceEngine, ThreatScorer,
SensorHealthMonitor, and NotificationManager into a single pipeline
that generates, routes, and displays notifications in real time.

Notification categories:
    - Geofence entry/exit alerts
    - Threat escalation warnings
    - Sensor health / offline alerts
    - Anomaly detections (behavioral + signal)

Run with:
    PYTHONPATH=src python3 src/tritium_lib/notifications/demos/notification_demo.py

Endpoints:
    GET  /                       — HTML dashboard with live notification feed
    GET  /api/notifications      — all notifications (JSON)
    GET  /api/notifications/unread — unread only
    POST /api/notifications/{id}/read — mark one read
    POST /api/notifications/read-all  — mark all read
    POST /api/notifications/{id}/dismiss — dismiss (mark read)
    GET  /api/targets            — current tracked targets
    GET  /api/geofence           — geofence zones and events
    GET  /api/health             — sensor health status
    GET  /api/status             — pipeline status summary
    WS   /ws                     — real-time notification WebSocket feed
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from tritium_lib.notifications import Notification, NotificationManager
from tritium_lib.tracking import (
    TargetTracker,
    GeofenceEngine,
    GeoZone,
    ThreatScorer,
    SensorHealthMonitor,
)


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections for real-time notification push."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, message: dict) -> None:
        """Send a JSON message to all connected WebSocket clients."""
        text = json.dumps(message)
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def count(self) -> int:
        return len(self._connections)


ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Notification pipeline
# ---------------------------------------------------------------------------

DEMO_PORT = 9092
TICK_INTERVAL = 2.0  # seconds between simulation ticks

# Synthetic sensor nodes
_SENSORS = ["node-alpha", "node-bravo", "node-charlie", "node-delta"]

# Synthetic BLE devices
_BLE_DEVICES = [
    {"mac": "AA:BB:CC:11:22:01", "name": "iPhone-Matt", "device_type": "phone"},
    {"mac": "AA:BB:CC:11:22:02", "name": "Galaxy-S24", "device_type": "phone"},
    {"mac": "DD:EE:FF:33:44:01", "name": "AirTag-Keys", "device_type": "tracker"},
    {"mac": "DD:EE:FF:33:44:02", "name": "Unknown-BLE", "device_type": "ble_device"},
    {"mac": "11:22:33:AA:BB:01", "name": "FitBit-Charge", "device_type": "wearable"},
    {"mac": "11:22:33:AA:BB:02", "name": "Laptop-WiFi", "device_type": "laptop"},
]

# Synthetic YOLO detections
_DETECTIONS = [
    {"class_name": "person", "confidence": 0.85},
    {"class_name": "car", "confidence": 0.92},
    {"class_name": "person", "confidence": 0.78},
]


@dataclass
class SimpleEventBus:
    """Minimal event bus that records events for the demo."""

    _log: list[tuple[str, dict]] = field(default_factory=list)

    def publish(self, topic: str, data: dict | None = None) -> None:
        self._log.append((topic, data or {}))

    @property
    def events(self) -> list[tuple[str, dict]]:
        return list(self._log)

    @property
    def event_count(self) -> int:
        return len(self._log)


class NotificationPipeline:
    """Full notification pipeline wiring all subsystems together.

    On each tick:
      1. Generate synthetic BLE/YOLO sightings (targets move)
      2. GeofenceEngine detects enter/exit transitions
      3. ThreatScorer evaluates behavioral threat scores
      4. SensorHealthMonitor checks sensor sighting rates
      5. Anomaly checks flag unusual patterns
      6. All events feed into NotificationManager
      7. NotificationManager broadcasts to WebSocket clients
    """

    def __init__(self) -> None:
        self.rng = random.Random(42)
        self.tick = 0

        # Event bus
        self.bus = SimpleEventBus()

        # Notification manager — broadcasts go to WebSocket clients
        self._pending_ws: list[dict] = []
        self.notif_mgr = NotificationManager(
            broadcast=self._on_notification,
            max_notifications=500,
        )

        # Tracking subsystems
        self.tracker = TargetTracker(event_bus=self.bus)
        self.geofence = GeofenceEngine(event_bus=self.bus)
        self.tracker.set_geofence_engine(self.geofence)
        self.health_monitor = SensorHealthMonitor(event_bus=self.bus)

        # Threat scorer with geofence integration
        def geofence_check(tid: str, pos: tuple[float, float]) -> bool:
            events = self.geofence.check(tid, pos)
            return any(e.event_type == "enter" for e in events)

        self.threat_scorer = ThreatScorer(
            geofence_checker=geofence_check,
            on_score_update=self._on_threat_update,
        )

        # Tracked state for anomaly detection
        self._prev_threat_levels: dict[str, str] = {}
        self._sensor_offline_notified: set[str] = set()

        # Set up geofence zones
        self.geofence.add_zone(GeoZone(
            zone_id="restricted-hq",
            name="HQ Restricted Area",
            polygon=[(20, 20), (80, 20), (80, 80), (20, 80)],
            zone_type="restricted",
            alert_on_enter=True,
            alert_on_exit=True,
        ))
        self.geofence.add_zone(GeoZone(
            zone_id="parking-south",
            name="South Parking Lot",
            polygon=[(0, -60), (100, -60), (100, -10), (0, -10)],
            zone_type="monitored",
        ))
        self.geofence.add_zone(GeoZone(
            zone_id="perimeter-north",
            name="North Perimeter Fence",
            polygon=[(-50, 100), (200, 100), (200, 130), (-50, 130)],
            zone_type="restricted",
        ))

    def _on_notification(self, msg: dict) -> None:
        """Callback from NotificationManager — queues for async WS broadcast."""
        self._pending_ws.append(msg)

    def _on_threat_update(self, target_id: str, score: float, profile: dict) -> None:
        """Callback from ThreatScorer on significant score change."""
        level = "critical" if score >= 0.7 else "warning" if score >= 0.3 else "info"
        prev_level = self._prev_threat_levels.get(target_id, "none")

        # Only notify on escalation
        if level != prev_level and level in ("warning", "critical"):
            target = self.tracker.get_target(target_id)
            name = target.name if target else target_id[:16]
            self.notif_mgr.add(
                title=f"Threat Escalation: {name}",
                message=(
                    f"Target {target_id} threat score escalated to "
                    f"{score:.0%} ({level}). "
                    f"Zone violations: {profile.get('zone_violations', 0)}, "
                    f"movement anomaly: {profile.get('movement_score', 0):.2f}"
                ),
                severity=level,
                source="threat_scorer",
                entity_id=target_id,
            )

        self._prev_threat_levels[target_id] = level

    def generate_tick(self) -> dict:
        """Run one simulation tick. Returns stats dict."""
        self.tick += 1
        stats = {
            "tick": self.tick,
            "ble_sightings": 0,
            "yolo_sightings": 0,
            "geofence_events": 0,
            "notifications_generated": 0,
        }
        notif_count_before = len(self.notif_mgr.get_all(limit=9999))

        # -- BLE sightings --
        for i, dev in enumerate(_BLE_DEVICES):
            # Devices wander on paths that cross geofence zones
            home_x = 10.0 + i * 30.0
            home_y = -20.0 + i * 25.0
            angle = self.tick * 0.2 + i * 1.0
            radius = 25.0 + 15.0 * math.sin(self.tick * 0.05 + i)
            x = home_x + radius * math.cos(angle)
            y = home_y + radius * math.sin(angle)
            rssi = self.rng.randint(-80, -25)

            self.tracker.update_from_ble({
                "mac": dev["mac"],
                "name": dev["name"],
                "rssi": rssi,
                "device_type": dev["device_type"],
                "node_position": {"x": x, "y": y},
            })
            stats["ble_sightings"] += 1

            # Record sensor sighting for health monitoring
            sensor = _SENSORS[i % len(_SENSORS)]
            self.health_monitor.record_sighting(sensor)

        # -- Simulate one sensor going offline after tick 15 --
        if self.tick <= 15:
            self.health_monitor.record_sighting("node-echo")
        elif self.tick == 20:
            # Check health and notify about offline sensor
            health = self.health_monitor.get_health()
            for h in health:
                if h["status"] == "offline" and h["sensor_id"] not in self._sensor_offline_notified:
                    self._sensor_offline_notified.add(h["sensor_id"])
                    self.notif_mgr.add(
                        title=f"Sensor Offline: {h['sensor_id']}",
                        message=(
                            f"Sensor {h['sensor_id']} has not reported for "
                            f"{h['last_seen_seconds_ago']:.0f}s. "
                            f"Last baseline rate: {h['baseline_rate']:.1f}/min."
                        ),
                        severity="critical",
                        source="sensor_health",
                        entity_id=h["sensor_id"],
                    )

        # -- YOLO detections --
        for det in _DETECTIONS:
            base_idx = self.rng.randint(0, len(_BLE_DEVICES) - 1)
            home_x = 10.0 + base_idx * 30.0
            home_y = -20.0 + base_idx * 25.0
            offset = self.rng.uniform(3.0, 25.0)
            a = self.rng.uniform(0, 2 * math.pi)
            cx = home_x + offset * math.cos(a)
            cy = home_y + offset * math.sin(a)

            self.tracker.update_from_detection({
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "center_x": cx,
                "center_y": cy,
            })
            stats["yolo_sightings"] += 1

        # -- Geofence checks (done automatically via tracker) --
        # Collect geofence events and generate notifications
        geo_events = self.geofence.get_events(limit=50)
        recent_geo = [
            e for e in geo_events
            if e.timestamp > time.time() - TICK_INTERVAL * 1.5
        ]
        for ev in recent_geo:
            if ev.event_type == "enter":
                severity = "critical" if ev.zone_type == "restricted" else "warning"
                self.notif_mgr.add(
                    title=f"Geofence Entry: {ev.zone_name}",
                    message=(
                        f"Target {ev.target_id} entered {ev.zone_type} zone "
                        f"'{ev.zone_name}' at position ({ev.position[0]:.1f}, "
                        f"{ev.position[1]:.1f})"
                    ),
                    severity=severity,
                    source="geofence",
                    entity_id=ev.target_id,
                )
                stats["geofence_events"] += 1
            elif ev.event_type == "exit":
                self.notif_mgr.add(
                    title=f"Geofence Exit: {ev.zone_name}",
                    message=(
                        f"Target {ev.target_id} exited zone '{ev.zone_name}' "
                        f"at ({ev.position[0]:.1f}, {ev.position[1]:.1f})"
                    ),
                    severity="info",
                    source="geofence",
                    entity_id=ev.target_id,
                )
                stats["geofence_events"] += 1

        # -- Threat scoring --
        all_targets = self.tracker.get_all()
        target_dicts = [
            {
                "target_id": t.target_id,
                "position": t.position,
                "heading": t.heading,
                "speed": t.speed,
                "source": t.source,
                "alliance": t.alliance,
            }
            for t in all_targets
        ]
        self.threat_scorer.evaluate(target_dicts)

        # -- Anomaly detection: flag targets with erratic behavior --
        if self.tick % 5 == 0:
            profiles = self.threat_scorer.get_all_profiles(min_score=0.0)
            for p in profiles:
                if p["movement_score"] > 0.5:
                    target = self.tracker.get_target(p["target_id"])
                    name = target.name if target else p["target_id"][:16]
                    self.notif_mgr.add(
                        title=f"Anomaly: Erratic Movement",
                        message=(
                            f"Target {name} ({p['target_id']}) showing erratic "
                            f"movement pattern. Movement score: "
                            f"{p['movement_score']:.2f}, "
                            f"threat score: {p['threat_score']:.2f}"
                        ),
                        severity="warning",
                        source="anomaly_detector",
                        entity_id=p["target_id"],
                    )

        # -- Periodic sensor health check --
        if self.tick % 10 == 0:
            health = self.health_monitor.get_health()
            for h in health:
                if h["status"] == "degraded":
                    self.notif_mgr.add(
                        title=f"Sensor Degraded: {h['sensor_id']}",
                        message=(
                            f"Sensor {h['sensor_id']} sighting rate "
                            f"deviated {h['deviation_pct']:.1f}% from baseline "
                            f"({h['sighting_rate']:.1f}/min vs "
                            f"{h['baseline_rate']:.1f}/min)"
                        ),
                        severity="warning",
                        source="sensor_health",
                        entity_id=h["sensor_id"],
                    )

        notif_count_after = len(self.notif_mgr.get_all(limit=9999))
        stats["notifications_generated"] = notif_count_after - notif_count_before
        return stats

    def get_pending_ws_messages(self) -> list[dict]:
        """Drain pending WebSocket messages."""
        msgs = self._pending_ws[:]
        self._pending_ws.clear()
        return msgs


# Singleton pipeline
pipeline = NotificationPipeline()
_bg_task: asyncio.Task | None = None


async def _tick_loop() -> None:
    """Background loop running simulation ticks."""
    while True:
        try:
            pipeline.generate_tick()
            # Broadcast pending notifications over WebSocket
            for msg in pipeline.get_pending_ws_messages():
                await ws_manager.broadcast(msg)
        except Exception as e:
            print(f"Tick error: {e}")
        await asyncio.sleep(TICK_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tick loop on startup."""
    global _bg_task
    _bg_task = asyncio.create_task(_tick_loop())
    print(f"Notification demo running on http://localhost:{DEMO_PORT}")
    print(f"  Tick interval: {TICK_INTERVAL}s")
    print(f"  {len(_BLE_DEVICES)} BLE devices, {len(_DETECTIONS)} YOLO sources")
    print(f"  {len(_SENSORS)} sensor nodes, 3 geofence zones")
    yield
    if _bg_task:
        _bg_task.cancel()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tritium Notification Demo",
    description="Unified notification pipeline — geofence, threats, health, anomalies",
    lifespan=lifespan,
)


def _target_to_dict(t) -> dict:
    """Convert TrackedTarget to JSON-safe dict without geo deps."""
    return {
        "target_id": t.target_id,
        "name": t.name,
        "alliance": t.alliance,
        "asset_type": t.asset_type,
        "position": {"x": round(t.position[0], 1), "y": round(t.position[1], 1)},
        "heading": round(t.heading, 1),
        "speed": round(t.speed, 2),
        "source": t.source,
        "signal_count": t.signal_count,
        "threat_score": round(t.threat_score, 3),
    }


# -- REST endpoints --

@app.get("/api/notifications")
async def get_notifications(limit: int = 100, since: float | None = None):
    """Return all notifications, newest first."""
    return {
        "notifications": pipeline.notif_mgr.get_all(limit=limit, since=since),
        "unread_count": pipeline.notif_mgr.count_unread(),
    }


@app.get("/api/notifications/unread")
async def get_unread():
    """Return unread notifications only."""
    return {
        "notifications": pipeline.notif_mgr.get_unread(),
        "count": pipeline.notif_mgr.count_unread(),
    }


@app.post("/api/notifications/{notification_id}/read")
async def mark_read(notification_id: str):
    """Mark a notification as read."""
    found = pipeline.notif_mgr.mark_read(notification_id)
    if not found:
        return JSONResponse({"error": "Notification not found"}, status_code=404)
    return {"ok": True, "unread_count": pipeline.notif_mgr.count_unread()}


@app.post("/api/notifications/read-all")
async def mark_all_read():
    """Mark all notifications as read."""
    count = pipeline.notif_mgr.mark_all_read()
    return {"ok": True, "marked": count}


@app.post("/api/notifications/{notification_id}/dismiss")
async def dismiss_notification(notification_id: str):
    """Dismiss (acknowledge) a notification — marks it read."""
    found = pipeline.notif_mgr.mark_read(notification_id)
    if not found:
        return JSONResponse({"error": "Notification not found"}, status_code=404)
    return {"ok": True, "dismissed": notification_id}


@app.get("/api/targets")
async def get_targets():
    """Return all tracked targets."""
    return [_target_to_dict(t) for t in pipeline.tracker.get_all()]


@app.get("/api/geofence")
async def get_geofence():
    """Return geofence zones and recent events."""
    zones = pipeline.geofence.list_zones()
    events = pipeline.geofence.get_events(limit=50)
    return {
        "zones": [z.to_dict() for z in zones],
        "events": [e.to_dict() for e in events],
    }


@app.get("/api/health")
async def get_health():
    """Return sensor health status."""
    return {
        "sensors": pipeline.health_monitor.get_health(),
    }


@app.get("/api/status")
async def get_status():
    """Pipeline status summary."""
    targets = pipeline.tracker.get_all()
    return {
        "tick": pipeline.tick,
        "total_targets": len(targets),
        "ble_targets": sum(1 for t in targets if t.source == "ble"),
        "yolo_targets": sum(1 for t in targets if t.source == "yolo"),
        "geofence_zones": len(pipeline.geofence.list_zones()),
        "total_notifications": len(pipeline.notif_mgr.get_all(limit=9999)),
        "unread_notifications": pipeline.notif_mgr.count_unread(),
        "ws_clients": ws_manager.count,
        "event_bus_events": pipeline.bus.event_count,
        "threat_status": pipeline.threat_scorer.get_status(),
    }


# -- WebSocket endpoint --

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Real-time notification feed via WebSocket.

    Sends JSON messages: {"type": "notification:new", "data": {...}}
    On connect, sends the last 10 unread notifications as a burst.
    """
    await ws_manager.connect(ws)
    try:
        # Send initial burst of recent unread notifications
        unread = pipeline.notif_mgr.get_unread()
        for n in unread[:10]:
            await ws.send_text(json.dumps({
                "type": "notification:new",
                "data": n,
            }))
        # Keep connection alive, listen for client messages
        while True:
            data = await ws.receive_text()
            # Client can send {"action": "mark_read", "id": "..."}
            try:
                msg = json.loads(data)
                if msg.get("action") == "mark_read" and msg.get("id"):
                    pipeline.notif_mgr.mark_read(msg["id"])
                elif msg.get("action") == "mark_all_read":
                    pipeline.notif_mgr.mark_all_read()
            except (json.JSONDecodeError, KeyError):
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tritium Notification Demo</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0a0a0a; color: #c0c0c0;
    font-family: 'Courier New', monospace; font-size: 13px;
}
h1 {
    color: #00f0ff; text-align: center; padding: 14px; font-size: 18px;
    text-shadow: 0 0 12px #00f0ff44;
    border-bottom: 1px solid #1a1a1a;
    letter-spacing: 2px;
}
.layout { display: grid; grid-template-columns: 320px 1fr; height: calc(100vh - 52px); }
.sidebar {
    border-right: 1px solid #1a1a1a; overflow-y: auto;
    background: #0c0c0c;
}
.main { overflow-y: auto; padding: 12px; }
.section { margin-bottom: 12px; }
.section h2 {
    color: #05ffa1; font-size: 12px; padding: 8px 12px;
    border-bottom: 1px solid #1a1a1a; text-transform: uppercase;
    letter-spacing: 1px;
}
.controls {
    padding: 8px 12px; display: flex; gap: 6px;
    border-bottom: 1px solid #1a1a1a;
}
.btn {
    background: #111; color: #00f0ff; border: 1px solid #00f0ff44;
    padding: 4px 10px; border-radius: 3px; cursor: pointer;
    font-family: inherit; font-size: 11px;
}
.btn:hover { background: #00f0ff22; }
.btn-danger { color: #ff2a6d; border-color: #ff2a6d44; }
.btn-danger:hover { background: #ff2a6d22; }
.badge-count {
    background: #ff2a6d; color: #fff; border-radius: 10px;
    padding: 1px 7px; font-size: 11px; font-weight: bold;
    margin-left: 6px;
}

/* Notification list */
.notif-list { list-style: none; }
.notif-item {
    padding: 10px 12px; border-bottom: 1px solid #111;
    cursor: pointer; transition: background 0.15s;
    position: relative;
}
.notif-item:hover { background: #151515; }
.notif-item.unread { border-left: 3px solid #00f0ff; }
.notif-item.read { opacity: 0.5; border-left: 3px solid transparent; }
.notif-title {
    font-size: 12px; font-weight: bold; margin-bottom: 3px;
    display: flex; align-items: center; gap: 6px;
}
.notif-msg { font-size: 11px; color: #888; line-height: 1.4; }
.notif-meta { font-size: 10px; color: #555; margin-top: 4px; }
.notif-dismiss {
    position: absolute; right: 8px; top: 8px;
    background: none; border: none; color: #555; cursor: pointer;
    font-size: 14px; line-height: 1;
}
.notif-dismiss:hover { color: #ff2a6d; }

/* Severity badges */
.sev { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }
.sev-info { background: #00f0ff22; color: #00f0ff; border: 1px solid #00f0ff44; }
.sev-warning { background: #fcee0a22; color: #fcee0a; border: 1px solid #fcee0a44; }
.sev-critical { background: #ff2a6d22; color: #ff2a6d; border: 1px solid #ff2a6d44; }

/* Source badges */
.src { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 9px; }
.src-geofence { background: #05ffa122; color: #05ffa1; }
.src-threat_scorer { background: #ff2a6d22; color: #ff2a6d; }
.src-sensor_health { background: #fcee0a22; color: #fcee0a; }
.src-anomaly_detector { background: #a855f722; color: #a855f7; }
.src-system { background: #00f0ff22; color: #00f0ff; }

/* Stats grid */
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 12px; }
.stat-card {
    background: #111; border: 1px solid #1a1a1a; border-radius: 4px; padding: 12px;
}
.stat-label { font-size: 10px; color: #666; text-transform: uppercase; }
.stat-value { font-size: 22px; color: #00f0ff; font-weight: bold; margin-top: 2px; }
.stat-value.critical { color: #ff2a6d; }
.stat-value.warning { color: #fcee0a; }
.stat-value.healthy { color: #05ffa1; }

/* Target table */
table { width: 100%; border-collapse: collapse; font-size: 11px; }
th { color: #ff2a6d; text-align: left; padding: 6px; border-bottom: 1px solid #222; }
td { padding: 4px 6px; border-bottom: 1px solid #111; }
tr:hover { background: #151515; }
.hostile { color: #ff2a6d; }
.friendly { color: #05ffa1; }
.unknown { color: #fcee0a; }

/* Health bars */
.health-bar { display: flex; align-items: center; gap: 8px; padding: 6px 12px; border-bottom: 1px solid #111; }
.health-name { width: 100px; font-size: 11px; }
.health-status { width: 60px; font-size: 10px; font-weight: bold; }
.health-status.healthy { color: #05ffa1; }
.health-status.degraded { color: #fcee0a; }
.health-status.critical { color: #ff2a6d; }
.health-status.offline { color: #ff2a6d; }
.health-status.unknown { color: #555; }
.bar-bg { flex: 1; height: 6px; background: #1a1a1a; border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }

/* WS status */
.ws-status {
    padding: 4px 12px; font-size: 10px; text-align: center;
    border-bottom: 1px solid #1a1a1a;
}
.ws-connected { color: #05ffa1; }
.ws-disconnected { color: #ff2a6d; }

/* Toast animation */
@keyframes slideIn {
    from { transform: translateX(-100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
.notif-item.new { animation: slideIn 0.3s ease-out; }
</style>
</head>
<body>
<h1>TRITIUM NOTIFICATION PIPELINE</h1>
<div class="layout">
    <div class="sidebar">
        <div class="ws-status" id="ws-status">
            <span class="ws-disconnected">DISCONNECTED</span>
        </div>
        <div class="controls">
            <button class="btn" onclick="markAllRead()">Mark All Read</button>
            <span style="flex:1"></span>
            <span id="unread-badge" class="badge-count">0</span>
        </div>
        <div class="section">
            <h2>Notifications</h2>
            <ul class="notif-list" id="notif-list"></ul>
        </div>
    </div>
    <div class="main">
        <div class="section">
            <h2>Pipeline Status</h2>
            <div class="stats-grid" id="stats-grid"></div>
        </div>
        <div class="section">
            <h2>Sensor Health</h2>
            <div id="health-panel"></div>
        </div>
        <div class="section">
            <h2>Tracked Targets</h2>
            <table>
                <thead><tr>
                    <th>ID</th><th>Name</th><th>Alliance</th><th>Source</th>
                    <th>Position</th><th>Signals</th><th>Threat</th>
                </tr></thead>
                <tbody id="targets-body"></tbody>
            </table>
        </div>
    </div>
</div>

<script>
// -- State --
let notifications = [];
let ws = null;
let wsRetry = 0;

// -- WebSocket --
function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws');
    ws.onopen = () => {
        document.getElementById('ws-status').innerHTML =
            '<span class="ws-connected">CONNECTED (WebSocket)</span>';
        wsRetry = 0;
    };
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'notification:new') {
            addNotification(msg.data, true);
        }
    };
    ws.onclose = () => {
        document.getElementById('ws-status').innerHTML =
            '<span class="ws-disconnected">DISCONNECTED</span>';
        wsRetry++;
        setTimeout(connectWS, Math.min(5000, 1000 * wsRetry));
    };
    ws.onerror = () => ws.close();
}

// -- Notification rendering --
function sevClass(s) {
    if (s === 'critical') return 'sev-critical';
    if (s === 'warning') return 'sev-warning';
    return 'sev-info';
}

function srcClass(s) {
    return 'src-' + (s || 'system');
}

function timeAgo(ts) {
    const diff = (Date.now() / 1000) - ts;
    if (diff < 60) return Math.floor(diff) + 's ago';
    if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
    return Math.floor(diff / 3600) + 'h ago';
}

function addNotification(n, isNew) {
    // Avoid duplicates
    if (notifications.find(x => x.id === n.id)) return;
    notifications.unshift(n);
    if (notifications.length > 200) notifications.pop();
    renderNotifications(isNew ? n.id : null);
}

function renderNotifications(newId) {
    const list = document.getElementById('notif-list');
    list.innerHTML = notifications.slice(0, 100).map(n => {
        const cls = n.read ? 'read' : 'unread';
        const extra = (newId === n.id) ? ' new' : '';
        return `<li class="notif-item ${cls}${extra}" data-id="${n.id}">
            <button class="notif-dismiss" onclick="dismissNotif(event, '${n.id}')">&times;</button>
            <div class="notif-title">
                <span class="sev ${sevClass(n.severity)}">${n.severity.toUpperCase()}</span>
                <span class="src ${srcClass(n.source)}">${n.source}</span>
                ${n.title}
            </div>
            <div class="notif-msg">${n.message}</div>
            <div class="notif-meta">${timeAgo(n.timestamp)}${n.entity_id ? ' | ' + n.entity_id.substring(0, 24) : ''}</div>
        </li>`;
    }).join('');
    updateBadge();
}

function updateBadge() {
    const count = notifications.filter(n => !n.read).length;
    const badge = document.getElementById('unread-badge');
    badge.textContent = count;
    badge.style.display = count > 0 ? 'inline' : 'none';
}

async function dismissNotif(event, id) {
    event.stopPropagation();
    await fetch('/api/notifications/' + id + '/dismiss', { method: 'POST' });
    const n = notifications.find(x => x.id === id);
    if (n) n.read = true;
    if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ action: 'mark_read', id: id }));
    }
    renderNotifications();
}

async function markAllRead() {
    await fetch('/api/notifications/read-all', { method: 'POST' });
    notifications.forEach(n => n.read = true);
    renderNotifications();
}

// -- Status updates --
async function updateStatus() {
    try {
        const s = await (await fetch('/api/status')).json();
        const grid = document.getElementById('stats-grid');
        grid.innerHTML = [
            { label: 'Tick', value: s.tick, cls: '' },
            { label: 'Total Targets', value: s.total_targets, cls: '' },
            { label: 'Notifications', value: s.total_notifications, cls: '' },
            { label: 'Unread', value: s.unread_notifications, cls: s.unread_notifications > 0 ? 'critical' : 'healthy' },
            { label: 'BLE Targets', value: s.ble_targets, cls: '' },
            { label: 'YOLO Targets', value: s.yolo_targets, cls: '' },
            { label: 'Geofence Zones', value: s.geofence_zones, cls: '' },
            { label: 'WS Clients', value: s.ws_clients, cls: s.ws_clients > 0 ? 'healthy' : 'warning' },
        ].map(x => `<div class="stat-card">
            <div class="stat-label">${x.label}</div>
            <div class="stat-value ${x.cls}">${x.value}</div>
        </div>`).join('');
    } catch (e) { console.error('Status error:', e); }
}

async function updateHealth() {
    try {
        const data = await (await fetch('/api/health')).json();
        const panel = document.getElementById('health-panel');
        if (!data.sensors.length) { panel.innerHTML = '<div style="padding:12px;color:#555">No sensor data yet</div>'; return; }
        panel.innerHTML = data.sensors.map(h => {
            const pct = h.baseline_rate > 0 ? Math.max(0, Math.min(100, (h.sighting_rate / h.baseline_rate) * 100)) : 50;
            const color = h.status === 'healthy' ? '#05ffa1' : h.status === 'degraded' ? '#fcee0a' : '#ff2a6d';
            return `<div class="health-bar">
                <span class="health-name">${h.sensor_id}</span>
                <span class="health-status ${h.status}">${h.status.toUpperCase()}</span>
                <div class="bar-bg"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
                <span style="width:60px;text-align:right;font-size:10px;color:#666">${h.sighting_rate.toFixed(1)}/min</span>
            </div>`;
        }).join('');
    } catch (e) { console.error('Health error:', e); }
}

async function updateTargets() {
    try {
        const targets = await (await fetch('/api/targets')).json();
        const tbody = document.getElementById('targets-body');
        tbody.innerHTML = targets.slice(0, 20).map(t => `<tr>
            <td>${t.target_id.substring(0, 20)}</td>
            <td>${t.name}</td>
            <td class="${t.alliance}">${t.alliance}</td>
            <td>${t.source}</td>
            <td>${t.position.x}, ${t.position.y}</td>
            <td>${t.signal_count}</td>
            <td style="color:${t.threat_score > 0.5 ? '#ff2a6d' : t.threat_score > 0.2 ? '#fcee0a' : '#05ffa1'}">${(t.threat_score * 100).toFixed(0)}%</td>
        </tr>`).join('');
    } catch (e) { console.error('Targets error:', e); }
}

// -- Initial load --
async function loadNotifications() {
    try {
        const data = await (await fetch('/api/notifications?limit=100')).json();
        notifications = data.notifications;
        renderNotifications();
    } catch (e) { console.error('Load error:', e); }
}

async function refresh() {
    await Promise.all([updateStatus(), updateHealth(), updateTargets()]);
}

loadNotifications();
refresh();
setInterval(refresh, 3000);
connectWS();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the notification demo dashboard."""
    return _DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEMO_PORT, log_level="warning")
