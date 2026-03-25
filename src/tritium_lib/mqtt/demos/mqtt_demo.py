# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Standalone MQTT integration demo — the core Tritium data flow.

Demonstrates the full sensor-to-fusion pipeline:
  sensor -> MQTT topic -> EventBus -> TargetTracker -> fusion

Works without a real MQTT broker (MockMQTTClient simulation mode).
If paho-mqtt is available and a broker is reachable, optionally connects live.

Run with:
    PYTHONPATH=src python3 src/tritium_lib/mqtt/demos/mqtt_demo.py

Endpoints:
    GET /            — HTML dashboard
    GET /topics      — active MQTT topics and subscription state
    GET /messages    — recent messages with topic, payload, timestamp
    GET /targets     — fused targets from TargetTracker
    GET /status      — pipeline health summary
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from tritium_lib.events import EventBus, Event
from tritium_lib.mqtt import TritiumTopics, parse_topic, parse_site_topic
from tritium_lib.tracking import TargetTracker, TargetCorrelator


# ---------------------------------------------------------------------------
# MockMQTTClient — simulates MQTT pub/sub without a real broker
# ---------------------------------------------------------------------------

@dataclass
class MQTTMessage:
    """A captured MQTT message."""
    topic: str
    payload: dict
    timestamp: float = field(default_factory=time.time)
    qos: int = 0
    retain: bool = False

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "qos": self.qos,
            "retain": self.retain,
        }


class MockMQTTClient:
    """In-process MQTT simulator — routes publishes to matching subscribers.

    Supports MQTT-style wildcards:
      - ``+`` matches a single topic level
      - ``#`` matches any remaining levels (must be last)

    Thread-safe. Drop-in replacement for a real paho-mqtt client in demos.
    """

    def __init__(self) -> None:
        self._subscriptions: dict[str, list[Callable[[str, dict], None]]] = {}
        self._retained: dict[str, MQTTMessage] = {}
        self._message_log: deque[MQTTMessage] = deque(maxlen=500)
        self._lock = threading.Lock()
        self._connected = False
        self._publish_count = 0
        self._subscribe_count = 0

    def connect(self, host: str = "localhost", port: int = 1883) -> None:
        """Simulate connecting to a broker."""
        self._connected = True

    def disconnect(self) -> None:
        """Simulate disconnecting from a broker."""
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def subscribe(self, topic_filter: str, callback: Callable[[str, dict], None]) -> None:
        """Subscribe to a topic filter with a callback."""
        with self._lock:
            if topic_filter not in self._subscriptions:
                self._subscriptions[topic_filter] = []
            self._subscriptions[topic_filter].append(callback)
            self._subscribe_count += 1

        # Deliver retained messages matching this filter
        with self._lock:
            for rtopic, rmsg in self._retained.items():
                if self._topic_matches(topic_filter, rtopic):
                    callback(rtopic, rmsg.payload)

    def unsubscribe(self, topic_filter: str, callback: Optional[Callable] = None) -> None:
        """Unsubscribe from a topic filter."""
        with self._lock:
            if callback is None:
                self._subscriptions.pop(topic_filter, None)
            elif topic_filter in self._subscriptions:
                self._subscriptions[topic_filter] = [
                    cb for cb in self._subscriptions[topic_filter] if cb is not callback
                ]

    def publish(self, topic: str, payload: dict, qos: int = 0, retain: bool = False) -> MQTTMessage:
        """Publish a message to a topic. Routes to matching subscribers."""
        msg = MQTTMessage(
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
        )
        with self._lock:
            self._message_log.append(msg)
            self._publish_count += 1
            if retain:
                self._retained[topic] = msg

        # Find and invoke matching subscribers
        callbacks = self._match_subscribers(topic)
        for cb in callbacks:
            try:
                cb(topic, payload)
            except Exception:
                pass  # Don't let one bad subscriber break the bus

        return msg

    def _match_subscribers(self, topic: str) -> list[Callable]:
        """Find all subscriber callbacks whose filter matches the given topic."""
        with self._lock:
            matched = []
            for pattern, callbacks in self._subscriptions.items():
                if self._topic_matches(pattern, topic):
                    matched.extend(callbacks)
            return matched

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        """Check if an MQTT topic filter pattern matches a concrete topic.

        Supports ``+`` (single-level) and ``#`` (multi-level, trailing only).
        """
        p_parts = pattern.split("/")
        t_parts = topic.split("/")

        for i, p in enumerate(p_parts):
            if p == "#":
                return True  # matches everything remaining
            if i >= len(t_parts):
                return False  # pattern is longer than topic
            if p != "+" and p != t_parts[i]:
                return False  # mismatch
        return len(p_parts) == len(t_parts)

    @property
    def messages(self) -> list[MQTTMessage]:
        """Return recent messages (newest first)."""
        with self._lock:
            return list(reversed(self._message_log))

    @property
    def subscriptions(self) -> dict[str, int]:
        """Return topic filters and their subscriber counts."""
        with self._lock:
            return {k: len(v) for k, v in self._subscriptions.items()}

    @property
    def publish_count(self) -> int:
        return self._publish_count

    @property
    def subscribe_count(self) -> int:
        return self._subscribe_count

    @property
    def retained_topics(self) -> list[str]:
        with self._lock:
            return list(self._retained.keys())


# ---------------------------------------------------------------------------
# MQTTEventBridge — wires MockMQTTClient <-> EventBus <-> TargetTracker
# ---------------------------------------------------------------------------

class MQTTEventBridge:
    """Bridges MQTT messages to the EventBus and routes sensor data to TargetTracker.

    This is the core integration layer of Tritium. Every sensor publishes
    to an MQTT topic, the bridge translates the message into an EventBus event,
    and the appropriate TargetTracker method ingests it.

    Data flow:
        sensor -> MQTT topic -> MQTTEventBridge -> EventBus -> TargetTracker -> fusion
    """

    def __init__(
        self,
        mqtt: MockMQTTClient,
        event_bus: EventBus,
        tracker: TargetTracker,
        topics: TritiumTopics,
    ) -> None:
        self.mqtt = mqtt
        self.bus = event_bus
        self.tracker = tracker
        self.topics = topics
        self._routed_count = 0
        self._error_count = 0

        # Subscribe to all sensor domains
        self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Wire MQTT subscriptions to the EventBus and TargetTracker."""

        # Edge heartbeats -> bus event
        self.mqtt.subscribe(
            self.topics.all_edge(),
            self._on_edge_message,
        )

        # Sensor data -> bus event + tracker
        self.mqtt.subscribe(
            self.topics.all_sensors(),
            self._on_sensor_message,
        )

        # Camera detections -> bus event + tracker
        self.mqtt.subscribe(
            self.topics.all_cameras(),
            self._on_camera_message,
        )

        # Meshtastic mesh nodes -> bus event + tracker
        self.mqtt.subscribe(
            self.topics.all_meshtastic(),
            self._on_meshtastic_message,
        )

        # SDR spectrum -> bus event
        self.mqtt.subscribe(
            self.topics.all_sdr(),
            self._on_sdr_message,
        )

    def _on_edge_message(self, topic: str, payload: dict) -> None:
        """Handle edge device messages (heartbeats, telemetry, wifi probes)."""
        try:
            parsed = parse_site_topic(topic)
            if parsed is None:
                return

            data_type = parsed.data_type or ""
            device_id = parsed.device_id

            if data_type == "heartbeat":
                self.bus.publish(
                    "mqtt.edge.heartbeat",
                    data={"device_id": device_id, **payload},
                    source="mqtt",
                )
            elif data_type == "telemetry":
                self.bus.publish(
                    "mqtt.edge.telemetry",
                    data={"device_id": device_id, **payload},
                    source="mqtt",
                )
            elif data_type == "wifi_probe":
                self.bus.publish(
                    "mqtt.edge.wifi_probe",
                    data={"device_id": device_id, **payload},
                    source="mqtt",
                )
            elif data_type == "wifi_scan":
                self.bus.publish(
                    "mqtt.edge.wifi_scan",
                    data={"device_id": device_id, **payload},
                    source="mqtt",
                )

            self._routed_count += 1
        except Exception:
            self._error_count += 1

    def _on_sensor_message(self, topic: str, payload: dict) -> None:
        """Handle sensor data — route BLE sightings to tracker."""
        try:
            parsed = parse_site_topic(topic)
            if parsed is None:
                return

            data_type = parsed.data_type or ""
            device_id = parsed.device_id

            if data_type == "ble":
                # BLE sighting -> TargetTracker
                self.tracker.update_from_ble(payload)
                self.bus.publish(
                    "mqtt.sensor.ble",
                    data={"observer": device_id, **payload},
                    source="mqtt",
                )
            elif data_type == "wifi":
                self.bus.publish(
                    "mqtt.sensor.wifi",
                    data={"observer": device_id, **payload},
                    source="mqtt",
                )
            else:
                self.bus.publish(
                    f"mqtt.sensor.{data_type}",
                    data={"observer": device_id, **payload},
                    source="mqtt",
                )

            self._routed_count += 1
        except Exception:
            self._error_count += 1

    def _on_camera_message(self, topic: str, payload: dict) -> None:
        """Handle camera messages — route YOLO detections to tracker."""
        try:
            parsed = parse_site_topic(topic)
            if parsed is None:
                return

            data_type = parsed.data_type or ""
            device_id = parsed.device_id

            if data_type == "detections":
                detections = payload.get("detections", [payload])
                for det in detections:
                    self.tracker.update_from_detection(det)
                self.bus.publish(
                    "mqtt.camera.detections",
                    data={"camera_id": device_id, "count": len(detections), **payload},
                    source="mqtt",
                )

            self._routed_count += 1
        except Exception:
            self._error_count += 1

    def _on_meshtastic_message(self, topic: str, payload: dict) -> None:
        """Handle Meshtastic mesh messages — route nodes to tracker."""
        try:
            parsed = parse_site_topic(topic)
            if parsed is None:
                return

            data_type = parsed.data_type or ""
            device_id = parsed.device_id

            if data_type == "nodes":
                nodes = payload.get("nodes", [payload])
                for node in nodes:
                    self.tracker.update_from_mesh(node)
                self.bus.publish(
                    "mqtt.meshtastic.nodes",
                    data={"bridge_id": device_id, "count": len(nodes), **payload},
                    source="mqtt",
                )
            elif data_type == "position":
                self.tracker.update_from_mesh(payload)
                self.bus.publish(
                    "mqtt.meshtastic.position",
                    data={"bridge_id": device_id, **payload},
                    source="mqtt",
                )

            self._routed_count += 1
        except Exception:
            self._error_count += 1

    def _on_sdr_message(self, topic: str, payload: dict) -> None:
        """Handle SDR messages — forward as bus events."""
        try:
            parsed = parse_site_topic(topic)
            if parsed is None:
                return

            data_type = parsed.data_type or ""
            device_id = parsed.device_id

            self.bus.publish(
                f"mqtt.sdr.{data_type}",
                data={"device_id": device_id, **payload},
                source="mqtt",
            )
            self._routed_count += 1
        except Exception:
            self._error_count += 1

    @property
    def routed_count(self) -> int:
        return self._routed_count

    @property
    def error_count(self) -> int:
        return self._error_count


# ---------------------------------------------------------------------------
# SensorSimulator — generates realistic MQTT traffic from virtual sensors
# ---------------------------------------------------------------------------

# Simulated sensor nodes
_EDGE_NODES = [
    {"device_id": "esp32-alpha", "sensors": ["ble", "wifi"]},
    {"device_id": "esp32-bravo", "sensors": ["ble"]},
    {"device_id": "esp32-charlie", "sensors": ["ble", "wifi"]},
]

_CAMERAS = [
    {"device_id": "cam-front-door", "classes": ["person", "car"]},
    {"device_id": "cam-parking-lot", "classes": ["car", "motorcycle", "bicycle"]},
]

_MESH_NODES = [
    {"target_id": "mesh_tlora01", "name": "TLoRa-Pager-01"},
    {"target_id": "mesh_tlora02", "name": "TLoRa-Pager-02"},
    {"target_id": "mesh_heltec03", "name": "Heltec-V3-03"},
]

_BLE_DEVICES = [
    {"mac": "AA:BB:CC:11:22:01", "name": "iPhone-Matt", "device_type": "phone"},
    {"mac": "AA:BB:CC:11:22:02", "name": "Galaxy-S24", "device_type": "phone"},
    {"mac": "DD:EE:FF:33:44:01", "name": "AirTag-Keys", "device_type": "tracker"},
    {"mac": "DD:EE:FF:33:44:02", "name": "Tile-Wallet", "device_type": "tracker"},
    {"mac": "11:22:33:AA:BB:01", "name": "Fitbit-Charge6", "device_type": "wearable"},
    {"mac": "11:22:33:AA:BB:02", "name": "AirPods-Pro", "device_type": "audio"},
]


class SensorSimulator:
    """Generates synthetic MQTT sensor traffic through a MockMQTTClient.

    Each tick simulates:
      - Edge node heartbeats
      - BLE sightings from edge nodes
      - Camera YOLO detections
      - Meshtastic mesh node updates
      - SDR spectrum sweeps

    Devices wander on realistic paths with configurable noise.
    """

    def __init__(
        self,
        mqtt: MockMQTTClient,
        topics: TritiumTopics,
        seed: int = 42,
    ) -> None:
        self.mqtt = mqtt
        self.topics = topics
        self.rng = random.Random(seed)
        self.tick = 0

        # Each BLE device has a base position and wanders
        self._ble_positions: dict[str, tuple[float, float]] = {}
        for i, dev in enumerate(_BLE_DEVICES):
            self._ble_positions[dev["mac"]] = (
                30.0 + i * 25.0,
                40.0 + (i % 3) * 20.0,
            )

        # Mesh node positions (further out, GPS-based)
        self._mesh_positions: dict[str, tuple[float, float]] = {}
        for i, node in enumerate(_MESH_NODES):
            self._mesh_positions[node["target_id"]] = (
                100.0 + i * 50.0,
                150.0 + (i % 2) * 30.0,
            )

    def generate_tick(self) -> dict:
        """Generate one batch of simulated sensor messages. Returns stats."""
        self.tick += 1
        stats = {"heartbeats": 0, "ble": 0, "camera": 0, "mesh": 0, "sdr": 0}

        # --- Edge node heartbeats ---
        for node in _EDGE_NODES:
            self.mqtt.publish(
                self.topics.edge_heartbeat(node["device_id"]),
                {
                    "device_id": node["device_id"],
                    "uptime_s": self.tick * 2,
                    "free_heap": self.rng.randint(80000, 120000),
                    "wifi_rssi": self.rng.randint(-70, -30),
                    "sensors": node["sensors"],
                    "timestamp": time.time(),
                },
                retain=True,
            )
            stats["heartbeats"] += 1

        # --- BLE sightings from edge nodes ---
        for dev in _BLE_DEVICES:
            # Wander the device
            base = self._ble_positions[dev["mac"]]
            angle = self.tick * 0.2 + hash(dev["mac"]) % 100
            radius = 8.0 + 4.0 * math.sin(self.tick * 0.05 + hash(dev["mac"]) % 50)
            x = base[0] + radius * math.cos(angle)
            y = base[1] + radius * math.sin(angle)
            self._ble_positions[dev["mac"]] = (
                base[0] + 0.1 * (x - base[0]),
                base[1] + 0.1 * (y - base[1]),
            )

            # Pick a random edge node as the observer
            observer = self.rng.choice(_EDGE_NODES)
            rssi = self.rng.randint(-85, -25)

            self.mqtt.publish(
                self.topics.sensor(observer["device_id"], "ble"),
                {
                    "mac": dev["mac"],
                    "name": dev["name"],
                    "rssi": rssi,
                    "device_type": dev["device_type"],
                    "node_position": {"x": x, "y": y},
                    "observer": observer["device_id"],
                    "timestamp": time.time(),
                },
            )
            stats["ble"] += 1

        # --- Camera YOLO detections ---
        for cam in _CAMERAS:
            # Each camera sees 1-3 detections per tick
            n_detections = self.rng.randint(1, 3)
            detections = []
            for _ in range(n_detections):
                cls = self.rng.choice(cam["classes"])
                # Place detection near a random BLE device (for correlation)
                ref_mac = self.rng.choice(_BLE_DEVICES)["mac"]
                ref_pos = self._ble_positions[ref_mac]
                # Some close (correlatable), some far
                offset = self.rng.uniform(1.0, 15.0)
                angle = self.rng.uniform(0, 2 * math.pi)
                cx = ref_pos[0] + offset * math.cos(angle)
                cy = ref_pos[1] + offset * math.sin(angle)

                detections.append({
                    "class_name": cls,
                    "confidence": round(self.rng.uniform(0.5, 0.98), 2),
                    "center_x": round(cx, 1),
                    "center_y": round(cy, 1),
                    "bbox": {
                        "x": round(cx - 1, 1),
                        "y": round(cy - 1, 1),
                        "w": round(self.rng.uniform(1.5, 4.0), 1),
                        "h": round(self.rng.uniform(2.0, 5.0), 1),
                    },
                })

            self.mqtt.publish(
                self.topics.camera_detections(cam["device_id"]),
                {
                    "camera_id": cam["device_id"],
                    "detections": detections,
                    "frame_id": self.tick,
                    "timestamp": time.time(),
                },
            )
            stats["camera"] += len(detections)

        # --- Meshtastic mesh node updates ---
        for node in _MESH_NODES:
            base = self._mesh_positions[node["target_id"]]
            # Mesh nodes move slowly (walking speed)
            dx = self.rng.uniform(-0.5, 0.5)
            dy = self.rng.uniform(-0.5, 0.5)
            new_pos = (base[0] + dx, base[1] + dy)
            self._mesh_positions[node["target_id"]] = new_pos

            self.mqtt.publish(
                self.topics.meshtastic_nodes("bridge-01"),
                {
                    "nodes": [{
                        "target_id": node["target_id"],
                        "name": node["name"],
                        "position": {"x": new_pos[0], "y": new_pos[1]},
                        "battery": round(self.rng.uniform(0.3, 1.0), 2),
                        "snr": round(self.rng.uniform(5.0, 15.0), 1),
                        "last_heard": time.time(),
                    }],
                },
            )
            stats["mesh"] += 1

        # --- SDR spectrum sweep (every 3rd tick) ---
        if self.tick % 3 == 0:
            self.mqtt.publish(
                self.topics.sdr_spectrum("hackrf-01"),
                {
                    "device_id": "hackrf-01",
                    "center_freq_mhz": 915.0,
                    "bandwidth_mhz": 2.0,
                    "peak_dbm": round(self.rng.uniform(-80.0, -20.0), 1),
                    "noise_floor_dbm": round(self.rng.uniform(-110.0, -90.0), 1),
                    "num_bins": 256,
                    "timestamp": time.time(),
                },
            )
            stats["sdr"] += 1

        return stats


# ---------------------------------------------------------------------------
# MQTTPipeline — wires everything together
# ---------------------------------------------------------------------------

class MQTTPipeline:
    """Full MQTT integration pipeline: simulator -> broker -> bridge -> tracker.

    This is the demo-able proof of the core Tritium data flow.
    """

    def __init__(self, site_id: str = "demo", seed: int = 42) -> None:
        # Infrastructure
        self.topics = TritiumTopics(site_id=site_id)
        self.mqtt = MockMQTTClient()
        self.bus = EventBus()
        self.tracker = TargetTracker(event_bus=self.bus)

        # Bridge MQTT -> EventBus -> Tracker
        self.bridge = MQTTEventBridge(
            mqtt=self.mqtt,
            event_bus=self.bus,
            tracker=self.tracker,
            topics=self.topics,
        )

        # Correlator — fuse multi-source sightings
        self.correlator = TargetCorrelator(
            self.tracker,
            radius=15.0,
            confidence_threshold=0.25,
        )
        self.correlation_log: list[dict] = []

        # Event log — captures all bus events for the dashboard
        self._bus_events: deque[dict] = deque(maxlen=200)
        self.bus.subscribe("#", self._capture_bus_event)

        # Simulator
        self.simulator = SensorSimulator(
            mqtt=self.mqtt,
            topics=self.topics,
            seed=seed,
        )

        # Connect the mock broker
        self.mqtt.connect("localhost", 1883)

    def _capture_bus_event(self, event: Event) -> None:
        """Store bus events for the dashboard API."""
        self._bus_events.append({
            "topic": event.topic,
            "source": event.source,
            "timestamp": event.timestamp,
            "data_keys": list(event.data.keys()) if isinstance(event.data, dict) else [],
        })

    def tick(self) -> dict:
        """Run one simulation tick. Returns stats from the simulator."""
        stats = self.simulator.generate_tick()

        # Run correlator after ingestion
        new_correlations = self.correlator.correlate()
        for rec in new_correlations:
            self.correlation_log.append({
                "primary_id": rec.primary_id,
                "secondary_id": rec.secondary_id,
                "confidence": round(rec.confidence, 3),
                "reason": rec.reason,
            })
        stats["correlations"] = len(new_correlations)

        return stats

    @property
    def bus_events(self) -> list[dict]:
        return list(self._bus_events)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

DEMO_PORT = 9092
TICK_INTERVAL = 2.0

pipeline = MQTTPipeline()
_bg_task: asyncio.Task | None = None


async def _tick_loop() -> None:
    """Background loop running simulation ticks."""
    while True:
        try:
            pipeline.tick()
        except Exception as e:
            print(f"Tick error: {e}")
        await asyncio.sleep(TICK_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bg_task
    _bg_task = asyncio.create_task(_tick_loop())
    print(f"MQTT integration demo running on http://localhost:{DEMO_PORT}")
    print(f"  Site: {pipeline.topics.site}")
    print(f"  Tick interval: {TICK_INTERVAL}s")
    print(f"  Edge nodes: {len(_EDGE_NODES)}")
    print(f"  Cameras: {len(_CAMERAS)}")
    print(f"  BLE devices: {len(_BLE_DEVICES)}")
    print(f"  Mesh nodes: {len(_MESH_NODES)}")
    yield
    _bg_task.cancel()


app = FastAPI(
    title="Tritium MQTT Integration Demo",
    description="Core data flow: sensor -> MQTT -> EventBus -> TargetTracker -> fusion",
    lifespan=lifespan,
)


@app.get("/topics")
async def get_topics():
    """Show active MQTT topics and subscription state."""
    return {
        "site": pipeline.topics.site,
        "subscriptions": pipeline.mqtt.subscriptions,
        "retained_topics": pipeline.mqtt.retained_topics,
        "publish_count": pipeline.mqtt.publish_count,
        "subscribe_count": pipeline.mqtt.subscribe_count,
        "topic_tree": {
            "edge": {
                "heartbeat": [pipeline.topics.edge_heartbeat(n["device_id"]) for n in _EDGE_NODES],
                "telemetry": [pipeline.topics.edge_telemetry(n["device_id"]) for n in _EDGE_NODES],
                "command": [pipeline.topics.edge_command(n["device_id"]) for n in _EDGE_NODES],
            },
            "sensors": {
                "ble": [pipeline.topics.sensor(n["device_id"], "ble") for n in _EDGE_NODES],
                "wifi": [pipeline.topics.sensor(n["device_id"], "wifi") for n in _EDGE_NODES if "wifi" in n["sensors"]],
            },
            "cameras": {
                "detections": [pipeline.topics.camera_detections(c["device_id"]) for c in _CAMERAS],
            },
            "meshtastic": {
                "nodes": [pipeline.topics.meshtastic_nodes("bridge-01")],
            },
            "sdr": {
                "spectrum": [pipeline.topics.sdr_spectrum("hackrf-01")],
            },
        },
    }


@app.get("/messages")
async def get_messages(limit: int = 50):
    """Return recent MQTT messages."""
    msgs = pipeline.mqtt.messages[:limit]
    return {
        "count": len(msgs),
        "total_published": pipeline.mqtt.publish_count,
        "messages": [m.to_dict() for m in msgs],
    }


@app.get("/targets")
async def get_targets():
    """Return all fused targets from the TargetTracker."""
    targets = pipeline.tracker.get_all()
    result = []
    for t in targets:
        result.append({
            "target_id": t.target_id,
            "name": t.name,
            "alliance": t.alliance,
            "asset_type": t.asset_type,
            "position": {"x": round(t.position[0], 1), "y": round(t.position[1], 1)},
            "heading": round(t.heading, 1),
            "speed": round(t.speed, 2),
            "source": t.source,
            "signal_count": t.signal_count,
            "position_confidence": round(t.effective_confidence, 3),
            "confirming_sources": list(t.confirming_sources),
            "correlated_ids": list(t.correlated_ids),
            "correlation_confidence": round(t.correlation_confidence, 3),
            "classification": t.classification,
            "classification_confidence": round(t.classification_confidence, 3),
        })
    return {
        "count": len(result),
        "targets": result,
        "correlations": pipeline.correlation_log[-20:],
    }


@app.get("/status")
async def get_status():
    """Pipeline health summary."""
    targets = pipeline.tracker.get_all()
    return {
        "tick": pipeline.simulator.tick,
        "mqtt": {
            "connected": pipeline.mqtt.is_connected,
            "publish_count": pipeline.mqtt.publish_count,
            "subscribe_count": pipeline.mqtt.subscribe_count,
            "subscription_filters": len(pipeline.mqtt.subscriptions),
            "retained_topics": len(pipeline.mqtt.retained_topics),
            "message_log_size": len(pipeline.mqtt.messages),
        },
        "bridge": {
            "routed_count": pipeline.bridge.routed_count,
            "error_count": pipeline.bridge.error_count,
        },
        "event_bus": {
            "event_count": len(pipeline.bus_events),
        },
        "tracker": {
            "total_targets": len(targets),
            "ble_targets": sum(1 for t in targets if t.source == "ble"),
            "yolo_targets": sum(1 for t in targets if t.source == "yolo"),
            "mesh_targets": sum(1 for t in targets if t.source == "mesh"),
            "hostiles": len(pipeline.tracker.get_hostiles()),
            "friendlies": len(pipeline.tracker.get_friendlies()),
        },
        "correlations": len(pipeline.correlation_log),
    }


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tritium MQTT Integration Demo</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0a; color: #c0c0c0; font-family: 'Courier New', monospace; }
h1 { color: #00f0ff; text-align: center; padding: 16px; font-size: 20px;
     text-shadow: 0 0 10px #00f0ff44; border-bottom: 1px solid #1a1a1a; }
.subtitle { text-align: center; color: #666; font-size: 11px; padding: 4px 0 12px; }
.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; padding: 8px; }
.panel { background: #111; border: 1px solid #1a1a1a; border-radius: 4px; padding: 12px; }
.panel h2 { color: #05ffa1; font-size: 13px; margin-bottom: 8px; }
.stat { display: flex; justify-content: space-between; padding: 2px 0;
        border-bottom: 1px solid #0a0a0a; font-size: 12px; }
.stat .val { color: #00f0ff; font-weight: bold; }
.fullwidth { grid-column: 1 / -1; }
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
.badge-mesh { background: #05ffa122; color: #05ffa1; border: 1px solid #05ffa144; }
.msg-row { font-size: 10px; padding: 2px 4px; border-bottom: 1px solid #0a0a0a;
           overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.msg-topic { color: #fcee0a; }
.flow-arrow { color: #ff2a6d; font-weight: bold; margin: 0 4px; }
.flow-box { display: inline-block; padding: 4px 8px; border: 1px solid #333;
            border-radius: 4px; font-size: 11px; margin: 2px; }
.flow-active { border-color: #00f0ff; color: #00f0ff; background: #00f0ff11; }
</style>
</head>
<body>
<h1>TRITIUM MQTT INTEGRATION DEMO</h1>
<div class="subtitle">sensor -> MQTT topic -> EventBus -> TargetTracker -> fusion</div>
<div class="grid">
  <div class="panel fullwidth" style="text-align:center; padding:8px;">
    <span class="flow-box flow-active" id="flow-sensors">SENSORS</span>
    <span class="flow-arrow">--></span>
    <span class="flow-box flow-active" id="flow-mqtt">MQTT (<span id="pub-count">0</span>)</span>
    <span class="flow-arrow">--></span>
    <span class="flow-box flow-active" id="flow-bridge">BRIDGE (<span id="route-count">0</span>)</span>
    <span class="flow-arrow">--></span>
    <span class="flow-box flow-active" id="flow-bus">EVENT BUS (<span id="bus-count">0</span>)</span>
    <span class="flow-arrow">--></span>
    <span class="flow-box flow-active" id="flow-tracker">TRACKER (<span id="target-count">0</span>)</span>
    <span class="flow-arrow">--></span>
    <span class="flow-box flow-active" id="flow-fusion">FUSION (<span id="corr-count">0</span>)</span>
  </div>
  <div class="panel">
    <h2>MQTT STATUS</h2>
    <div id="mqtt-status">Loading...</div>
  </div>
  <div class="panel">
    <h2>BRIDGE STATUS</h2>
    <div id="bridge-status">Loading...</div>
  </div>
  <div class="panel">
    <h2>TRACKER STATUS</h2>
    <div id="tracker-status">Loading...</div>
  </div>
  <div class="panel fullwidth">
    <h2>TARGETS</h2>
    <table>
      <thead><tr><th>ID</th><th>Name</th><th>Alliance</th><th>Source</th>
        <th>Position</th><th>Signals</th><th>Confidence</th><th>Sources</th></tr></thead>
      <tbody id="targets-body"></tbody>
    </table>
  </div>
  <div class="panel">
    <h2>TOPIC SUBSCRIPTIONS</h2>
    <div id="subscriptions">Loading...</div>
  </div>
  <div class="panel">
    <h2>RECENT MQTT MESSAGES</h2>
    <div id="messages" style="max-height:300px; overflow-y:auto;">Loading...</div>
  </div>
  <div class="panel">
    <h2>CORRELATIONS</h2>
    <div id="correlations">None yet</div>
  </div>
</div>
<script>
async function fetchJSON(url) { return (await fetch(url)).json(); }

function sourceBadge(s) {
    const cls = s === 'ble' ? 'badge-ble' : s === 'yolo' ? 'badge-yolo' : 'badge-mesh';
    return `<span class="badge ${cls}">${s}</span>`;
}

function allianceCls(a) {
    return a === 'hostile' ? 'hostile' : a === 'friendly' ? 'friendly' : 'unknown';
}

function statHTML(pairs) {
    return pairs.map(([k,v]) =>
        `<div class="stat"><span>${k}</span><span class="val">${v}</span></div>`
    ).join('');
}

async function refresh() {
    try {
        const [status, topics, messages, targets] = await Promise.all([
            fetchJSON('/status'), fetchJSON('/topics'),
            fetchJSON('/messages?limit=20'), fetchJSON('/targets'),
        ]);

        // Flow bar
        document.getElementById('pub-count').textContent = status.mqtt.publish_count;
        document.getElementById('route-count').textContent = status.bridge.routed_count;
        document.getElementById('bus-count').textContent = status.event_bus.event_count;
        document.getElementById('target-count').textContent = status.tracker.total_targets;
        document.getElementById('corr-count').textContent = status.correlations;

        // MQTT status
        document.getElementById('mqtt-status').innerHTML = statHTML([
            ['Connected', status.mqtt.connected ? 'YES' : 'NO'],
            ['Published', status.mqtt.publish_count],
            ['Subscriptions', status.mqtt.subscription_filters],
            ['Retained', status.mqtt.retained_topics],
            ['Log Size', status.mqtt.message_log_size],
        ]);

        // Bridge status
        document.getElementById('bridge-status').innerHTML = statHTML([
            ['Routed', status.bridge.routed_count],
            ['Errors', status.bridge.error_count],
            ['Tick', status.tick],
        ]);

        // Tracker status
        document.getElementById('tracker-status').innerHTML = statHTML([
            ['Total', status.tracker.total_targets],
            ['BLE', status.tracker.ble_targets],
            ['YOLO', status.tracker.yolo_targets],
            ['Mesh', status.tracker.mesh_targets],
            ['Hostiles', status.tracker.hostiles],
            ['Correlations', status.correlations],
        ]);

        // Targets table
        const tbody = document.getElementById('targets-body');
        tbody.innerHTML = targets.targets.slice(0, 30).map(t => `<tr>
            <td>${t.target_id.substring(0,22)}</td>
            <td>${t.name}</td>
            <td class="${allianceCls(t.alliance)}">${t.alliance}</td>
            <td>${sourceBadge(t.source)}</td>
            <td>${t.position.x.toFixed(1)}, ${t.position.y.toFixed(1)}</td>
            <td>${t.signal_count}</td>
            <td>${(t.position_confidence * 100).toFixed(0)}%</td>
            <td>${t.confirming_sources.join(', ')}</td>
        </tr>`).join('');

        // Subscriptions
        document.getElementById('subscriptions').innerHTML =
            Object.entries(topics.subscriptions).map(([k,v]) =>
                `<div class="stat"><span class="msg-topic">${k}</span><span class="val">${v}</span></div>`
            ).join('');

        // Messages
        document.getElementById('messages').innerHTML =
            messages.messages.slice(0, 15).map(m => {
                const t = new Date(m.timestamp * 1000).toLocaleTimeString();
                return `<div class="msg-row"><span style="color:#666">${t}</span> <span class="msg-topic">${m.topic}</span></div>`;
            }).join('');

        // Correlations
        const el = document.getElementById('correlations');
        if (targets.correlations.length === 0) {
            el.textContent = 'No correlations yet';
        } else {
            el.innerHTML = targets.correlations.slice(-10).reverse().map(c =>
                `<div class="stat"><span>${c.primary_id.substring(0,16)} + ${c.secondary_id.substring(0,16)}</span><span class="val">${(c.confidence*100).toFixed(0)}%</span></div>`
            ).join('');
        }
    } catch(e) { console.error('Refresh error:', e); }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the MQTT integration demo dashboard."""
    return _DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DEMO_PORT, log_level="warning")
