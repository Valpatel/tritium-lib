# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SensorPipeline — event-bus-driven connector for FusionEngine.

Subscribes to sensor event topics on an EventBus and routes incoming
data to the appropriate FusionEngine.ingest_*() method.  Publishes
fusion results on standardized topics for downstream consumers.

Input topics (subscribed):
    sensor.ble.sighting         — BLE device sighting
    sensor.wifi.probe           — WiFi probe request
    sensor.camera.detection     — Camera/YOLO detection
    sensor.acoustic.event       — Acoustic classification event
    sensor.mesh.node            — Meshtastic mesh node
    sensor.adsb.detection       — ADS-B aircraft detection
    sensor.rf_motion.event      — RF motion detection

Output topics (published by FusionEngine):
    fusion.sensor.ingested      — raw sensor accepted
    fusion.target.correlated    — two targets merged
    fusion.target.updated       — target state changed (after correlation)
    fusion.zone.entered         — target entered a geofence zone
    fusion.zone.exited          — target exited a geofence zone
    fusion.snapshot             — periodic full-state snapshot

Usage::

    from tritium_lib.events.bus import EventBus
    from tritium_lib.fusion import FusionEngine, SensorPipeline

    bus = EventBus()
    engine = FusionEngine(event_bus=bus)
    pipeline = SensorPipeline(engine, bus)
    pipeline.start()

    # Now publish sensor data on the bus:
    bus.publish("sensor.ble.sighting", {"mac": "AA:BB:CC:DD:EE:FF", "rssi": -50})
    # ... FusionEngine processes it automatically.

    pipeline.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from tritium_lib.events.bus import EventBus, Event
from .engine import FusionEngine

logger = logging.getLogger("sensor_pipeline")

# Topic -> FusionEngine method mapping
_TOPIC_MAP: dict[str, str] = {
    "sensor.ble.sighting": "ingest_ble",
    "sensor.wifi.probe": "ingest_wifi",
    "sensor.camera.detection": "ingest_camera",
    "sensor.acoustic.event": "ingest_acoustic",
    "sensor.mesh.node": "ingest_mesh",
    "sensor.adsb.detection": "ingest_adsb",
    "sensor.rf_motion.event": "ingest_rf_motion",
}


class SensorPipeline:
    """Event-bus bridge between sensor event topics and FusionEngine.

    Subscribes to all sensor.* topics and dispatches to the engine.
    Optionally runs a periodic correlation + snapshot loop.

    Args:
        engine: The FusionEngine to route sensor data to.
        event_bus: The EventBus to subscribe/publish on.
        correlation_interval: Seconds between auto-correlation passes (0 to disable).
        snapshot_interval: Seconds between snapshot publications (0 to disable).
    """

    def __init__(
        self,
        engine: FusionEngine,
        event_bus: EventBus,
        *,
        correlation_interval: float = 5.0,
        snapshot_interval: float = 30.0,
    ) -> None:
        self._engine = engine
        self._bus = event_bus
        self._correlation_interval = correlation_interval
        self._snapshot_interval = snapshot_interval
        self._running = False
        self._thread: threading.Thread | None = None

        # Stats
        self._ingested_count: int = 0
        self._error_count: int = 0
        self._lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Subscribe to sensor topics and start the background loop."""
        if self._started:
            return

        for topic in _TOPIC_MAP:
            self._bus.subscribe(topic, self._on_sensor_event)

        # Also subscribe to geofence events from the engine and republish
        self._bus.subscribe("geofence:enter", self._on_geofence_enter)
        self._bus.subscribe("geofence:exit", self._on_geofence_exit)

        self._started = True

        if self._correlation_interval > 0 or self._snapshot_interval > 0:
            self._running = True
            self._thread = threading.Thread(
                target=self._background_loop,
                name="sensor_pipeline",
                daemon=True,
            )
            self._thread.start()

        logger.info(
            "SensorPipeline started (correlation=%.1fs, snapshot=%.1fs)",
            self._correlation_interval,
            self._snapshot_interval,
        )

    def stop(self) -> None:
        """Unsubscribe from topics and stop the background loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=max(
                self._correlation_interval, self._snapshot_interval
            ) + 2)
            self._thread = None

        for topic in _TOPIC_MAP:
            try:
                self._bus.unsubscribe(topic, self._on_sensor_event)
            except Exception:
                pass

        try:
            self._bus.unsubscribe("geofence:enter", self._on_geofence_enter)
            self._bus.unsubscribe("geofence:exit", self._on_geofence_exit)
        except Exception:
            pass

        self._started = False
        logger.info("SensorPipeline stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_sensor_event(self, event: Event) -> None:
        """Route a sensor event to the appropriate ingest method."""
        method_name = _TOPIC_MAP.get(event.topic)
        if method_name is None:
            return

        data = event.data
        if not isinstance(data, dict):
            logger.warning("Non-dict data on %s, skipping", event.topic)
            return

        method = getattr(self._engine, method_name, None)
        if method is None:
            return

        try:
            result = method(data)
            with self._lock:
                self._ingested_count += 1

            if result is not None:
                self._bus.publish("fusion.target.updated", {
                    "target_id": result,
                    "source_topic": event.topic,
                    "timestamp": time.time(),
                })
        except Exception as exc:
            with self._lock:
                self._error_count += 1
            logger.warning(
                "Error ingesting %s data: %s", event.topic, exc,
            )

    def _on_geofence_enter(self, event: Event) -> None:
        """Republish geofence enter events on fusion topic."""
        self._bus.publish("fusion.zone.entered", event.data)

    def _on_geofence_exit(self, event: Event) -> None:
        """Republish geofence exit events on fusion topic."""
        self._bus.publish("fusion.zone.exited", event.data)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _background_loop(self) -> None:
        """Periodic correlation and snapshot loop."""
        last_correlate = 0.0
        last_snapshot = 0.0

        while self._running:
            now = time.monotonic()

            if self._correlation_interval > 0 and (now - last_correlate) >= self._correlation_interval:
                try:
                    new_corr = self._engine.run_correlation()
                    if new_corr:
                        logger.debug(
                            "Correlation pass: %d new correlations", len(new_corr),
                        )
                except Exception as exc:
                    logger.warning("Correlation pass error: %s", exc)
                last_correlate = now

            if self._snapshot_interval > 0 and (now - last_snapshot) >= self._snapshot_interval:
                try:
                    snapshot = self._engine.get_snapshot()
                    self._bus.publish("fusion.snapshot", snapshot.to_dict())
                except Exception as exc:
                    logger.warning("Snapshot publish error: %s", exc)
                last_snapshot = now

            time.sleep(min(
                self._correlation_interval if self._correlation_interval > 0 else 999,
                self._snapshot_interval if self._snapshot_interval > 0 else 999,
                1.0,
            ))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the pipeline background loop is active."""
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Pipeline operational status."""
        with self._lock:
            return {
                "running": self._running,
                "subscribed": self._started,
                "ingested_count": self._ingested_count,
                "error_count": self._error_count,
                "correlation_interval": self._correlation_interval,
                "snapshot_interval": self._snapshot_interval,
                "topics": list(_TOPIC_MAP.keys()),
            }
