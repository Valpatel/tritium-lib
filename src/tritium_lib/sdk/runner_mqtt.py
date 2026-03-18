# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.

"""Lightweight MQTT client wrapper for runner-to-SC communication."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class RunnerMQTTClient:
    """Thin wrapper around paho-mqtt v2 for Tritium agent communication."""

    def __init__(
        self,
        client_id: str,
        host: str = "localhost",
        port: int = 1883,
    ) -> None:
        self.client_id = client_id
        self.host = host
        self.port = port
        self._client: mqtt.Client | None = None
        self._connected = False
        self._subscriptions: dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to the MQTT broker. Returns True on success."""
        try:
            # Support both paho-mqtt v1 and v2
            if hasattr(mqtt, "CallbackAPIVersion"):
                self._client = mqtt.Client(
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                    client_id=self.client_id,
                )
            else:
                self._client = mqtt.Client(client_id=self.client_id)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message
            self._client.connect(self.host, self.port, keepalive=60)
            self._client.loop_start()
            return True
        except Exception:
            logger.exception("MQTT connect failed")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                logger.exception("MQTT disconnect error")
            finally:
                self._connected = False

    # ------------------------------------------------------------------
    # Pub / Sub
    # ------------------------------------------------------------------

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """JSON-serialize *payload* and publish to *topic*."""
        if self._client is None:
            logger.warning("publish called before connect")
            return
        raw = json.dumps(payload)
        self._client.publish(topic, raw)

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to *topic* and dispatch incoming messages to *callback*."""
        self._subscriptions[topic] = callback
        if self._client is not None and self._connected:
            self._client.subscribe(topic)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        rc: Any,
        properties: Any = None,
    ) -> None:
        self._connected = True
        logger.info("MQTT connected to %s:%d", self.host, self.port)
        # Re-subscribe after reconnect
        for topic in self._subscriptions:
            client.subscribe(topic)

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any = None,
        rc: Any = None,
        properties: Any = None,
    ) -> None:
        self._connected = False
        logger.warning("MQTT disconnected (rc=%s)", rc)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        callback = self._subscriptions.get(msg.topic)
        if callback is None:
            return
        try:
            payload = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"raw": msg.payload.decode("utf-8", errors="replace")}
        try:
            callback(msg.topic, payload)
        except Exception:
            logger.exception("Error in subscription callback for %s", msg.topic)
