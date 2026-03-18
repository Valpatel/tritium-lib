# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.

"""BaseRunner ABC — headless device runner for Tritium addons."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from .runner_mqtt import RunnerMQTTClient

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """Abstract base class that every Tritium device agent must extend.

    Subclasses implement device-specific discovery, start/stop, and
    command handling.  The base class provides MQTT wiring, topic
    formatting, and the main run-loop.
    """

    def __init__(
        self,
        agent_id: str,
        device_type: str,
        site_id: str = "home",
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
    ) -> None:
        self.agent_id = agent_id
        self.device_type = device_type
        self.site_id = site_id
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port

        self._mqtt: RunnerMQTTClient | None = None
        self._running: bool = False
        self._devices: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Topic helpers
    # ------------------------------------------------------------------

    @property
    def status_topic(self) -> str:
        """MQTT topic for agent status messages."""
        return f"tritium/{self.site_id}/{self.device_type}/{self.agent_id}/status"

    def data_topic(self, data_type: str, device_id: str = "") -> str:
        """MQTT topic for publishing device data."""
        dev = device_id or self.agent_id
        return f"tritium/{self.site_id}/{self.device_type}/{dev}/{data_type}"

    @property
    def command_topic(self) -> str:
        """MQTT topic to receive remote commands."""
        return f"tritium/{self.site_id}/{self.device_type}/{self.agent_id}/command"

    # ------------------------------------------------------------------
    # Abstract — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    async def discover_devices(self) -> list[dict[str, Any]]:
        """Scan for locally-attached USB / serial devices."""
        ...

    @abstractmethod
    async def start_device(self, device_info: dict[str, Any]) -> bool:
        """Begin streaming from *device_info*. Return True on success."""
        ...

    @abstractmethod
    async def stop_device(self, device_id: str) -> bool:
        """Stop streaming from the device identified by *device_id*."""
        ...

    @abstractmethod
    async def on_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle a remote command and return a response dict."""
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def _publish_status(self, status: dict[str, Any]) -> None:
        """Publish an agent status update over MQTT."""
        if self._mqtt is None:
            return
        status.setdefault("agent_id", self.agent_id)
        status.setdefault("device_type", self.device_type)
        status.setdefault("site_id", self.site_id)
        status.setdefault("timestamp", time.time())
        self._mqtt.publish(self.status_topic, status)

    def _publish_data(
        self,
        data_type: str,
        data: dict[str, Any],
        device_id: str = "",
    ) -> None:
        """Publish sensor / device data over MQTT."""
        if self._mqtt is None:
            return
        data.setdefault("timestamp", time.time())
        self._mqtt.publish(self.data_topic(data_type, device_id), data)

    def _subscribe_commands(self) -> None:
        """Subscribe to the agent's command topic."""
        if self._mqtt is None:
            return
        self._mqtt.subscribe(self.command_topic, self._handle_command_message)

    def _handle_command_message(self, topic: str, payload: dict[str, Any]) -> None:
        """Dispatch an incoming MQTT command into the async handler."""
        command = payload.get("command", "")
        if not command:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._dispatch_command(command, payload))
        except RuntimeError:
            # No running loop — run synchronously as a fallback
            asyncio.run(self._dispatch_command(command, payload))

    async def _dispatch_command(self, command: str, payload: dict[str, Any]) -> None:
        """Await the subclass command handler and publish the response."""
        try:
            result = await self.on_command(command, payload)
            self._publish_status({"command_result": command, "result": result})
        except Exception:
            logger.exception("Error handling command %s", command)
            self._publish_status({"command_result": command, "error": "handler_exception"})

    # ------------------------------------------------------------------
    # Main lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop: connect MQTT, discover devices, stream, handle commands."""
        self._mqtt = RunnerMQTTClient(
            client_id=f"tritium-agent-{self.agent_id}",
            host=self.mqtt_host,
            port=self.mqtt_port,
        )
        connected = self._mqtt.connect()
        if not connected:
            logger.error(
                "Failed to connect to MQTT broker at %s:%d",
                self.mqtt_host,
                self.mqtt_port,
            )

        self._subscribe_commands()
        self._running = True

        self._publish_status({"state": "starting"})

        # Discover and start devices
        self._devices = await self.discover_devices()
        logger.info("Discovered %d device(s)", len(self._devices))
        self._publish_status({"state": "discovered", "device_count": len(self._devices)})

        for dev in self._devices:
            ok = await self.start_device(dev)
            if ok:
                logger.info("Started device: %s", dev.get("id", "unknown"))
            else:
                logger.warning("Failed to start device: %s", dev.get("id", "unknown"))

        self._publish_status({"state": "running", "device_count": len(self._devices)})

        # Keep alive until shutdown
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully stop all devices and disconnect MQTT."""
        self._running = False
        logger.info("Shutting down agent %s", self.agent_id)

        for dev in self._devices:
            device_id = dev.get("id", "")
            if device_id:
                try:
                    await self.stop_device(device_id)
                except Exception:
                    logger.exception("Error stopping device %s", device_id)

        self._publish_status({"state": "offline"})

        if self._mqtt is not None:
            self._mqtt.disconnect()
            self._mqtt = None
