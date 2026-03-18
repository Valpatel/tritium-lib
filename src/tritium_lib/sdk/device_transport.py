# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""DeviceTransport abstraction for the Tritium Addon SDK.

Provides a uniform interface for communicating with devices regardless
of whether they are connected locally (USB/serial) or remotely (MQTT).
"""

from abc import ABC, abstractmethod
from typing import Any, Callable


class DeviceTransport(ABC):
    """Abstract base class for device communication transports."""

    def __init__(self) -> None:
        self._data_callbacks: list[Callable[[dict], None]] = []

    @property
    @abstractmethod
    def transport_type(self) -> str:
        """Transport identifier: 'local', 'mqtt', 'ssh', etc."""
        ...

    @abstractmethod
    async def connect(self) -> bool:
        """Establish transport connection. Returns True on success."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down the transport connection."""
        ...

    @abstractmethod
    async def send_command(
        self, command: str, payload: dict | None = None
    ) -> dict | None:
        """Send a command over the transport and return the response."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the transport is currently connected."""
        ...

    def on_data(self, callback: Callable[[dict], None]) -> None:
        """Register a callback to receive incoming data."""
        self._data_callbacks.append(callback)

    def _emit_data(self, data: dict) -> None:
        """Invoke all registered data callbacks."""
        for cb in self._data_callbacks:
            cb(data)


class LocalTransport(DeviceTransport):
    """Transport for USB/serial devices connected locally.

    Actual subprocess management is delegated to SubprocessManager;
    this class handles the transport abstraction layer only.
    """

    def __init__(self, device_path: str, device_type: str = "unknown") -> None:
        super().__init__()
        self._device_path = device_path
        self._device_type = device_type
        self._connected = False
        self._last_command: tuple[str, dict | None] | None = None

    @property
    def transport_type(self) -> str:
        return "local"

    @property
    def device_path(self) -> str:
        return self._device_path

    @property
    def device_type(self) -> str:
        return self._device_type

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def send_command(
        self, command: str, payload: dict | None = None
    ) -> dict | None:
        self._last_command = (command, payload)
        return {"status": "ok"}


class MQTTTransport(DeviceTransport):
    """Transport for remote devices reachable via an MQTT broker."""

    def __init__(
        self,
        device_id: str,
        domain: str,
        site_id: str = "home",
        mqtt_client: Any = None,
    ) -> None:
        super().__init__()
        self._device_id = device_id
        self._domain = domain
        self._site_id = site_id
        self._mqtt_client = mqtt_client
        self._connected = False
        self._last_command: tuple[str, dict | None] | None = None

    @property
    def transport_type(self) -> str:
        return "mqtt"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def command_topic(self) -> str:
        return f"tritium/{self._site_id}/{self._domain}/{self._device_id}/command"

    @property
    def status_topic(self) -> str:
        return f"tritium/{self._site_id}/{self._domain}/{self._device_id}/status"

    @property
    def data_topic(self) -> str:
        return f"tritium/{self._site_id}/{self._domain}/{self._device_id}/data"

    async def connect(self) -> bool:
        if self._mqtt_client is not None:
            try:
                self._mqtt_client.subscribe(self.status_topic)
                self._mqtt_client.subscribe(self.data_topic)
            except Exception:
                return False
        self._connected = True
        return True

    async def disconnect(self) -> None:
        if self._mqtt_client is not None:
            try:
                self._mqtt_client.unsubscribe(self.status_topic)
                self._mqtt_client.unsubscribe(self.data_topic)
            except Exception:
                pass
        self._connected = False

    async def send_command(
        self, command: str, payload: dict | None = None
    ) -> dict | None:
        import json

        message = {"command": command}
        if payload is not None:
            message["payload"] = payload

        if self._mqtt_client is not None:
            try:
                self._mqtt_client.publish(self.command_topic, json.dumps(message))
                return {"status": "sent"}
            except Exception:
                return {"status": "error"}
        else:
            self._last_command = (command, payload)
            return {"status": "queued"}

    def on_mqtt_message(self, topic: str, payload: dict) -> None:
        """Route an incoming MQTT message to registered data callbacks."""
        if topic == self.data_topic or topic == self.status_topic:
            self._emit_data(payload)
