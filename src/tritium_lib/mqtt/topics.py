# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""MQTT topic conventions — single source of truth for topic patterns.

Both tritium-sc and tritium-edge import these to ensure consistency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# --- Device-centric topic constants ---
# These follow the pattern: tritium/devices/{device_id}/{message_type}

TOPIC_HEARTBEAT = "heartbeat"
TOPIC_SENSORS = "sensors"
TOPIC_COMMANDS = "commands"
TOPIC_OTA_STATUS = "ota/status"
TOPIC_FLEET_BROADCAST = "tritium/fleet/broadcast"

_DEVICE_PREFIX = "tritium/devices"


def device_heartbeat(device_id: str) -> str:
    """Topic for device heartbeat messages."""
    return f"{_DEVICE_PREFIX}/{device_id}/{TOPIC_HEARTBEAT}"


def device_sensors(device_id: str, sensor_type: str) -> str:
    """Topic for device sensor readings."""
    return f"{_DEVICE_PREFIX}/{device_id}/{TOPIC_SENSORS}/{sensor_type}"


def device_commands(device_id: str) -> str:
    """Topic for commands sent to a device."""
    return f"{_DEVICE_PREFIX}/{device_id}/{TOPIC_COMMANDS}"


def device_ota_status(device_id: str) -> str:
    """Topic for OTA update status from a device."""
    return f"{_DEVICE_PREFIX}/{device_id}/{TOPIC_OTA_STATUS}"


def fleet_broadcast() -> str:
    """Topic for fleet-wide broadcast messages."""
    return TOPIC_FLEET_BROADCAST


# --- Topic parser ---

_DEVICE_TOPIC_RE = re.compile(
    r"^tritium/devices/(?P<device_id>[^/]+)/(?P<message_type>.+)$"
)

_SITE_TOPIC_RE = re.compile(
    r"^tritium/(?P<site>[^/]+)/(?P<domain>[^/]+)/(?P<device_id>[^/]+)/(?P<data_type>.+)$"
)


@dataclass
class ParsedTopic:
    """Result of parsing a Tritium MQTT topic."""
    device_id: str
    message_type: str
    sensor_type: Optional[str] = None
    # Site-scoped fields (populated by parse_site_topic)
    site: Optional[str] = None
    domain: Optional[str] = None
    data_type: Optional[str] = None


def parse_topic(topic: str) -> Optional[ParsedTopic]:
    """Extract device_id and message_type from a Tritium device topic.

    Returns None if the topic doesn't match the Tritium device pattern.

    Examples:
        >>> parse_topic("tritium/devices/esp32-001/heartbeat")
        ParsedTopic(device_id='esp32-001', message_type='heartbeat')

        >>> parse_topic("tritium/devices/esp32-001/sensors/temperature")
        ParsedTopic(device_id='esp32-001', message_type='sensors/temperature',
                    sensor_type='temperature')
    """
    m = _DEVICE_TOPIC_RE.match(topic)
    if not m:
        return None
    device_id = m.group("device_id")
    message_type = m.group("message_type")
    sensor_type = None
    if message_type.startswith("sensors/"):
        sensor_type = message_type[len("sensors/"):]
    return ParsedTopic(
        device_id=device_id,
        message_type=message_type,
        sensor_type=sensor_type,
    )


def parse_site_topic(topic: str) -> Optional[ParsedTopic]:
    """Parse a site-scoped Tritium topic.

    Handles topics of the form: tritium/{site}/{domain}/{device_id}/{data_type}

    Returns None if the topic doesn't match the site-scoped pattern.

    Examples:
        >>> parse_site_topic("tritium/home/sdr/hackrf-01/spectrum")
        ParsedTopic(device_id='hackrf-01', message_type='sdr/spectrum',
                    site='home', domain='sdr', data_type='spectrum')
    """
    m = _SITE_TOPIC_RE.match(topic)
    if not m:
        return None
    site = m.group("site")
    domain = m.group("domain")
    device_id = m.group("device_id")
    data_type = m.group("data_type")
    return ParsedTopic(
        device_id=device_id,
        message_type=f"{domain}/{data_type}",
        site=site,
        domain=domain,
        data_type=data_type,
    )


# --- Site-scoped topic builder (original API) ---

class TritiumTopics:
    """Topic builder for Tritium MQTT messages (site-scoped).

    This builder uses the site-scoped hierarchy:
      tritium/{site}/{domain}/{device_id}/{data_type}
    """

    def __init__(self, site_id: str = "home"):
        self.site = site_id
        self.prefix = f"tritium/{site_id}"

    # --- Edge device topics (tritium-edge) ---

    def edge_heartbeat(self, device_id: str) -> str:
        return f"{self.prefix}/edge/{device_id}/heartbeat"

    def edge_telemetry(self, device_id: str) -> str:
        return f"{self.prefix}/edge/{device_id}/telemetry"

    def edge_command(self, device_id: str) -> str:
        return f"{self.prefix}/edge/{device_id}/command"

    def edge_ota_status(self, device_id: str) -> str:
        return f"{self.prefix}/edge/{device_id}/ota"

    def edge_capabilities(self, device_id: str) -> str:
        """Topic for device capability advertisement (retained, published on boot)."""
        return f"{self.prefix}/edge/{device_id}/capabilities"

    # --- Sensor topics (shared) ---

    def sensor(self, device_id: str, sensor_type: str) -> str:
        return f"{self.prefix}/sensors/{device_id}/{sensor_type}"

    def sensor_wildcard(self, device_id: str = "+") -> str:
        return f"{self.prefix}/sensors/{device_id}/#"

    # --- Camera topics (shared) ---

    def camera_frame(self, device_id: str) -> str:
        return f"{self.prefix}/cameras/{device_id}/frame"

    def camera_detections(self, device_id: str) -> str:
        return f"{self.prefix}/cameras/{device_id}/detections"

    def camera_command(self, device_id: str) -> str:
        return f"{self.prefix}/cameras/{device_id}/command"

    # --- Audio topics (shared) ---

    def audio_stream(self, device_id: str) -> str:
        return f"{self.prefix}/audio/{device_id}/stream"

    def audio_vad(self, device_id: str) -> str:
        return f"{self.prefix}/audio/{device_id}/vad"

    # --- Mesh topics ---

    def mesh_peers(self, device_id: str) -> str:
        return f"{self.prefix}/mesh/{device_id}/peers"

    # --- Meshtastic topics ---

    def meshtastic_nodes(self, device_id: str) -> str:
        """Topic for Meshtastic node list updates from a bridge device."""
        return f"{self.prefix}/meshtastic/{device_id}/nodes"

    def meshtastic_message(self, device_id: str) -> str:
        """Topic for Meshtastic text messages received by a bridge device."""
        return f"{self.prefix}/meshtastic/{device_id}/message"

    def meshtastic_command(self, device_id: str) -> str:
        """Topic for commands sent to a Meshtastic bridge device."""
        return f"{self.prefix}/meshtastic/{device_id}/command"

    # --- Camera feed topics ---

    def camera_feed(self, device_id: str) -> str:
        """Topic for continuous camera MJPEG feed frames."""
        return f"{self.prefix}/cameras/{device_id}/feed"

    def camera_snapshot(self, device_id: str) -> str:
        """Topic for single camera snapshot requests/responses."""
        return f"{self.prefix}/cameras/{device_id}/snapshot"

    # --- WiFi passive fingerprinting topics ---

    def wifi_probe(self, device_id: str) -> str:
        """Topic for WiFi probe request observations from an edge node."""
        return f"{self.prefix}/edge/{device_id}/wifi_probe"

    def wifi_scan(self, device_id: str) -> str:
        """Topic for WiFi network scan results from an edge node."""
        return f"{self.prefix}/edge/{device_id}/wifi_scan"

    # --- Robot topics (tritium-sc) ---

    def robot_telemetry(self, robot_id: str) -> str:
        return f"{self.prefix}/robots/{robot_id}/telemetry"

    def robot_command(self, robot_id: str) -> str:
        return f"{self.prefix}/robots/{robot_id}/command"

    def robot_thoughts(self, robot_id: str) -> str:
        return f"{self.prefix}/robots/{robot_id}/thoughts"

    # --- System topics ---

    def alerts(self) -> str:
        return f"{self.prefix}/amy/alerts"

    def escalation(self) -> str:
        return f"{self.prefix}/escalation/change"

    # --- Wildcards for subscriptions ---

    def all_edge(self) -> str:
        return f"{self.prefix}/edge/+/#"

    def all_sensors(self) -> str:
        return f"{self.prefix}/sensors/+/#"

    def all_cameras(self) -> str:
        return f"{self.prefix}/cameras/+/#"

    def all_meshtastic(self) -> str:
        return f"{self.prefix}/meshtastic/+/#"

    # --- Meshtastic extended topics ---

    def meshtastic_status(self, device_id: str) -> str:
        """Topic for Meshtastic bridge device status."""
        return f"{self.prefix}/meshtastic/{device_id}/status"

    def meshtastic_position(self, device_id: str) -> str:
        """Topic for Meshtastic node position updates."""
        return f"{self.prefix}/meshtastic/{device_id}/position"

    # --- SDR topics ---

    def sdr_spectrum(self, device_id: str) -> str:
        """Topic for SDR spectrum data from a device."""
        return f"{self.prefix}/sdr/{device_id}/spectrum"

    def sdr_status(self, device_id: str) -> str:
        """Topic for SDR device status."""
        return f"{self.prefix}/sdr/{device_id}/status"

    def sdr_command(self, device_id: str) -> str:
        """Topic for commands sent to an SDR device."""
        return f"{self.prefix}/sdr/{device_id}/command"

    def all_sdr(self) -> str:
        """Wildcard subscription for all SDR device topics."""
        return f"{self.prefix}/sdr/+/#"

    # --- Generic addon device topics ---

    def addon_device(self, domain: str, device_id: str, data_type: str) -> str:
        """Generic topic for any addon domain device.

        Follows the standard pattern: tritium/{site}/{domain}/{device_id}/{data_type}
        """
        return f"{self.prefix}/{domain}/{device_id}/{data_type}"

    def addon_device_status(self, domain: str, device_id: str) -> str:
        """Status topic for a generic addon device."""
        return self.addon_device(domain, device_id, "status")

    def addon_device_command(self, domain: str, device_id: str) -> str:
        """Command topic for a generic addon device."""
        return self.addon_device(domain, device_id, "command")

    def all_addon_domain(self, domain: str) -> str:
        """Wildcard subscription for all devices in an addon domain."""
        return f"{self.prefix}/{domain}/+/#"
