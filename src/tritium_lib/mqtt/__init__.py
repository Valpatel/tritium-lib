# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""MQTT topic conventions and codecs for the Tritium ecosystem.

All Tritium services use the same topic hierarchy:
  tritium/{site}/{domain}/{device_id}/{data_type}

Device-centric topics (flat namespace):
  tritium/devices/{device_id}/{message_type}

This module defines the topic patterns and JSON codecs so that
tritium-sc and tritium-edge speak the same language.
"""

from .topics import (
    TritiumTopics,
    ParsedTopic,
    parse_topic,
    parse_site_topic,
    device_heartbeat,
    device_sensors,
    device_commands,
    device_ota_status,
    fleet_broadcast,
    TOPIC_HEARTBEAT,
    TOPIC_SENSORS,
    TOPIC_COMMANDS,
    TOPIC_OTA_STATUS,
    TOPIC_FLEET_BROADCAST,
)

__all__ = [
    "TritiumTopics",
    "ParsedTopic",
    "parse_topic",
    "parse_site_topic",
    "device_heartbeat",
    "device_sensors",
    "device_commands",
    "device_ota_status",
    "fleet_broadcast",
    "TOPIC_HEARTBEAT",
    "TOPIC_SENSORS",
    "TOPIC_COMMANDS",
    "TOPIC_OTA_STATUS",
    "TOPIC_FLEET_BROADCAST",
]
