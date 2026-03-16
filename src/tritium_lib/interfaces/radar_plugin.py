# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Generic radar plugin interface.

Extends SensorPlugin with radar-specific operations: range reporting,
track acquisition, scan control. Any radar system (Aeris-10, passive
bistatic radar, FMCW modules) implements this interface.

Plugin Architecture:
    SensorPlugin (base)
      +-- RadarPlugin (this file — generic radar)
            +-- Aeris10Plugin (specific)
            +-- PassiveRadarPlugin (specific)
            +-- FMCWRadarPlugin (specific)

MQTT topics (published by radar plugins):
    tritium/{site}/radar/{radar_id}/scan   — complete scan with tracks
    tritium/{site}/radar/{radar_id}/track  — individual track updates
    tritium/{site}/radar/{radar_id}/config — radar configuration
"""

from abc import abstractmethod

from tritium_lib.interfaces.sensor_plugin import SensorPlugin
from tritium_lib.models.radar import RadarConfig, RadarScan, RadarTrack


class RadarPlugin(SensorPlugin):
    """Generic radar plugin interface.

    Aeris-10, passive radar, FMCW modules, and other radar hardware
    implement this interface. Provides range/track reporting, scan
    control, and configuration management.

    Implementations must also satisfy SensorPlugin methods (get_name,
    get_sensor_type, start, stop, get_status, get_mqtt_topics, get_capabilities).
    """

    @abstractmethod
    def get_max_range(self) -> float:
        """Return maximum detection range in meters."""
        ...

    @abstractmethod
    def get_tracks(self) -> list[RadarTrack]:
        """Return currently tracked targets.

        Each RadarTrack includes range, azimuth, velocity, and classification.
        """
        ...

    @abstractmethod
    def start_scanning(self, config: RadarConfig) -> None:
        """Start radar scanning with the given configuration.

        The plugin will begin acquiring tracks and publishing to MQTT.
        """
        ...

    @abstractmethod
    def get_scan(self) -> RadarScan:
        """Return the most recent complete radar scan.

        A scan includes all tracks from one antenna rotation or dwell period.
        """
        ...
