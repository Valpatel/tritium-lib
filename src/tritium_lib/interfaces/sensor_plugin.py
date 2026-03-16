# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Base sensor plugin interface for the Tritium plugin system.

All sensor plugins inherit from SensorPlugin, which defines the minimal
contract every sensor must fulfill: lifecycle (start/stop), status reporting,
capability advertisement, and MQTT topic registration.

Plugin Architecture:
    SensorPlugin (base)
      +-- SDRPlugin (generic SDR)
      |     +-- HackRFPlugin (specific)
      |     +-- RTLSDRPlugin (specific)
      |     +-- AirspyPlugin (specific)
      |     +-- LimeSDRPlugin (specific)
      +-- RadarPlugin (generic radar)
      |     +-- Aeris10Plugin (specific)
      |     +-- PassiveRadarPlugin (specific)
      +-- CameraPlugin (generic camera)
      |     +-- RTSPCamera (specific)
      |     +-- USBCamera (specific)
      |     +-- MQTTCamera (specific)
      +-- BLEPlugin (future)
      +-- AcousticPlugin (future)
"""

from abc import ABC, abstractmethod


class SensorPlugin(ABC):
    """Base interface for all sensor plugins.

    Every sensor in the Tritium ecosystem — SDR receivers, radars, cameras,
    BLE scanners, acoustic arrays — implements this interface. It provides:

    - Identity: name and sensor type for registry/discovery
    - Capabilities: what this sensor can do (e.g., "spectrum_scan", "object_detection")
    - Lifecycle: start/stop control
    - Status: health and operational state
    - MQTT: which topics this plugin publishes to
    """

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable name of this plugin instance (e.g., 'Rooftop RTL-SDR')."""
        ...

    @abstractmethod
    def get_sensor_type(self) -> str:
        """Sensor category: 'sdr', 'radar', 'camera', 'ble', 'acoustic', etc."""
        ...

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """List of capability tags this plugin provides.

        Examples: ['spectrum_scan', 'signal_decode', 'ism_decode'],
                  ['track_detection', 'doppler'], ['object_detection', 'streaming'].
        """
        ...

    @abstractmethod
    def start(self) -> None:
        """Start the sensor plugin. Begins data acquisition."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the sensor plugin. Releases hardware resources."""
        ...

    @abstractmethod
    def get_status(self) -> dict:
        """Return current operational status.

        Should include at minimum:
            - 'running': bool
            - 'error': str or None
            - 'uptime_s': float (seconds since start)
        Implementations may add sensor-specific fields.
        """
        ...

    @abstractmethod
    def get_mqtt_topics(self) -> list[str]:
        """Return the MQTT topics this plugin publishes to.

        Used for topic registration and routing configuration.
        """
        ...
