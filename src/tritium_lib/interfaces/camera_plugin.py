# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Generic camera plugin interface.

Extends SensorPlugin with camera-specific operations: frame capture,
detection retrieval, and resolution control. RTSP, USB, MQTT, and
synthetic camera sources implement this interface.

Plugin Architecture:
    SensorPlugin (base)
      +-- CameraPlugin (this file — generic camera)
            +-- RTSPCamera (specific)
            +-- USBCamera (specific)
            +-- MQTTCamera (specific)
            +-- SyntheticCamera (specific)

MQTT topics (published by camera plugins):
    tritium/{site}/cameras/{id}/frame      — JPEG frames
    tritium/{site}/cameras/{id}/detections — YOLO detection results
"""

from abc import abstractmethod

from tritium_lib.interfaces.sensor_plugin import SensorPlugin


class CameraPlugin(SensorPlugin):
    """Generic camera plugin interface.

    RTSP, USB, MQTT, and synthetic cameras implement this interface.
    Provides frame capture, object detection retrieval, and resolution
    control.

    Implementations must also satisfy SensorPlugin methods (get_name,
    get_sensor_type, start, stop, get_status, get_mqtt_topics, get_capabilities).
    """

    @abstractmethod
    def get_frame(self) -> bytes:
        """Capture and return the current frame as JPEG bytes."""
        ...

    @abstractmethod
    def get_detections(self) -> list[dict]:
        """Return recent object detections.

        Each detection is a dict with at minimum:
            - 'class': str (e.g., 'person', 'vehicle')
            - 'confidence': float (0.0-1.0)
            - 'bbox': [x1, y1, x2, y2] (pixel coordinates)
        """
        ...

    @abstractmethod
    def set_resolution(self, width: int, height: int) -> None:
        """Set the camera capture resolution."""
        ...
