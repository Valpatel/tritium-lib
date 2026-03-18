# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Position anchoring and sensor fusion foundation models.

GPS-anchored devices create trust anchors. Every detection relationship
(BLE, LoRa, WiFi, camera, acoustic) creates edges that converge on
real-world positions. This is the core of the unified operating picture.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class PositionAnchor(BaseModel):
    """A known position with confidence — the trust anchor for sensor fusion.

    GPS-equipped devices (mesh nodes, phones, survey points) report their
    position. Each becomes an anchor that can locate nearby detections.
    Higher confidence = more trust in the position.
    """

    anchor_id: str
    lat: float
    lng: float
    alt: float | None = None
    source: str = "gps"  # "gps", "manual", "survey", "cell_tower", "wifi_rtt"
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="1.0 = surveyed, 0.9 = GPS fix, 0.5 = estimated",
    )
    timestamp: float = Field(default_factory=time.time)
    device_id: str | None = None
    fixed: bool = False  # True for stationary nodes (house, roof node)
    label: str = ""


class DetectionEdge(BaseModel):
    """A sensor detected something — the edge in the position graph.

    Every time a device detects another (BLE scan, LoRa packet, WiFi probe,
    camera frame, acoustic hit), it creates an edge. The detector's known
    position plus the detection's signal strength constrains the detected
    entity's location.
    """

    detector_id: str
    detected_id: str
    detection_type: str  # "ble", "lora", "wifi", "camera", "acoustic"
    rssi: float | None = None  # dBm
    snr: float | None = None  # signal-to-noise ratio
    distance_estimate_m: float | None = None
    timestamp: float = Field(default_factory=time.time)
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Detection reliability",
    )


class FusedPositionEstimate(BaseModel):
    """Computed position for a non-GPS device — the fusion output.

    When a target has no GPS but is detected by one or more anchored
    sensors, we estimate its position from the detection edges and
    anchor positions.
    """

    target_id: str
    lat: float
    lng: float
    accuracy_m: float = Field(
        default=50.0,
        ge=0.0,
        description="Estimated accuracy radius in meters",
    )
    method: str = "proximity"  # "proximity", "trilateration", "centroid", "rssi_model"
    anchor_count: int = 1
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    timestamp: float = Field(default_factory=time.time)
