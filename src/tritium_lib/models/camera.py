# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Camera source models — frame capture, detection, and source management.

These models define camera sources (RTSP, MJPEG, MQTT, USB, synthetic),
individual frames, and object detection results.  Used by the edge firmware
(camera capture + MQTT publish) and the command center (YOLO pipeline +
display).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CameraSourceType(str, Enum):
    """How a camera feed is ingested."""
    SYNTHETIC = "synthetic"
    RTSP = "rtsp"
    MJPEG = "mjpeg"
    MQTT = "mqtt"
    USB = "usb"


class CameraFrameFormat(str, Enum):
    """Pixel format of a camera frame."""
    JPEG = "jpeg"
    RGB565 = "rgb565"


class CameraPosition(BaseModel):
    """Geographic position and orientation of a camera."""
    lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    lng: Optional[float] = Field(None, ge=-180.0, le=180.0)
    alt: Optional[float] = None  # altitude in meters


class CameraSource(BaseModel):
    """A camera feed source — physical, virtual, or streamed.

    Each source has a unique ID, a type describing how frames are acquired,
    and optional positioning data for GIS integration.
    """
    source_id: str
    name: str = ""
    source_type: CameraSourceType = CameraSourceType.SYNTHETIC
    url: Optional[str] = None  # connection URL for rtsp/mjpeg sources
    enabled: bool = True
    position: CameraPosition = Field(default_factory=CameraPosition)
    fov_degrees: Optional[float] = None  # horizontal field of view
    rotation_degrees: float = 0.0  # clockwise rotation from north

    @property
    def has_position(self) -> bool:
        """True if the camera has a geographic position."""
        return self.position.lat is not None and self.position.lng is not None


class CameraFrame(BaseModel):
    """A single captured frame from a camera source.

    Frame data is not stored in this model — it travels via MQTT binary
    payload.  This metadata accompanies the frame for routing and display.
    """
    source_id: str
    timestamp: Optional[datetime] = None
    width: int = 0
    height: int = 0
    format: CameraFrameFormat = CameraFrameFormat.JPEG

    @property
    def resolution(self) -> str:
        """Human-readable resolution string."""
        return f"{self.width}x{self.height}"


class BoundingBox(BaseModel):
    """Axis-aligned bounding box for a detected object."""
    x: float = 0.0  # top-left x (pixels or normalized 0-1)
    y: float = 0.0  # top-left y
    w: float = 0.0  # width
    h: float = 0.0  # height

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        return (self.x + self.w / 2, self.y + self.h / 2)


class CameraDetection(BaseModel):
    """An object detected in a camera frame (e.g. from YOLO).

    Detections are produced by the SC vision pipeline and associated
    back to the source camera and timestamp for correlation.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "source_id": "cam-front-01",
                    "class_name": "person",
                    "confidence": 0.92,
                    "bbox": {"x": 100, "y": 50, "w": 80, "h": 200},
                }
            ]
        }
    )

    source_id: str = Field(..., min_length=1)
    class_name: str = ""  # detected object class, e.g. "person", "vehicle"
    confidence: float = Field(0.0, ge=0.0, le=1.0)  # 0.0 to 1.0
    bbox: BoundingBox = Field(default_factory=BoundingBox)
    timestamp: Optional[datetime] = None
    # Depth enrichment (optional — set by perception.depth when an aligned
    # depth frame is available, e.g. Isaac Sim or an RGB-D sensor).
    range_m: Optional[float] = Field(
        None, ge=0.0, description="Range to the object in metres, from depth"
    )
    camera_xyz: Optional[tuple[float, float, float]] = Field(
        None,
        description=(
            "Camera-frame 3D point (metres): +x right, +y down, +z forward"
        ),
    )
    # World placement (optional — set by perception.projection.
    # place_detections_on_map when a posed camera transform is available).
    world_enu: Optional[tuple[float, float, float]] = Field(
        None,
        description=(
            "World-frame point relative to the camera's ground footprint "
            "(metres): +east, +north, +up"
        ),
    )
    world_lat: Optional[float] = Field(
        None, ge=-90.0, le=90.0,
        description="Geographic latitude of the detection (degrees)",
    )
    world_lng: Optional[float] = Field(
        None, ge=-180.0, le=180.0,
        description="Geographic longitude of the detection (degrees)",
    )
    # Provenance — WHO produced class_name.  Without this a geometric guess
    # is indistinguishable from a model's output, and a background-subtraction
    # blob reaches the tactical map wearing a COCO class it never earned.
    class_source: str = Field(
        "",
        description=(
            "Origin of class_name: 'classifier' (a model produced it), "
            "'heuristic' (a non-classifying backend guessed it), or '' "
            "(unknown -- treated as unclassified)"
        ),
    )
    shape_hint: Optional[str] = Field(
        None,
        description=(
            "Coarse geometry of the blob ('tall' / 'wide') from a "
            "non-classifying backend.  A hint for downstream fusion, never "
            "an identity -- do NOT render it as a class."
        ),
    )

    @property
    def is_classified(self) -> bool:
        """True only if a real classifier produced ``class_name``.

        Anything else -- a motion blob's aspect-ratio guess, an unstamped
        detection -- is unclassified, and consumers that assign identity
        (tracker, tactical map, alerting) must not treat it as a class.
        """
        return self.class_source == "classifier"

    @property
    def display_label(self) -> str:
        """The badge an operator may honestly be shown.

        An unclassified detection shows what actually happened (``MOTION``),
        not what its geometry resembled.
        """
        return (self.class_name or "unknown").upper()

    @property
    def is_high_confidence(self) -> bool:
        """True if detection confidence is above 0.7."""
        return self.confidence > 0.7

    def to_summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"Detection [{self.source_id}] {self.class_name} "
            f"conf={self.confidence:.2f} bbox=({self.bbox.x:.0f},{self.bbox.y:.0f},"
            f"{self.bbox.w:.0f},{self.bbox.h:.0f})"
        )
