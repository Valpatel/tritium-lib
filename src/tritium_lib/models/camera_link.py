# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Camera-to-target detection link models.

Structured associations between camera detections and tracked targets.
When a YOLO detection occurs within a camera's known FOV, a
CameraDetectionLink records the association for the target dossier
and downstream analytics.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field


class FramePosition(BaseModel):
    """Normalized position within a camera frame (0.0-1.0)."""
    x: float = 0.0  # horizontal position (0=left, 1=right)
    y: float = 0.0  # vertical position (0=top, 1=bottom)


class CameraDetectionLink(BaseModel):
    """Links a camera detection to a target via camera FOV geometry.

    Created when a YOLO detection (person, vehicle, etc.) occurs within
    a camera's known field of view.  The link records which camera saw
    the target, where in the frame, and with what confidence.

    Used for:
    - Dossier enrichment (camera sighting history)
    - Cross-camera correlation (same target seen by multiple cameras)
    - Coverage analysis (which cameras are producing useful detections)
    """

    link_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    detection_id: str = ""  # ID of the YOLO detection event
    camera_id: str = ""     # source_id of the CameraSource
    target_id: str = ""     # target_id in TargetTracker or dossier_id
    class_name: str = ""    # detected class: person, vehicle, etc.
    position_in_frame: FramePosition = Field(default_factory=FramePosition)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: float = Field(default_factory=time.time)
    camera_fov_degrees: Optional[float] = None  # camera FOV at time of detection
    camera_rotation: Optional[float] = None  # camera rotation at time of detection
    bbox_area: float = 0.0  # normalized bounding box area (0.0-1.0)

    @property
    def is_high_confidence(self) -> bool:
        """True if the link confidence is above 0.7."""
        return self.confidence > 0.7

    def to_signal_dict(self) -> dict:
        """Convert to a DossierSignal-compatible data dict."""
        return {
            "link_id": self.link_id,
            "detection_id": self.detection_id,
            "camera_id": self.camera_id,
            "class_name": self.class_name,
            "position_in_frame": self.position_in_frame.model_dump(),
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "camera_fov_degrees": self.camera_fov_degrees,
            "camera_rotation": self.camera_rotation,
            "bbox_area": self.bbox_area,
        }


class CameraLinkSummary(BaseModel):
    """Summary of camera-target links for a specific camera or target."""

    entity_id: str = ""  # camera_id or target_id
    total_links: int = 0
    unique_targets: int = 0
    unique_cameras: int = 0
    avg_confidence: float = 0.0
    first_link: float = 0.0  # timestamp of first link
    last_link: float = 0.0   # timestamp of most recent link
    class_distribution: dict[str, int] = Field(default_factory=dict)
