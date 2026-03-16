# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""License Plate Recognition (LPR) models for vehicle tracking.

LPR integrates with YOLO detection pipeline — when a vehicle is detected,
the plate region is cropped and OCR'd to extract text. Plates feed into
TargetTracker as vehicle identifiers for correlation with BLE/WiFi signals.

MQTT topics:
    tritium/{site}/lpr/{camera_id}/plate  — plate detection events
    tritium/{site}/lpr/alerts             — wanted/flagged plate alerts
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PlateRegion(str, Enum):
    """Geographic region for plate format validation."""

    US = "us"
    EU = "eu"
    UK = "uk"
    ASIA = "asia"
    MIDDLE_EAST = "middle_east"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class PlateColor(str, Enum):
    """Plate background color (can indicate vehicle type)."""

    WHITE = "white"
    YELLOW = "yellow"
    BLUE = "blue"
    GREEN = "green"
    RED = "red"
    BLACK = "black"
    UNKNOWN = "unknown"


class PlateAlert(str, Enum):
    """Alert categories for flagged plates."""

    STOLEN = "stolen"
    WANTED = "wanted"
    BOLO = "bolo"  # be on the lookout
    EXPIRED = "expired"
    AMBER_ALERT = "amber_alert"
    CUSTOM = "custom"
    NONE = "none"


class PlateDetection(BaseModel):
    """A single license plate detection from a camera frame.

    Produced by the LPR pipeline (YOLO vehicle detection + OCR).
    """

    plate_text: str  # normalized text (e.g., "ABC1234")
    plate_text_raw: str = ""  # raw OCR output before normalization
    confidence: float = 0.0  # OCR confidence 0.0-1.0
    region: PlateRegion = PlateRegion.UNKNOWN
    plate_color: PlateColor = PlateColor.UNKNOWN

    # Bounding box in the camera frame (pixels)
    bbox_x: int = 0
    bbox_y: int = 0
    bbox_w: int = 0
    bbox_h: int = 0

    # Source camera and frame
    camera_id: str = ""
    frame_id: str = ""
    frame_timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )

    # Vehicle context from YOLO
    vehicle_type: str = ""  # car, truck, bus, motorcycle
    vehicle_color: str = ""
    vehicle_confidence: float = 0.0  # YOLO confidence for the vehicle detection
    vehicle_bbox_x: int = 0
    vehicle_bbox_y: int = 0
    vehicle_bbox_w: int = 0
    vehicle_bbox_h: int = 0

    # Tritium target mapping
    target_id: str = ""  # e.g., "lpr_{plate_text}"

    def compute_target_id(self) -> str:
        """Generate Tritium target ID from plate text."""
        clean = self.plate_text.replace(" ", "").replace("-", "").upper()
        return f"lpr_{clean}"

    def to_target_dict(self) -> dict:
        """Convert to dict for TargetTracker ingestion."""
        return {
            "target_id": self.compute_target_id(),
            "name": self.plate_text,
            "source": "lpr",
            "asset_type": self.vehicle_type or "vehicle",
            "alliance": "unknown",
            "classification": "vehicle",
            "metadata": {
                "plate_text": self.plate_text,
                "plate_region": self.region.value,
                "vehicle_type": self.vehicle_type,
                "vehicle_color": self.vehicle_color,
                "camera_id": self.camera_id,
                "confidence": self.confidence,
            },
        }


class PlateRecord(BaseModel):
    """Historical record of a plate sighting with position context."""

    plate_text: str
    camera_id: str = ""
    site_id: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )
    confidence: float = 0.0
    vehicle_type: str = ""
    vehicle_color: str = ""
    direction: str = ""  # inbound, outbound, unknown
    speed_estimate_mph: float = 0.0  # if dual-camera speed trap


class PlateWatchlist(BaseModel):
    """A watchlist of flagged license plates."""

    name: str = "default"
    description: str = ""
    entries: list["PlateWatchEntry"] = Field(default_factory=list)

    def check_plate(self, plate_text: str) -> Optional["PlateWatchEntry"]:
        """Check if a plate is on this watchlist."""
        normalized = plate_text.replace(" ", "").replace("-", "").upper()
        for entry in self.entries:
            entry_norm = entry.plate_text.replace(" ", "").replace("-", "").upper()
            if entry_norm == normalized:
                return entry
        return None


class PlateWatchEntry(BaseModel):
    """An entry in a plate watchlist."""

    plate_text: str
    alert_type: PlateAlert = PlateAlert.NONE
    description: str = ""
    added_by: str = ""
    added_at: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )
    expires_at: Optional[float] = None
    notify: bool = True

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now().timestamp() > self.expires_at


class LPRStats(BaseModel):
    """Statistics for the LPR pipeline."""

    total_detections: int = 0
    unique_plates: int = 0
    watchlist_hits: int = 0
    avg_confidence: float = 0.0
    detections_per_camera: dict[str, int] = Field(default_factory=dict)
    top_plates: list[dict] = Field(default_factory=list)  # [{plate, count}]
    last_detection_time: float = 0.0
