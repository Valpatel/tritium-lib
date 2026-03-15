# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""AcousticFeatureVector model for edge-to-SC MFCC feature transport.

Provides a compact, MQTT-serializable representation of audio features
extracted on ESP32 edge devices (via hal_acoustic). The SC receives these
feature vectors for ML classification without needing raw audio.

MQTT topic:
    tritium/{site}/acoustic/{device_id}/features — published by edge
"""

import json
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AcousticFeatureVector(BaseModel):
    """MFCC and spectral features extracted from I2S microphone data on edge.

    This is the transport model for acoustic ML features. Edge devices compute
    MFCCs and spectral descriptors locally, then publish this compact payload
    over MQTT for SC-side classification and intelligence pipelines.

    Attributes:
        device_id: Edge device identifier that captured the audio.
        timestamp: Unix timestamp of the audio capture.
        mfcc_coefficients: 13 Mel-frequency cepstral coefficients.
        energy: RMS energy of the audio segment (0.0-1.0).
        zero_crossing_rate: Zero crossings per sample (0.0-1.0).
        spectral_centroid: Center of mass of the spectrum in Hz.
        duration_ms: Duration of the audio segment in milliseconds.
        sample_rate: Sample rate used for capture (typically 16000).
        classification: Optional classification label from edge rule-based classifier.
        confidence: Optional classification confidence (0.0-1.0).
    """

    device_id: str
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )
    mfcc_coefficients: list[float] = Field(
        default_factory=lambda: [0.0] * 13,
        description="13 Mel-frequency cepstral coefficients",
    )
    energy: float = 0.0
    zero_crossing_rate: float = 0.0
    spectral_centroid: float = 0.0
    duration_ms: int = 0
    sample_rate: int = 16000
    classification: Optional[str] = None
    confidence: Optional[float] = None

    def to_mqtt_payload(self) -> str:
        """Serialize to compact JSON string for MQTT publishing.

        Returns:
            JSON string suitable for MQTT payload. Uses short keys where
            possible to minimize bandwidth on constrained links.
        """
        payload: dict = {
            "device_id": self.device_id,
            "ts": self.timestamp,
            "mfcc": self.mfcc_coefficients,
            "energy": round(self.energy, 4),
            "zcr": round(self.zero_crossing_rate, 4),
            "sc": round(self.spectral_centroid, 1),
            "dur_ms": self.duration_ms,
            "sr": self.sample_rate,
        }
        if self.classification is not None:
            payload["cls"] = self.classification
        if self.confidence is not None:
            payload["conf"] = round(self.confidence, 3)
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def from_mqtt_payload(cls, payload: str) -> "AcousticFeatureVector":
        """Deserialize from compact MQTT JSON payload.

        Args:
            payload: JSON string from MQTT message (compact key format).

        Returns:
            AcousticFeatureVector instance.
        """
        data = json.loads(payload)
        return cls(
            device_id=data["device_id"],
            timestamp=data.get("ts", 0.0),
            mfcc_coefficients=data.get("mfcc", [0.0] * 13),
            energy=data.get("energy", 0.0),
            zero_crossing_rate=data.get("zcr", 0.0),
            spectral_centroid=data.get("sc", 0.0),
            duration_ms=data.get("dur_ms", 0),
            sample_rate=data.get("sr", 16000),
            classification=data.get("cls"),
            confidence=data.get("conf"),
        )
