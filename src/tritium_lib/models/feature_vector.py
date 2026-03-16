# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Feature vector models for edge-to-cloud ML intelligence pipeline.

FeatureVector represents a compact feature extraction from an edge sensor
(BLE, WiFi, etc.) that can be aggregated and fed to ML classifiers
without transmitting raw advertisement data.

ClassificationFeedback represents the SC-to-edge feedback loop: when SC
classifies a device, it sends the result back to the edge node so the
edge can cache and include it in future sightings.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FeatureSource(str, Enum):
    """Source sensor type for feature extraction."""
    BLE = "ble"
    WIFI = "wifi"
    ACOUSTIC = "acoustic"
    RF = "rf"
    CAMERA = "camera"


class FeatureVector(BaseModel):
    """Compact feature vector extracted by an edge node from sensor data.

    Attributes:
        source_id: Edge node identifier (device_id).
        mac: Target device MAC address (for BLE/WiFi features).
        source_type: Sensor type that produced this feature vector.
        features: Dictionary of named feature values (floats).
        version: Feature extraction algorithm version for compatibility.
        timestamp: When the features were extracted.
    """
    source_id: str
    mac: str = ""
    source_type: FeatureSource = FeatureSource.BLE
    features: dict[str, float] = Field(default_factory=dict)
    version: int = 1
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def feature_list(self, keys: list[str] | None = None) -> list[float]:
        """Return features as an ordered list of floats.

        Args:
            keys: Ordered list of feature names. If None, uses sorted keys.

        Returns:
            List of float values in the specified order.
        """
        if keys is None:
            keys = sorted(self.features.keys())
        return [self.features.get(k, 0.0) for k in keys]


class AggregatedFeatures(BaseModel):
    """Aggregated feature vectors for a single device across edge nodes.

    Attributes:
        mac: Target device MAC address.
        vectors: List of individual feature vectors from different nodes/times.
        node_count: Number of unique edge nodes that contributed features.
        first_seen: Earliest feature extraction timestamp.
        last_seen: Most recent feature extraction timestamp.
        mean_features: Averaged feature values across all vectors.
    """
    mac: str
    vectors: list[FeatureVector] = Field(default_factory=list)
    node_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    mean_features: dict[str, float] = Field(default_factory=dict)

    def compute_mean(self) -> dict[str, float]:
        """Compute mean feature values across all vectors.

        Returns:
            Dictionary of feature name to mean value.
        """
        if not self.vectors:
            return {}

        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        for v in self.vectors:
            for k, val in v.features.items():
                sums[k] = sums.get(k, 0.0) + val
                counts[k] = counts.get(k, 0) + 1

        result = {k: sums[k] / counts[k] for k in sums}
        self.mean_features = result
        return result


class ClassificationFeedback(BaseModel):
    """Classification result sent from SC back to an edge node.

    When the SC classifier determines a device type, this feedback
    is published via MQTT so the edge node can cache the classification
    and include it in future sighting reports.

    Attributes:
        mac: Device MAC address that was classified.
        predicted_type: Predicted device type (phone, watch, laptop, etc.).
        confidence: Classification confidence (0.0 to 1.0).
        confirmed_by: How the classification was confirmed.
        model_version: ML model version that made this prediction.
        timestamp: When the classification was made.
    """
    mac: str
    predicted_type: str
    confidence: float = 0.0
    confirmed_by: str = "ml_classifier"
    model_version: str = ""
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class EdgeIntelligenceMetrics(BaseModel):
    """Per-edge-node ML intelligence metrics.

    Tracks how many devices an edge node has seen, how many have been
    classified, and the feedback loop health.

    Attributes:
        node_id: Edge device identifier.
        total_devices_seen: Total unique MACs observed.
        devices_classified: Devices with ML classification.
        feedback_received: Number of classification feedback messages received.
        accuracy_rate: Estimated accuracy (confirmed correct / total classified).
        feature_vectors_sent: Total feature vectors published to SC.
        last_feedback_ts: When last classification feedback was received.
        last_feature_ts: When last feature vector was sent.
    """
    node_id: str
    total_devices_seen: int = 0
    devices_classified: int = 0
    feedback_received: int = 0
    accuracy_rate: float = 0.0
    feature_vectors_sent: int = 0
    last_feedback_ts: Optional[datetime] = None
    last_feature_ts: Optional[datetime] = None
