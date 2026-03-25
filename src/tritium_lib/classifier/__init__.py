# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Device classifier — multi-signal BLE and WiFi device type classification.

Uses BLE fingerprint data (GAP appearance, service UUIDs, company IDs, name
patterns, Apple continuity, Google Fast Pair) to produce a rich device type
classification from whatever signals are available.
"""

from tritium_lib.classifier.device_classifier import (
    DeviceClassifier,
    DeviceClassification,
    _is_mac_randomized as is_mac_randomized,
)

__all__ = ["DeviceClassifier", "DeviceClassification", "is_mac_randomized"]
