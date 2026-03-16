# AGPL-3.0 — Copyright 2026 Valpatel Software LLC
"""Static data files for tritium-lib (BLE fingerprints, lookup tables, etc.)."""

import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent


def load_ble_fingerprints() -> dict:
    """Load the BLE device fingerprinting lookup tables."""
    with open(_DATA_DIR / "ble_fingerprints.json") as f:
        return json.load(f)
