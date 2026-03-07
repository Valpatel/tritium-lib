# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""CoT (Cursor on Target) XML codec for the Tritium ecosystem.

Generates MIL-STD-2045 CoT XML so that edge devices appear in
TAK clients (ATAK, WinTAK, WebTAK) alongside tritium-sc targets.

Zero external dependencies — uses only xml.etree.ElementTree.
"""

from .codec import device_to_cot, sensor_to_cot, parse_cot

__all__ = ["device_to_cot", "sensor_to_cot", "parse_cot"]
