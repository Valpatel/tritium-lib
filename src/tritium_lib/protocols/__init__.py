# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.protocols — Pure-Python parsers for common radio protocols.

Each parser accepts raw bytes, hex strings, or text and returns structured
data (dataclasses or dicts).  All parsers handle malformed input gracefully
by raising ``ParseError`` with a descriptive message.

Supported protocols:
    - BLE advertisement parsing (flags, service UUIDs, manufacturer data)
    - WiFi probe request field parsing
    - AIS (Automatic Identification System) — vessel position reports
    - ADS-B (Automatic Dependent Surveillance-Broadcast) — aircraft tracking
    - Meshtastic — LoRa mesh protocol message parsing
    - NMEA — GPS sentence parsing (GGA, RMC, GSA, VTG, GLL)
"""

from tritium_lib.protocols.ble_advert import BLEAdvertParser
from tritium_lib.protocols.wifi_probe import WiFiProbeParser
from tritium_lib.protocols.ais import AISParser
from tritium_lib.protocols.adsb import ADSBParser
from tritium_lib.protocols.meshtastic import MeshtasticParser
from tritium_lib.protocols.nmea import NMEAParser
from tritium_lib.protocols.errors import ParseError

__all__ = [
    "BLEAdvertParser",
    "WiFiProbeParser",
    "AISParser",
    "ADSBParser",
    "MeshtasticParser",
    "NMEAParser",
    "ParseError",
]
