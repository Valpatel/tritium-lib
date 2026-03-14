# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""DeviceClassifier — multi-signal BLE and WiFi device type classification.

Combines all available signals (MAC OUI, device name, GAP appearance,
service UUIDs, company ID, Apple continuity data, Google Fast Pair model ID)
to produce the best possible device type classification.

Each signal contributes a (device_type, confidence) vote.  The final
classification picks the highest-confidence vote, with ties broken by
signal priority (appearance > service_uuid > company_id > name_pattern > oui).

Usage::

    from tritium_lib.classifier import DeviceClassifier

    dc = DeviceClassifier()
    result = dc.classify_ble(mac="AC:BC:32:AA:BB:CC", name="iPhone 15")
    print(result.device_type)   # "phone"
    print(result.confidence)    # 0.9
    print(result.manufacturer)  # "Apple"

    wifi = dc.classify_wifi(ssid="DIRECT-HP-Printer", bssid="00:17:88:AA:BB:CC")
    print(wifi.device_type)     # "printer"
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("device-classifier")

# Path to the bundled fingerprint data
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_FINGERPRINTS_PATH = os.path.join(_DATA_DIR, "ble_fingerprints.json")


@dataclass
class DeviceClassification:
    """Result of device classification."""

    device_type: str = "unknown"
    device_name: str = ""
    manufacturer: str = ""
    confidence: float = 0.0
    signals: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "manufacturer": self.manufacturer,
            "confidence": self.confidence,
            "signals": self.signals,
        }


# OUI prefix -> manufacturer (common entries for offline lookup)
_OUI_MANUFACTURERS: dict[str, str] = {
    "00:17:88": "Philips Lighting",
    "DC:A6:32": "Raspberry Pi",
    "B8:27:EB": "Raspberry Pi",
    "E4:5F:01": "Raspberry Pi",
    "28:CD:C1": "Raspberry Pi",
    "24:0A:C4": "Espressif",
    "30:AE:A4": "Espressif",
    "3C:61:05": "Espressif",
    "3C:71:BF": "Espressif",
    "40:F5:20": "Espressif",
    "48:3F:DA": "Espressif",
    "58:CF:79": "Espressif",
    "7C:9E:BD": "Espressif",
    "84:0D:8E": "Espressif",
    "84:CC:A8": "Espressif",
    "8C:AA:B5": "Espressif",
    "94:3C:C6": "Espressif",
    "A4:CF:12": "Espressif",
    "AC:67:B2": "Espressif",
    "B4:E6:2D": "Espressif",
    "BC:DD:C2": "Espressif",
    "C4:4F:33": "Espressif",
    "CC:50:E3": "Espressif",
    "D8:A0:1D": "Espressif",
    "E8:68:E7": "Espressif",
    "F0:08:D1": "Espressif",
    "F4:12:FA": "Espressif",
    "00:0A:95": "Apple",
    "3C:E0:72": "Apple",
    "F0:18:98": "Apple",
    "AC:BC:32": "Apple",
    "F8:FF:C2": "Apple",
    "00:25:00": "Apple",
    "A4:83:E7": "Apple",
    "40:B4:CD": "Samsung",
    "8C:F5:A3": "Samsung",
    "00:1E:75": "Samsung",
    "78:47:1D": "Samsung",
    "60:AF:6D": "Samsung",
}

# OUI manufacturer -> likely device types
_OUI_DEVICE_HINTS: dict[str, tuple[str, float]] = {
    "Espressif": ("microcontroller", 0.6),
    "Raspberry Pi": ("microcontroller", 0.6),
    "Philips Lighting": ("smart_home", 0.7),
}

# BLE device name patterns -> (device_type, confidence)
_BLE_NAME_PATTERNS: list[tuple[str, str, float]] = [
    (r"(?i)^iPhone", "phone", 0.9),
    (r"(?i)^Samsung", "phone", 0.8),
    (r"(?i)^Pixel", "phone", 0.85),
    (r"(?i)^Galaxy\s?(S|A|Z|Note|Fold|Flip)", "phone", 0.85),
    (r"(?i)^Galaxy\s?Watch", "watch", 0.9),
    (r"(?i)^Galaxy\s?Buds", "earbuds", 0.9),
    (r"(?i)^Galaxy\s?Tab", "tablet", 0.85),
    (r"(?i)Watch", "watch", 0.8),
    (r"(?i)^Fitbit", "fitness", 0.9),
    (r"(?i)^Garmin", "watch", 0.9),
    (r"(?i)AirPod", "earbuds", 0.95),
    (r"(?i)^Bose", "headphones", 0.85),
    (r"(?i)^Sony.*WH", "headphones", 0.85),
    (r"(?i)^Sony.*WF", "earbuds", 0.85),
    (r"(?i)^JBL", "speaker", 0.8),
    (r"(?i)^UE.*BOOM", "speaker", 0.85),
    (r"(?i)^Marshall", "headphones", 0.8),
    (r"(?i)MacBook", "computer", 0.9),
    (r"(?i)^iMac", "computer", 0.9),
    (r"(?i)^iPad", "tablet", 0.9),
    (r"(?i)^Fire.*HD", "tablet", 0.8),
    (r"(?i)^Tile", "tag", 0.9),
    (r"(?i)^AirTag", "tag", 0.95),
    (r"(?i)^Chipolo", "tag", 0.9),
    (r"(?i)^Tesla", "vehicle", 0.8),
    (r"(?i)^Govee", "smart_home", 0.85),
    (r"(?i)^Wyze", "camera", 0.8),
    (r"(?i)^Ring", "camera", 0.8),
    (r"(?i)^ESP32", "microcontroller", 0.9),
    (r"(?i)^Raspberry", "microcontroller", 0.85),
    (r"(?i)^Nintendo", "gamepad", 0.9),
    (r"(?i)^Xbox", "gamepad", 0.9),
    (r"(?i)^DualSense", "gamepad", 0.9),
    (r"(?i)^Meta\s?Quest", "vr_headset", 0.9),
    (r"(?i)^Oculus", "vr_headset", 0.9),
    (r"(?i)^OnePlus", "phone", 0.85),
    (r"(?i)^Xiaomi", "phone", 0.7),
    (r"(?i)^Huawei", "phone", 0.7),
    (r"(?i)^OPPO", "phone", 0.7),
    (r"(?i)^Sonos", "speaker", 0.9),
    (r"(?i)^HomePod", "smart_speaker", 0.95),
    (r"(?i)^Echo", "smart_speaker", 0.85),
    (r"(?i)^Google Home", "smart_speaker", 0.9),
    (r"(?i)^Nest", "smart_home", 0.8),
]

# WiFi SSID patterns -> (device_type, confidence)
_WIFI_SSID_PATTERNS: list[tuple[str, str, float]] = [
    (r"(?i)^iPhone", "phone", 0.9),
    (r"(?i)^Android[_\- ]", "phone", 0.85),
    (r"(?i)^Galaxy[_\- ]", "phone", 0.85),
    (r"(?i)^Pixel[_\- ]", "phone", 0.85),
    (r"(?i)^DIRECT-", "printer", 0.7),
    (r"(?i)^HP-", "printer", 0.7),
    (r"(?i)^ChromeCast", "media_player", 0.8),
    (r"(?i)^Roku", "media_player", 0.8),
    (r"(?i)^FireTV", "media_player", 0.8),
    (r"(?i)^Ring[_\- ]", "camera", 0.8),
    (r"(?i)^Nest[_\- ]", "smart_home", 0.75),
    (r"(?i)^Amazon[_\- ]", "smart_home", 0.6),
    (r"(?i)^Echo[_\- ]", "smart_home", 0.75),
    (r"(?i)MacBook", "computer", 0.85),
    (r"(?i)^LAPTOP-", "computer", 0.8),
    (r"(?i)^DESKTOP-", "computer", 0.8),
    (r"(?i)Tesla", "vehicle", 0.7),
    (r"(?i)^xfinitywifi$", "hotspot", 0.5),
    (r"(?i)^ATT.*Hotspot", "hotspot", 0.5),
]


def _normalize_hex_key(value: str) -> str:
    """Normalize a hex key to match JSON format: 0x prefix with uppercase hex digits.

    JSON keys in ble_fingerprints.json use the format ``0xABCD`` (lowercase ``0x``
    prefix, uppercase hex digits).  Input may be ``0X02``, ``02``, ``0x02`` etc.
    """
    v = value.strip()
    # Strip any 0x/0X prefix
    if v.upper().startswith("0X"):
        v = v[2:]
    # Re-add lowercase 0x prefix with uppercase hex digits
    return f"0x{v.upper()}"


class DeviceClassifier:
    """Multi-signal device type classifier using BLE fingerprint data.

    Loads fingerprint lookup tables from tritium-lib's bundled JSON data.
    Combines all available signals to produce the best classification.

    Parameters
    ----------
    fingerprints_path:
        Override path to ble_fingerprints.json.  Defaults to the bundled copy.
    """

    def __init__(self, fingerprints_path: str | None = None) -> None:
        self._fp_path = fingerprints_path or _FINGERPRINTS_PATH
        self._data: dict[str, Any] = {}
        self._loaded = False
        self._load_fingerprints()

    def _load_fingerprints(self) -> None:
        """Load BLE fingerprint data from JSON."""
        try:
            with open(self._fp_path, "r") as f:
                self._data = json.load(f)
            self._loaded = True
            logger.debug("Loaded BLE fingerprints from %s", self._fp_path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load BLE fingerprints: %s", exc)
            self._data = {}

    @property
    def loaded(self) -> bool:
        """Whether fingerprint data was successfully loaded."""
        return self._loaded

    # ------------------------------------------------------------------
    # BLE classification
    # ------------------------------------------------------------------

    def classify_ble(
        self,
        mac: str = "",
        name: str = "",
        company_id: int | None = None,
        appearance: int | None = None,
        service_uuids: list[str] | None = None,
        fast_pair_model_id: str | None = None,
        apple_device_class: str | None = None,
    ) -> DeviceClassification:
        """Classify a BLE device using all available signals.

        Each signal produces a (device_type, confidence) vote.  The final
        result uses the highest-confidence vote.

        Args:
            mac: Device MAC address (for OUI lookup).
            name: Advertised device name.
            company_id: BLE company identifier (16-bit).
            appearance: GAP appearance value (16-bit).
            service_uuids: List of advertised service UUIDs.
            fast_pair_model_id: Google Fast Pair model ID hex string.
            apple_device_class: Apple continuity device class hex string.

        Returns:
            DeviceClassification with best device_type, confidence, and
            all contributing signals.
        """
        signals: list[dict[str, Any]] = []
        manufacturer = ""

        # 1. OUI manufacturer lookup
        if mac:
            oui_result = self._classify_oui(mac)
            if oui_result:
                manufacturer = oui_result.get("manufacturer", "")
                if oui_result.get("device_type"):
                    signals.append(oui_result)

        # 2. GAP Appearance (highest priority — official BT SIG classification)
        if appearance is not None:
            app_result = self._classify_appearance(appearance)
            if app_result:
                signals.append(app_result)

        # 3. Service UUIDs
        if service_uuids:
            for uuid in service_uuids:
                uuid_result = self._classify_service_uuid(uuid)
                if uuid_result:
                    signals.append(uuid_result)

        # 4. Company ID
        if company_id is not None:
            cid_result = self._classify_company_id(company_id)
            if cid_result:
                if not manufacturer and cid_result.get("manufacturer"):
                    manufacturer = cid_result["manufacturer"]
                signals.append(cid_result)

        # 5. Google Fast Pair model ID
        if fast_pair_model_id:
            fp_result = self._classify_fast_pair(fast_pair_model_id)
            if fp_result:
                signals.append(fp_result)

        # 6. Apple continuity device class
        if apple_device_class:
            apple_result = self._classify_apple_device(apple_device_class)
            if apple_result:
                if not manufacturer:
                    manufacturer = "Apple"
                signals.append(apple_result)

        # 7. Name pattern matching (lowest priority — regex heuristic)
        if name:
            name_result = self._classify_name(name)
            if name_result:
                signals.append(name_result)

        # Pick highest confidence signal
        if not signals:
            return DeviceClassification(
                manufacturer=manufacturer,
                signals=[],
            )

        best = max(signals, key=lambda s: s.get("confidence", 0))
        return DeviceClassification(
            device_type=best.get("device_type", "unknown"),
            device_name=best.get("device_name", name or ""),
            manufacturer=manufacturer,
            confidence=best.get("confidence", 0.0),
            signals=signals,
        )

    # ------------------------------------------------------------------
    # WiFi classification
    # ------------------------------------------------------------------

    def classify_wifi(
        self,
        ssid: str = "",
        bssid: str = "",
        probed_ssids: list[str] | None = None,
    ) -> DeviceClassification:
        """Classify a WiFi device from SSID, BSSID, and probe requests.

        Args:
            ssid: The network SSID (or device hotspot name).
            bssid: The BSSID MAC address.
            probed_ssids: List of SSIDs the device has probed for.

        Returns:
            DeviceClassification with best device_type and confidence.
        """
        signals: list[dict[str, Any]] = []
        manufacturer = ""

        # OUI from BSSID
        if bssid:
            oui_result = self._classify_oui(bssid)
            if oui_result:
                manufacturer = oui_result.get("manufacturer", "")
                if oui_result.get("device_type"):
                    signals.append(oui_result)

        # SSID pattern matching
        all_ssids = []
        if ssid:
            all_ssids.append(ssid)
        if probed_ssids:
            all_ssids.extend(probed_ssids)

        for s in all_ssids:
            for pattern, device_type, confidence in _WIFI_SSID_PATTERNS:
                if re.search(pattern, s):
                    signals.append({
                        "signal": "wifi_ssid_pattern",
                        "device_type": device_type,
                        "confidence": confidence,
                        "matched_ssid": s,
                    })
                    break  # one match per SSID is enough

        if not signals:
            return DeviceClassification(
                manufacturer=manufacturer,
                signals=[],
            )

        best = max(signals, key=lambda s: s.get("confidence", 0))
        return DeviceClassification(
            device_type=best.get("device_type", "unknown"),
            device_name=best.get("device_name", ssid or ""),
            manufacturer=manufacturer,
            confidence=best.get("confidence", 0.0),
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Internal signal classifiers
    # ------------------------------------------------------------------

    def _classify_oui(self, mac: str) -> dict[str, Any] | None:
        """Classify from MAC OUI prefix."""
        mac_clean = mac.upper().replace("-", ":").replace(".", ":")
        if len(mac_clean) < 8:
            return None
        prefix = mac_clean[:8]

        manufacturer = _OUI_MANUFACTURERS.get(prefix, "")
        if not manufacturer:
            # Try tritium-lib's full OUI store
            try:
                from tritium_lib.store.ble import oui_lookup
                manufacturer = oui_lookup(mac_clean) or ""
            except (ImportError, Exception):
                pass

        if not manufacturer:
            return None

        result: dict[str, Any] = {
            "signal": "oui",
            "manufacturer": manufacturer,
            "prefix": prefix,
            "confidence": 0.3,
        }

        # Some OUI prefixes imply device type
        hint = _OUI_DEVICE_HINTS.get(manufacturer)
        if hint:
            result["device_type"] = hint[0]
            result["confidence"] = hint[1]

        return result

    def _classify_appearance(self, appearance: int) -> dict[str, Any] | None:
        """Classify from GAP Appearance value."""
        gap_values = self._data.get("gap_appearance_values", {})
        key = _normalize_hex_key(f"{appearance:04X}")
        entry = gap_values.get(key)
        if not entry:
            # Try category (upper byte)
            category_key = _normalize_hex_key(f"{(appearance & 0xFFC0):04X}")
            entry = gap_values.get(category_key)
        if not entry or entry.get("type") == "unknown":
            return None

        return {
            "signal": "gap_appearance",
            "device_type": entry["type"],
            "device_name": entry.get("name", ""),
            "appearance_code": key,
            "confidence": 0.9,
        }

    def _classify_service_uuid(self, uuid: str) -> dict[str, Any] | None:
        """Classify from advertised service UUID."""
        service_uuids = self._data.get("service_uuids", {})
        # Normalize short UUIDs (4 or 6 chars with 0x prefix)
        stripped = uuid.strip()
        if len(stripped) <= 6:
            uuid_key = _normalize_hex_key(stripped)
        else:
            # 128-bit UUID — keep as-is for vendor pattern matching below
            uuid_key = stripped

        entry = service_uuids.get(uuid_key)
        if not entry or not entry.get("device_type"):
            # Try vendor UUID patterns (128-bit UUIDs)
            vendor_patterns = self._data.get("vendor_uuid_patterns", {})
            for pattern, vendor_info in vendor_patterns.items():
                regex = pattern.replace("?", ".")
                if re.match(regex, uuid.lower()):
                    return {
                        "signal": "vendor_uuid",
                        "device_type": vendor_info.get("types", ["unknown"])[0],
                        "device_name": vendor_info.get("vendor", ""),
                        "uuid": uuid,
                        "confidence": 0.75,
                    }
            return None

        return {
            "signal": "service_uuid",
            "device_type": entry["device_type"],
            "device_name": entry.get("name", ""),
            "uuid": uuid_key,
            "confidence": 0.8,
        }

    def _classify_company_id(self, company_id: int) -> dict[str, Any] | None:
        """Classify from BLE company identifier."""
        company_ids = self._data.get("company_ids", {})
        entry = company_ids.get(str(company_id))
        if not entry:
            return None

        types = entry.get("types", [])
        device_type = types[0] if types else "unknown"

        return {
            "signal": "company_id",
            "device_type": device_type,
            "manufacturer": entry.get("name", ""),
            "company_id": company_id,
            "confidence": 0.65,
        }

    def _classify_fast_pair(self, model_id: str) -> dict[str, Any] | None:
        """Classify from Google Fast Pair model ID."""
        fast_pair = self._data.get("fast_pair_models", {})
        model_key = _normalize_hex_key(model_id)

        entry = fast_pair.get(model_key)
        if not entry:
            return None

        return {
            "signal": "fast_pair",
            "device_type": entry.get("type", "unknown"),
            "device_name": entry.get("name", ""),
            "model_id": model_key,
            "confidence": 0.92,
        }

    def _classify_apple_device(self, device_class: str) -> dict[str, Any] | None:
        """Classify from Apple continuity device class byte."""
        apple_classes = self._data.get("apple_device_classes", {})
        dc_key = _normalize_hex_key(device_class)

        entry = apple_classes.get(dc_key)
        if not entry:
            return None

        return {
            "signal": "apple_device_class",
            "device_type": entry.get("type", "unknown"),
            "device_name": entry.get("name", ""),
            "apple_class": dc_key,
            "confidence": 0.93,
        }

    def _classify_name(self, name: str) -> dict[str, Any] | None:
        """Classify from device name using regex patterns."""
        for pattern, device_type, confidence in _BLE_NAME_PATTERNS:
            if re.search(pattern, name):
                return {
                    "signal": "name_pattern",
                    "device_type": device_type,
                    "device_name": name,
                    "confidence": confidence,
                }
        return None
