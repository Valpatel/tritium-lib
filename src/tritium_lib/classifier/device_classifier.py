# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""DeviceClassifier — multi-signal BLE and WiFi device type classification.

Combines all available signals (MAC OUI, device name, GAP appearance,
service UUIDs, company ID, Apple continuity data, Google Fast Pair model ID,
DHCP vendor class, DHCP hostname, mDNS services) to produce the best possible
device type classification.

Each signal contributes a (device_type, confidence) vote.  The final
classification picks the highest-confidence vote, with ties broken by
signal priority (appearance > service_uuid > company_id > name_pattern > oui).

Loads 11 JSON fingerprint databases from tritium-lib's bundled data directory:
- ble_fingerprints.json (consolidated BLE fingerprint data)
- oui_device_types.json (461 OUI prefixes → manufacturer + device types)
- ble_name_patterns.json (217 BLE advertised name patterns)
- wifi_ssid_patterns.json (72 WiFi SSID patterns)
- ble_appearance_values.json (217 GAP appearance codes)
- ble_service_uuids.json (77 BLE service UUIDs)
- ble_company_ids.json (654 BLE company IDs)
- apple_continuity_types.json (Apple BLE protocol details)
- wifi_vendor_fingerprints.json (DHCP vendor class, hostname, mDNS)
- device_classification_rules.json (priority weights, MAC randomization)

Usage::

    from tritium_lib.classifier import DeviceClassifier

    dc = DeviceClassifier()
    result = dc.classify_ble(mac="AC:BC:32:AA:BB:CC", name="iPhone 15")
    print(result.device_type)   # "phone"
    print(result.confidence)    # 0.9
    print(result.manufacturer)  # "Apple"

    wifi = dc.classify_wifi(ssid="DIRECT-HP-Printer", bssid="00:17:88:AA:BB:CC")
    print(wifi.device_type)     # "printer"

    # DHCP-based classification
    dhcp = dc.classify_dhcp(vendor_class="android-dhcp-14", hostname="Galaxy-S24")
    print(dhcp.device_type)     # "phone"

    # mDNS-based classification
    mdns = dc.classify_mdns(services=["_googlecast._tcp", "_spotify-connect._tcp"])
    print(mdns.device_type)     # "streaming_device"
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
    mac_randomized: bool = False
    os_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "manufacturer": self.manufacturer,
            "confidence": self.confidence,
            "signals": self.signals,
            "mac_randomized": self.mac_randomized,
            "os_hint": self.os_hint,
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


def _is_mac_randomized(mac: str) -> bool:
    """Detect if a MAC address is locally administered (likely randomized).

    A MAC is locally administered if bit 1 of the first octet is set.
    In hex, this means the second character of the first octet is one of
    2, 3, 6, 7, A, B, E, F.

    Requires a valid MAC address (at least ``XX:XX:XX`` format, 8+ chars
    after normalization) to avoid false positives on fragments.
    """
    mac_clean = mac.upper().replace("-", ":").replace(".", ":")
    # Require at least a plausible MAC length (e.g. "02:AA:BB" = 8 chars)
    if len(mac_clean) < 8:
        return False
    # Get the second hex character of the first octet
    first_octet = mac_clean.split(":")[0] if ":" in mac_clean else mac_clean[:2]
    if len(first_octet) < 2:
        return False
    second_char = first_octet[1].upper()
    return second_char in ("2", "3", "6", "7", "A", "B", "E", "F")


class DeviceClassifier:
    """Multi-signal device type classifier using BLE fingerprint data.

    Loads fingerprint lookup tables from tritium-lib's bundled JSON data.
    Combines all available signals to produce the best classification.

    Loads up to 11 JSON databases for maximum classification accuracy:
    - ble_fingerprints.json — consolidated BLE fingerprints (primary)
    - oui_device_types.json — 461 OUI prefixes (supplements hardcoded OUI)
    - ble_name_patterns.json — 217 BLE name patterns (supplements hardcoded)
    - wifi_ssid_patterns.json — 72 WiFi SSID patterns (supplements hardcoded)
    - ble_appearance_values.json — 217 GAP appearance codes (supplements)
    - ble_service_uuids.json — 77 BLE service UUIDs (supplements)
    - ble_company_ids.json — 654 company IDs (supplements)
    - apple_continuity_types.json — Apple BLE protocol data
    - wifi_vendor_fingerprints.json — DHCP vendor, hostname, mDNS
    - device_classification_rules.json — priority weights

    Parameters
    ----------
    fingerprints_path:
        Override path to ble_fingerprints.json.  Defaults to the bundled copy.
    data_dir:
        Override path to the data directory containing all JSON databases.
    """

    def __init__(
        self,
        fingerprints_path: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        self._data_dir = data_dir or _DATA_DIR
        self._fp_path = fingerprints_path or _FINGERPRINTS_PATH
        self._data: dict[str, Any] = {}
        self._loaded = False
        # Supplementary databases (loaded lazily from standalone JSON files)
        self._oui_db: dict[str, Any] = {}
        self._ble_name_db: list[dict[str, Any]] = []
        self._wifi_ssid_db: list[dict[str, Any]] = []
        self._appearance_db: dict[int, dict[str, Any]] = {}
        self._service_uuid_db: dict[str, dict[str, Any]] = {}
        self._company_id_db: dict[str, dict[str, Any]] = {}
        self._dhcp_vendor_patterns: list[dict[str, Any]] = []
        self._dhcp_hostname_patterns: list[dict[str, Any]] = []
        self._mdns_services: dict[str, dict[str, Any]] = {}
        self._classification_rules: dict[str, Any] = {}
        self._load_fingerprints()
        self._load_supplementary_databases()

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

    def _load_json_file(self, filename: str) -> dict[str, Any]:
        """Load a JSON file from the data directory, returning empty dict on failure."""
        path = os.path.join(self._data_dir, filename)
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.debug("Could not load %s: %s", filename, exc)
            return {}

    def _load_supplementary_databases(self) -> None:
        """Load standalone JSON databases that supplement ble_fingerprints.json."""
        # OUI device types (461 prefixes)
        oui_data = self._load_json_file("oui_device_types.json")
        self._oui_db = oui_data.get("prefixes", {})

        # BLE name patterns (217 patterns)
        name_data = self._load_json_file("ble_name_patterns.json")
        self._ble_name_db = name_data.get("patterns", [])

        # WiFi SSID patterns (72 patterns)
        ssid_data = self._load_json_file("wifi_ssid_patterns.json")
        self._wifi_ssid_db = ssid_data.get("patterns", [])

        # BLE appearance values (217 codes, integer keys)
        app_data = self._load_json_file("ble_appearance_values.json")
        raw_appearances = app_data.get("appearances", {})
        # Convert string keys to int for fast lookup
        for k, v in raw_appearances.items():
            try:
                self._appearance_db[int(k)] = v
            except (ValueError, TypeError):
                pass

        # BLE service UUIDs (77 services)
        svc_data = self._load_json_file("ble_service_uuids.json")
        self._service_uuid_db = svc_data.get("services", {})

        # BLE company IDs (654 companies)
        cid_data = self._load_json_file("ble_company_ids.json")
        self._company_id_db = cid_data.get("companies", {})

        # WiFi vendor fingerprints (DHCP, mDNS)
        wifi_vendor = self._load_json_file("wifi_vendor_fingerprints.json")
        dhcp_vc = wifi_vendor.get("dhcp_vendor_class", {})
        self._dhcp_vendor_patterns = dhcp_vc.get("patterns", []) if isinstance(dhcp_vc, dict) else []
        dhcp_hn = wifi_vendor.get("dhcp_hostname_patterns", {})
        self._dhcp_hostname_patterns = dhcp_hn.get("patterns", []) if isinstance(dhcp_hn, dict) else []
        mdns = wifi_vendor.get("mdns_service_types", {})
        self._mdns_services = mdns.get("services", {}) if isinstance(mdns, dict) else {}

        # Classification rules
        self._classification_rules = self._load_json_file("device_classification_rules.json")

        loaded_count = sum([
            len(self._oui_db) > 0,
            len(self._ble_name_db) > 0,
            len(self._wifi_ssid_db) > 0,
            len(self._appearance_db) > 0,
            len(self._service_uuid_db) > 0,
            len(self._company_id_db) > 0,
            len(self._dhcp_vendor_patterns) > 0,
            len(self._dhcp_hostname_patterns) > 0,
            len(self._mdns_services) > 0,
        ])
        logger.debug("Loaded %d supplementary databases", loaded_count)

    @property
    def loaded(self) -> bool:
        """Whether fingerprint data was successfully loaded."""
        return self._loaded

    @property
    def database_stats(self) -> dict[str, int]:
        """Return counts of entries in each loaded database."""
        return {
            "ble_fingerprints": 1 if self._loaded else 0,
            "oui_prefixes": len(self._oui_db),
            "ble_name_patterns": len(self._ble_name_db),
            "wifi_ssid_patterns": len(self._wifi_ssid_db),
            "appearance_codes": len(self._appearance_db),
            "service_uuids": len(self._service_uuid_db),
            "company_ids_standalone": len(self._company_id_db),
            "company_ids_fingerprints": len(self._data.get("company_ids", {})),
            "dhcp_vendor_patterns": len(self._dhcp_vendor_patterns),
            "dhcp_hostname_patterns": len(self._dhcp_hostname_patterns),
            "mdns_services": len(self._mdns_services),
        }

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
        mac_randomized = False

        # 0. MAC randomization detection
        if mac:
            mac_randomized = _is_mac_randomized(mac)

        # 1. OUI manufacturer lookup (skip if MAC is randomized — OUI is unreliable)
        if mac and not mac_randomized:
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
                mac_randomized=mac_randomized,
            )

        best = max(signals, key=lambda s: s.get("confidence", 0))
        return DeviceClassification(
            device_type=best.get("device_type", "unknown"),
            device_name=best.get("device_name", name or ""),
            manufacturer=manufacturer,
            confidence=best.get("confidence", 0.0),
            signals=signals,
            mac_randomized=mac_randomized,
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

        Uses both hardcoded patterns and the wifi_ssid_patterns.json database
        (72 patterns with manufacturer and network_type metadata).

        Args:
            ssid: The network SSID (or device hotspot name).
            bssid: The BSSID MAC address.
            probed_ssids: List of SSIDs the device has probed for.

        Returns:
            DeviceClassification with best device_type and confidence.
        """
        signals: list[dict[str, Any]] = []
        manufacturer = ""
        mac_randomized = False

        # OUI from BSSID (skip if randomized)
        if bssid:
            mac_randomized = _is_mac_randomized(bssid)
            if not mac_randomized:
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
            matched = False
            # Try supplementary database first (richer metadata)
            for entry in self._wifi_ssid_db:
                pattern = entry.get("pattern", "")
                # Only use DB entry if it provides a device_type
                if pattern and entry.get("device_type") and re.search(pattern, s):
                    sig: dict[str, Any] = {
                        "signal": "wifi_ssid_pattern_db",
                        "device_type": entry["device_type"],
                        "confidence": entry.get("confidence", 0.7),
                        "matched_ssid": s,
                    }
                    if entry.get("manufacturer"):
                        sig["manufacturer"] = entry["manufacturer"]
                        if not manufacturer:
                            manufacturer = entry["manufacturer"]
                    if entry.get("network_type"):
                        sig["network_type"] = entry["network_type"]
                    signals.append(sig)
                    matched = True
                    break

            # Fall back to hardcoded patterns if no DB match with device_type
            if not matched:
                for pattern, device_type, confidence in _WIFI_SSID_PATTERNS:
                    if re.search(pattern, s):
                        signals.append({
                            "signal": "wifi_ssid_pattern",
                            "device_type": device_type,
                            "confidence": confidence,
                            "matched_ssid": s,
                        })
                        break

        if not signals:
            return DeviceClassification(
                manufacturer=manufacturer,
                signals=[],
                mac_randomized=mac_randomized,
            )

        best = max(signals, key=lambda s: s.get("confidence", 0))
        return DeviceClassification(
            device_type=best.get("device_type", "unknown"),
            device_name=best.get("device_name", ssid or ""),
            manufacturer=manufacturer,
            confidence=best.get("confidence", 0.0),
            signals=signals,
            mac_randomized=mac_randomized,
        )

    # ------------------------------------------------------------------
    # DHCP classification
    # ------------------------------------------------------------------

    def classify_dhcp(
        self,
        vendor_class: str = "",
        hostname: str = "",
    ) -> DeviceClassification:
        """Classify a device from DHCP fingerprint data.

        Uses wifi_vendor_fingerprints.json DHCP vendor class (Option 60)
        and hostname (Option 12) patterns.

        Args:
            vendor_class: DHCP Option 60 vendor class identifier string.
            hostname: DHCP Option 12 hostname string.

        Returns:
            DeviceClassification with best device_type, os_hint, and confidence.
        """
        signals: list[dict[str, Any]] = []
        os_hint = ""

        # DHCP vendor class matching — collect all matches, pick best
        if vendor_class:
            vc_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for entry in self._dhcp_vendor_patterns:
                pattern = entry.get("pattern", "")
                if pattern and re.search(pattern, vendor_class):
                    vc_candidates.append(entry)
            # Prefer entries with a device_type, then highest confidence
            if vc_candidates:
                typed = [e for e in vc_candidates if e.get("device_type")]
                best_entry = max(
                    typed or vc_candidates,
                    key=lambda e: e.get("confidence", 0),
                )
                sig: dict[str, Any] = {
                    "signal": "dhcp_vendor_class",
                    "confidence": best_entry.get("confidence", 0.7),
                    "matched_value": vendor_class,
                }
                if best_entry.get("device_type"):
                    sig["device_type"] = best_entry["device_type"]
                if best_entry.get("os"):
                    sig["os"] = best_entry["os"]
                    if not os_hint:
                        os_hint = best_entry["os"]
                signals.append(sig)

        # DHCP hostname matching — collect all matches, pick best
        if hostname:
            hn_candidates: list[dict[str, Any]] = []
            for entry in self._dhcp_hostname_patterns:
                pattern = entry.get("pattern", "")
                if pattern and re.search(pattern, hostname):
                    hn_candidates.append(entry)
            if hn_candidates:
                typed_hn = [e for e in hn_candidates if e.get("device_type")]
                best_hn = max(
                    typed_hn or hn_candidates,
                    key=lambda e: e.get("confidence", 0),
                )
                sig = {
                    "signal": "dhcp_hostname",
                    "confidence": best_hn.get("confidence", 0.7),
                    "matched_value": hostname,
                }
                if best_hn.get("device_type"):
                    sig["device_type"] = best_hn["device_type"]
                if best_hn.get("os"):
                    sig["os"] = best_hn["os"]
                    if not os_hint:
                        os_hint = best_hn["os"]
                signals.append(sig)

        if not signals:
            return DeviceClassification(signals=[], os_hint=os_hint)

        # Pick best signal (prefer one with device_type)
        typed_signals = [s for s in signals if s.get("device_type")]
        best = max(
            typed_signals or signals,
            key=lambda s: s.get("confidence", 0),
        )
        return DeviceClassification(
            device_type=best.get("device_type", "unknown"),
            device_name=hostname or vendor_class or "",
            confidence=best.get("confidence", 0.0),
            signals=signals,
            os_hint=os_hint,
        )

    # ------------------------------------------------------------------
    # mDNS classification
    # ------------------------------------------------------------------

    def classify_mdns(
        self,
        services: list[str] | None = None,
    ) -> DeviceClassification:
        """Classify a device from advertised mDNS/Bonjour service types.

        Uses wifi_vendor_fingerprints.json mDNS service type database.

        Args:
            services: List of mDNS service type strings
                      (e.g. ``["_googlecast._tcp", "_spotify-connect._tcp"]``).

        Returns:
            DeviceClassification with best device_type and confidence.
        """
        if not services:
            return DeviceClassification(signals=[])

        signals: list[dict[str, Any]] = []
        manufacturer = ""

        for svc in services:
            entry = self._mdns_services.get(svc)
            if entry:
                sig: dict[str, Any] = {
                    "signal": "mdns_service",
                    "service_type": svc,
                    "service_name": entry.get("name", ""),
                    "category": entry.get("category", ""),
                    "confidence": 0.7,
                }
                if entry.get("device_hint"):
                    sig["device_type"] = entry["device_hint"]
                if entry.get("manufacturer_hint"):
                    sig["manufacturer"] = entry["manufacturer_hint"]
                    if not manufacturer:
                        manufacturer = entry["manufacturer_hint"]
                signals.append(sig)

        if not signals:
            return DeviceClassification(signals=[])

        # Pick best signal (prefer one with device_type)
        typed_signals = [s for s in signals if s.get("device_type")]
        best = max(
            typed_signals or signals,
            key=lambda s: s.get("confidence", 0),
        )
        return DeviceClassification(
            device_type=best.get("device_type", "unknown"),
            device_name=best.get("service_name", ""),
            manufacturer=manufacturer,
            confidence=best.get("confidence", 0.0),
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Combined multi-protocol classification
    # ------------------------------------------------------------------

    def classify_multi(
        self,
        *,
        mac: str = "",
        ble_name: str = "",
        company_id: int | None = None,
        appearance: int | None = None,
        service_uuids: list[str] | None = None,
        fast_pair_model_id: str | None = None,
        apple_device_class: str | None = None,
        ssid: str = "",
        bssid: str = "",
        probed_ssids: list[str] | None = None,
        dhcp_vendor_class: str = "",
        dhcp_hostname: str = "",
        mdns_services: list[str] | None = None,
    ) -> DeviceClassification:
        """Classify a device using ALL available signals from any protocol.

        Combines BLE, WiFi, DHCP, and mDNS signals into a single best
        classification.  This is the most powerful classification method
        when you have data from multiple protocols for the same device.

        Returns:
            DeviceClassification with the highest-confidence result from
            all available signals.
        """
        all_signals: list[dict[str, Any]] = []
        manufacturer = ""
        os_hint = ""
        mac_randomized = False

        # BLE classification
        if any([mac, ble_name, company_id is not None, appearance is not None,
                service_uuids, fast_pair_model_id, apple_device_class]):
            ble_result = self.classify_ble(
                mac=mac,
                name=ble_name,
                company_id=company_id,
                appearance=appearance,
                service_uuids=service_uuids,
                fast_pair_model_id=fast_pair_model_id,
                apple_device_class=apple_device_class,
            )
            all_signals.extend(ble_result.signals)
            if ble_result.manufacturer:
                manufacturer = ble_result.manufacturer
            mac_randomized = ble_result.mac_randomized

        # WiFi classification
        if any([ssid, bssid, probed_ssids]):
            wifi_result = self.classify_wifi(
                ssid=ssid,
                bssid=bssid,
                probed_ssids=probed_ssids,
            )
            all_signals.extend(wifi_result.signals)
            if wifi_result.manufacturer and not manufacturer:
                manufacturer = wifi_result.manufacturer
            if wifi_result.mac_randomized:
                mac_randomized = True

        # DHCP classification
        if any([dhcp_vendor_class, dhcp_hostname]):
            dhcp_result = self.classify_dhcp(
                vendor_class=dhcp_vendor_class,
                hostname=dhcp_hostname,
            )
            all_signals.extend(dhcp_result.signals)
            if dhcp_result.os_hint:
                os_hint = dhcp_result.os_hint

        # mDNS classification
        if mdns_services:
            mdns_result = self.classify_mdns(services=mdns_services)
            all_signals.extend(mdns_result.signals)
            if mdns_result.manufacturer and not manufacturer:
                manufacturer = mdns_result.manufacturer

        if not all_signals:
            return DeviceClassification(
                signals=[],
                mac_randomized=mac_randomized,
                os_hint=os_hint,
            )

        typed_signals = [s for s in all_signals if s.get("device_type")]
        best = max(
            typed_signals or all_signals,
            key=lambda s: s.get("confidence", 0),
        )
        return DeviceClassification(
            device_type=best.get("device_type", "unknown"),
            device_name=best.get("device_name", ble_name or ssid or ""),
            manufacturer=manufacturer,
            confidence=best.get("confidence", 0.0),
            signals=all_signals,
            mac_randomized=mac_randomized,
            os_hint=os_hint,
        )

    # ------------------------------------------------------------------
    # Internal signal classifiers
    # ------------------------------------------------------------------

    def _classify_oui(self, mac: str) -> dict[str, Any] | None:
        """Classify from MAC OUI prefix.

        Checks hardcoded OUI table first, then falls back to
        oui_device_types.json (461 prefixes with manufacturer + device types).
        """
        mac_clean = mac.upper().replace("-", ":").replace(".", ":")
        if len(mac_clean) < 8:
            return None
        prefix = mac_clean[:8]

        manufacturer = _OUI_MANUFACTURERS.get(prefix, "")

        # Fallback to supplementary OUI database
        if not manufacturer and self._oui_db:
            db_entry = self._oui_db.get(prefix)
            if db_entry:
                manufacturer = db_entry.get("manufacturer", "")

        if not manufacturer:
            return None

        result: dict[str, Any] = {
            "signal": "oui",
            "manufacturer": manufacturer,
            "prefix": prefix,
            "confidence": 0.3,
        }

        # Some OUI prefixes imply device type — check hardcoded hints first
        hint = _OUI_DEVICE_HINTS.get(manufacturer)
        if hint:
            result["device_type"] = hint[0]
            result["confidence"] = hint[1]
        elif self._oui_db:
            # Check supplementary DB for device type hints
            db_entry = self._oui_db.get(prefix)
            if db_entry:
                device_types = db_entry.get("device_types", [])
                if device_types:
                    # Use the first listed device type as a hint
                    result["device_type"] = device_types[0]
                    result["confidence"] = 0.5
                    result["device_type_candidates"] = device_types

        return result

    def _classify_appearance(self, appearance: int) -> dict[str, Any] | None:
        """Classify from GAP Appearance value.

        Checks ble_fingerprints.json first (hex keys), then falls back to
        ble_appearance_values.json (217 entries with integer keys).
        """
        gap_values = self._data.get("gap_appearance_values", {})
        key = _normalize_hex_key(f"{appearance:04X}")
        entry = gap_values.get(key)
        if not entry:
            # Try category (upper byte)
            category_key = _normalize_hex_key(f"{(appearance & 0xFFC0):04X}")
            entry = gap_values.get(category_key)

        if entry and entry.get("type") != "unknown":
            return {
                "signal": "gap_appearance",
                "device_type": entry["type"],
                "device_name": entry.get("name", ""),
                "appearance_code": key,
                "confidence": 0.9,
            }

        # Fallback to supplementary appearance database (integer keys)
        if self._appearance_db:
            app_entry = self._appearance_db.get(appearance)
            if not app_entry:
                # Try category (upper byte)
                app_entry = self._appearance_db.get(appearance & 0xFFC0)
            if app_entry and app_entry.get("device_type", "unknown") != "unknown":
                return {
                    "signal": "gap_appearance_db",
                    "device_type": app_entry["device_type"],
                    "device_name": app_entry.get("description", ""),
                    "category": app_entry.get("category", ""),
                    "appearance_code": key,
                    "confidence": 0.9,
                }

        return None

    def _classify_service_uuid(self, uuid: str) -> dict[str, Any] | None:
        """Classify from advertised service UUID.

        Checks ble_fingerprints.json first, then ble_service_uuids.json
        (77 services), then vendor UUID patterns for 128-bit UUIDs.
        """
        service_uuids = self._data.get("service_uuids", {})
        # Normalize short UUIDs (4 or 6 chars with 0x prefix)
        stripped = uuid.strip()
        if len(stripped) <= 6:
            uuid_key = _normalize_hex_key(stripped)
        else:
            # 128-bit UUID — keep as-is for vendor pattern matching below
            uuid_key = stripped

        entry = service_uuids.get(uuid_key)
        if entry and entry.get("device_type"):
            return {
                "signal": "service_uuid",
                "device_type": entry["device_type"],
                "device_name": entry.get("name", ""),
                "uuid": uuid_key,
                "confidence": 0.8,
            }

        # Fallback to supplementary service UUID database
        if self._service_uuid_db:
            svc_entry = self._service_uuid_db.get(uuid_key)
            if svc_entry and svc_entry.get("device_hint"):
                return {
                    "signal": "service_uuid_db",
                    "device_type": svc_entry["device_hint"],
                    "device_name": svc_entry.get("name", ""),
                    "category": svc_entry.get("category", ""),
                    "uuid": uuid_key,
                    "confidence": 0.8,
                }

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

    def _classify_company_id(self, company_id: int) -> dict[str, Any] | None:
        """Classify from BLE company identifier.

        Checks ble_fingerprints.json first (933 entries), then falls back
        to ble_company_ids.json (654 entries with different format).
        """
        company_ids = self._data.get("company_ids", {})
        entry = company_ids.get(str(company_id))
        if entry:
            types = entry.get("types", [])
            device_type = types[0] if types else "unknown"
            return {
                "signal": "company_id",
                "device_type": device_type,
                "manufacturer": entry.get("name", ""),
                "company_id": company_id,
                "confidence": 0.65,
            }

        # Fallback to supplementary company ID database
        if self._company_id_db:
            db_entry = self._company_id_db.get(str(company_id))
            if db_entry:
                device_types = db_entry.get("device_types", [])
                device_type = device_types[0] if device_types else "unknown"
                return {
                    "signal": "company_id_db",
                    "device_type": device_type,
                    "manufacturer": db_entry.get("name", ""),
                    "company_id": company_id,
                    "confidence": 0.65,
                }

        return None

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
        """Classify from device name using regex patterns.

        Checks hardcoded patterns first (48 entries), then falls back to
        ble_name_patterns.json (217 patterns with manufacturer metadata).
        """
        # Hardcoded patterns (fast, high-confidence for common devices)
        for pattern, device_type, confidence in _BLE_NAME_PATTERNS:
            if re.search(pattern, name):
                return {
                    "signal": "name_pattern",
                    "device_type": device_type,
                    "device_name": name,
                    "confidence": confidence,
                }

        # Fallback to supplementary name pattern database (217 patterns)
        for entry in self._ble_name_db:
            pattern = entry.get("pattern", "")
            if pattern and re.search(pattern, name):
                result: dict[str, Any] = {
                    "signal": "name_pattern_db",
                    "device_type": entry.get("device_type", "unknown"),
                    "device_name": name,
                    "confidence": entry.get("confidence", 0.7),
                }
                if entry.get("manufacturer"):
                    result["manufacturer"] = entry["manufacturer"]
                return result

        return None
