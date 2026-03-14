# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Unified device classifier — identifies BLE and WiFi devices using
comprehensive public lookup databases (OUI, Bluetooth SIG, Apple Continuity,
Google Fast Pair, SSID patterns, DHCP/mDNS fingerprints)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class DeviceClassification:
    """Result of classifying a device from BLE or WiFi signals."""

    device_type: str = "unknown"
    manufacturer: Optional[str] = None
    model_hint: Optional[str] = None
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    os_hint: Optional[str] = None
    is_randomized_mac: bool = False

    def merge(self, other: DeviceClassification) -> DeviceClassification:
        """Merge another classification into this one, keeping highest
        confidence values and accumulating sources."""
        sources = list(set(self.sources + other.sources))
        if other.confidence > self.confidence:
            return DeviceClassification(
                device_type=other.device_type,
                manufacturer=other.manufacturer or self.manufacturer,
                model_hint=other.model_hint or self.model_hint,
                confidence=other.confidence,
                sources=sources,
                os_hint=other.os_hint or self.os_hint,
                is_randomized_mac=self.is_randomized_mac or other.is_randomized_mac,
            )
        return DeviceClassification(
            device_type=self.device_type,
            manufacturer=self.manufacturer or other.manufacturer,
            model_hint=self.model_hint or other.model_hint,
            confidence=self.confidence,
            sources=sources,
            os_hint=self.os_hint or other.os_hint,
            is_randomized_mac=self.is_randomized_mac or other.is_randomized_mac,
        )


class DeviceClassifier:
    """Classifies BLE and WiFi devices using loaded JSON lookup databases.

    Usage::

        classifier = DeviceClassifier()
        result = classifier.classify_ble(
            mac="AA:BB:CC:DD:EE:FF",
            name="iPhone",
            appearance=64,
        )
        print(result.device_type, result.manufacturer, result.confidence)
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = data_dir or DATA_DIR
        self._oui: dict = {}
        self._company_ids: dict = {}
        self._appearances: dict = {}
        self._service_uuids: dict = {}
        self._name_patterns: list[dict] = []
        self._apple_types: dict = {}
        self._wifi_ssid_patterns: list[dict] = []
        self._dhcp_vendor_patterns: list[dict] = []
        self._dhcp_hostname_patterns: list[dict] = []
        self._mdns_services: dict = {}
        self._fast_pair_models: dict = {}
        self._loaded = False

    def _load_json(self, filename: str) -> dict:
        """Load a JSON data file from the data directory."""
        path = self._data_dir / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_loaded(self) -> None:
        """Lazy-load all data files on first use."""
        if self._loaded:
            return

        oui_data = self._load_json("oui_device_types.json")
        self._oui = oui_data.get("prefixes", {})

        company_data = self._load_json("ble_company_ids.json")
        self._company_ids = company_data.get("companies", {})

        appearance_data = self._load_json("ble_appearance_values.json")
        self._appearances = appearance_data.get("appearances", {})

        service_data = self._load_json("ble_service_uuids.json")
        self._service_uuids = service_data.get("services", {})

        name_data = self._load_json("ble_name_patterns.json")
        self._name_patterns = name_data.get("patterns", [])
        # Pre-compile regexes
        for p in self._name_patterns:
            p["_re"] = re.compile(p["pattern"])

        apple_data = self._load_json("apple_continuity_types.json")
        self._apple_types = apple_data
        self._fast_pair_models = apple_data.get("google_fast_pair_models", {})

        ssid_data = self._load_json("wifi_ssid_patterns.json")
        self._wifi_ssid_patterns = ssid_data.get("patterns", [])
        for p in self._wifi_ssid_patterns:
            p["_re"] = re.compile(p["pattern"])

        vendor_data = self._load_json("wifi_vendor_fingerprints.json")
        dhcp_vc = vendor_data.get("dhcp_vendor_class", {})
        self._dhcp_vendor_patterns = dhcp_vc.get("patterns", [])
        for p in self._dhcp_vendor_patterns:
            p["_re"] = re.compile(p["pattern"])

        dhcp_hn = vendor_data.get("dhcp_hostname_patterns", {})
        self._dhcp_hostname_patterns = dhcp_hn.get("patterns", [])
        for p in self._dhcp_hostname_patterns:
            p["_re"] = re.compile(p["pattern"])

        self._mdns_services = vendor_data.get("mdns_service_types", {}).get(
            "services", {}
        )

        self._loaded = True

    # ------------------------------------------------------------------
    # MAC address utilities
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_mac(mac: str) -> str:
        """Normalize a MAC address to uppercase colon-separated format."""
        mac = mac.upper().replace("-", ":").replace(".", ":")
        # Handle no-separator MACs like AABBCCDDEEFF
        if len(mac) == 12 and ":" not in mac:
            mac = ":".join(mac[i : i + 2] for i in range(0, 12, 2))
        return mac

    @staticmethod
    def is_randomized_mac(mac: str) -> bool:
        """Check if a MAC address is locally administered (likely randomized).

        The second hex digit of the first octet determines this:
        if bit 1 is set (digit is 2,3,6,7,A,B,E,F), it's locally administered.
        """
        mac = mac.upper().replace("-", ":").replace(".", ":")
        if len(mac) < 2:
            return False
        try:
            first_octet = int(mac[:2].replace(":", ""), 16)
            return bool(first_octet & 0x02)
        except ValueError:
            return False

    @staticmethod
    def get_oui_prefix(mac: str) -> str:
        """Extract the OUI prefix (first 3 octets) from a MAC address."""
        mac = DeviceClassifier.normalize_mac(mac)
        parts = mac.split(":")
        if len(parts) >= 3:
            return ":".join(parts[:3])
        return mac

    # ------------------------------------------------------------------
    # BLE classification
    # ------------------------------------------------------------------

    def classify_ble(
        self,
        mac: str = "",
        name: str = "",
        company_id: Optional[int] = None,
        appearance: Optional[int] = None,
        service_uuids: Optional[list[str]] = None,
        raw_adv: Optional[bytes] = None,
    ) -> DeviceClassification:
        """Classify a BLE device using all available signal data.

        Args:
            mac: BLE MAC address (may be randomized).
            name: BLE advertised device name.
            company_id: Bluetooth SIG company identifier from manufacturer data.
            appearance: BLE GAP Appearance value.
            service_uuids: List of advertised service UUIDs.
            raw_adv: Raw advertising data bytes (for Apple Continuity parsing).

        Returns:
            DeviceClassification with best-effort identification.
        """
        self._ensure_loaded()
        result = DeviceClassification()

        if mac:
            mac = self.normalize_mac(mac)
            result.is_randomized_mac = self.is_randomized_mac(mac)

        # 1. Appearance value (highest priority — explicit declaration)
        if appearance is not None:
            app_result = self._classify_by_appearance(appearance)
            if app_result:
                result = result.merge(app_result)

        # 2. Apple Continuity device type from raw advertising data
        if raw_adv:
            apple_result = self._classify_apple_continuity(raw_adv)
            if apple_result:
                result = result.merge(apple_result)

        # 3. Device name pattern matching
        if name:
            name_result = self._classify_by_name(name)
            if name_result:
                result = result.merge(name_result)

        # 4. Service UUIDs
        if service_uuids:
            uuid_result = self._classify_by_service_uuids(service_uuids)
            if uuid_result:
                result = result.merge(uuid_result)

        # 5. Company ID
        if company_id is not None:
            cid_result = self._classify_by_company_id(company_id)
            if cid_result:
                result = result.merge(cid_result)

        # 6. OUI lookup (lowest priority — only manufacturer hint)
        if mac and not result.is_randomized_mac:
            oui_result = self._classify_by_oui(mac)
            if oui_result:
                result = result.merge(oui_result)

        return result

    def _classify_by_appearance(self, appearance: int) -> Optional[DeviceClassification]:
        """Look up BLE GAP Appearance value."""
        key = str(appearance)
        entry = self._appearances.get(key)
        if not entry:
            return None
        return DeviceClassification(
            device_type=entry.get("device_type", "unknown"),
            manufacturer=None,
            model_hint=entry.get("description"),
            confidence=0.95,
            sources=["ble_appearance"],
        )

    def _classify_apple_continuity(
        self, raw_adv: bytes
    ) -> Optional[DeviceClassification]:
        """Parse Apple Continuity protocol from raw BLE advertising data.

        Looks for Apple company ID (0x004C) followed by message type byte.
        """
        apple_company = b"\x4c\x00"
        idx = raw_adv.find(apple_company)
        if idx < 0 or idx + 3 >= len(raw_adv):
            return None

        msg_type = raw_adv[idx + 2]
        msg_type_hex = f"0x{msg_type:02X}"

        # Nearby Info message (0x10) — contains device type
        if msg_type == 0x10:
            return DeviceClassification(
                device_type="phone",
                manufacturer="Apple",
                model_hint="Apple device (Nearby Info)",
                confidence=0.85,
                sources=["apple_continuity_nearby_info"],
                os_hint="iOS/macOS",
            )

        # Proximity Pairing message (0x05) — AirPods/Beats
        if msg_type == 0x05:
            model_bytes = raw_adv[idx + 3 : idx + 5]
            if len(model_bytes) == 2:
                model_id = f"0x{model_bytes[0]:02X}{model_bytes[1]:02X}"
                pp_models = self._apple_types.get("proximity_pairing_models", {})
                model_info = pp_models.get(model_id)
                if model_info:
                    return DeviceClassification(
                        device_type=model_info["device_type"],
                        manufacturer="Apple",
                        model_hint=model_info["name"],
                        confidence=0.95,
                        sources=["apple_proximity_pairing"],
                        os_hint="iOS",
                    )
            return DeviceClassification(
                device_type="earbud",
                manufacturer="Apple",
                model_hint="Apple audio device",
                confidence=0.80,
                sources=["apple_proximity_pairing"],
                os_hint="iOS",
            )

        # Nearby Action (0x0F)
        if msg_type == 0x0F:
            return DeviceClassification(
                device_type="phone",
                manufacturer="Apple",
                model_hint="Apple device (Nearby Action)",
                confidence=0.85,
                sources=["apple_continuity_nearby_action"],
                os_hint="iOS/macOS",
            )

        # Find My (0x12)
        if msg_type == 0x12:
            return DeviceClassification(
                device_type="tracker",
                manufacturer="Apple",
                model_hint="Find My device",
                confidence=0.85,
                sources=["apple_continuity_findmy"],
                os_hint="iOS",
            )

        # Any other Apple Continuity message
        msg_types = self._apple_types.get("message_types", {})
        msg_info = msg_types.get(msg_type_hex, {})
        return DeviceClassification(
            device_type="phone",
            manufacturer="Apple",
            model_hint=msg_info.get("name", "Apple device"),
            confidence=0.75,
            sources=["apple_continuity"],
            os_hint="iOS/macOS",
        )

    def _classify_by_name(self, name: str) -> Optional[DeviceClassification]:
        """Match BLE device name against known patterns."""
        best: Optional[DeviceClassification] = None
        for p in self._name_patterns:
            regex = p.get("_re")
            if regex and regex.search(name):
                conf = p.get("confidence", 0.5)
                candidate = DeviceClassification(
                    device_type=p.get("device_type", "unknown"),
                    manufacturer=p.get("manufacturer"),
                    model_hint=name,
                    confidence=conf,
                    sources=["ble_name_pattern"],
                )
                if best is None or conf > best.confidence:
                    best = candidate
        return best

    def _classify_by_service_uuids(
        self, uuids: list[str]
    ) -> Optional[DeviceClassification]:
        """Classify based on advertised BLE service UUIDs."""
        best: Optional[DeviceClassification] = None
        for uuid_str in uuids:
            uuid_upper = uuid_str.upper()
            # Normalize to 0xNNNN format for 16-bit UUIDs
            if len(uuid_upper) == 4:
                uuid_upper = f"0x{uuid_upper}"
            elif len(uuid_upper) == 6 and uuid_upper.startswith("0X"):
                uuid_upper = f"0x{uuid_upper[2:]}"

            entry = self._service_uuids.get(uuid_upper)
            if not entry:
                # Try lowercase variant
                entry = self._service_uuids.get(uuid_str)
            if not entry:
                continue

            hint = entry.get("device_hint")
            if not hint:
                continue

            candidate = DeviceClassification(
                device_type=hint,
                manufacturer=None,
                model_hint=entry.get("name"),
                confidence=0.80,
                sources=["ble_service_uuid"],
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        return best

    def _classify_by_company_id(
        self, company_id: int
    ) -> Optional[DeviceClassification]:
        """Look up Bluetooth SIG company identifier."""
        entry = self._company_ids.get(str(company_id))
        if not entry:
            return None

        device_types = entry.get("device_types", [])
        primary_type = device_types[0] if device_types else "unknown"
        return DeviceClassification(
            device_type=primary_type,
            manufacturer=entry.get("name"),
            model_hint=None,
            confidence=0.70,
            sources=["ble_company_id"],
        )

    def _classify_by_oui(self, mac: str) -> Optional[DeviceClassification]:
        """Look up MAC OUI prefix for manufacturer and device type hints."""
        prefix = self.get_oui_prefix(mac)
        entry = self._oui.get(prefix)
        if not entry:
            return None

        device_types = entry.get("device_types", [])
        primary_type = device_types[0] if device_types else "unknown"
        return DeviceClassification(
            device_type=primary_type,
            manufacturer=entry.get("manufacturer"),
            model_hint=None,
            confidence=0.55,
            sources=["oui_lookup"],
        )

    # ------------------------------------------------------------------
    # WiFi classification
    # ------------------------------------------------------------------

    def classify_wifi(
        self,
        mac: str = "",
        ssid: str = "",
        vendor_class: str = "",
        hostname: str = "",
        mdns_services: Optional[list[str]] = None,
    ) -> DeviceClassification:
        """Classify a WiFi device using all available signal data.

        Args:
            mac: WiFi MAC address (may be randomized).
            ssid: SSID the device is broadcasting or connected to.
            vendor_class: DHCP Option 60 Vendor Class Identifier.
            hostname: DHCP Option 12 hostname.
            mdns_services: List of mDNS/Bonjour service types advertised.

        Returns:
            DeviceClassification with best-effort identification.
        """
        self._ensure_loaded()
        result = DeviceClassification()

        if mac:
            mac = self.normalize_mac(mac)
            result.is_randomized_mac = self.is_randomized_mac(mac)

        # 1. SSID pattern matching
        if ssid:
            ssid_result = self._classify_by_ssid(ssid)
            if ssid_result:
                result = result.merge(ssid_result)

        # 2. DHCP vendor class
        if vendor_class:
            vc_result = self._classify_by_dhcp_vendor(vendor_class)
            if vc_result:
                result = result.merge(vc_result)

        # 3. Hostname pattern
        if hostname:
            hn_result = self._classify_by_hostname(hostname)
            if hn_result:
                result = result.merge(hn_result)

        # 4. mDNS services
        if mdns_services:
            mdns_result = self._classify_by_mdns(mdns_services)
            if mdns_result:
                result = result.merge(mdns_result)

        # 5. OUI lookup (lowest priority)
        if mac and not result.is_randomized_mac:
            oui_result = self._classify_by_oui(mac)
            if oui_result:
                result = result.merge(oui_result)

        return result

    def _classify_by_ssid(self, ssid: str) -> Optional[DeviceClassification]:
        """Match SSID against known patterns."""
        best: Optional[DeviceClassification] = None
        for p in self._wifi_ssid_patterns:
            regex = p.get("_re")
            if regex and regex.search(ssid):
                conf = p.get("confidence", 0.5)
                candidate = DeviceClassification(
                    device_type=p.get("device_type", "unknown") or "unknown",
                    manufacturer=p.get("manufacturer"),
                    model_hint=ssid,
                    confidence=conf,
                    sources=["wifi_ssid_pattern"],
                )
                if best is None or conf > best.confidence:
                    best = candidate
        return best

    def _classify_by_dhcp_vendor(
        self, vendor_class: str
    ) -> Optional[DeviceClassification]:
        """Match DHCP vendor class identifier."""
        for p in self._dhcp_vendor_patterns:
            regex = p.get("_re")
            if regex and regex.search(vendor_class):
                return DeviceClassification(
                    device_type=p.get("device_type", "unknown") or "unknown",
                    manufacturer=None,
                    model_hint=vendor_class,
                    confidence=p.get("confidence", 0.7),
                    sources=["dhcp_vendor_class"],
                    os_hint=p.get("os"),
                )
        return None

    def _classify_by_hostname(self, hostname: str) -> Optional[DeviceClassification]:
        """Match DHCP hostname against known patterns."""
        best: Optional[DeviceClassification] = None
        for p in self._dhcp_hostname_patterns:
            regex = p.get("_re")
            if regex and regex.search(hostname):
                conf = p.get("confidence", 0.5)
                candidate = DeviceClassification(
                    device_type=p.get("device_type", "unknown"),
                    manufacturer=None,
                    model_hint=hostname,
                    confidence=conf,
                    sources=["dhcp_hostname"],
                    os_hint=p.get("os"),
                )
                if best is None or conf > best.confidence:
                    best = candidate
        return best

    def _classify_by_mdns(
        self, services: list[str]
    ) -> Optional[DeviceClassification]:
        """Classify device by advertised mDNS/Bonjour service types."""
        best: Optional[DeviceClassification] = None
        for svc in services:
            # Normalize: strip .local suffix
            svc_key = svc.replace(".local", "").strip(".")
            entry = self._mdns_services.get(svc_key)
            if not entry:
                continue

            hint = entry.get("device_hint")
            if not hint:
                continue

            candidate = DeviceClassification(
                device_type=hint,
                manufacturer=entry.get("manufacturer_hint"),
                model_hint=entry.get("name"),
                confidence=0.70,
                sources=["mdns_service"],
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        return best

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def lookup_oui(self, mac: str) -> Optional[dict]:
        """Look up OUI entry for a MAC address. Returns dict with
        manufacturer and device_types or None."""
        self._ensure_loaded()
        mac = self.normalize_mac(mac)
        prefix = self.get_oui_prefix(mac)
        return self._oui.get(prefix)

    def lookup_company_id(self, company_id: int) -> Optional[dict]:
        """Look up a Bluetooth SIG company identifier."""
        self._ensure_loaded()
        return self._company_ids.get(str(company_id))

    def lookup_appearance(self, appearance: int) -> Optional[dict]:
        """Look up a BLE GAP Appearance value."""
        self._ensure_loaded()
        return self._appearances.get(str(appearance))

    def lookup_service_uuid(self, uuid: str) -> Optional[dict]:
        """Look up a BLE service UUID."""
        self._ensure_loaded()
        uuid_upper = uuid.upper()
        if len(uuid_upper) == 4:
            uuid_upper = f"0x{uuid_upper}"
        return self._service_uuids.get(uuid_upper)
