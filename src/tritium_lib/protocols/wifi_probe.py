# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""WiFi probe request parser.

Parses 802.11 probe request frames from raw bytes or a structured
dict of tagged parameters.  Probe requests are management frames
(type 0, subtype 4) that devices broadcast to discover access points.

Frame layout:
    [2 frame control][2 duration][6 DA][6 SA][6 BSSID][2 seq]
    [Tagged Parameters: [1 tag][1 len][N data] ...]

Supported tagged parameters:
    Tag 0   — SSID
    Tag 1   — Supported Rates
    Tag 50  — Extended Supported Rates
    Tag 45  — HT Capabilities
    Tag 127 — Extended Capabilities
    Tag 221 — Vendor Specific (WPS, Microsoft, Apple)
    Tag 107 — Interworking (Hotspot 2.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.protocols.errors import ParseError

# 802.11 frame control: probe request = type 0 (management), subtype 4
_FC_PROBE_REQUEST = 0x0040  # little-endian: 0x40, 0x00

# OUI prefixes for vendor-specific IEs
_OUI_MICROSOFT = bytes([0x00, 0x50, 0xF2])
_OUI_WFA = bytes([0x50, 0x6F, 0x9A])  # Wi-Fi Alliance
_OUI_APPLE = bytes([0x00, 0x17, 0xF2])


@dataclass
class WiFiRate:
    """A single supported rate."""

    rate_mbps: float
    is_basic: bool = False  # BSS basic rate (mandatory)


@dataclass
class WiFiHTCapabilities:
    """Parsed HT (802.11n) capabilities."""

    channel_width_40mhz: bool = False
    short_gi_20mhz: bool = False
    short_gi_40mhz: bool = False
    tx_stbc: bool = False
    rx_stbc_streams: int = 0
    max_amsdu_length: int = 3839  # 3839 or 7935
    raw: int = 0


@dataclass
class WiFiProbeRequest:
    """Fully parsed WiFi probe request frame."""

    # MAC addresses
    source_mac: str = ""
    destination_mac: str = "ff:ff:ff:ff:ff:ff"  # usually broadcast
    bssid: str = "ff:ff:ff:ff:ff:ff"

    # Sequence control
    sequence_number: int = 0
    fragment_number: int = 0

    # SSID
    ssid: str = ""
    is_broadcast_probe: bool = True  # True if SSID is empty (wildcard)

    # Rates
    supported_rates: list[WiFiRate] = field(default_factory=list)
    extended_rates: list[WiFiRate] = field(default_factory=list)

    # Capabilities
    ht_capabilities: Optional[WiFiHTCapabilities] = None
    has_ht: bool = False  # 802.11n capable
    has_interworking: bool = False  # Hotspot 2.0

    # Vendor IEs
    vendor_ies: list[dict] = field(default_factory=list)

    # Raw tagged parameters
    raw_tags: list[dict] = field(default_factory=list)

    @property
    def all_rates_mbps(self) -> list[float]:
        """All supported rates combined."""
        return [r.rate_mbps for r in self.supported_rates + self.extended_rates]

    @property
    def max_rate_mbps(self) -> float:
        """Highest supported rate."""
        rates = self.all_rates_mbps
        return max(rates) if rates else 0.0

    @property
    def oui(self) -> str:
        """OUI prefix from source MAC (first 3 octets)."""
        parts = self.source_mac.split(":")
        if len(parts) >= 3:
            return ":".join(parts[:3]).upper()
        return ""

    @property
    def is_randomized_mac(self) -> bool:
        """Check if source MAC is locally administered (randomized).

        Bit 1 of the first octet set = locally administered.
        """
        if not self.source_mac:
            return False
        try:
            first_byte = int(self.source_mac.split(":")[0], 16)
            return bool(first_byte & 0x02)
        except (ValueError, IndexError):
            return False


class WiFiProbeParser:
    """Parser for WiFi probe request frames.

    Accepts either raw 802.11 frame bytes or a dict of pre-extracted
    tagged parameters (for integration with capture tools that already
    decode the frame header).

    Usage::

        parser = WiFiProbeParser()

        # From raw frame bytes
        probe = parser.parse(raw_frame_bytes)

        # From pre-parsed tagged parameters
        probe = parser.from_fields(
            source_mac="AA:BB:CC:DD:EE:FF",
            ssid="MyNetwork",
            tags={0: b"MyNetwork", 1: b"\\x82\\x84\\x8b\\x96"},
        )
    """

    @staticmethod
    def _normalize_mac(mac_bytes: bytes) -> str:
        """Convert 6 bytes to colon-separated MAC string."""
        if len(mac_bytes) < 6:
            return ""
        return ":".join(f"{b:02x}" for b in mac_bytes[:6])

    @staticmethod
    def _parse_rates(data: bytes) -> list[WiFiRate]:
        """Parse supported rates IE data."""
        rates = []
        for byte in data:
            is_basic = bool(byte & 0x80)
            rate_val = byte & 0x7F
            rate_mbps = rate_val * 0.5
            rates.append(WiFiRate(rate_mbps=rate_mbps, is_basic=is_basic))
        return rates

    @staticmethod
    def _parse_ht_capabilities(data: bytes) -> WiFiHTCapabilities:
        """Parse HT capabilities IE (tag 45)."""
        if len(data) < 2:
            return WiFiHTCapabilities()
        cap_info = int.from_bytes(data[0:2], "little")
        return WiFiHTCapabilities(
            channel_width_40mhz=bool(cap_info & 0x0002),
            short_gi_20mhz=bool(cap_info & 0x0020),
            short_gi_40mhz=bool(cap_info & 0x0040),
            tx_stbc=bool(cap_info & 0x0080),
            rx_stbc_streams=(cap_info >> 8) & 0x03,
            max_amsdu_length=7935 if (cap_info & 0x0800) else 3839,
            raw=cap_info,
        )

    @staticmethod
    def _parse_vendor_ie(data: bytes) -> dict:
        """Parse a vendor-specific IE (tag 221)."""
        if len(data) < 3:
            return {"oui": "", "data": data.hex()}
        oui = data[0:3]
        oui_str = ":".join(f"{b:02x}" for b in oui)
        vendor_data = data[3:]

        vendor_name = "Unknown"
        if oui == _OUI_MICROSOFT:
            vendor_name = "Microsoft"
        elif oui == _OUI_WFA:
            vendor_name = "Wi-Fi Alliance"
        elif oui == _OUI_APPLE:
            vendor_name = "Apple"

        return {
            "oui": oui_str,
            "vendor": vendor_name,
            "type": vendor_data[0] if len(vendor_data) > 0 else 0,
            "data": vendor_data.hex(),
        }

    def _parse_tagged_params(
        self, data: bytes, result: WiFiProbeRequest
    ) -> None:
        """Parse the tagged parameter section of a probe request."""
        offset = 0
        while offset + 1 < len(data):
            tag_id = data[offset]
            tag_len = data[offset + 1]
            if offset + 2 + tag_len > len(data):
                break  # truncated

            tag_data = data[offset + 2 : offset + 2 + tag_len]

            result.raw_tags.append({
                "tag": tag_id,
                "length": tag_len,
                "data": tag_data.hex(),
            })

            if tag_id == 0:  # SSID
                try:
                    result.ssid = tag_data.decode("utf-8", errors="replace")
                except Exception:
                    result.ssid = tag_data.hex()
                result.is_broadcast_probe = len(tag_data) == 0

            elif tag_id == 1:  # Supported Rates
                result.supported_rates = self._parse_rates(tag_data)

            elif tag_id == 50:  # Extended Supported Rates
                result.extended_rates = self._parse_rates(tag_data)

            elif tag_id == 45:  # HT Capabilities
                result.ht_capabilities = self._parse_ht_capabilities(tag_data)
                result.has_ht = True

            elif tag_id == 107:  # Interworking
                result.has_interworking = True

            elif tag_id == 221:  # Vendor Specific
                result.vendor_ies.append(self._parse_vendor_ie(tag_data))

            offset += 2 + tag_len

    def parse(self, data: bytes | str) -> WiFiProbeRequest:
        """Parse a raw 802.11 probe request frame.

        Args:
            data: Raw frame bytes or hex string.

        Returns:
            WiFiProbeRequest with parsed fields.

        Raises:
            ParseError: If frame is too short or not a probe request.
        """
        if isinstance(data, str):
            cleaned = data.strip().replace(" ", "").replace("-", "")
            if cleaned.startswith(("0x", "0X")):
                cleaned = cleaned[2:]
            try:
                raw = bytes.fromhex(cleaned)
            except ValueError as exc:
                raise ParseError("WiFi", f"Invalid hex string: {exc}", data) from exc
        elif isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        else:
            raise ParseError("WiFi", f"Expected bytes or hex string, got {type(data).__name__}", data)

        # Minimum frame: FC(2) + Duration(2) + DA(6) + SA(6) + BSSID(6) + Seq(2) = 24
        if len(raw) < 24:
            raise ParseError("WiFi", f"Frame too short ({len(raw)} bytes, need >= 24)", data)

        # Check frame control for probe request
        fc = int.from_bytes(raw[0:2], "little")
        frame_type = (fc >> 2) & 0x03
        frame_subtype = (fc >> 4) & 0x0F

        if frame_type != 0 or frame_subtype != 4:
            raise ParseError(
                "WiFi",
                f"Not a probe request (type={frame_type}, subtype={frame_subtype})",
                data,
            )

        result = WiFiProbeRequest()
        result.destination_mac = self._normalize_mac(raw[4:10])
        result.source_mac = self._normalize_mac(raw[10:16])
        result.bssid = self._normalize_mac(raw[16:22])

        seq_ctrl = int.from_bytes(raw[22:24], "little")
        result.fragment_number = seq_ctrl & 0x0F
        result.sequence_number = (seq_ctrl >> 4) & 0x0FFF

        # Parse tagged parameters (starting at byte 24)
        if len(raw) > 24:
            self._parse_tagged_params(raw[24:], result)

        return result

    def from_fields(
        self,
        source_mac: str = "",
        ssid: str = "",
        tags: dict[int, bytes] | None = None,
        destination_mac: str = "ff:ff:ff:ff:ff:ff",
        bssid: str = "ff:ff:ff:ff:ff:ff",
    ) -> WiFiProbeRequest:
        """Build a WiFiProbeRequest from pre-extracted fields.

        This is useful when a capture tool has already decoded the frame
        header and you just want to parse the information elements.

        Args:
            source_mac: Source MAC address string.
            ssid: Network SSID being probed (empty = broadcast).
            tags: Dict mapping tag ID to raw tag data bytes.
            destination_mac: Destination MAC (usually broadcast).
            bssid: BSSID (usually broadcast for probes).

        Returns:
            WiFiProbeRequest.
        """
        result = WiFiProbeRequest(
            source_mac=source_mac.lower(),
            destination_mac=destination_mac.lower(),
            bssid=bssid.lower(),
            ssid=ssid,
            is_broadcast_probe=len(ssid) == 0,
        )

        if tags:
            # Reconstruct a tagged parameter buffer
            buf = bytearray()
            for tag_id, tag_data in sorted(tags.items()):
                buf.append(tag_id)
                buf.append(len(tag_data))
                buf.extend(tag_data)
            self._parse_tagged_params(bytes(buf), result)

        return result
