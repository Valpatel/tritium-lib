# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""BLE advertisement data parser.

Parses raw BLE advertisement bytes (AD structures) as defined in
Bluetooth Core Specification v5.x, Vol 3, Part C, Section 11.

Each AD structure is: [length] [type] [data...]

Supported AD types:
    0x01 — Flags
    0x02 / 0x03 — 16-bit Service UUIDs (incomplete / complete)
    0x04 / 0x05 — 32-bit Service UUIDs (incomplete / complete)
    0x06 / 0x07 — 128-bit Service UUIDs (incomplete / complete)
    0x08 / 0x09 — Shortened / Complete Local Name
    0x0A — TX Power Level
    0xFF — Manufacturer Specific Data
    0x16 — Service Data (16-bit UUID)
    0x20 — Service Data (32-bit UUID)
    0x21 — Service Data (128-bit UUID)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.protocols.errors import ParseError

# BLE Flags bits
FLAG_LE_LIMITED_DISC = 0x01
FLAG_LE_GENERAL_DISC = 0x02
FLAG_BR_EDR_NOT_SUPPORTED = 0x04
FLAG_LE_BR_EDR_CONTROLLER = 0x08
FLAG_LE_BR_EDR_HOST = 0x10

# Well-known 16-bit company IDs (subset)
COMPANY_IDS: dict[int, str] = {
    0x004C: "Apple",
    0x0006: "Microsoft",
    0x00E0: "Google",
    0x0075: "Samsung",
    0x0059: "Nordic Semiconductor",
    0x000F: "Texas Instruments",
    0x0131: "Espressif",
    0x0002: "Intel",
    0x001D: "Qualcomm",
    0x0046: "MediaTek",
    0x0310: "Xiaomi",
}


@dataclass
class BLEFlags:
    """Parsed BLE advertisement flags."""

    raw: int = 0
    le_limited_discoverable: bool = False
    le_general_discoverable: bool = False
    br_edr_not_supported: bool = False
    le_br_edr_controller: bool = False
    le_br_edr_host: bool = False


@dataclass
class ManufacturerData:
    """Manufacturer-specific data from a BLE advertisement."""

    company_id: int = 0
    company_name: str = "Unknown"
    data: bytes = b""


@dataclass
class ServiceData:
    """Service data from a BLE advertisement."""

    uuid: str = ""
    data: bytes = b""


@dataclass
class BLEAdvertisement:
    """Fully parsed BLE advertisement."""

    flags: Optional[BLEFlags] = None
    local_name: str = ""
    shortened_name: str = ""
    tx_power: Optional[int] = None
    service_uuids_16: list[str] = field(default_factory=list)
    service_uuids_32: list[str] = field(default_factory=list)
    service_uuids_128: list[str] = field(default_factory=list)
    manufacturer_data: list[ManufacturerData] = field(default_factory=list)
    service_data: list[ServiceData] = field(default_factory=list)
    raw_structures: list[dict] = field(default_factory=list)

    @property
    def all_service_uuids(self) -> list[str]:
        """Return all service UUIDs from all sizes combined."""
        return self.service_uuids_16 + self.service_uuids_32 + self.service_uuids_128

    @property
    def is_connectable(self) -> bool:
        """Heuristic: if BR/EDR not supported and general discoverable, likely connectable."""
        if self.flags is None:
            return False
        return self.flags.le_general_discoverable or self.flags.le_limited_discoverable

    @property
    def display_name(self) -> str:
        """Best available name for this device."""
        return self.local_name or self.shortened_name or ""


class BLEAdvertParser:
    """Parser for raw BLE advertisement data (AD structures).

    Usage::

        parser = BLEAdvertParser()
        advert = parser.parse(b"\\x02\\x01\\x06\\x03\\x03\\x0f\\x18")
        print(advert.flags.le_general_discoverable)  # True
        print(advert.service_uuids_16)               # ['180f']
    """

    @staticmethod
    def _to_bytes(data: bytes | str) -> bytes:
        """Normalize input to bytes."""
        if isinstance(data, str):
            # Strip whitespace and common prefixes
            cleaned = data.strip().replace(" ", "").replace("-", "")
            if cleaned.startswith("0x") or cleaned.startswith("0X"):
                cleaned = cleaned[2:]
            try:
                return bytes.fromhex(cleaned)
            except ValueError as exc:
                raise ParseError("BLE", f"Invalid hex string: {exc}", data) from exc
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        raise ParseError("BLE", f"Expected bytes or hex string, got {type(data).__name__}", data)

    @staticmethod
    def _parse_flags(data: bytes) -> BLEFlags:
        """Parse a flags AD structure."""
        if len(data) < 1:
            return BLEFlags()
        val = data[0]
        return BLEFlags(
            raw=val,
            le_limited_discoverable=bool(val & FLAG_LE_LIMITED_DISC),
            le_general_discoverable=bool(val & FLAG_LE_GENERAL_DISC),
            br_edr_not_supported=bool(val & FLAG_BR_EDR_NOT_SUPPORTED),
            le_br_edr_controller=bool(val & FLAG_LE_BR_EDR_CONTROLLER),
            le_br_edr_host=bool(val & FLAG_LE_BR_EDR_HOST),
        )

    @staticmethod
    def _parse_uuid16_list(data: bytes) -> list[str]:
        """Parse a list of 16-bit UUIDs (little-endian)."""
        uuids = []
        for i in range(0, len(data) - 1, 2):
            val = int.from_bytes(data[i : i + 2], "little")
            uuids.append(f"{val:04x}")
        return uuids

    @staticmethod
    def _parse_uuid32_list(data: bytes) -> list[str]:
        """Parse a list of 32-bit UUIDs (little-endian)."""
        uuids = []
        for i in range(0, len(data) - 3, 4):
            val = int.from_bytes(data[i : i + 4], "little")
            uuids.append(f"{val:08x}")
        return uuids

    @staticmethod
    def _parse_uuid128_list(data: bytes) -> list[str]:
        """Parse a list of 128-bit UUIDs (little-endian)."""
        uuids = []
        for i in range(0, len(data) - 15, 16):
            raw = data[i : i + 16]
            # 128-bit UUIDs are stored little-endian; display big-endian
            be = raw[::-1]
            uuid_str = (
                f"{be[0:4].hex()}-{be[4:6].hex()}-{be[6:8].hex()}-"
                f"{be[8:10].hex()}-{be[10:16].hex()}"
            )
            uuids.append(uuid_str)
        return uuids

    @staticmethod
    def _parse_manufacturer_data(data: bytes) -> ManufacturerData:
        """Parse manufacturer-specific data."""
        if len(data) < 2:
            return ManufacturerData(data=data)
        company_id = int.from_bytes(data[0:2], "little")
        company_name = COMPANY_IDS.get(company_id, "Unknown")
        return ManufacturerData(
            company_id=company_id,
            company_name=company_name,
            data=data[2:],
        )

    @staticmethod
    def _parse_service_data_16(data: bytes) -> ServiceData:
        """Parse service data with 16-bit UUID."""
        if len(data) < 2:
            return ServiceData(data=data)
        uuid_val = int.from_bytes(data[0:2], "little")
        return ServiceData(uuid=f"{uuid_val:04x}", data=data[2:])

    @staticmethod
    def _parse_service_data_32(data: bytes) -> ServiceData:
        """Parse service data with 32-bit UUID."""
        if len(data) < 4:
            return ServiceData(data=data)
        uuid_val = int.from_bytes(data[0:4], "little")
        return ServiceData(uuid=f"{uuid_val:08x}", data=data[4:])

    @staticmethod
    def _parse_service_data_128(data: bytes) -> ServiceData:
        """Parse service data with 128-bit UUID."""
        if len(data) < 16:
            return ServiceData(data=data)
        raw = data[0:16]
        be = raw[::-1]
        uuid_str = (
            f"{be[0:4].hex()}-{be[4:6].hex()}-{be[6:8].hex()}-"
            f"{be[8:10].hex()}-{be[10:16].hex()}"
        )
        return ServiceData(uuid=uuid_str, data=data[16:])

    def parse(self, data: bytes | str) -> BLEAdvertisement:
        """Parse raw BLE advertisement data into structured fields.

        Args:
            data: Raw advertisement bytes or hex string.

        Returns:
            BLEAdvertisement with parsed fields.

        Raises:
            ParseError: If data is fundamentally invalid.
        """
        raw = self._to_bytes(data)
        if len(raw) == 0:
            raise ParseError("BLE", "Empty advertisement data", data)

        result = BLEAdvertisement()
        offset = 0

        while offset < len(raw):
            if offset + 1 > len(raw):
                break  # no room for length byte

            length = raw[offset]
            if length == 0:
                offset += 1
                continue  # skip zero-length padding

            if offset + 1 + length > len(raw):
                # Truncated — stop parsing but keep what we have
                break

            ad_type = raw[offset + 1]
            ad_data = raw[offset + 2 : offset + 1 + length]

            # Record raw structure
            result.raw_structures.append({
                "type": ad_type,
                "length": length,
                "data": ad_data.hex(),
            })

            # Parse by type
            if ad_type == 0x01:  # Flags
                result.flags = self._parse_flags(ad_data)
            elif ad_type in (0x02, 0x03):  # 16-bit UUIDs
                result.service_uuids_16.extend(self._parse_uuid16_list(ad_data))
            elif ad_type in (0x04, 0x05):  # 32-bit UUIDs
                result.service_uuids_32.extend(self._parse_uuid32_list(ad_data))
            elif ad_type in (0x06, 0x07):  # 128-bit UUIDs
                result.service_uuids_128.extend(self._parse_uuid128_list(ad_data))
            elif ad_type == 0x08:  # Shortened Local Name
                try:
                    result.shortened_name = ad_data.decode("utf-8", errors="replace")
                except Exception:
                    result.shortened_name = ad_data.hex()
            elif ad_type == 0x09:  # Complete Local Name
                try:
                    result.local_name = ad_data.decode("utf-8", errors="replace")
                except Exception:
                    result.local_name = ad_data.hex()
            elif ad_type == 0x0A:  # TX Power Level
                if len(ad_data) >= 1:
                    # Signed 8-bit
                    val = ad_data[0]
                    if val > 127:
                        val -= 256
                    result.tx_power = val
            elif ad_type == 0xFF:  # Manufacturer Specific Data
                result.manufacturer_data.append(self._parse_manufacturer_data(ad_data))
            elif ad_type == 0x16:  # Service Data (16-bit)
                result.service_data.append(self._parse_service_data_16(ad_data))
            elif ad_type == 0x20:  # Service Data (32-bit)
                result.service_data.append(self._parse_service_data_32(ad_data))
            elif ad_type == 0x21:  # Service Data (128-bit)
                result.service_data.append(self._parse_service_data_128(ad_data))

            offset += 1 + length

        return result

    def parse_hex(self, hex_str: str) -> BLEAdvertisement:
        """Convenience: parse a hex string advertisement."""
        return self.parse(hex_str)
