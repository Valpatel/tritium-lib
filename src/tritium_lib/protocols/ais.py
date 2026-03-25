# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""AIS (Automatic Identification System) message parser.

Parses AIVDM/AIVDO NMEA-encapsulated AIS sentences.  AIS uses 6-bit
ASCII armor encoding over VHF Data Link.

Supported message types:
    1, 2, 3  — Position Report (Class A)
    5        — Static and Voyage Related Data
    18       — Standard Class B Position Report
    24       — Class B Static Data (Part A + Part B)

Sentence format:
    !AIVDM,<fragment_count>,<fragment_num>,<seq_id>,<channel>,<payload>,<pad>*<checksum>

The payload is 6-bit ASCII armored: each character maps to a 6-bit value,
and the resulting bit stream encodes the message fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.protocols.errors import ParseError

# Navigation status lookup (ITU-R M.1371)
_NAV_STATUS = {
    0: "under_way_engine",
    1: "at_anchor",
    2: "not_under_command",
    3: "restricted_maneuverability",
    4: "constrained_by_draught",
    5: "moored",
    6: "aground",
    7: "fishing",
    8: "under_way_sailing",
    9: "reserved_hsc",
    10: "reserved_wig",
    11: "reserved_11",
    12: "reserved_12",
    13: "reserved_13",
    14: "ais_sart",
    15: "default",
}

# Ship type lookup (first digit = category)
_SHIP_TYPE_CATEGORY = {
    0: "not_available",
    1: "reserved",
    2: "wing_in_ground",
    3: "special_craft",
    4: "high_speed_craft",
    5: "special_craft",
    6: "passenger",
    7: "cargo",
    8: "tanker",
    9: "other",
}


@dataclass
class AISPositionReport:
    """Position data from an AIS message (types 1, 2, 3, 18)."""

    message_type: int = 0
    mmsi: int = 0
    navigation_status: int = 15  # default
    navigation_status_text: str = "default"
    rate_of_turn: float = 0.0  # degrees/min
    speed_over_ground: float = 0.0  # knots (1/10 knot resolution)
    position_accuracy: bool = False  # True = DGPS/high accuracy
    longitude: float = 181.0  # degrees (181 = not available)
    latitude: float = 91.0  # degrees (91 = not available)
    course_over_ground: float = 360.0  # degrees (360 = not available)
    true_heading: int = 511  # degrees (511 = not available)
    timestamp: int = 60  # UTC second (60 = not available)
    repeat_indicator: int = 0
    raim_flag: bool = False

    @property
    def has_valid_position(self) -> bool:
        return -180.0 <= self.longitude <= 180.0 and -90.0 <= self.latitude <= 90.0


@dataclass
class AISStaticData:
    """Static and voyage data from AIS type 5 or 24 messages."""

    message_type: int = 0
    mmsi: int = 0
    imo_number: int = 0
    call_sign: str = ""
    vessel_name: str = ""
    ship_type: int = 0
    ship_type_category: str = "not_available"
    dimension_bow: int = 0  # meters
    dimension_stern: int = 0  # meters
    dimension_port: int = 0  # meters
    dimension_starboard: int = 0  # meters
    draught: float = 0.0  # meters (1/10 m resolution)
    destination: str = ""
    eta_month: int = 0
    eta_day: int = 0
    eta_hour: int = 24
    eta_minute: int = 60
    repeat_indicator: int = 0

    @property
    def length(self) -> int:
        """Total vessel length in meters."""
        return self.dimension_bow + self.dimension_stern

    @property
    def beam(self) -> int:
        """Total vessel beam (width) in meters."""
        return self.dimension_port + self.dimension_starboard

    @property
    def eta_string(self) -> str:
        """ETA as MM-DD HH:MM string."""
        if self.eta_month == 0 and self.eta_day == 0:
            return ""
        return f"{self.eta_month:02d}-{self.eta_day:02d} {self.eta_hour:02d}:{self.eta_minute:02d}"


@dataclass
class AISSentence:
    """Parsed AIVDM/AIVDO sentence wrapper."""

    sentence_type: str = ""  # "AIVDM" or "AIVDO"
    fragment_count: int = 1
    fragment_number: int = 1
    sequential_id: str = ""
    channel: str = ""  # "A" or "B"
    payload: str = ""
    pad_bits: int = 0
    checksum: str = ""
    checksum_valid: bool = False


class AISParser:
    """Parser for AIVDM/AIVDO AIS messages.

    Usage::

        parser = AISParser()
        # Single sentence
        result = parser.parse("!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*24")
        print(result.mmsi)  # 265538450

        # Decode sentence structure
        sentence = parser.decode_sentence("!AIVDM,1,1,,B,13u@Dt002s000000000000000000,0*24")
        print(sentence.channel)  # "B"
    """

    @staticmethod
    def _char_to_payload(c: str) -> int:
        """Convert a single 6-bit ASCII armored character to its value."""
        val = ord(c) - 48
        if val > 40:
            val -= 8
        return val & 0x3F

    @staticmethod
    def _payload_to_bits(payload: str) -> list[int]:
        """Convert AIS payload string to a list of bits."""
        bits: list[int] = []
        for ch in payload:
            val = ord(ch) - 48
            if val > 40:
                val -= 8
            val &= 0x3F
            for i in range(5, -1, -1):
                bits.append((val >> i) & 1)
        return bits

    @staticmethod
    def _bits_to_uint(bits: list[int], start: int, length: int) -> int:
        """Extract an unsigned integer from bit array."""
        val = 0
        for i in range(start, min(start + length, len(bits))):
            val = (val << 1) | bits[i]
        return val

    @staticmethod
    def _bits_to_int(bits: list[int], start: int, length: int) -> int:
        """Extract a signed integer from bit array (two's complement)."""
        val = 0
        for i in range(start, min(start + length, len(bits))):
            val = (val << 1) | bits[i]
        if length > 0 and val >= (1 << (length - 1)):
            val -= 1 << length
        return val

    @staticmethod
    def _bits_to_string(bits: list[int], start: int, length: int) -> str:
        """Extract a 6-bit encoded string from bit array."""
        chars = []
        for i in range(start, min(start + length, len(bits)), 6):
            val = 0
            for j in range(6):
                if i + j < len(bits):
                    val = (val << 1) | bits[i + j]
                else:
                    val <<= 1
            if val < 32:
                val += 64  # '@' offset for AIS 6-bit ASCII
            chars.append(chr(val))
        return "".join(chars).rstrip("@").strip()

    @staticmethod
    def _compute_checksum(sentence: str) -> str:
        """Compute NMEA XOR checksum for the part between ! and *."""
        start = sentence.find("!") + 1
        if start == 0:
            start = sentence.find("$") + 1
        end = sentence.find("*")
        if end == -1:
            end = len(sentence)
        chk = 0
        for c in sentence[start:end]:
            chk ^= ord(c)
        return f"{chk:02X}"

    def decode_sentence(self, sentence: str) -> AISSentence:
        """Decode the NMEA wrapper of an AIS sentence.

        Args:
            sentence: Raw NMEA sentence string.

        Returns:
            AISSentence with parsed fields.

        Raises:
            ParseError: If sentence format is invalid.
        """
        sentence = sentence.strip()
        if not sentence:
            raise ParseError("AIS", "Empty sentence")

        # Extract checksum
        checksum = ""
        checksum_valid = False
        if "*" in sentence:
            parts = sentence.rsplit("*", 1)
            checksum = parts[1].strip().upper()
            expected = self._compute_checksum(sentence)
            checksum_valid = checksum == expected

        # Strip sentence delimiters
        body = sentence
        if body.startswith(("!", "$")):
            body = body[1:]
        if "*" in body:
            body = body[: body.index("*")]

        fields = body.split(",")
        if len(fields) < 7:
            raise ParseError("AIS", f"Too few fields ({len(fields)}, need 7)", sentence)

        sentence_type = fields[0]
        if sentence_type not in ("AIVDM", "AIVDO"):
            raise ParseError("AIS", f"Unknown sentence type: {sentence_type}", sentence)

        try:
            # The last field contains payload and pad bits
            payload_field = fields[5]
            pad_bits = int(fields[6]) if fields[6] else 0

            return AISSentence(
                sentence_type=sentence_type,
                fragment_count=int(fields[1]),
                fragment_number=int(fields[2]),
                sequential_id=fields[3],
                channel=fields[4],
                payload=payload_field,
                pad_bits=pad_bits,
                checksum=checksum,
                checksum_valid=checksum_valid,
            )
        except (ValueError, IndexError) as exc:
            raise ParseError("AIS", f"Failed to parse sentence fields: {exc}", sentence) from exc

    def _parse_position_report(self, bits: list[int], msg_type: int) -> AISPositionReport:
        """Parse AIS message types 1, 2, 3 (Class A position) or 18 (Class B)."""
        result = AISPositionReport(message_type=msg_type)

        result.repeat_indicator = self._bits_to_uint(bits, 6, 2)
        result.mmsi = self._bits_to_uint(bits, 8, 30)

        if msg_type in (1, 2, 3):
            nav_status = self._bits_to_uint(bits, 38, 4)
            result.navigation_status = nav_status
            result.navigation_status_text = _NAV_STATUS.get(nav_status, "unknown")

            rot_raw = self._bits_to_int(bits, 42, 8)
            if rot_raw == -128:
                result.rate_of_turn = 0.0  # not available
            elif rot_raw > 0:
                result.rate_of_turn = (rot_raw / 4.733) ** 2
            elif rot_raw < 0:
                result.rate_of_turn = -((abs(rot_raw) / 4.733) ** 2)
            else:
                result.rate_of_turn = 0.0

            result.speed_over_ground = self._bits_to_uint(bits, 50, 10) / 10.0
            result.position_accuracy = bool(self._bits_to_uint(bits, 60, 1))
            result.longitude = self._bits_to_int(bits, 61, 28) / 600000.0
            result.latitude = self._bits_to_int(bits, 89, 27) / 600000.0
            result.course_over_ground = self._bits_to_uint(bits, 116, 12) / 10.0
            result.true_heading = self._bits_to_uint(bits, 128, 9)
            result.timestamp = self._bits_to_uint(bits, 137, 6)
            result.raim_flag = bool(self._bits_to_uint(bits, 148, 1))

        elif msg_type == 18:
            result.speed_over_ground = self._bits_to_uint(bits, 46, 10) / 10.0
            result.position_accuracy = bool(self._bits_to_uint(bits, 56, 1))
            result.longitude = self._bits_to_int(bits, 57, 28) / 600000.0
            result.latitude = self._bits_to_int(bits, 85, 27) / 600000.0
            result.course_over_ground = self._bits_to_uint(bits, 112, 12) / 10.0
            result.true_heading = self._bits_to_uint(bits, 124, 9)
            result.timestamp = self._bits_to_uint(bits, 133, 6)
            result.raim_flag = bool(self._bits_to_uint(bits, 141, 1))

        return result

    def _parse_static_data(self, bits: list[int], msg_type: int) -> AISStaticData:
        """Parse AIS message type 5 (Class A static + voyage)."""
        result = AISStaticData(message_type=msg_type)

        result.repeat_indicator = self._bits_to_uint(bits, 6, 2)
        result.mmsi = self._bits_to_uint(bits, 8, 30)

        if msg_type == 5:
            result.imo_number = self._bits_to_uint(bits, 40, 30)
            result.call_sign = self._bits_to_string(bits, 70, 42)
            result.vessel_name = self._bits_to_string(bits, 112, 120)
            result.ship_type = self._bits_to_uint(bits, 232, 8)
            result.ship_type_category = _SHIP_TYPE_CATEGORY.get(
                result.ship_type // 10, "not_available"
            )
            result.dimension_bow = self._bits_to_uint(bits, 240, 9)
            result.dimension_stern = self._bits_to_uint(bits, 249, 9)
            result.dimension_port = self._bits_to_uint(bits, 258, 6)
            result.dimension_starboard = self._bits_to_uint(bits, 264, 6)
            result.eta_month = self._bits_to_uint(bits, 274, 4)
            result.eta_day = self._bits_to_uint(bits, 278, 5)
            result.eta_hour = self._bits_to_uint(bits, 283, 5)
            result.eta_minute = self._bits_to_uint(bits, 288, 6)
            result.draught = self._bits_to_uint(bits, 294, 8) / 10.0
            result.destination = self._bits_to_string(bits, 302, 120)

        elif msg_type == 24:
            part_number = self._bits_to_uint(bits, 38, 2)
            if part_number == 0:
                # Part A: name only
                result.vessel_name = self._bits_to_string(bits, 40, 120)
            elif part_number == 1:
                # Part B: type, vendor, call sign, dimensions
                result.ship_type = self._bits_to_uint(bits, 40, 8)
                result.ship_type_category = _SHIP_TYPE_CATEGORY.get(
                    result.ship_type // 10, "not_available"
                )
                result.call_sign = self._bits_to_string(bits, 90, 42)
                result.dimension_bow = self._bits_to_uint(bits, 132, 9)
                result.dimension_stern = self._bits_to_uint(bits, 141, 9)
                result.dimension_port = self._bits_to_uint(bits, 150, 6)
                result.dimension_starboard = self._bits_to_uint(bits, 156, 6)

        return result

    def parse(self, sentence: str) -> AISPositionReport | AISStaticData:
        """Parse an AIS sentence and return typed data.

        Args:
            sentence: Raw AIVDM/AIVDO NMEA sentence.

        Returns:
            AISPositionReport for types 1/2/3/18, AISStaticData for 5/24.

        Raises:
            ParseError: If sentence is malformed or message type unsupported.
        """
        decoded = self.decode_sentence(sentence)
        bits = self._payload_to_bits(decoded.payload)

        if len(bits) < 6:
            raise ParseError("AIS", "Payload too short", sentence)

        msg_type = self._bits_to_uint(bits, 0, 6)

        if msg_type in (1, 2, 3, 18):
            if len(bits) < 149 and msg_type in (1, 2, 3):
                raise ParseError(
                    "AIS",
                    f"Type {msg_type} payload too short ({len(bits)} bits, need >= 149)",
                    sentence,
                )
            return self._parse_position_report(bits, msg_type)

        elif msg_type in (5, 24):
            return self._parse_static_data(bits, msg_type)

        else:
            raise ParseError("AIS", f"Unsupported message type: {msg_type}", sentence)

    def get_message_type(self, sentence: str) -> int:
        """Quick peek at message type without full parsing.

        Args:
            sentence: Raw AIVDM/AIVDO sentence.

        Returns:
            Integer message type (1-27).
        """
        decoded = self.decode_sentence(sentence)
        bits = self._payload_to_bits(decoded.payload)
        if len(bits) < 6:
            raise ParseError("AIS", "Payload too short to read message type", sentence)
        return self._bits_to_uint(bits, 0, 6)
