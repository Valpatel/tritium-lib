# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ADS-B (Automatic Dependent Surveillance-Broadcast) message parser.

Parses Mode S downlink format (DF) messages, focusing on ADS-B
extended squitter (DF17/DF18).  Input can be:
    - Raw hex string of the 14-byte (112-bit) message
    - Beast binary frame
    - SBS/BaseStation CSV line

Supported DF17 type codes:
    TC 1-4   — Aircraft Identification
    TC 5-8   — Surface Position
    TC 9-18  — Airborne Position (barometric altitude)
    TC 19    — Airborne Velocity
    TC 20-22 — Airborne Position (GNSS altitude)
    TC 28    — Aircraft Status (emergency/priority)
    TC 29    — Target State and Status
    TC 31    — Operational Status

Also supports:
    DF4/DF20  — Altitude Reply / Comm-B with Altitude
    DF5/DF21  — Squawk Reply / Comm-B with Identity
    DF11      — All-Call Reply (ICAO address extraction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.protocols.errors import ParseError

# Downlink format descriptions
_DF_NAMES = {
    0: "Short Air-Air Surveillance",
    4: "Surveillance Altitude Reply",
    5: "Surveillance Identity Reply",
    11: "All-Call Reply",
    16: "Long Air-Air Surveillance",
    17: "Extended Squitter",
    18: "Extended Squitter (Non-Transponder)",
    19: "Military Extended Squitter",
    20: "Comm-B Altitude Reply",
    21: "Comm-B Identity Reply",
    24: "Comm-D (ELM)",
}

# Aircraft category sets
_CATEGORY_SETS = {
    (0, 0): "No category info",
    (1, 0): "No category info",
    (2, 1): "Surface Emergency Vehicle",
    (2, 3): "Surface Service Vehicle",
    (2, 4): "Ground Obstruction (4)",
    (2, 5): "Ground Obstruction (5)",
    (2, 6): "Ground Obstruction (6)",
    (2, 7): "Ground Obstruction (7)",
    (3, 1): "Glider/Sailplane",
    (3, 2): "Lighter-Than-Air",
    (3, 3): "Parachutist/Skydiver",
    (3, 4): "Ultralight/Hang-Glider",
    (3, 6): "UAV",
    (3, 7): "Space Vehicle",
    (4, 1): "Light (< 15500 lbs)",
    (4, 2): "Small (15500-75000 lbs)",
    (4, 3): "Large (75000-300000 lbs)",
    (4, 4): "High Vortex Large",
    (4, 5): "Heavy (> 300000 lbs)",
    (4, 6): "High Performance (> 5g, > 400 kts)",
    (4, 7): "Rotorcraft",
}

# Gillham code mapping for altitude decoding
_EMERGENCY_CODES = {
    0: "no_emergency",
    1: "general_emergency",
    2: "lifeguard_medical",
    3: "minimum_fuel",
    4: "no_communications",
    5: "unlawful_interference",
    6: "downed_aircraft",
    7: "reserved",
}

# ICAO character set for callsign decoding
_ICAO_CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"


@dataclass
class ADSBMessage:
    """Base parsed ADS-B / Mode S message."""

    downlink_format: int = 0
    df_name: str = ""
    icao_hex: str = ""  # 24-bit ICAO address as 6-char hex
    raw_hex: str = ""
    raw_bits: int = 0  # message length in bits

    # Type code (for DF17/18 extended squitter)
    type_code: int = 0
    subtype: int = 0

    # Capabilities
    capability: int = 0


@dataclass
class ADSBIdentification(ADSBMessage):
    """Aircraft identification (TC 1-4)."""

    callsign: str = ""
    category: int = 0
    category_set: int = 0
    category_description: str = ""


@dataclass
class ADSBAirbornePosition(ADSBMessage):
    """Airborne position report (TC 9-18, 20-22)."""

    altitude_ft: float = 0.0
    cpr_lat: int = 0  # CPR latitude (17 bits)
    cpr_lon: int = 0  # CPR longitude (17 bits)
    cpr_odd: bool = False  # True = odd frame, False = even frame
    surveillance_status: int = 0
    single_antenna: bool = False
    is_barometric: bool = True  # True for TC 9-18, False for 20-22


@dataclass
class ADSBSurfacePosition(ADSBMessage):
    """Surface position report (TC 5-8)."""

    ground_speed_kt: float = 0.0
    ground_track: float = 0.0
    ground_track_valid: bool = False
    cpr_lat: int = 0
    cpr_lon: int = 0
    cpr_odd: bool = False


@dataclass
class ADSBVelocity(ADSBMessage):
    """Airborne velocity report (TC 19)."""

    speed_kt: float = 0.0
    heading: float = 0.0
    heading_valid: bool = False
    vertical_rate_fpm: float = 0.0  # feet per minute
    vertical_rate_source: str = ""  # "barometric" or "gnss"
    speed_type: str = ""  # "ground_speed" or "indicated_airspeed"
    gnss_baro_diff_ft: float = 0.0


@dataclass
class ADSBSquawk(ADSBMessage):
    """Squawk/identity reply (DF5/DF21)."""

    squawk: str = "0000"

    @property
    def is_emergency(self) -> bool:
        return self.squawk in ("7500", "7600", "7700")

    @property
    def is_hijack(self) -> bool:
        return self.squawk == "7500"


@dataclass
class ADSBAltitudeReply(ADSBMessage):
    """Altitude reply (DF4/DF20)."""

    altitude_ft: float = 0.0
    flight_status: int = 0


@dataclass
class ADSBEmergency(ADSBMessage):
    """Aircraft status / emergency (TC 28)."""

    emergency_state: int = 0
    emergency_text: str = "no_emergency"
    squawk: str = "0000"


class ADSBParser:
    """Parser for ADS-B / Mode S hex messages.

    Usage::

        parser = ADSBParser()
        msg = parser.parse("8D4840D6202CC371C32CE0576098")
        if isinstance(msg, ADSBIdentification):
            print(msg.callsign)  # "KLM1023"
    """

    @staticmethod
    def _hex_to_bits(hex_str: str) -> list[int]:
        """Convert hex string to bit array."""
        bits: list[int] = []
        for ch in hex_str.upper():
            val = int(ch, 16)
            for i in range(3, -1, -1):
                bits.append((val >> i) & 1)
        return bits

    @staticmethod
    def _bits_to_uint(bits: list[int], start: int, length: int) -> int:
        val = 0
        for i in range(start, min(start + length, len(bits))):
            val = (val << 1) | bits[i]
        return val

    @staticmethod
    def _bits_to_int(bits: list[int], start: int, length: int) -> int:
        val = 0
        for i in range(start, min(start + length, len(bits))):
            val = (val << 1) | bits[i]
        if length > 0 and val >= (1 << (length - 1)):
            val -= 1 << length
        return val

    @staticmethod
    def _decode_callsign(bits: list[int], start: int) -> str:
        """Decode an 8-character callsign from 48 bits."""
        chars = []
        for i in range(8):
            idx = 0
            for j in range(6):
                pos = start + i * 6 + j
                if pos < len(bits):
                    idx = (idx << 1) | bits[pos]
                else:
                    idx <<= 1
            if 0 <= idx < len(_ICAO_CHARSET):
                ch = _ICAO_CHARSET[idx]
                if ch != "#":
                    chars.append(ch)
            else:
                chars.append("?")
        return "".join(chars).strip()

    @staticmethod
    def _decode_altitude(bits: list[int], start: int) -> float:
        """Decode 12-bit altitude code (Gillham/Mode C style for DF17).

        For TC 9-18 (barometric), the 12 bits at the position contain
        an encoded altitude.  Bit 8 (Q-bit) determines encoding:
            Q=1: 25-ft resolution → altitude = (N * 25) - 1000
            Q=0: 100-ft Gillham coding
        """
        # Extract 12 altitude bits
        alt_bits = []
        for i in range(12):
            pos = start + i
            if pos < len(bits):
                alt_bits.append(bits[pos])
            else:
                alt_bits.append(0)

        # Q-bit is bit index 7 (8th bit of the 12)
        q_bit = alt_bits[7]

        if q_bit == 1:
            # Remove Q-bit and compute
            n_bits = alt_bits[0:7] + alt_bits[8:12]
            n = 0
            for b in n_bits:
                n = (n << 1) | b
            return float(n * 25 - 1000)
        else:
            # Gillham (100-ft increments) — simplified
            n = 0
            for b in alt_bits:
                n = (n << 1) | b
            if n == 0:
                return 0.0
            return float(n * 100)

    @staticmethod
    def _decode_squawk_identity(bits: list[int], start: int) -> str:
        """Decode 13-bit identity (squawk) code.

        The 13 bits encode a 4-digit octal squawk code using
        a specific interleaving pattern.
        """
        if start + 13 > len(bits):
            return "0000"

        # Identity code bits layout (13 bits):
        # C1 A1 C2 A2 C4 A4 _ B1 D1 B2 D2 B4 D4
        c1 = bits[start + 0]
        a1 = bits[start + 1]
        c2 = bits[start + 2]
        a2 = bits[start + 3]
        c4 = bits[start + 4]
        a4 = bits[start + 5]
        # bit 6 is spare (SPI)
        b1 = bits[start + 7]
        d1 = bits[start + 8]
        b2 = bits[start + 9]
        d2 = bits[start + 10]
        b4 = bits[start + 11]
        d4 = bits[start + 12]

        a = a4 * 4 + a2 * 2 + a1
        b = b4 * 4 + b2 * 2 + b1
        c = c4 * 4 + c2 * 2 + c1
        d = d4 * 4 + d2 * 2 + d1

        return f"{a}{b}{c}{d}"

    @staticmethod
    def _decode_df4_altitude(bits: list[int]) -> float:
        """Decode altitude from DF4/DF20 (13-bit altitude code at bits 20-32)."""
        if len(bits) < 33:
            return 0.0

        alt_bits = []
        for i in range(20, 33):
            alt_bits.append(bits[i])

        # M-bit (bit 25, index 5 in the 13 alt bits) and Q-bit (index 7)
        m_bit = alt_bits[5]
        q_bit = alt_bits[7]

        if m_bit == 0 and q_bit == 1:
            # 25-ft resolution
            n_bits = alt_bits[0:5] + alt_bits[6:7] + alt_bits[8:13]
            n = 0
            for b in n_bits:
                n = (n << 1) | b
            return float(n * 25 - 1000)
        else:
            # 100-ft resolution (simplified)
            n = 0
            for b in alt_bits:
                n = (n << 1) | b
            return float(n * 100)

    def parse(self, hex_msg: str) -> ADSBMessage:
        """Parse a Mode S / ADS-B hex message.

        Args:
            hex_msg: Hex string (14 or 28 hex chars for 56/112-bit messages).
                     Leading ``*`` or trailing ``;`` are stripped.

        Returns:
            Typed message dataclass.

        Raises:
            ParseError: If message is malformed.
        """
        # Clean input
        cleaned = hex_msg.strip().upper()
        cleaned = cleaned.lstrip("*").rstrip(";")
        cleaned = cleaned.replace(" ", "")

        if not cleaned:
            raise ParseError("ADSB", "Empty message")

        # Validate hex
        try:
            int(cleaned, 16)
        except ValueError:
            raise ParseError("ADSB", f"Invalid hex: {cleaned}", hex_msg)

        bits = self._hex_to_bits(cleaned)
        n_bits = len(bits)

        if n_bits < 56:
            raise ParseError("ADSB", f"Message too short ({n_bits} bits, need >= 56)", hex_msg)

        # Downlink format (first 5 bits)
        df = self._bits_to_uint(bits, 0, 5)
        df_name = _DF_NAMES.get(df, f"Unknown DF{df}")

        # ICAO address extraction depends on DF
        if df in (11, 17, 18):
            icao_hex = f"{self._bits_to_uint(bits, 8, 24):06X}"
        elif df in (0, 4, 5, 16, 20, 21):
            # For short messages, ICAO is in the parity (last 24 bits)
            # but only recoverable with known aircraft — use raw for now
            icao_hex = f"{self._bits_to_uint(bits, n_bits - 24, 24):06X}"
        else:
            icao_hex = ""

        # DF4/DF20 — Altitude Reply
        if df in (4, 20):
            fs = self._bits_to_uint(bits, 5, 3)
            altitude = self._decode_df4_altitude(bits)
            return ADSBAltitudeReply(
                downlink_format=df,
                df_name=df_name,
                icao_hex=icao_hex,
                raw_hex=cleaned,
                raw_bits=n_bits,
                altitude_ft=altitude,
                flight_status=fs,
            )

        # DF5/DF21 — Identity (Squawk) Reply
        if df in (5, 21):
            squawk = self._decode_squawk_identity(bits, 20)
            return ADSBSquawk(
                downlink_format=df,
                df_name=df_name,
                icao_hex=icao_hex,
                raw_hex=cleaned,
                raw_bits=n_bits,
                squawk=squawk,
            )

        # DF11 — All-Call Reply
        if df == 11:
            ca = self._bits_to_uint(bits, 5, 3)
            return ADSBMessage(
                downlink_format=df,
                df_name=df_name,
                icao_hex=icao_hex,
                raw_hex=cleaned,
                raw_bits=n_bits,
                capability=ca,
            )

        # DF17/DF18 — Extended Squitter (ADS-B)
        if df in (17, 18):
            if n_bits < 112:
                raise ParseError("ADSB", f"Extended squitter too short ({n_bits} bits, need 112)", hex_msg)

            ca = self._bits_to_uint(bits, 5, 3)
            tc = self._bits_to_uint(bits, 32, 5)
            st = self._bits_to_uint(bits, 37, 3)

            # TC 1-4: Aircraft Identification
            if 1 <= tc <= 4:
                callsign = self._decode_callsign(bits, 40)
                cat_set = tc
                cat = st
                cat_desc = _CATEGORY_SETS.get((cat_set, cat), "Unknown")
                return ADSBIdentification(
                    downlink_format=df,
                    df_name=df_name,
                    icao_hex=icao_hex,
                    raw_hex=cleaned,
                    raw_bits=n_bits,
                    type_code=tc,
                    subtype=st,
                    capability=ca,
                    callsign=callsign,
                    category=cat,
                    category_set=cat_set,
                    category_description=cat_desc,
                )

            # TC 5-8: Surface Position
            if 5 <= tc <= 8:
                # Ground speed
                movement = self._bits_to_uint(bits, 37, 7)
                if movement == 0:
                    gs = 0.0
                elif movement == 1:
                    gs = 0.0  # stopped
                elif movement <= 8:
                    gs = 0.125 * (movement - 1)
                elif movement <= 12:
                    gs = 1.0 + 0.25 * (movement - 9)
                elif movement <= 38:
                    gs = 2.0 + 0.5 * (movement - 13)
                elif movement <= 93:
                    gs = 15.0 + 1.0 * (movement - 39)
                elif movement <= 108:
                    gs = 70.0 + 2.0 * (movement - 94)
                elif movement <= 123:
                    gs = 100.0 + 5.0 * (movement - 109)
                elif movement == 124:
                    gs = 175.0
                else:
                    gs = 0.0

                trk_valid = bool(self._bits_to_uint(bits, 44, 1))
                trk = self._bits_to_uint(bits, 45, 7) * (360.0 / 128.0)
                cpr_odd = bool(self._bits_to_uint(bits, 53, 1))
                cpr_lat = self._bits_to_uint(bits, 54, 17)
                cpr_lon = self._bits_to_uint(bits, 71, 17)

                return ADSBSurfacePosition(
                    downlink_format=df,
                    df_name=df_name,
                    icao_hex=icao_hex,
                    raw_hex=cleaned,
                    raw_bits=n_bits,
                    type_code=tc,
                    subtype=st,
                    capability=ca,
                    ground_speed_kt=gs,
                    ground_track=trk,
                    ground_track_valid=trk_valid,
                    cpr_lat=cpr_lat,
                    cpr_lon=cpr_lon,
                    cpr_odd=cpr_odd,
                )

            # TC 9-18: Airborne Position (barometric)
            # TC 20-22: Airborne Position (GNSS)
            if 9 <= tc <= 22 and tc not in (19,):
                ss = self._bits_to_uint(bits, 37, 2)
                saf = bool(self._bits_to_uint(bits, 39, 1))
                alt = self._decode_altitude(bits, 40)
                cpr_odd = bool(self._bits_to_uint(bits, 53, 1))
                cpr_lat = self._bits_to_uint(bits, 54, 17)
                cpr_lon = self._bits_to_uint(bits, 71, 17)

                return ADSBAirbornePosition(
                    downlink_format=df,
                    df_name=df_name,
                    icao_hex=icao_hex,
                    raw_hex=cleaned,
                    raw_bits=n_bits,
                    type_code=tc,
                    subtype=st,
                    capability=ca,
                    altitude_ft=alt,
                    cpr_lat=cpr_lat,
                    cpr_lon=cpr_lon,
                    cpr_odd=cpr_odd,
                    surveillance_status=ss,
                    single_antenna=saf,
                    is_barometric=(9 <= tc <= 18),
                )

            # TC 19: Airborne Velocity
            if tc == 19:
                vr_source = "gnss" if self._bits_to_uint(bits, 36 + 35, 1) else "barometric"
                speed = 0.0
                heading = 0.0
                heading_valid = False
                speed_type = ""

                if st in (1, 2):
                    # Ground speed (east-west / north-south components)
                    ew_sign = -1 if self._bits_to_uint(bits, 45, 1) else 1
                    ew_vel = self._bits_to_uint(bits, 46, 10) - 1
                    ns_sign = -1 if self._bits_to_uint(bits, 56, 1) else 1
                    ns_vel = self._bits_to_uint(bits, 57, 10) - 1

                    if ew_vel >= 0 and ns_vel >= 0:
                        vx = ew_sign * ew_vel
                        vy = ns_sign * ns_vel
                        speed = (vx ** 2 + vy ** 2) ** 0.5
                        if st == 2:
                            speed *= 4  # supersonic
                        import math
                        heading = (math.atan2(vx, vy) * 180.0 / math.pi) % 360.0
                        heading_valid = True
                    speed_type = "ground_speed"

                elif st in (3, 4):
                    # Indicated airspeed / heading
                    heading_valid = bool(self._bits_to_uint(bits, 45, 1))
                    heading = self._bits_to_uint(bits, 46, 10) * (360.0 / 1024.0)
                    speed = self._bits_to_uint(bits, 57, 10)
                    if st == 4:
                        speed *= 4
                    speed_type = "indicated_airspeed"

                # Vertical rate
                vr_sign = -1 if self._bits_to_uint(bits, 68, 1) else 1
                vr_val = self._bits_to_uint(bits, 69, 9)
                vr_fpm = vr_sign * (vr_val - 1) * 64.0 if vr_val > 0 else 0.0

                # GNSS-Baro difference
                gbd_sign = -1 if self._bits_to_uint(bits, 80, 1) else 1
                gbd_val = self._bits_to_uint(bits, 81, 7)
                gbd_ft = gbd_sign * gbd_val * 25.0

                return ADSBVelocity(
                    downlink_format=df,
                    df_name=df_name,
                    icao_hex=icao_hex,
                    raw_hex=cleaned,
                    raw_bits=n_bits,
                    type_code=tc,
                    subtype=st,
                    capability=ca,
                    speed_kt=speed,
                    heading=heading,
                    heading_valid=heading_valid,
                    vertical_rate_fpm=vr_fpm,
                    vertical_rate_source=vr_source,
                    speed_type=speed_type,
                    gnss_baro_diff_ft=gbd_ft,
                )

            # TC 28: Emergency / Priority Status
            if tc == 28:
                emergency_state = self._bits_to_uint(bits, 40, 3)
                squawk_a = self._bits_to_uint(bits, 43, 4)
                squawk_b = self._bits_to_uint(bits, 47, 4)
                squawk_c = self._bits_to_uint(bits, 51, 4)
                squawk_d = self._bits_to_uint(bits, 55, 4)
                squawk_str = f"{squawk_a}{squawk_b}{squawk_c}{squawk_d}"

                return ADSBEmergency(
                    downlink_format=df,
                    df_name=df_name,
                    icao_hex=icao_hex,
                    raw_hex=cleaned,
                    raw_bits=n_bits,
                    type_code=tc,
                    subtype=st,
                    capability=ca,
                    emergency_state=emergency_state,
                    emergency_text=_EMERGENCY_CODES.get(emergency_state, "unknown"),
                    squawk=squawk_str,
                )

            # Fallback for other TCs
            return ADSBMessage(
                downlink_format=df,
                df_name=df_name,
                icao_hex=icao_hex,
                raw_hex=cleaned,
                raw_bits=n_bits,
                type_code=tc,
                subtype=st,
                capability=ca,
            )

        # Unknown DF — return base message
        return ADSBMessage(
            downlink_format=df,
            df_name=df_name,
            icao_hex=icao_hex,
            raw_hex=cleaned,
            raw_bits=n_bits,
        )

    def parse_sbs(self, line: str) -> dict:
        """Parse an SBS/BaseStation CSV line (port 30003 format).

        SBS format: MSG,<type>,<session>,<aircraft>,<icao>,<flight>,
                    <gen_date>,<gen_time>,<log_date>,<log_time>,
                    <callsign>,<altitude>,<speed>,<heading>,
                    <lat>,<lon>,<vertical_rate>,<squawk>,
                    <alert>,<emergency>,<spi>,<on_ground>

        Args:
            line: Raw CSV line.

        Returns:
            Dict with parsed fields (only non-empty fields included).

        Raises:
            ParseError: If format is invalid.
        """
        line = line.strip()
        if not line:
            raise ParseError("ADSB", "Empty SBS line")

        fields = line.split(",")
        if len(fields) < 22:
            raise ParseError("ADSB", f"SBS line too short ({len(fields)} fields, need 22)", line)

        if fields[0] != "MSG":
            raise ParseError("ADSB", f"Not an SBS MSG line: {fields[0]}", line)

        result: dict = {
            "format": "sbs",
            "msg_type": int(fields[1]) if fields[1] else 0,
            "icao_hex": fields[4].strip().upper(),
        }

        # Callsign (field 10)
        if fields[10].strip():
            result["callsign"] = fields[10].strip()

        # Altitude (field 11)
        if fields[11].strip():
            try:
                result["altitude_ft"] = float(fields[11])
            except ValueError:
                pass

        # Ground speed (field 12)
        if fields[12].strip():
            try:
                result["ground_speed_kt"] = float(fields[12])
            except ValueError:
                pass

        # Track/heading (field 13)
        if fields[13].strip():
            try:
                result["heading"] = float(fields[13])
            except ValueError:
                pass

        # Latitude (field 14)
        if fields[14].strip():
            try:
                result["latitude"] = float(fields[14])
            except ValueError:
                pass

        # Longitude (field 15)
        if fields[15].strip():
            try:
                result["longitude"] = float(fields[15])
            except ValueError:
                pass

        # Vertical rate (field 16)
        if fields[16].strip():
            try:
                result["vertical_rate_fpm"] = float(fields[16])
            except ValueError:
                pass

        # Squawk (field 17)
        if fields[17].strip():
            result["squawk"] = fields[17].strip()

        # On ground (field 21)
        if fields[21].strip():
            result["on_ground"] = fields[21].strip() == "-1"

        return result
