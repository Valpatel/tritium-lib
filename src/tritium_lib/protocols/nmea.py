# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""NMEA 0183 GPS sentence parser.

Parses standard NMEA sentences from GPS receivers.

Supported sentence types:
    $GPGGA / $GNGGA — Global Positioning System Fix Data
    $GPRMC / $GNRMC — Recommended Minimum Navigation Information
    $GPGSA / $GNGSA — DOP and Active Satellites
    $GPVTG / $GNVTG — Course Over Ground and Ground Speed
    $GPGLL / $GNGLL — Geographic Position (lat/lon)
    $GPGSV / $GNGSV — Satellites in View

Talker IDs: GP (GPS), GL (GLONASS), GA (Galileo), GB/BD (BeiDou), GN (multi-GNSS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.protocols.errors import ParseError

# GPS fix quality (GGA field 6)
_FIX_QUALITY = {
    0: "invalid",
    1: "gps",
    2: "dgps",
    3: "pps",
    4: "rtk_fixed",
    5: "rtk_float",
    6: "estimated",
    7: "manual",
    8: "simulation",
}

# RMC status
_RMC_STATUS = {
    "A": "active",
    "V": "void",
}

# GSA fix mode
_GSA_FIX_MODE = {
    1: "no_fix",
    2: "2d",
    3: "3d",
}


@dataclass
class NMEAPosition:
    """A parsed GPS position."""

    latitude: float = 0.0  # decimal degrees (positive = N)
    longitude: float = 0.0  # decimal degrees (positive = E)
    altitude_m: float = 0.0  # meters above geoid
    geoid_separation_m: float = 0.0  # meters

    @property
    def is_valid(self) -> bool:
        return not (self.latitude == 0.0 and self.longitude == 0.0)


@dataclass
class NMEATime:
    """Parsed UTC time from NMEA."""

    hours: int = 0
    minutes: int = 0
    seconds: float = 0.0

    def __str__(self) -> str:
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:06.3f}"


@dataclass
class NMEADate:
    """Parsed date from NMEA (DDMMYY)."""

    day: int = 0
    month: int = 0
    year: int = 0  # 4-digit

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"


@dataclass
class NMEASatellite:
    """A satellite in view (from GSV)."""

    prn: int = 0  # satellite PRN number
    elevation: int = 0  # degrees (0-90)
    azimuth: int = 0  # degrees (0-359)
    snr: int = 0  # dB-Hz (0-99, 0 = not tracked)


@dataclass
class NMEAGGA:
    """Parsed GGA sentence — fix data."""

    sentence_type: str = "GGA"
    talker_id: str = "GP"
    time: Optional[NMEATime] = None
    position: NMEAPosition = field(default_factory=NMEAPosition)
    fix_quality: int = 0
    fix_quality_text: str = "invalid"
    satellites_used: int = 0
    hdop: float = 0.0
    dgps_age: float = 0.0
    dgps_station_id: str = ""


@dataclass
class NMEARMC:
    """Parsed RMC sentence — recommended minimum."""

    sentence_type: str = "RMC"
    talker_id: str = "GP"
    time: Optional[NMEATime] = None
    date: Optional[NMEADate] = None
    status: str = "void"
    position: NMEAPosition = field(default_factory=NMEAPosition)
    speed_knots: float = 0.0
    course_degrees: float = 0.0
    magnetic_variation: float = 0.0
    mode: str = ""  # A=autonomous, D=differential, E=estimated, N=not valid


@dataclass
class NMEAGSA:
    """Parsed GSA sentence — DOP and active satellites."""

    sentence_type: str = "GSA"
    talker_id: str = "GP"
    mode: str = ""  # M=manual, A=automatic
    fix_type: int = 1  # 1=no fix, 2=2D, 3=3D
    fix_type_text: str = "no_fix"
    satellite_prns: list[int] = field(default_factory=list)
    pdop: float = 0.0
    hdop: float = 0.0
    vdop: float = 0.0


@dataclass
class NMEAVTG:
    """Parsed VTG sentence — course and speed over ground."""

    sentence_type: str = "VTG"
    talker_id: str = "GP"
    course_true: float = 0.0
    course_magnetic: float = 0.0
    speed_knots: float = 0.0
    speed_kmh: float = 0.0
    mode: str = ""  # A=autonomous, D=differential, E=estimated, N=not valid


@dataclass
class NMEAGLL:
    """Parsed GLL sentence — geographic position."""

    sentence_type: str = "GLL"
    talker_id: str = "GP"
    time: Optional[NMEATime] = None
    position: NMEAPosition = field(default_factory=NMEAPosition)
    status: str = "void"
    mode: str = ""


@dataclass
class NMEAGSV:
    """Parsed GSV sentence — satellites in view."""

    sentence_type: str = "GSV"
    talker_id: str = "GP"
    total_messages: int = 1
    message_number: int = 1
    total_satellites: int = 0
    satellites: list[NMEASatellite] = field(default_factory=list)


class NMEAParser:
    """Parser for NMEA 0183 GPS sentences.

    Usage::

        parser = NMEAParser()
        result = parser.parse("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*47")
        if isinstance(result, NMEAGGA):
            print(result.position.latitude)   # 48.1173
            print(result.fix_quality_text)    # "gps"
    """

    @staticmethod
    def _compute_checksum(sentence: str) -> str:
        """Compute NMEA XOR checksum for content between $ and *."""
        start = sentence.find("$") + 1
        if start == 0:
            start = sentence.find("!") + 1
        end = sentence.find("*")
        if end == -1:
            end = len(sentence)
        chk = 0
        for c in sentence[start:end]:
            chk ^= ord(c)
        return f"{chk:02X}"

    @staticmethod
    def _parse_time(field_val: str) -> Optional[NMEATime]:
        """Parse HHMMSS.sss time field."""
        if not field_val or len(field_val) < 6:
            return None
        try:
            hours = int(field_val[0:2])
            minutes = int(field_val[2:4])
            seconds = float(field_val[4:])
            return NMEATime(hours=hours, minutes=minutes, seconds=seconds)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _parse_date(field_val: str) -> Optional[NMEADate]:
        """Parse DDMMYY date field."""
        if not field_val or len(field_val) < 6:
            return None
        try:
            day = int(field_val[0:2])
            month = int(field_val[2:4])
            year_2d = int(field_val[4:6])
            # 2000+ if < 80, 1900+ otherwise
            year = 2000 + year_2d if year_2d < 80 else 1900 + year_2d
            return NMEADate(day=day, month=month, year=year)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _parse_lat(value: str, direction: str) -> float:
        """Parse DDMM.MMMM latitude to decimal degrees."""
        if not value:
            return 0.0
        try:
            degrees = int(value[:2])
            minutes = float(value[2:])
            decimal = degrees + minutes / 60.0
            if direction == "S":
                decimal = -decimal
            return decimal
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def _parse_lon(value: str, direction: str) -> float:
        """Parse DDDMM.MMMM longitude to decimal degrees."""
        if not value:
            return 0.0
        try:
            degrees = int(value[:3])
            minutes = float(value[3:])
            decimal = degrees + minutes / 60.0
            if direction == "W":
                decimal = -decimal
            return decimal
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def _safe_float(val: str, default: float = 0.0) -> float:
        """Safe float conversion."""
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    @staticmethod
    def _safe_int(val: str, default: int = 0) -> int:
        """Safe int conversion."""
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def validate_checksum(self, sentence: str) -> bool:
        """Validate NMEA sentence checksum.

        Args:
            sentence: Complete NMEA sentence with checksum.

        Returns:
            True if checksum matches or no checksum present.
        """
        if "*" not in sentence:
            return True
        expected = sentence.rsplit("*", 1)[1].strip().upper()
        computed = self._compute_checksum(sentence)
        return expected == computed

    def _parse_gga(self, fields: list[str], talker: str) -> NMEAGGA:
        """Parse GGA sentence fields."""
        result = NMEAGGA(talker_id=talker)

        if len(fields) < 15:
            return result

        result.time = self._parse_time(fields[1])

        lat = self._parse_lat(fields[2], fields[3])
        lon = self._parse_lon(fields[4], fields[5])
        alt = self._safe_float(fields[9])
        geoid = self._safe_float(fields[11])

        result.position = NMEAPosition(
            latitude=lat, longitude=lon, altitude_m=alt, geoid_separation_m=geoid
        )

        fq = self._safe_int(fields[6])
        result.fix_quality = fq
        result.fix_quality_text = _FIX_QUALITY.get(fq, "unknown")
        result.satellites_used = self._safe_int(fields[7])
        result.hdop = self._safe_float(fields[8])
        result.dgps_age = self._safe_float(fields[13])
        result.dgps_station_id = fields[14] if len(fields) > 14 else ""

        return result

    def _parse_rmc(self, fields: list[str], talker: str) -> NMEARMC:
        """Parse RMC sentence fields."""
        result = NMEARMC(talker_id=talker)

        if len(fields) < 12:
            return result

        result.time = self._parse_time(fields[1])
        result.status = _RMC_STATUS.get(fields[2], "void")

        lat = self._parse_lat(fields[3], fields[4])
        lon = self._parse_lon(fields[5], fields[6])
        result.position = NMEAPosition(latitude=lat, longitude=lon)

        result.speed_knots = self._safe_float(fields[7])
        result.course_degrees = self._safe_float(fields[8])
        result.date = self._parse_date(fields[9])

        mag_var = self._safe_float(fields[10])
        if len(fields) > 11 and fields[11] == "W":
            mag_var = -mag_var
        result.magnetic_variation = mag_var

        if len(fields) > 12:
            result.mode = fields[12]

        return result

    def _parse_gsa(self, fields: list[str], talker: str) -> NMEAGSA:
        """Parse GSA sentence fields."""
        result = NMEAGSA(talker_id=talker)

        if len(fields) < 18:
            return result

        result.mode = fields[1]
        ft = self._safe_int(fields[2], 1)
        result.fix_type = ft
        result.fix_type_text = _GSA_FIX_MODE.get(ft, "unknown")

        # PRNs in fields 3-14
        prns = []
        for i in range(3, 15):
            if i < len(fields) and fields[i]:
                prn = self._safe_int(fields[i])
                if prn > 0:
                    prns.append(prn)
        result.satellite_prns = prns

        result.pdop = self._safe_float(fields[15])
        result.hdop = self._safe_float(fields[16])
        result.vdop = self._safe_float(fields[17])

        return result

    def _parse_vtg(self, fields: list[str], talker: str) -> NMEAVTG:
        """Parse VTG sentence fields."""
        result = NMEAVTG(talker_id=talker)

        if len(fields) < 9:
            return result

        result.course_true = self._safe_float(fields[1])
        result.course_magnetic = self._safe_float(fields[3])
        result.speed_knots = self._safe_float(fields[5])
        result.speed_kmh = self._safe_float(fields[7])

        if len(fields) > 9:
            result.mode = fields[9]

        return result

    def _parse_gll(self, fields: list[str], talker: str) -> NMEAGLL:
        """Parse GLL sentence fields."""
        result = NMEAGLL(talker_id=talker)

        if len(fields) < 7:
            return result

        lat = self._parse_lat(fields[1], fields[2])
        lon = self._parse_lon(fields[3], fields[4])
        result.position = NMEAPosition(latitude=lat, longitude=lon)
        result.time = self._parse_time(fields[5])
        result.status = _RMC_STATUS.get(fields[6], "void")

        if len(fields) > 7:
            result.mode = fields[7]

        return result

    def _parse_gsv(self, fields: list[str], talker: str) -> NMEAGSV:
        """Parse GSV sentence fields."""
        result = NMEAGSV(talker_id=talker)

        if len(fields) < 4:
            return result

        result.total_messages = self._safe_int(fields[1], 1)
        result.message_number = self._safe_int(fields[2], 1)
        result.total_satellites = self._safe_int(fields[3])

        # Satellites: each uses 4 fields (PRN, elevation, azimuth, SNR)
        idx = 4
        while idx + 3 < len(fields):
            prn = self._safe_int(fields[idx])
            elev = self._safe_int(fields[idx + 1])
            azim = self._safe_int(fields[idx + 2])
            snr = self._safe_int(fields[idx + 3])
            if prn > 0:
                result.satellites.append(
                    NMEASatellite(prn=prn, elevation=elev, azimuth=azim, snr=snr)
                )
            idx += 4

        return result

    def parse(self, sentence: str) -> NMEAGGA | NMEARMC | NMEAGSA | NMEAVTG | NMEAGLL | NMEAGSV:
        """Parse an NMEA sentence.

        Args:
            sentence: Raw NMEA sentence (e.g., "$GPGGA,...*47").

        Returns:
            Typed dataclass for the sentence type.

        Raises:
            ParseError: If sentence is malformed or unsupported.
        """
        sentence = sentence.strip()
        if not sentence:
            raise ParseError("NMEA", "Empty sentence")

        if not sentence.startswith(("$", "!")):
            raise ParseError("NMEA", f"Sentence must start with $ or !, got: {sentence[:10]}", sentence)

        # Validate checksum if present
        if "*" in sentence:
            if not self.validate_checksum(sentence):
                raise ParseError(
                    "NMEA",
                    f"Checksum mismatch (expected {self._compute_checksum(sentence)})",
                    sentence,
                )

        # Strip checksum for field parsing
        body = sentence[1:]  # remove $ or !
        if "*" in body:
            body = body[: body.index("*")]

        fields = body.split(",")
        if len(fields) < 2:
            raise ParseError("NMEA", "Too few fields", sentence)

        # Extract talker ID and sentence type
        sentence_id = fields[0]
        if len(sentence_id) < 4:
            raise ParseError("NMEA", f"Invalid sentence ID: {sentence_id}", sentence)

        talker = sentence_id[:2]
        stype = sentence_id[2:]

        if stype == "GGA":
            return self._parse_gga(fields, talker)
        elif stype == "RMC":
            return self._parse_rmc(fields, talker)
        elif stype == "GSA":
            return self._parse_gsa(fields, talker)
        elif stype == "VTG":
            return self._parse_vtg(fields, talker)
        elif stype == "GLL":
            return self._parse_gll(fields, talker)
        elif stype == "GSV":
            return self._parse_gsv(fields, talker)
        else:
            raise ParseError("NMEA", f"Unsupported sentence type: {stype}", sentence)

    def parse_multi(self, data: str) -> list:
        """Parse multiple NMEA sentences from a block of text.

        Args:
            data: Multi-line text with one sentence per line.

        Returns:
            List of parsed sentence objects (skips invalid lines).
        """
        results = []
        for line in data.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith(("$", "!")):
                continue
            try:
                results.append(self.parse(line))
            except ParseError:
                continue  # skip unparseable lines
        return results
