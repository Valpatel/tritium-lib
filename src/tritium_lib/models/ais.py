# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""AIS and ADS-B models for maritime and aviation tracking.

AIS (Automatic Identification System) — ship/vessel tracking via VHF
ADS-B (Automatic Dependent Surveillance-Broadcast) — aircraft tracking

These models represent entities received from AIS/ADS-B receivers
(e.g., RTL-SDR dongle + dump1090/rtl_ais) and feed into TargetTracker
as maritime or aviation targets in the unified operating picture.

MQTT topics:
    tritium/{site}/ais/{receiver}/vessel   — vessel position reports
    tritium/{site}/adsb/{receiver}/flight  — aircraft position reports
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# -- AIS (Maritime) --------------------------------------------------------


class VesselType(str, Enum):
    """AIS vessel type categories."""

    CARGO = "cargo"
    TANKER = "tanker"
    PASSENGER = "passenger"
    FISHING = "fishing"
    MILITARY = "military"
    SAILING = "sailing"
    PLEASURE = "pleasure"
    TUG = "tug"
    PILOT = "pilot"
    SAR = "search_and_rescue"
    LAW_ENFORCEMENT = "law_enforcement"
    OTHER = "other"
    UNKNOWN = "unknown"


class NavigationStatus(str, Enum):
    """AIS navigation status."""

    UNDERWAY_ENGINE = "underway_engine"
    AT_ANCHOR = "at_anchor"
    NOT_UNDER_COMMAND = "not_under_command"
    RESTRICTED_MANEUVERABILITY = "restricted_maneuverability"
    CONSTRAINED_BY_DRAUGHT = "constrained_by_draught"
    MOORED = "moored"
    AGROUND = "aground"
    FISHING = "fishing"
    UNDERWAY_SAILING = "underway_sailing"
    UNKNOWN = "unknown"


class AISPosition(BaseModel):
    """Position report from an AIS message."""

    latitude: float = 0.0
    longitude: float = 0.0
    heading: float = 0.0  # degrees true north (0-360)
    course_over_ground: float = 0.0  # degrees
    speed_over_ground: float = 0.0  # knots
    rate_of_turn: float = 0.0  # degrees/min (positive = right)
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


class AISVessel(BaseModel):
    """An AIS-tracked vessel.

    Fields map to ITU-R M.1371 standard AIS message types 1-5.
    MMSI is the unique Maritime Mobile Service Identity.
    """

    mmsi: int  # 9-digit Maritime Mobile Service Identity
    name: str = ""
    call_sign: str = ""
    imo_number: int = 0  # IMO ship identification number
    vessel_type: VesselType = VesselType.UNKNOWN
    vessel_type_code: int = 0  # raw AIS type code (0-99)

    # Dimensions (meters)
    length: float = 0.0
    beam: float = 0.0  # width
    draught: float = 0.0

    # Current state
    position: AISPosition = Field(default_factory=AISPosition)
    navigation_status: NavigationStatus = NavigationStatus.UNKNOWN
    destination: str = ""
    eta: Optional[str] = None  # ISO format or MM-DD HH:MM

    # Receiver metadata
    receiver_id: str = ""
    signal_strength: float = 0.0  # dBm
    last_seen: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )

    # Tritium target mapping
    target_id: str = ""  # e.g., "ais_{mmsi}"

    def compute_target_id(self) -> str:
        """Generate the Tritium target ID for this vessel."""
        return f"ais_{self.mmsi}"

    def to_target_dict(self) -> dict:
        """Convert to a dict suitable for TargetTracker ingestion."""
        return {
            "target_id": self.compute_target_id(),
            "name": self.name or f"MMSI:{self.mmsi}",
            "source": "ais",
            "asset_type": self.vessel_type.value,
            "alliance": "unknown",
            "position": {
                "lat": self.position.latitude,
                "lng": self.position.longitude,
            },
            "heading": self.position.heading,
            "speed": self.position.speed_over_ground,
            "classification": self.vessel_type.value,
            "metadata": {
                "mmsi": self.mmsi,
                "call_sign": self.call_sign,
                "imo": self.imo_number,
                "destination": self.destination,
                "nav_status": self.navigation_status.value,
                "length": self.length,
                "beam": self.beam,
            },
        }


# -- ADS-B (Aviation) -----------------------------------------------------


class FlightCategory(str, Enum):
    """Aircraft category from ADS-B message."""

    NO_INFO = "no_info"
    LIGHT = "light"  # < 15500 lbs
    SMALL = "small"  # 15500-75000 lbs
    LARGE = "large"  # 75000-300000 lbs
    HIGH_VORTEX_LARGE = "high_vortex_large"
    HEAVY = "heavy"  # > 300000 lbs
    HIGH_PERFORMANCE = "high_performance"  # > 5G, > 400 kts
    ROTORCRAFT = "rotorcraft"
    GLIDER = "glider"
    LIGHTER_THAN_AIR = "lighter_than_air"
    PARACHUTIST = "parachutist"
    ULTRALIGHT = "ultralight"
    UAV = "uav"  # unmanned aerial vehicle
    SPACE_VEHICLE = "space_vehicle"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


class ADSBPosition(BaseModel):
    """Position report from an ADS-B message."""

    latitude: float = 0.0
    longitude: float = 0.0
    altitude_ft: float = 0.0  # feet MSL
    altitude_m: float = 0.0  # meters (computed)
    heading: float = 0.0  # degrees true north
    ground_speed: float = 0.0  # knots
    vertical_rate: float = 0.0  # feet/min (positive = climbing)
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    def compute_altitude_m(self) -> float:
        """Convert altitude from feet to meters."""
        return self.altitude_ft * 0.3048


class SquawkCode(BaseModel):
    """Transponder squawk code with meaning."""

    code: str = "0000"  # 4-digit octal
    emergency: bool = False
    ident: bool = False  # IDENT button pressed

    @property
    def is_hijack(self) -> bool:
        return self.code == "7500"

    @property
    def is_radio_failure(self) -> bool:
        return self.code == "7600"

    @property
    def is_emergency(self) -> bool:
        return self.code == "7700" or self.emergency


class ADSBFlight(BaseModel):
    """An ADS-B-tracked aircraft/flight.

    ICAO 24-bit address is the unique identifier.
    """

    icao_hex: str  # 6-character hex ICAO 24-bit address
    callsign: str = ""  # flight number (e.g., "UAL123")
    registration: str = ""  # aircraft registration (e.g., "N12345")
    category: FlightCategory = FlightCategory.UNKNOWN

    # Aircraft info
    aircraft_type: str = ""  # ICAO type designator (e.g., "B738")
    operator: str = ""  # airline or operator name

    # Current state
    position: ADSBPosition = Field(default_factory=ADSBPosition)
    squawk: SquawkCode = Field(default_factory=SquawkCode)
    on_ground: bool = False

    # Signal metadata
    receiver_id: str = ""
    signal_strength: float = 0.0  # dBm
    messages_received: int = 0
    last_seen: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )

    # Tritium target mapping
    target_id: str = ""  # e.g., "adsb_{icao_hex}"

    def compute_target_id(self) -> str:
        """Generate the Tritium target ID for this flight."""
        return f"adsb_{self.icao_hex.lower()}"

    def is_emergency(self) -> bool:
        """Check if the flight is squawking emergency."""
        return self.squawk.is_emergency or self.squawk.is_hijack

    def to_target_dict(self) -> dict:
        """Convert to a dict suitable for TargetTracker ingestion."""
        alliance = "unknown"
        if self.squawk.is_hijack:
            alliance = "hostile"
        elif self.squawk.is_emergency:
            alliance = "neutral"

        return {
            "target_id": self.compute_target_id(),
            "name": self.callsign or f"ICAO:{self.icao_hex}",
            "source": "adsb",
            "asset_type": self.category.value,
            "alliance": alliance,
            "position": {
                "lat": self.position.latitude,
                "lng": self.position.longitude,
                "alt_m": self.position.compute_altitude_m(),
            },
            "heading": self.position.heading,
            "speed": self.position.ground_speed,
            "classification": self.category.value,
            "metadata": {
                "icao": self.icao_hex,
                "callsign": self.callsign,
                "registration": self.registration,
                "aircraft_type": self.aircraft_type,
                "operator": self.operator,
                "squawk": self.squawk.code,
                "on_ground": self.on_ground,
                "altitude_ft": self.position.altitude_ft,
                "vertical_rate": self.position.vertical_rate,
            },
        }
