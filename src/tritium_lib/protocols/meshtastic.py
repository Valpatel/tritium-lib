# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Meshtastic protocol message parser.

Parses Meshtastic mesh packets from raw bytes or structured dicts
(as received via serial/BLE/TCP from a Meshtastic radio).

Meshtastic packet structure (over the air):
    [4 destination][4 source][4 packet_id][1 flags][1 channel_hash][payload...]

PortNum (application layer) determines payload interpretation:
    1   — TEXT_MESSAGE_APP
    3   — POSITION_APP
    4   — NODEINFO_APP
    5   — ROUTING_APP
    6   — ADMIN_APP
    7   — TEXT_MESSAGE_COMPRESSED_APP
    33  — RANGE_TEST_APP
    67  — TELEMETRY_APP
    68  — ZPS_APP (zone position service)
    70  — SIMULATOR_APP
    71  — TRACEROUTE_APP
    72  — NEIGHBORINFO_APP
    73  — ATAK_PLUGIN
    256 — PRIVATE_APP
    257 — ATAK_FORWARDER
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.protocols.errors import ParseError

# Port number to application name mapping
PORTNUM_MAP: dict[int, str] = {
    0: "UNKNOWN_APP",
    1: "TEXT_MESSAGE_APP",
    3: "POSITION_APP",
    4: "NODEINFO_APP",
    5: "ROUTING_APP",
    6: "ADMIN_APP",
    7: "TEXT_MESSAGE_COMPRESSED_APP",
    33: "RANGE_TEST_APP",
    67: "TELEMETRY_APP",
    68: "ZPS_APP",
    70: "SIMULATOR_APP",
    71: "TRACEROUTE_APP",
    72: "NEIGHBORINFO_APP",
    73: "ATAK_PLUGIN",
    256: "PRIVATE_APP",
    257: "ATAK_FORWARDER",
}

# Hardware model IDs
HW_MODEL_MAP: dict[int, str] = {
    0: "UNSET",
    1: "TLORA_V2",
    2: "TLORA_V1",
    3: "TLORA_V2_1_1P6",
    4: "TBEAM",
    5: "HELTEC_V2_0",
    6: "TBEAM_V0P7",
    7: "T_ECHO",
    8: "TLORA_V1_1P3",
    9: "RAK4631",
    10: "HELTEC_V2_1",
    11: "HELTEC_V1",
    12: "LILYGO_TBEAM_S3_CORE",
    25: "RAK11200",
    26: "NANO_G1",
    33: "TLORA_V2_1_1P8",
    34: "TLORA_T3_S3",
    37: "NANO_G1_EXPLORER",
    39: "NANO_G2_ULTRA",
    40: "TBEAM_S3_CORE",
    41: "RAK11310",
    43: "HELTEC_V3",
    44: "HELTEC_WSL_V3",
    47: "BETAFPV_2400_TX",
    48: "BETAFPV_900_NANO_TX",
    49: "RPI_PICO",
    50: "HELTEC_WIRELESS_TRACKER",
    51: "HELTEC_WIRELESS_PAPER",
    52: "T_DECK",
    53: "T_WATCH_S3",
    54: "PICOMPUTER_S3",
    55: "HELTEC_HT62",
    56: "EBYTE_ESP32_S3",
    255: "PRIVATE_HW",
}


@dataclass
class MeshtasticPosition:
    """Parsed position from a POSITION_APP message."""

    latitude_i: int = 0  # latitude in 1e-7 degrees
    longitude_i: int = 0  # longitude in 1e-7 degrees
    altitude: int = 0  # meters above MSL
    time: int = 0  # unix timestamp
    sats_in_view: int = 0
    precision_bits: int = 0
    ground_speed: int = 0  # m/s
    ground_track: int = 0  # degrees * 1e-5

    @property
    def latitude(self) -> float:
        """Latitude in decimal degrees."""
        return self.latitude_i / 1e7

    @property
    def longitude(self) -> float:
        """Longitude in decimal degrees."""
        return self.longitude_i / 1e7

    @property
    def has_valid_position(self) -> bool:
        return self.latitude_i != 0 or self.longitude_i != 0


@dataclass
class MeshtasticNodeInfo:
    """Parsed node info from a NODEINFO_APP message."""

    node_id: str = ""  # "!aabbccdd" format
    long_name: str = ""
    short_name: str = ""
    hw_model: int = 0
    hw_model_name: str = "UNSET"
    role: int = 0
    firmware_version: str = ""


@dataclass
class MeshtasticTelemetry:
    """Parsed telemetry from a TELEMETRY_APP message."""

    time: int = 0
    # Device metrics
    battery_level: int = 0  # 0-100
    voltage: float = 0.0
    channel_utilization: float = 0.0
    air_util_tx: float = 0.0
    uptime_seconds: int = 0
    # Environment metrics
    temperature: float = 0.0
    relative_humidity: float = 0.0
    barometric_pressure: float = 0.0


@dataclass
class MeshtasticRouting:
    """Parsed routing info from a ROUTING_APP message."""

    error_reason: int = 0
    error_text: str = ""


@dataclass
class MeshtasticNeighborInfo:
    """Parsed neighbor info from a NEIGHBORINFO_APP message."""

    node_id: int = 0
    neighbors: list[dict] = field(default_factory=list)  # [{node_id, snr}]


@dataclass
class MeshtasticPacket:
    """A fully parsed Meshtastic mesh packet."""

    # Packet header
    source: int = 0  # 32-bit node number
    destination: int = 0  # 32-bit node number (0xFFFFFFFF = broadcast)
    packet_id: int = 0
    channel_index: int = 0
    want_ack: bool = False
    hop_limit: int = 3
    hop_start: int = 3

    # Application layer
    portnum: int = 0
    portnum_name: str = "UNKNOWN_APP"
    payload: bytes = b""

    # Decoded payload (populated based on portnum)
    text: str = ""  # for TEXT_MESSAGE_APP
    position: Optional[MeshtasticPosition] = None
    node_info: Optional[MeshtasticNodeInfo] = None
    telemetry: Optional[MeshtasticTelemetry] = None
    routing: Optional[MeshtasticRouting] = None
    neighbor_info: Optional[MeshtasticNeighborInfo] = None

    # Metadata
    rx_time: int = 0
    rx_snr: float = 0.0
    rx_rssi: int = 0
    encrypted: bool = False

    @property
    def source_hex(self) -> str:
        """Source node ID in !aabbccdd format."""
        return f"!{self.source:08x}"

    @property
    def destination_hex(self) -> str:
        """Destination node ID in !aabbccdd format."""
        return f"!{self.destination:08x}"

    @property
    def is_broadcast(self) -> bool:
        return self.destination == 0xFFFFFFFF


# Routing error codes
_ROUTING_ERRORS = {
    0: "NONE",
    1: "NO_ROUTE",
    2: "GOT_NAK",
    3: "TIMEOUT",
    4: "NO_INTERFACE",
    5: "MAX_RETRANSMIT",
    6: "NO_CHANNEL",
    7: "TOO_LARGE",
    8: "NO_RESPONSE",
    9: "DUTY_CYCLE_LIMIT",
    10: "BAD_REQUEST",
    11: "NOT_AUTHORIZED",
}


class MeshtasticParser:
    """Parser for Meshtastic mesh protocol messages.

    Supports parsing from:
    - Raw packet bytes (as captured over BLE/serial)
    - Structured dicts (as received from the Python Meshtastic API)

    Usage::

        parser = MeshtasticParser()

        # From dict (most common — Meshtastic Python API output)
        packet = parser.from_dict({
            "from": 0xAABBCCDD,
            "to": 0xFFFFFFFF,
            "id": 12345,
            "decoded": {
                "portnum": 1,
                "payload": b"Hello mesh!",
                "text": "Hello mesh!",
            },
        })
        print(packet.text)  # "Hello mesh!"

        # From raw bytes
        packet = parser.parse(raw_bytes)
    """

    @staticmethod
    def _to_bytes(data: bytes | str) -> bytes:
        if isinstance(data, str):
            cleaned = data.strip().replace(" ", "")
            if cleaned.startswith(("0x", "0X")):
                cleaned = cleaned[2:]
            try:
                return bytes.fromhex(cleaned)
            except ValueError as exc:
                raise ParseError("Meshtastic", f"Invalid hex string: {exc}", data) from exc
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        raise ParseError("Meshtastic", f"Expected bytes or hex, got {type(data).__name__}", data)

    @staticmethod
    def _decode_position_payload(data: bytes) -> MeshtasticPosition:
        """Decode a protobuf-lite position payload.

        Meshtastic uses protobuf, but we do a simplified parse of the
        most common fields using varint + field tag decoding.
        """
        pos = MeshtasticPosition()
        offset = 0

        while offset < len(data):
            if offset >= len(data):
                break

            # Read varint tag
            tag_byte = data[offset]
            field_number = tag_byte >> 3
            wire_type = tag_byte & 0x07
            offset += 1

            if wire_type == 0:  # varint
                val = 0
                shift = 0
                while offset < len(data):
                    b = data[offset]
                    offset += 1
                    val |= (b & 0x7F) << shift
                    if not (b & 0x80):
                        break
                    shift += 7

                # Zigzag decode for signed fields
                if field_number in (1, 2):  # lat_i, lng_i are sint32
                    val = (val >> 1) ^ -(val & 1)

                if field_number == 1:
                    pos.latitude_i = val
                elif field_number == 2:
                    pos.longitude_i = val
                elif field_number == 3:
                    pos.altitude = val
                elif field_number == 4:
                    pos.time = val
                elif field_number == 9:
                    pos.sats_in_view = val
                elif field_number == 12:
                    pos.ground_speed = val
                elif field_number == 13:
                    pos.ground_track = val

            elif wire_type == 5:  # 32-bit fixed
                if offset + 4 <= len(data):
                    offset += 4
                else:
                    break
            elif wire_type == 1:  # 64-bit fixed
                if offset + 8 <= len(data):
                    offset += 8
                else:
                    break
            elif wire_type == 2:  # length-delimited
                length = 0
                shift = 0
                while offset < len(data):
                    b = data[offset]
                    offset += 1
                    length |= (b & 0x7F) << shift
                    if not (b & 0x80):
                        break
                    shift += 7
                offset += length
            else:
                break  # unknown wire type

        return pos

    @staticmethod
    def _decode_nodeinfo_payload(data: bytes) -> MeshtasticNodeInfo:
        """Simplified protobuf decode for NodeInfo."""
        info = MeshtasticNodeInfo()
        offset = 0

        while offset < len(data):
            tag_byte = data[offset]
            field_number = tag_byte >> 3
            wire_type = tag_byte & 0x07
            offset += 1

            if wire_type == 0:  # varint
                val = 0
                shift = 0
                while offset < len(data):
                    b = data[offset]
                    offset += 1
                    val |= (b & 0x7F) << shift
                    if not (b & 0x80):
                        break
                    shift += 7
                if field_number == 2:  # hw_model
                    info.hw_model = val
                    info.hw_model_name = HW_MODEL_MAP.get(val, f"UNKNOWN({val})")
                elif field_number == 6:  # role
                    info.role = val

            elif wire_type == 2:  # length-delimited
                length = 0
                shift = 0
                while offset < len(data):
                    b = data[offset]
                    offset += 1
                    length |= (b & 0x7F) << shift
                    if not (b & 0x80):
                        break
                    shift += 7
                sub_data = data[offset : offset + length]
                offset += length

                if field_number == 1:  # id (string)
                    try:
                        info.node_id = sub_data.decode("utf-8")
                    except UnicodeDecodeError:
                        info.node_id = sub_data.hex()
                elif field_number == 3:  # long_name
                    try:
                        info.long_name = sub_data.decode("utf-8")
                    except UnicodeDecodeError:
                        info.long_name = sub_data.hex()
                elif field_number == 4:  # short_name
                    try:
                        info.short_name = sub_data.decode("utf-8")
                    except UnicodeDecodeError:
                        info.short_name = sub_data.hex()
                elif field_number == 7:  # firmware_version
                    try:
                        info.firmware_version = sub_data.decode("utf-8")
                    except UnicodeDecodeError:
                        info.firmware_version = ""

            elif wire_type == 5:
                offset += 4
            elif wire_type == 1:
                offset += 8
            else:
                break

        return info

    @staticmethod
    def _decode_telemetry_payload(data: bytes) -> MeshtasticTelemetry:
        """Simplified protobuf decode for Telemetry."""
        tel = MeshtasticTelemetry()
        offset = 0

        while offset < len(data):
            if offset >= len(data):
                break
            tag_byte = data[offset]
            field_number = tag_byte >> 3
            wire_type = tag_byte & 0x07
            offset += 1

            if wire_type == 0:  # varint
                val = 0
                shift = 0
                while offset < len(data):
                    b = data[offset]
                    offset += 1
                    val |= (b & 0x7F) << shift
                    if not (b & 0x80):
                        break
                    shift += 7
                if field_number == 1:
                    tel.time = val
                elif field_number == 2:
                    tel.battery_level = val
                elif field_number == 6:
                    tel.uptime_seconds = val

            elif wire_type == 5:  # 32-bit fixed (float)
                if offset + 4 <= len(data):
                    val = struct.unpack("<f", data[offset : offset + 4])[0]
                    offset += 4
                    if field_number == 3:
                        tel.voltage = val
                    elif field_number == 4:
                        tel.channel_utilization = val
                    elif field_number == 5:
                        tel.air_util_tx = val
                    elif field_number == 7:
                        tel.temperature = val
                    elif field_number == 8:
                        tel.relative_humidity = val
                    elif field_number == 9:
                        tel.barometric_pressure = val
                else:
                    break

            elif wire_type == 2:  # length-delimited (sub-message)
                length = 0
                shift = 0
                while offset < len(data):
                    b = data[offset]
                    offset += 1
                    length |= (b & 0x7F) << shift
                    if not (b & 0x80):
                        break
                    shift += 7
                # Recursively parse sub-message for device/environment metrics
                if field_number == 2 and length > 0:
                    sub_data = data[offset : offset + length]
                    sub_tel = MeshtasticParser._decode_telemetry_payload(sub_data)
                    # Merge sub fields
                    if sub_tel.battery_level:
                        tel.battery_level = sub_tel.battery_level
                    if sub_tel.voltage:
                        tel.voltage = sub_tel.voltage
                    if sub_tel.channel_utilization:
                        tel.channel_utilization = sub_tel.channel_utilization
                    if sub_tel.air_util_tx:
                        tel.air_util_tx = sub_tel.air_util_tx
                    if sub_tel.uptime_seconds:
                        tel.uptime_seconds = sub_tel.uptime_seconds
                elif field_number == 3 and length > 0:
                    sub_data = data[offset : offset + length]
                    sub_tel = MeshtasticParser._decode_telemetry_payload(sub_data)
                    if sub_tel.temperature:
                        tel.temperature = sub_tel.temperature
                    if sub_tel.relative_humidity:
                        tel.relative_humidity = sub_tel.relative_humidity
                    if sub_tel.barometric_pressure:
                        tel.barometric_pressure = sub_tel.barometric_pressure
                offset += length

            elif wire_type == 1:
                offset += 8
            else:
                break

        return tel

    def parse(self, data: bytes | str) -> MeshtasticPacket:
        """Parse a raw Meshtastic packet from bytes.

        The minimal packet header is:
            [4 dest][4 src][4 id][1 flags][1 channel][payload...]

        Args:
            data: Raw packet bytes or hex string.

        Returns:
            MeshtasticPacket with parsed header and decoded payload.

        Raises:
            ParseError: If packet is too short or malformed.
        """
        raw = self._to_bytes(data)

        if len(raw) < 14:
            raise ParseError(
                "Meshtastic",
                f"Packet too short ({len(raw)} bytes, need >= 14)",
                data,
            )

        pkt = MeshtasticPacket()

        # Header: destination(4) + source(4) + id(4) + flags(1) + channel(1)
        pkt.destination = struct.unpack("<I", raw[0:4])[0]
        pkt.source = struct.unpack("<I", raw[4:8])[0]
        pkt.packet_id = struct.unpack("<I", raw[8:12])[0]

        flags = raw[12]
        pkt.hop_limit = flags & 0x07
        pkt.want_ack = bool(flags & 0x08)

        pkt.channel_index = raw[13]
        pkt.payload = raw[14:]

        # Try to determine portnum from first byte of payload if present
        if len(pkt.payload) > 0:
            # In encrypted packets the payload is opaque
            # For unencrypted, first varint is usually the portnum tag
            first_byte = pkt.payload[0]
            if first_byte >> 3 == 1 and (first_byte & 0x07) == 0:
                # Field 1, varint — likely portnum
                if len(pkt.payload) > 1:
                    pkt.portnum = pkt.payload[1]
                    pkt.portnum_name = PORTNUM_MAP.get(pkt.portnum, f"UNKNOWN({pkt.portnum})")

        return pkt

    def from_dict(self, data: dict) -> MeshtasticPacket:
        """Parse a Meshtastic packet from a Python dict.

        This is the most common input — the Meshtastic Python API
        returns packets as dicts with 'from', 'to', 'decoded', etc.

        Args:
            data: Dict from Meshtastic Python API.

        Returns:
            MeshtasticPacket with decoded fields.

        Raises:
            ParseError: If dict is missing required fields.
        """
        if not isinstance(data, dict):
            raise ParseError("Meshtastic", f"Expected dict, got {type(data).__name__}", data)

        pkt = MeshtasticPacket()

        pkt.source = data.get("from", data.get("fromId", 0))
        if isinstance(pkt.source, str):
            # Handle "!aabbccdd" format
            try:
                pkt.source = int(pkt.source.lstrip("!"), 16)
            except ValueError:
                pkt.source = 0

        pkt.destination = data.get("to", data.get("toId", 0xFFFFFFFF))
        if isinstance(pkt.destination, str):
            try:
                pkt.destination = int(pkt.destination.lstrip("!"), 16)
            except ValueError:
                pkt.destination = 0xFFFFFFFF

        pkt.packet_id = data.get("id", 0)
        pkt.hop_limit = data.get("hopLimit", 3)
        pkt.hop_start = data.get("hopStart", 3)
        pkt.want_ack = data.get("wantAck", False)
        pkt.channel_index = data.get("channel", 0)
        pkt.rx_time = data.get("rxTime", 0)
        pkt.rx_snr = data.get("rxSnr", 0.0)
        pkt.rx_rssi = data.get("rxRssi", 0)
        pkt.encrypted = data.get("encrypted", False)

        decoded = data.get("decoded", {})
        if decoded:
            pkt.portnum = decoded.get("portnum", 0)
            if isinstance(pkt.portnum, str):
                # Handle string portnum names
                for k, v in PORTNUM_MAP.items():
                    if v == pkt.portnum:
                        pkt.portnum = k
                        break
                else:
                    try:
                        pkt.portnum = int(pkt.portnum)
                    except ValueError:
                        pkt.portnum = 0

            pkt.portnum_name = PORTNUM_MAP.get(pkt.portnum, f"UNKNOWN({pkt.portnum})")

            raw_payload = decoded.get("payload", b"")
            if isinstance(raw_payload, str):
                pkt.payload = raw_payload.encode("utf-8")
            elif isinstance(raw_payload, bytes):
                pkt.payload = raw_payload
            else:
                pkt.payload = b""

            # TEXT_MESSAGE_APP
            if pkt.portnum == 1:
                pkt.text = decoded.get("text", "")
                if not pkt.text and pkt.payload:
                    try:
                        pkt.text = pkt.payload.decode("utf-8")
                    except UnicodeDecodeError:
                        pkt.text = ""

            # POSITION_APP
            elif pkt.portnum == 3:
                pos_dict = decoded.get("position", {})
                if pos_dict:
                    pkt.position = MeshtasticPosition(
                        latitude_i=int(pos_dict.get("latitudeI", 0)),
                        longitude_i=int(pos_dict.get("longitudeI", 0)),
                        altitude=int(pos_dict.get("altitude", 0)),
                        time=int(pos_dict.get("time", 0)),
                        sats_in_view=int(pos_dict.get("satsInView", 0)),
                        ground_speed=int(pos_dict.get("groundSpeed", 0)),
                        ground_track=int(pos_dict.get("groundTrack", 0)),
                    )
                elif pkt.payload:
                    pkt.position = self._decode_position_payload(pkt.payload)

            # NODEINFO_APP
            elif pkt.portnum == 4:
                user_dict = decoded.get("user", {})
                if user_dict:
                    hw = user_dict.get("hwModel", 0)
                    if isinstance(hw, str):
                        # Reverse lookup
                        for k, v in HW_MODEL_MAP.items():
                            if v == hw:
                                hw = k
                                break
                        else:
                            hw = 0
                    pkt.node_info = MeshtasticNodeInfo(
                        node_id=user_dict.get("id", ""),
                        long_name=user_dict.get("longName", ""),
                        short_name=user_dict.get("shortName", ""),
                        hw_model=hw,
                        hw_model_name=HW_MODEL_MAP.get(hw, f"UNKNOWN({hw})"),
                        role=user_dict.get("role", 0),
                    )
                elif pkt.payload:
                    pkt.node_info = self._decode_nodeinfo_payload(pkt.payload)

            # TELEMETRY_APP
            elif pkt.portnum == 67:
                tel_dict = decoded.get("telemetry", decoded.get("deviceMetrics", {}))
                if tel_dict:
                    dev = tel_dict.get("deviceMetrics", tel_dict)
                    env = tel_dict.get("environmentMetrics", {})
                    pkt.telemetry = MeshtasticTelemetry(
                        time=int(tel_dict.get("time", 0)),
                        battery_level=int(dev.get("batteryLevel", 0)),
                        voltage=float(dev.get("voltage", 0.0)),
                        channel_utilization=float(dev.get("channelUtilization", 0.0)),
                        air_util_tx=float(dev.get("airUtilTx", 0.0)),
                        uptime_seconds=int(dev.get("uptimeSeconds", 0)),
                        temperature=float(env.get("temperature", 0.0)),
                        relative_humidity=float(env.get("relativeHumidity", 0.0)),
                        barometric_pressure=float(env.get("barometricPressure", 0.0)),
                    )
                elif pkt.payload:
                    pkt.telemetry = self._decode_telemetry_payload(pkt.payload)

            # ROUTING_APP
            elif pkt.portnum == 5:
                routing_dict = decoded.get("routing", {})
                err = routing_dict.get("errorReason", 0)
                if isinstance(err, str):
                    # Reverse lookup
                    for k, v in _ROUTING_ERRORS.items():
                        if v == err:
                            err = k
                            break
                    else:
                        err = 0
                pkt.routing = MeshtasticRouting(
                    error_reason=err,
                    error_text=_ROUTING_ERRORS.get(err, f"UNKNOWN({err})"),
                )

            # NEIGHBORINFO_APP
            elif pkt.portnum == 72:
                ni_dict = decoded.get("neighborinfo", decoded.get("neighbors", {}))
                if isinstance(ni_dict, dict):
                    neighbors = []
                    for n in ni_dict.get("neighbors", []):
                        neighbors.append({
                            "node_id": n.get("nodeId", 0),
                            "snr": n.get("snr", 0.0),
                        })
                    pkt.neighbor_info = MeshtasticNeighborInfo(
                        node_id=ni_dict.get("nodeId", pkt.source),
                        neighbors=neighbors,
                    )

        return pkt
