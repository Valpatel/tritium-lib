# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""CoT XML codec — converts Tritium edge devices to/from CoT XML.

Follows the same conventions as tritium-sc's engine/comms/cot.py:
  - MIL-STD-2045 CoT event format
  - Alliance-based type codes (a-f-G for friendly, a-h-G for hostile)
  - Team colors in __group detail
  - Stale time for position reports

Edge devices are represented as friendly ground sensors (a-f-G-E-S)
by default, with type refinement based on capabilities.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional


# CoT type codes for edge device roles
_EDGE_COT_TYPES: dict[str, str] = {
    "sensor":   "a-f-G-E-S",    # friendly ground sensor
    "camera":   "a-f-G-E-S-C",  # friendly ground sensor camera
    "gateway":  "a-f-G-E-S",    # friendly ground sensor
    "relay":    "a-f-G-E-S",    # friendly ground sensor (mesh relay)
    "display":  "a-f-G-U-C",    # friendly ground unit command
    "default":  "a-f-G-E-S",    # friendly ground sensor
}

_TEAM_COLORS: dict[str, str] = {
    "friendly": "Cyan",
    "hostile": "Red",
    "neutral": "White",
    "unknown": "Yellow",
}


def _infer_role(capabilities: list[str]) -> str:
    """Infer device role from capabilities for CoT type mapping."""
    if "camera" in capabilities:
        return "camera"
    if "lora" in capabilities or "mesh" in capabilities:
        return "relay"
    if "display" in capabilities:
        return "display"
    return "sensor"


def device_to_cot(
    device_id: str,
    lat: float,
    lng: float,
    alt: float = 0.0,
    capabilities: list[str] = None,
    alliance: str = "friendly",
    callsign: str = "",
    stale_seconds: int = 300,
    extra: dict = None,
) -> str:
    """Convert edge device position to CoT SA (situational awareness) XML.

    Args:
        device_id: Unique device ID (becomes CoT uid).
        lat, lng, alt: WGS84 position.
        capabilities: Device capabilities list (for type inference).
        alliance: friendly/hostile/neutral/unknown.
        callsign: Display name in TAK (defaults to device_id).
        stale_seconds: Seconds until this report goes stale.
        extra: Additional detail fields to embed.

    Returns:
        CoT XML string.
    """
    now = datetime.now(timezone.utc)
    stale = now + timedelta(seconds=stale_seconds)

    role = _infer_role(capabilities or [])
    cot_type = _EDGE_COT_TYPES.get(role, _EDGE_COT_TYPES["default"])

    # Swap alliance prefix if not friendly
    if alliance != "friendly" and cot_type.startswith("a-f-"):
        prefix_map = {"hostile": "a-h-", "neutral": "a-n-", "unknown": "a-u-"}
        cot_type = prefix_map.get(alliance, "a-u-") + cot_type[4:]

    event = ET.Element("event", {
        "version": "2.0",
        "uid": f"tritium-edge-{device_id}",
        "type": cot_type,
        "how": "m-g",  # machine-GPS
        "time": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "start": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "stale": stale.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    })

    ET.SubElement(event, "point", {
        "lat": f"{lat:.7f}",
        "lon": f"{lng:.7f}",
        "hae": f"{alt:.1f}",
        "ce": "10.0",
        "le": "10.0",
    })

    detail = ET.SubElement(event, "detail")

    # Contact (callsign)
    ET.SubElement(detail, "contact", {
        "callsign": callsign or device_id,
    })

    # Team color
    team_color = _TEAM_COLORS.get(alliance, "Yellow")
    ET.SubElement(detail, "__group", {
        "name": team_color,
        "role": "Sensor",
    })

    # Edge device metadata
    edge_detail = ET.SubElement(detail, "tritium_edge", {
        "device_id": device_id,
        "role": role,
    })
    if capabilities:
        edge_detail.set("capabilities", ",".join(capabilities))

    # Extra fields
    if extra:
        for k, v in extra.items():
            edge_detail.set(str(k), str(v))

    return ET.tostring(event, encoding="unicode", xml_declaration=True)


def sensor_to_cot(
    device_id: str,
    sensor_type: str,
    value: float,
    lat: float,
    lng: float,
    alt: float = 0.0,
    unit: str = "",
    stale_seconds: int = 120,
) -> str:
    """Convert a sensor reading to CoT XML with sensor detail.

    Creates a sensor point that TAK clients can display with
    the reading value as a remark.
    """
    now = datetime.now(timezone.utc)
    stale = now + timedelta(seconds=stale_seconds)

    event = ET.Element("event", {
        "version": "2.0",
        "uid": f"tritium-sensor-{device_id}-{sensor_type}",
        "type": "a-f-G-E-S",
        "how": "m-r",  # machine-reported
        "time": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "start": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "stale": stale.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    })

    ET.SubElement(event, "point", {
        "lat": f"{lat:.7f}",
        "lon": f"{lng:.7f}",
        "hae": f"{alt:.1f}",
        "ce": "10.0",
        "le": "10.0",
    })

    detail = ET.SubElement(event, "detail")
    ET.SubElement(detail, "contact", {
        "callsign": f"{device_id}/{sensor_type}",
    })
    ET.SubElement(detail, "remarks").text = (
        f"{sensor_type}: {value}{' ' + unit if unit else ''}"
    )

    return ET.tostring(event, encoding="unicode", xml_declaration=True)


def parse_cot(xml_str: str) -> Optional[dict]:
    """Parse a CoT XML event into a dict.

    Returns None if the XML is not valid CoT.
    """
    try:
        root = ET.fromstring(xml_str)
        if root.tag != "event":
            return None

        result = {
            "uid": root.get("uid", ""),
            "type": root.get("type", ""),
            "how": root.get("how", ""),
            "time": root.get("time", ""),
            "stale": root.get("stale", ""),
        }

        point = root.find("point")
        if point is not None:
            result["lat"] = float(point.get("lat", 0))
            result["lng"] = float(point.get("lon", 0))
            result["alt"] = float(point.get("hae", 0))

        detail = root.find("detail")
        if detail is not None:
            contact = detail.find("contact")
            if contact is not None:
                result["callsign"] = contact.get("callsign", "")

            edge = detail.find("tritium_edge")
            if edge is not None:
                result["device_id"] = edge.get("device_id", "")
                result["role"] = edge.get("role", "")
                caps = edge.get("capabilities", "")
                result["capabilities"] = caps.split(",") if caps else []

        return result
    except ET.ParseError:
        return None
