# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Cursor-on-Target (CoT) data models — MIL-STD-2045 compatible.

Structured Pydantic models for CoT events, with XML generation and parsing.
These complement the existing tritium_lib.cot codec (which converts Tritium
devices to CoT) by providing general-purpose CoT event handling.

CoT type string reference:
  a-f-G-U-C      friendly ground unit command
  a-f-G-U-C-I    friendly ground unit command infantry
  a-f-A-M-F-Q    friendly air military fixed-wing UAV
  a-h-G-U-C      hostile ground unit command
  a-n-G           neutral ground
  a-u-G           unknown ground
  b-m-p-s-m       bits - map point - spot - marker
"""

from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _default_stale() -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=300)


class CotPoint(BaseModel):
    """CoT point — WGS84 position with circular/linear error."""
    lat: float = 0.0
    lon: float = 0.0
    hae: float = 0.0   # height above ellipsoid (meters)
    ce: float = 9999999.0   # circular error (meters)
    le: float = 9999999.0   # linear error (meters)


class CotContact(BaseModel):
    """CoT contact detail — callsign and endpoint."""
    callsign: str = ""
    endpoint: str = ""  # e.g. "*:-1:stcp"


class CotDetail(BaseModel):
    """CoT detail element — extensible key-value metadata.

    The detail element is the main extension point for CoT events.
    Standard sub-elements (contact, __group, remarks) are modeled
    explicitly; additional fields go in 'extra'.
    """
    contact: Optional[CotContact] = None
    group_name: str = ""   # __group name (team color)
    group_role: str = ""   # __group role
    remarks: str = ""
    extra: dict[str, dict[str, str]] = Field(default_factory=dict)


class CotEvent(BaseModel):
    """A complete CoT event — the fundamental unit of SA data.

    Follows MIL-STD-2045 / CoT specification version 2.0.
    """
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "a-f-G-U-C"  # CoT type string
    how: str = "m-g"  # how the event was generated (m-g = machine GPS)
    time: datetime = Field(default_factory=_utcnow)
    start: datetime = Field(default_factory=_utcnow)
    stale: datetime = Field(default_factory=_default_stale)
    point: CotPoint = Field(default_factory=CotPoint)
    detail: CotDetail = Field(default_factory=CotDetail)
    version: str = "2.0"

    @property
    def is_stale(self) -> bool:
        return _utcnow() > self.stale

    @property
    def alliance(self) -> str:
        """Extract alliance from type string (f=friendly, h=hostile, etc.)."""
        parts = self.type.split("-")
        if len(parts) >= 2:
            mapping = {"f": "friendly", "h": "hostile", "n": "neutral", "u": "unknown"}
            return mapping.get(parts[1], "unknown")
        return "unknown"


# ---------------------------------------------------------------------------
# Common CoT type constants
# ---------------------------------------------------------------------------

COT_FRIENDLY_GROUND_UNIT = "a-f-G-U-C"
COT_FRIENDLY_GROUND_INFANTRY = "a-f-G-U-C-I"
COT_FRIENDLY_UAV = "a-f-A-M-F-Q"
COT_FRIENDLY_GROUND_SENSOR = "a-f-G-E-S"
COT_FRIENDLY_GROUND_SENSOR_CAMERA = "a-f-G-E-S-C"
COT_HOSTILE_GROUND_UNIT = "a-h-G-U-C"
COT_NEUTRAL_GROUND = "a-n-G"
COT_UNKNOWN_GROUND = "a-u-G"
COT_MAP_MARKER = "b-m-p-s-m"


# ---------------------------------------------------------------------------
# XML generation / parsing
# ---------------------------------------------------------------------------

_DT_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def cot_to_xml(event: CotEvent) -> str:
    """Serialize a CotEvent to MIL-STD-2045 CoT XML string."""
    root = ET.Element("event", {
        "version": event.version,
        "uid": event.uid,
        "type": event.type,
        "how": event.how,
        "time": event.time.strftime(_DT_FMT),
        "start": event.start.strftime(_DT_FMT),
        "stale": event.stale.strftime(_DT_FMT),
    })

    pt = event.point
    ET.SubElement(root, "point", {
        "lat": f"{pt.lat:.7f}",
        "lon": f"{pt.lon:.7f}",
        "hae": f"{pt.hae:.1f}",
        "ce": f"{pt.ce:.1f}",
        "le": f"{pt.le:.1f}",
    })

    detail_el = ET.SubElement(root, "detail")

    d = event.detail
    if d.contact:
        attrs = {"callsign": d.contact.callsign}
        if d.contact.endpoint:
            attrs["endpoint"] = d.contact.endpoint
        ET.SubElement(detail_el, "contact", attrs)

    if d.group_name:
        attrs = {"name": d.group_name}
        if d.group_role:
            attrs["role"] = d.group_role
        ET.SubElement(detail_el, "__group", attrs)

    if d.remarks:
        ET.SubElement(detail_el, "remarks").text = d.remarks

    for tag, attrs in d.extra.items():
        ET.SubElement(detail_el, tag, attrs)

    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def xml_to_cot(xml_str: str) -> Optional[CotEvent]:
    """Parse a CoT XML string into a CotEvent.

    Returns None if the XML is not a valid CoT event.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None

    if root.tag != "event":
        return None

    def _parse_dt(s: str) -> datetime:
        try:
            return datetime.strptime(s, _DT_FMT).replace(tzinfo=timezone.utc)
        except ValueError:
            # Try without microseconds
            try:
                return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                return _utcnow()

    # Parse point
    point = CotPoint()
    pt_el = root.find("point")
    if pt_el is not None:
        point = CotPoint(
            lat=float(pt_el.get("lat", 0)),
            lon=float(pt_el.get("lon", 0)),
            hae=float(pt_el.get("hae", 0)),
            ce=float(pt_el.get("ce", 9999999)),
            le=float(pt_el.get("le", 9999999)),
        )

    # Parse detail
    detail = CotDetail()
    detail_el = root.find("detail")
    if detail_el is not None:
        contact_el = detail_el.find("contact")
        if contact_el is not None:
            detail.contact = CotContact(
                callsign=contact_el.get("callsign", ""),
                endpoint=contact_el.get("endpoint", ""),
            )

        group_el = detail_el.find("__group")
        if group_el is not None:
            detail.group_name = group_el.get("name", "")
            detail.group_role = group_el.get("role", "")

        remarks_el = detail_el.find("remarks")
        if remarks_el is not None and remarks_el.text:
            detail.remarks = remarks_el.text

        # Collect extra elements
        known_tags = {"contact", "__group", "remarks"}
        for child in detail_el:
            if child.tag not in known_tags:
                detail.extra[child.tag] = dict(child.attrib)

    return CotEvent(
        uid=root.get("uid", ""),
        type=root.get("type", ""),
        how=root.get("how", "m-g"),
        time=_parse_dt(root.get("time", "")),
        start=_parse_dt(root.get("start", "")),
        stale=_parse_dt(root.get("stale", "")),
        version=root.get("version", "2.0"),
        point=point,
        detail=detail,
    )
