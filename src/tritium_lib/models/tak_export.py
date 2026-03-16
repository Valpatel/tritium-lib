# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""CoT export models — generate Cursor on Target XML from target data.

Provides CoTExportEvent and CoTExportPoint for converting Tritium target
dicts into MIL-STD-2045 CoT XML suitable for ATAK/WinTAK consumption.

Follows the same patterns as tritium_lib.models.cot (CotEvent/CotPoint)
and tritium_lib.cot.codec, but focused on the export use case: converting
a batch of TrackedTargets into a CoT XML document or stream.
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


_DT_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# Alliance to CoT affiliation character
_ALLIANCE_CHAR: dict[str, str] = {
    "friendly": "f",
    "hostile": "h",
    "neutral": "n",
    "unknown": "u",
}

# Alliance to team color for __group detail
_TEAM_COLORS: dict[str, str] = {
    "friendly": "Cyan",
    "hostile": "Red",
    "neutral": "White",
    "unknown": "Yellow",
}

# Asset type to CoT type suffix (after a-{affil}-)
_ASSET_TYPE_SUFFIX: dict[str, str] = {
    "person": "G-U-C",
    "vehicle": "G-E-V",
    "drone": "A-M-F-Q",
    "rover": "G-E-V",
    "turret": "G-E-W",
    "sensor": "G-E-S",
    "camera": "G-E-S-C",
    "mesh_radio": "G-E-S",
    "ble_device": "G-E-S",
    "phone": "G-U-C",
    "watch": "G-U-C",
    "computer": "G-E-C-I",
    "animal": "G-U",
    "unknown": "G",
}

# How codes by source
_HOW_MAP: dict[str, str] = {
    "simulation": "m-s",
    "yolo": "m-r",
    "mqtt": "m-g",
    "ble": "m-r",
    "wifi": "m-r",
    "tak": "m-g",
    "manual": "h-e",
    "mesh": "m-g",
}


class CoTExportPoint(BaseModel):
    """WGS84 position for a CoT export event."""
    lat: float = 0.0
    lon: float = 0.0
    hae: float = 0.0
    ce: float = 10.0
    le: float = 10.0


class CoTExportEvent(BaseModel):
    """A CoT event optimized for target export.

    Converts a Tritium target dict into a structured model that can be
    serialized to MIL-STD-2045 CoT XML.
    """
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cot_type: str = "a-u-G"
    how: str = "m-g"
    time: datetime = Field(default_factory=_utcnow)
    start: datetime = Field(default_factory=_utcnow)
    stale: datetime = Field(default_factory=_default_stale)
    point: CoTExportPoint = Field(default_factory=CoTExportPoint)
    callsign: str = ""
    team_color: str = "Yellow"
    team_role: str = "Team Member"
    speed: float = 0.0
    course: float = 0.0
    battery_pct: float = 100.0
    remarks: str = ""
    version: str = "2.0"

    @classmethod
    def from_target_dict(
        cls,
        target: dict,
        stale_seconds: int = 300,
    ) -> CoTExportEvent:
        """Create a CoTExportEvent from a Tritium target dict.

        Args:
            target: Target dict from TrackedTarget.to_dict() or similar.
            stale_seconds: Seconds until the event goes stale.

        Returns:
            Populated CoTExportEvent ready for XML serialization.
        """
        now = _utcnow()
        alliance = target.get("alliance", "unknown")
        asset_type = target.get("asset_type", target.get("type", "unknown"))
        source = target.get("source", "simulation")

        # Build CoT type
        affil = _ALLIANCE_CHAR.get(alliance, "u")
        suffix = _ASSET_TYPE_SUFFIX.get(asset_type, "G")
        cot_type = f"a-{affil}-{suffix}"

        # Position
        lat = target.get("lat", 0.0) or 0.0
        lon = target.get("lng", target.get("lon", 0.0)) or 0.0
        hae = target.get("alt", target.get("altitude", 0.0)) or 0.0

        # Fall back to position dict x/y
        if lat == 0.0 and lon == 0.0:
            pos = target.get("position", {})
            if isinstance(pos, dict):
                lon = pos.get("x", 0.0)
                lat = pos.get("y", 0.0)

        # Battery
        battery_raw = target.get("battery", 1.0)
        if battery_raw is not None and battery_raw <= 1.0:
            battery_pct = battery_raw * 100.0
        else:
            battery_pct = battery_raw or 100.0

        # Callsign
        callsign = target.get("name", target.get("target_id", "unknown"))

        # Remarks
        health = target.get("health", "")
        max_health = target.get("max_health", "")
        status = target.get("status", "active")
        kills = target.get("kills", 0)
        remarks = f"source:{source} status:{status}"
        if health:
            remarks += f" health:{health}/{max_health}"
        if kills:
            remarks += f" kills:{kills}"

        return cls(
            uid=target.get("target_id", str(uuid.uuid4())),
            cot_type=cot_type,
            how=_HOW_MAP.get(source, "m-g"),
            time=now,
            start=now,
            stale=now + timedelta(seconds=stale_seconds),
            point=CoTExportPoint(lat=lat, lon=lon, hae=hae),
            callsign=callsign,
            team_color=_TEAM_COLORS.get(alliance, "Yellow"),
            team_role="Team Member",
            speed=target.get("speed", 0.0) or 0.0,
            course=target.get("heading", 0.0) or 0.0,
            battery_pct=battery_pct,
            remarks=remarks,
        )

    def to_xml(self) -> str:
        """Serialize to MIL-STD-2045 CoT XML string.

        Returns:
            CoT XML string (no XML declaration for compatibility with TAK).
        """
        event = ET.Element("event")
        event.set("version", self.version)
        event.set("uid", self.uid)
        event.set("type", self.cot_type)
        event.set("how", self.how)
        event.set("time", self.time.strftime(_DT_FMT))
        event.set("start", self.start.strftime(_DT_FMT))
        event.set("stale", self.stale.strftime(_DT_FMT))

        pt = ET.SubElement(event, "point")
        pt.set("lat", f"{self.point.lat:.7f}")
        pt.set("lon", f"{self.point.lon:.7f}")
        pt.set("hae", f"{self.point.hae:.1f}")
        pt.set("ce", f"{self.point.ce:.1f}")
        pt.set("le", f"{self.point.le:.1f}")

        detail = ET.SubElement(event, "detail")

        contact = ET.SubElement(detail, "contact")
        contact.set("callsign", self.callsign)

        group = ET.SubElement(detail, "__group")
        group.set("name", self.team_color)
        group.set("role", self.team_role)

        status_el = ET.SubElement(detail, "status")
        status_el.set("battery", f"{self.battery_pct:.1f}")

        track = ET.SubElement(detail, "track")
        track.set("speed", f"{self.speed:.1f}")
        track.set("course", f"{self.course:.1f}")

        if self.remarks:
            remarks_el = ET.SubElement(detail, "remarks")
            remarks_el.text = self.remarks

        uid_el = ET.SubElement(detail, "uid")
        uid_el.set("Droid", self.callsign)

        return ET.tostring(event, encoding="unicode", xml_declaration=False)


def targets_to_cot_xml(
    targets: list[dict],
    stale_seconds: int = 300,
) -> str:
    """Convert a list of target dicts to a concatenated CoT XML string.

    Each target becomes a separate CoT event XML element. The output is
    suitable for streaming to a TAK server or saving to a .cot file.

    Args:
        targets: List of target dicts from TrackedTarget.to_dict().
        stale_seconds: Seconds until each event goes stale.

    Returns:
        Concatenated XML string with one event per target.
    """
    parts = []
    for t in targets:
        evt = CoTExportEvent.from_target_dict(t, stale_seconds=stale_seconds)
        parts.append(evt.to_xml())
    return "\n".join(parts)


def targets_to_cot_file(
    targets: list[dict],
    stale_seconds: int = 300,
) -> str:
    """Convert targets to a complete CoT XML document with wrapper.

    Wraps individual events in a <cot-events> root element for file export.
    This is not standard CoT wire format (which sends individual events),
    but is useful for file-based export/import.

    Args:
        targets: List of target dicts.
        stale_seconds: Seconds until each event goes stale.

    Returns:
        Complete XML document string.
    """
    events_xml = targets_to_cot_xml(targets, stale_seconds=stale_seconds)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<cot-events version="1.0" '
        f'count="{len(targets)}" '
        f'generated="{_utcnow().strftime(_DT_FMT)}">\n'
        f'{events_xml}\n'
        '</cot-events>\n'
    )
