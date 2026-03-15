# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""KML 2.2 XML generation models.

Provides simple builder classes for generating valid KML 2.2 XML output.
Used for exporting target movement trails to Google Earth, ATAK, and any
KML-compatible GIS application.

KML Spec: https://developers.google.com/kml/documentation/kmlreference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from xml.etree.ElementTree import Element, SubElement, tostring


KML_NS = "http://www.opengis.net/kml/2.2"


@dataclass
class KMLPoint:
    """A single point in KML format.

    Attributes:
        lat: Latitude in decimal degrees (WGS84).
        lon: Longitude in decimal degrees (WGS84).
        alt: Altitude in meters (optional).
        time: UTC timestamp (optional).
        name: Point name/label (optional).
        desc: Description text (optional).
    """
    lat: float
    lon: float
    alt: Optional[float] = None
    time: Optional[datetime] = None
    name: Optional[str] = None
    desc: Optional[str] = None

    def coord_string(self) -> str:
        """Return KML coordinate string: lon,lat[,alt]."""
        if self.alt is not None:
            return f"{self.lon:.8f},{self.lat:.8f},{self.alt:.2f}"
        return f"{self.lon:.8f},{self.lat:.8f},0"

    def to_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "alt": self.alt,
            "time": self.time.isoformat() if self.time else None,
            "name": self.name,
            "desc": self.desc,
        }

    @classmethod
    def from_dict(cls, data: dict) -> KMLPoint:
        t = data.get("time")
        if isinstance(t, str):
            t = datetime.fromisoformat(t)
        return cls(
            lat=float(data.get("lat", 0.0)),
            lon=float(data.get("lon", 0.0)),
            alt=data.get("alt"),
            time=t,
            name=data.get("name"),
            desc=data.get("desc"),
        )


@dataclass
class KMLTrack:
    """A line track in KML format, rendered as a LineString with optional
    TimeSpan.

    Attributes:
        name: Track name.
        desc: Track description.
        points: Ordered list of track points.
        line_color: ABGR hex color for the line (KML convention).
        line_width: Line width in pixels.
    """
    name: str = ""
    desc: str = ""
    points: list[KMLPoint] = field(default_factory=list)
    line_color: str = "ff00f0ff"  # cyan, fully opaque (ABGR)
    line_width: float = 3.0

    def add_point(
        self,
        lat: float,
        lon: float,
        alt: Optional[float] = None,
        time: Optional[datetime] = None,
        name: Optional[str] = None,
    ) -> KMLPoint:
        """Add a track point and return it."""
        pt = KMLPoint(lat=lat, lon=lon, alt=alt, time=time, name=name)
        self.points.append(pt)
        return pt

    def to_element(self) -> Element:
        """Build a KML Placemark element with LineString geometry."""
        pm = Element("Placemark")
        if self.name:
            SubElement(pm, "name").text = self.name
        if self.desc:
            SubElement(pm, "description").text = self.desc

        # TimeSpan from first to last point
        times = [p.time for p in self.points if p.time is not None]
        if len(times) >= 2:
            ts = SubElement(pm, "TimeSpan")
            SubElement(ts, "begin").text = min(times).isoformat()
            SubElement(ts, "end").text = max(times).isoformat()

        # Style
        style = SubElement(pm, "Style")
        ls = SubElement(style, "LineStyle")
        SubElement(ls, "color").text = self.line_color
        SubElement(ls, "width").text = str(self.line_width)

        # LineString
        linestring = SubElement(pm, "LineString")
        SubElement(linestring, "altitudeMode").text = "clampToGround"
        coords = " ".join(p.coord_string() for p in self.points)
        SubElement(linestring, "coordinates").text = coords

        return pm

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "desc": self.desc,
            "points": [p.to_dict() for p in self.points],
            "line_color": self.line_color,
            "line_width": self.line_width,
        }

    @classmethod
    def from_dict(cls, data: dict) -> KMLTrack:
        return cls(
            name=data.get("name", ""),
            desc=data.get("desc", ""),
            points=[KMLPoint.from_dict(p) for p in data.get("points", [])],
            line_color=data.get("line_color", "ff00f0ff"),
            line_width=data.get("line_width", 3.0),
        )


@dataclass
class KMLDocument:
    """Complete KML 2.2 document builder.

    Composes placemarks (points), tracks (lines), and folders into a
    valid KML 2.2 XML document. Use ``to_xml()`` for the full XML string.

    Attributes:
        name: Document name (optional).
        desc: Document description (optional).
        placemarks: Standalone point placemarks.
        tracks: Line tracks.
    """
    name: str = ""
    desc: str = ""
    placemarks: list[KMLPoint] = field(default_factory=list)
    tracks: list[KMLTrack] = field(default_factory=list)

    def add_placemark(self, lat: float, lon: float, **kwargs) -> KMLPoint:
        """Add a standalone point placemark."""
        pt = KMLPoint(lat=lat, lon=lon, **kwargs)
        self.placemarks.append(pt)
        return pt

    def add_track(self, name: str = "", desc: str = "") -> KMLTrack:
        """Add a new line track and return it for adding points."""
        trk = KMLTrack(name=name, desc=desc)
        self.tracks.append(trk)
        return trk

    def to_element(self) -> Element:
        """Build the root <kml> element."""
        kml = Element("kml", attrib={"xmlns": KML_NS})
        doc = SubElement(kml, "Document")

        if self.name:
            SubElement(doc, "name").text = self.name
        if self.desc:
            SubElement(doc, "description").text = self.desc

        # Point placemarks
        for pt in self.placemarks:
            pm = SubElement(doc, "Placemark")
            if pt.name:
                SubElement(pm, "name").text = pt.name
            if pt.desc:
                SubElement(pm, "description").text = pt.desc
            point_el = SubElement(pm, "Point")
            SubElement(point_el, "coordinates").text = pt.coord_string()

        # Track placemarks (LineString)
        for trk in self.tracks:
            doc.append(trk.to_element())

        return kml

    def to_xml(self, xml_declaration: bool = True) -> str:
        """Generate complete KML 2.2 XML string."""
        root = self.to_element()
        xml_bytes = tostring(root, encoding="unicode")
        if xml_declaration:
            return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes
        return xml_bytes

    def to_bytes(self) -> bytes:
        """Generate KML XML as UTF-8 bytes."""
        return self.to_xml().encode("utf-8")
