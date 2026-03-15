# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""GPX 1.1 XML generation models.

Provides simple builder classes for generating valid GPX 1.1 XML output.
Used for exporting target movement trails to external mapping tools such
as ATAK, Google Earth, and any GPX-compatible GIS application.

GPX Spec: https://www.topografix.com/gpx/1/1/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from xml.etree.ElementTree import Element, SubElement, tostring


GPX_NS = "http://www.topografix.com/GPX/1/1"
GPX_XSI = "http://www.w3.org/2001/XMLSchema-instance"
GPX_SCHEMA = "http://www.topografix.com/GPX/1/1/gpx.xsd"


@dataclass
class GPXWaypoint:
    """A single waypoint (wpt) in GPX format.

    Attributes:
        lat: Latitude in decimal degrees (WGS84).
        lon: Longitude in decimal degrees (WGS84).
        ele: Elevation in meters (optional).
        time: UTC timestamp (optional).
        name: Waypoint name/label (optional).
        desc: Description text (optional).
        sym: Symbol name for rendering (optional).
    """
    lat: float
    lon: float
    ele: Optional[float] = None
    time: Optional[datetime] = None
    name: Optional[str] = None
    desc: Optional[str] = None
    sym: Optional[str] = None

    def to_element(self, tag: str = "wpt") -> Element:
        """Build an XML Element for this waypoint."""
        el = Element(tag, attrib={"lat": f"{self.lat:.8f}", "lon": f"{self.lon:.8f}"})
        if self.ele is not None:
            SubElement(el, "ele").text = f"{self.ele:.2f}"
        if self.time is not None:
            SubElement(el, "time").text = self.time.isoformat()
        if self.name:
            SubElement(el, "name").text = self.name
        if self.desc:
            SubElement(el, "desc").text = self.desc
        if self.sym:
            SubElement(el, "sym").text = self.sym
        return el

    def to_dict(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "ele": self.ele,
            "time": self.time.isoformat() if self.time else None,
            "name": self.name,
            "desc": self.desc,
            "sym": self.sym,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GPXWaypoint:
        t = data.get("time")
        if isinstance(t, str):
            t = datetime.fromisoformat(t)
        return cls(
            lat=float(data.get("lat", 0.0)),
            lon=float(data.get("lon", 0.0)),
            ele=data.get("ele"),
            time=t,
            name=data.get("name"),
            desc=data.get("desc"),
            sym=data.get("sym"),
        )


@dataclass
class GPXTrack:
    """A track (trk) composed of track segments (trkseg) of waypoints.

    For simplicity, each GPXTrack has a single trkseg containing all its
    points. Multi-segment tracks can be built by appending points with
    gaps (though most consumers treat the full track as continuous).

    Attributes:
        name: Track name.
        desc: Track description.
        points: Ordered list of track points.
    """
    name: str = ""
    desc: str = ""
    points: list[GPXWaypoint] = field(default_factory=list)

    def add_point(
        self,
        lat: float,
        lon: float,
        ele: Optional[float] = None,
        time: Optional[datetime] = None,
        name: Optional[str] = None,
    ) -> GPXWaypoint:
        """Add a track point and return it."""
        pt = GPXWaypoint(lat=lat, lon=lon, ele=ele, time=time, name=name)
        self.points.append(pt)
        return pt

    def to_element(self) -> Element:
        """Build an XML Element for this track."""
        trk = Element("trk")
        if self.name:
            SubElement(trk, "name").text = self.name
        if self.desc:
            SubElement(trk, "desc").text = self.desc
        trkseg = SubElement(trk, "trkseg")
        for pt in self.points:
            trkseg.append(pt.to_element("trkpt"))
        return trk

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "desc": self.desc,
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict) -> GPXTrack:
        return cls(
            name=data.get("name", ""),
            desc=data.get("desc", ""),
            points=[GPXWaypoint.from_dict(p) for p in data.get("points", [])],
        )


@dataclass
class GPXRoute:
    """A route (rte) composed of ordered route points.

    Routes differ from tracks: routes are planned navigation paths
    while tracks are recorded movement trails. Both use the same
    waypoint structure.

    Attributes:
        name: Route name.
        desc: Route description.
        points: Ordered list of route points.
    """
    name: str = ""
    desc: str = ""
    points: list[GPXWaypoint] = field(default_factory=list)

    def add_point(
        self,
        lat: float,
        lon: float,
        ele: Optional[float] = None,
        name: Optional[str] = None,
    ) -> GPXWaypoint:
        """Add a route point and return it."""
        pt = GPXWaypoint(lat=lat, lon=lon, ele=ele, name=name)
        self.points.append(pt)
        return pt

    def to_element(self) -> Element:
        """Build an XML Element for this route."""
        rte = Element("rte")
        if self.name:
            SubElement(rte, "name").text = self.name
        if self.desc:
            SubElement(rte, "desc").text = self.desc
        for pt in self.points:
            rte.append(pt.to_element("rtept"))
        return rte

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "desc": self.desc,
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict) -> GPXRoute:
        return cls(
            name=data.get("name", ""),
            desc=data.get("desc", ""),
            points=[GPXWaypoint.from_dict(p) for p in data.get("points", [])],
        )


@dataclass
class GPXDocument:
    """Complete GPX 1.1 document builder.

    Composes waypoints, tracks, and routes into a valid GPX 1.1 XML
    document. Use ``to_xml()`` to get the full XML string.

    Attributes:
        creator: Application name (default: "Tritium").
        name: Document name (optional).
        desc: Document description (optional).
        waypoints: Standalone waypoints.
        tracks: Movement tracks.
        routes: Navigation routes.
    """
    creator: str = "Tritium"
    name: str = ""
    desc: str = ""
    waypoints: list[GPXWaypoint] = field(default_factory=list)
    tracks: list[GPXTrack] = field(default_factory=list)
    routes: list[GPXRoute] = field(default_factory=list)

    def add_waypoint(self, lat: float, lon: float, **kwargs) -> GPXWaypoint:
        """Add a standalone waypoint."""
        wpt = GPXWaypoint(lat=lat, lon=lon, **kwargs)
        self.waypoints.append(wpt)
        return wpt

    def add_track(self, name: str = "", desc: str = "") -> GPXTrack:
        """Add a new track and return it for adding points."""
        trk = GPXTrack(name=name, desc=desc)
        self.tracks.append(trk)
        return trk

    def add_route(self, name: str = "", desc: str = "") -> GPXRoute:
        """Add a new route and return it for adding points."""
        rte = GPXRoute(name=name, desc=desc)
        self.routes.append(rte)
        return rte

    def to_element(self) -> Element:
        """Build the root <gpx> element."""
        gpx = Element("gpx", attrib={
            "version": "1.1",
            "creator": self.creator,
            "xmlns": GPX_NS,
            "xmlns:xsi": GPX_XSI,
            "xsi:schemaLocation": f"{GPX_NS} {GPX_SCHEMA}",
        })

        # Metadata
        metadata = SubElement(gpx, "metadata")
        if self.name:
            SubElement(metadata, "name").text = self.name
        if self.desc:
            SubElement(metadata, "desc").text = self.desc
        SubElement(metadata, "time").text = datetime.now(timezone.utc).isoformat()

        # Waypoints
        for wpt in self.waypoints:
            gpx.append(wpt.to_element("wpt"))

        # Routes
        for rte in self.routes:
            gpx.append(rte.to_element())

        # Tracks
        for trk in self.tracks:
            gpx.append(trk.to_element())

        return gpx

    def to_xml(self, xml_declaration: bool = True) -> str:
        """Generate complete GPX 1.1 XML string."""
        root = self.to_element()
        xml_bytes = tostring(root, encoding="unicode")
        if xml_declaration:
            return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes
        return xml_bytes

    def to_bytes(self) -> bytes:
        """Generate GPX XML as UTF-8 bytes."""
        return self.to_xml().encode("utf-8")
