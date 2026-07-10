# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Value types for the real-GIS layer stack.

Stdlib + dataclasses only (no pydantic).  These are the wire/raster contracts
that the costmap lane and the SC frontend both consume, so the conventions
baked in here are load-bearing:

    - ``GeoBBox`` is WGS-84, ``west, south, east, north`` order.
    - ``ElevationGrid`` is a row-major raster with **row 0 = the NORTH edge**
      and ``None`` marking NoData cells.  ``cell_lon`` / ``cell_lat`` treat the
      grid as an *inclusive-edge* sampling: column 0 sits exactly on ``west``,
      column ``ncols - 1`` exactly on ``east`` (same for rows / north-south).

See ``README.md`` in this package for the full documented interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tritium_lib.geo import METERS_PER_DEG_LAT

__all__ = ["GeoBBox", "ElevationGrid"]


@dataclass
class GeoBBox:
    """A WGS-84 bounding box in ``west, south, east, north`` order (degrees)."""

    west: float
    south: float
    east: float
    north: float

    @classmethod
    def from_string(cls, text: str) -> "GeoBBox":
        """Parse a ``"west,south,east,north"`` string.

        Raises ``ValueError`` on the wrong number of parts or non-numeric
        values so callers can surface a clean 400 rather than a stack trace.
        """
        if text is None:
            raise ValueError("bbox string is required")
        parts = [p.strip() for p in str(text).split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"bbox must have 4 comma-separated values (w,s,e,n), got {len(parts)}"
            )
        try:
            west, south, east, north = (float(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"bbox values must be numeric: {text!r}") from exc
        return cls(west=west, south=south, east=east, north=north)

    def to_string(self) -> str:
        """Serialize back to ``"west,south,east,north"``."""
        return f"{self.west},{self.south},{self.east},{self.north}"

    def center(self) -> tuple[float, float]:
        """Return the box centre as ``(lon, lat)`` (GeoJSON lon-first order)."""
        return ((self.west + self.east) / 2.0, (self.south + self.north) / 2.0)

    def contains(self, lon: float, lat: float) -> bool:
        """True when ``(lon, lat)`` lies inside (inclusive) the box."""
        return (self.west <= lon <= self.east) and (self.south <= lat <= self.north)


@dataclass
class ElevationGrid:
    """A regularly-sampled elevation raster over a WGS-84 bounding box.

    ``values`` is row-major, length ``ncols * nrows``, with **row 0 on the
    NORTH edge** and values increasing eastward within a row.  ``None`` marks a
    NoData cell.  This row-0=north convention is the single source of truth the
    costmap lane reads — do not silently flip it.
    """

    west: float
    south: float
    east: float
    north: float
    ncols: int
    nrows: int
    values: list = field(default_factory=list)
    source: str = "usgs"
    resolution_m: float | None = None

    def cell_lon(self, ix: int) -> float:
        """Longitude of the cell-centre in column ``ix`` (0 == west edge)."""
        if self.ncols <= 1:
            return self.west
        return self.west + (self.east - self.west) * ix / (self.ncols - 1)

    def cell_lat(self, iy: int) -> float:
        """Latitude of the cell-centre in row ``iy`` (row 0 == north edge)."""
        if self.nrows <= 1:
            return self.north
        return self.north - (self.north - self.south) * iy / (self.nrows - 1)

    def value_at(self, ix: int, iy: int):
        """Elevation (metres) at column ``ix``, row ``iy`` — ``None`` if NoData."""
        if not (0 <= ix < self.ncols and 0 <= iy < self.nrows):
            raise IndexError(f"cell ({ix}, {iy}) out of {self.ncols}x{self.nrows} grid")
        idx = iy * self.ncols + ix
        if idx >= len(self.values):
            return None
        return self.values[idx]

    def min_max(self) -> tuple:
        """Return ``(min, max)`` over non-None cells, or ``(None, None)``."""
        present = [v for v in self.values if v is not None]
        if not present:
            return (None, None)
        return (min(present), max(present))

    def to_dict(self) -> dict:
        """JSON-serializable dict; round-trips exactly through ``from_dict``."""
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
            "ncols": self.ncols,
            "nrows": self.nrows,
            "values": list(self.values),
            "source": self.source,
            "resolution_m": self.resolution_m,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ElevationGrid":
        """Rebuild from ``to_dict`` output; tolerates extra keys (e.g. ``fixture``)."""
        return cls(
            west=data["west"],
            south=data["south"],
            east=data["east"],
            north=data["north"],
            ncols=data["ncols"],
            nrows=data["nrows"],
            values=list(data.get("values", [])),
            source=data.get("source", "usgs"),
            resolution_m=data.get("resolution_m"),
        )

    def slope_deg(self) -> list:
        """Per-cell terrain slope in degrees (max gradient, central differences).

        Metres are derived from the *sampled cell spacing* (bbox / (n - 1)),
        not from ``resolution_m`` (which is the source DEM's native resolution).
        Longitude spacing is scaled by ``cos(lat)`` via ``METERS_PER_DEG_LAT``.
        A cell whose central-difference stencil touches a NoData neighbour (or
        is itself NoData) yields ``None``.
        """
        slopes: list = [None] * (self.ncols * self.nrows)
        if self.ncols < 2 or self.nrows < 2:
            return slopes

        # Cell spacing in metres.
        dlon_deg = (self.east - self.west) / (self.ncols - 1)
        dlat_deg = (self.north - self.south) / (self.nrows - 1)
        dy_m = abs(dlat_deg) * METERS_PER_DEG_LAT

        for iy in range(self.nrows):
            lat = self.cell_lat(iy)
            m_per_deg_lon = METERS_PER_DEG_LAT * math.cos(math.radians(lat))
            dx_m = abs(dlon_deg) * m_per_deg_lon
            for ix in range(self.ncols):
                z = self.value_at(ix, iy)
                if z is None or dx_m == 0.0 or dy_m == 0.0:
                    continue

                # East-west gradient (central diff, forward/back at edges).
                if ix == 0:
                    z_e = self.value_at(ix + 1, iy)
                    dzdx = None if z_e is None else (z_e - z) / dx_m
                elif ix == self.ncols - 1:
                    z_w = self.value_at(ix - 1, iy)
                    dzdx = None if z_w is None else (z - z_w) / dx_m
                else:
                    z_w = self.value_at(ix - 1, iy)
                    z_e = self.value_at(ix + 1, iy)
                    if z_w is None or z_e is None:
                        dzdx = None
                    else:
                        dzdx = (z_e - z_w) / (2.0 * dx_m)

                # North-south gradient (rows increase southward).
                if iy == 0:
                    z_s = self.value_at(ix, iy + 1)
                    dzdy = None if z_s is None else (z - z_s) / dy_m
                elif iy == self.nrows - 1:
                    z_n = self.value_at(ix, iy - 1)
                    dzdy = None if z_n is None else (z_n - z) / dy_m
                else:
                    z_n = self.value_at(ix, iy - 1)
                    z_s = self.value_at(ix, iy + 1)
                    if z_n is None or z_s is None:
                        dzdy = None
                    else:
                        dzdy = (z_n - z_s) / (2.0 * dy_m)

                if dzdx is None or dzdy is None:
                    continue

                grad = math.hypot(dzdx, dzdy)
                slopes[iy * self.ncols + ix] = math.degrees(math.atan(grad))

        return slopes
