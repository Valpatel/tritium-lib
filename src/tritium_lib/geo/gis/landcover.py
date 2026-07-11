# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""USGS / MRLC National Land Cover Database (NLCD 2021) classification.

Turns a rendered NLCD land-cover raster into two things Tritium consumes:

    * a **classified grid** (:class:`LandCoverGrid`) — one NLCD class code per
      cell, mirroring :class:`~tritium_lib.geo.gis.models.ElevationGrid`'s
      row-major, **row 0 = NORTH edge** geometry so the two rasters stack.
    * a **tactical profile** per class (cover, concealment, mobility cost,
      passability) — the production half: a costmap / route planner reads this
      to weight movement and a threat model reads cover/concealment to reason
      about fields of fire and observation.

The MRLC WMS renders an 8-bit colormap PNG whose palette is *close to but not
pixel-exact* with the canonical NLCD table (rendering / antialias), so
:func:`classify_rgb` picks the **nearest** canonical colour (min squared
Euclidean distance in RGB), never an exact match.

Pure stdlib — no numpy, no Pillow.  PNG decoding lives in the fetcher (behind an
optional Pillow import); this module only ever sees already-decoded ``(r, g, b)``
triples, so it stays dependency-free and fully unit-testable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

__all__ = [
    "LandCoverClass",
    "NLCD_CLASSES",
    "classify_rgb",
    "tactical_profile",
    "LandCoverGrid",
]


@dataclass(frozen=True)
class LandCoverClass:
    """One NLCD land-cover class + its tactical doctrine.

    Attributes:
        code: NLCD class code (e.g. ``42`` = Evergreen Forest).
        name: Human-readable class name.
        rgb: Canonical NLCD colormap colour ``(r, g, b)`` (0-255).
        category: Coarse grouping (``water``/``forest``/``developed`` ...).
        cover: Protection from fire, 0..1 (1 = full hard cover).
        concealment: Protection from observation, 0..1 (1 = fully hidden).
        mobility_cost: Movement penalty multiplier, >= 1.0 (a route planner
            multiplies base traversal cost by this).
        passable: Whether ground units can enter the cell at all.
    """

    code: int
    name: str
    rgb: tuple[int, int, int]
    category: str
    cover: float
    concealment: float
    mobility_cost: float
    passable: bool


# ---------------------------------------------------------------------------
# Canonical NLCD 2021 class table (code -> LandCoverClass).
#
# RGB values are the canonical NLCD colormap.  Tactical doctrine (cover /
# concealment / mobility_cost / passable) is the production-validating half:
# a costmap multiplies base cost by mobility_cost and refuses impassable cells;
# a threat model reads cover (fire) and concealment (observation).
# ---------------------------------------------------------------------------
NLCD_CLASSES: dict[int, LandCoverClass] = {
    11: LandCoverClass(11, "Open Water", (70, 107, 159), "water",
                       cover=0.0, concealment=0.0, mobility_cost=999.0, passable=False),
    12: LandCoverClass(12, "Perennial Ice/Snow", (209, 222, 248), "snow",
                       cover=0.0, concealment=0.0, mobility_cost=2.5, passable=True),
    21: LandCoverClass(21, "Developed, Open Space", (222, 197, 197), "developed",
                       cover=0.2, concealment=0.2, mobility_cost=1.0, passable=True),
    22: LandCoverClass(22, "Developed, Low Intensity", (217, 146, 130), "developed",
                       cover=0.5, concealment=0.5, mobility_cost=1.2, passable=True),
    23: LandCoverClass(23, "Developed, Medium Intensity", (235, 0, 0), "developed",
                       cover=0.75, concealment=0.7, mobility_cost=1.5, passable=True),
    24: LandCoverClass(24, "Developed, High Intensity", (171, 0, 0), "developed",
                       cover=0.9, concealment=0.8, mobility_cost=1.8, passable=True),
    31: LandCoverClass(31, "Barren Land", (179, 172, 159), "barren",
                       cover=0.0, concealment=0.0, mobility_cost=1.6, passable=True),
    41: LandCoverClass(41, "Deciduous Forest", (104, 171, 95), "forest",
                       cover=0.2, concealment=0.85, mobility_cost=3.0, passable=True),
    42: LandCoverClass(42, "Evergreen Forest", (28, 95, 44), "forest",
                       cover=0.2, concealment=0.9, mobility_cost=3.0, passable=True),
    43: LandCoverClass(43, "Mixed Forest", (181, 197, 143), "forest",
                       cover=0.2, concealment=0.85, mobility_cost=3.0, passable=True),
    51: LandCoverClass(51, "Dwarf Scrub", (166, 140, 48), "shrub",
                       cover=0.15, concealment=0.5, mobility_cost=1.8, passable=True),
    52: LandCoverClass(52, "Shrub/Scrub", (204, 186, 124), "shrub",
                       cover=0.15, concealment=0.5, mobility_cost=1.8, passable=True),
    71: LandCoverClass(71, "Grassland/Herbaceous", (227, 227, 194), "herbaceous",
                       cover=0.0, concealment=0.2, mobility_cost=1.1, passable=True),
    72: LandCoverClass(72, "Sedge/Herbaceous", (202, 202, 120), "herbaceous",
                       cover=0.0, concealment=0.2, mobility_cost=1.1, passable=True),
    73: LandCoverClass(73, "Lichens", (138, 146, 44), "herbaceous",
                       cover=0.0, concealment=0.2, mobility_cost=1.1, passable=True),
    74: LandCoverClass(74, "Moss", (189, 204, 148), "herbaceous",
                       cover=0.0, concealment=0.2, mobility_cost=1.1, passable=True),
    81: LandCoverClass(81, "Pasture/Hay", (220, 217, 57), "cultivated",
                       cover=0.0, concealment=0.2, mobility_cost=1.1, passable=True),
    82: LandCoverClass(82, "Cultivated Crops", (171, 108, 40), "cultivated",
                       cover=0.0, concealment=0.4, mobility_cost=1.4, passable=True),
    90: LandCoverClass(90, "Woody Wetlands", (184, 217, 235), "wetland",
                       cover=0.1, concealment=0.6, mobility_cost=4.0, passable=True),
    95: LandCoverClass(95, "Emergent Herbaceous Wetlands", (108, 159, 184), "wetland",
                       cover=0.1, concealment=0.4, mobility_cost=6.0, passable=True),
}

#: Neutral profile for a code that is not in :data:`NLCD_CLASSES` — safe,
#: non-blocking defaults so an unknown class never makes ground impassable or
#: fabricates cover.
_NEUTRAL_PROFILE: dict = {
    "cover": 0.0,
    "concealment": 0.0,
    "mobility_cost": 1.0,
    "passable": True,
    "name": "Unknown",
    "category": "unknown",
}


def classify_rgb(r: int, g: int, b: int) -> int:
    """Return the NLCD class code whose canonical colour is nearest ``(r,g,b)``.

    Nearest by **squared Euclidean distance** in RGB (no sqrt needed — it is
    monotonic).  The MRLC WMS palette is close to but not pixel-exact with the
    canonical table, so an exact match would misclassify antialiased pixels;
    nearest-colour is the correct decode.
    """
    best_code = 11
    best_dist = None
    for code, cls in NLCD_CLASSES.items():
        cr, cg, cb = cls.rgb
        d = (r - cr) * (r - cr) + (g - cg) * (g - cg) + (b - cb) * (b - cb)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_code = code
    return best_code


def tactical_profile(code) -> dict:
    """Return the tactical doctrine dict for an NLCD class code.

    Keys: ``cover``/``concealment``/``mobility_cost``/``passable``/``name``/
    ``category``.  An unknown code (or ``None``) yields the neutral profile
    (no cover, no concealment, cost 1.0, passable) so a costmap degrades safely.
    """
    cls = NLCD_CLASSES.get(code)
    if cls is None:
        return dict(_NEUTRAL_PROFILE)
    return {
        "cover": cls.cover,
        "concealment": cls.concealment,
        "mobility_cost": cls.mobility_cost,
        "passable": cls.passable,
        "name": cls.name,
        "category": cls.category,
    }


@dataclass
class LandCoverGrid:
    """A regularly-sampled NLCD land-cover raster over a WGS-84 bounding box.

    ``codes`` is row-major, length ``ncols * nrows``, with **row 0 on the NORTH
    edge** and codes increasing eastward within a row.  ``None`` marks a NoData
    cell.  The geometry (``cell_lon`` / ``cell_lat``) is identical to
    :class:`~tritium_lib.geo.gis.models.ElevationGrid` so a land-cover grid and
    an elevation grid captured over the same bbox at the same resolution line up
    cell-for-cell — the costmap lane fuses them directly.
    """

    west: float
    south: float
    east: float
    north: float
    ncols: int
    nrows: int
    codes: list = field(default_factory=list)
    source: str = "nlcd"

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

    def code_at(self, ix: int, iy: int):
        """NLCD class code at column ``ix``, row ``iy`` — ``None`` if NoData."""
        if not (0 <= ix < self.ncols and 0 <= iy < self.nrows):
            raise IndexError(f"cell ({ix}, {iy}) out of {self.ncols}x{self.nrows} grid")
        idx = iy * self.ncols + ix
        if idx >= len(self.codes):
            return None
        return self.codes[idx]

    def dominant_category(self):
        """Most common non-None tactical category over the grid, or ``None``.

        Ties broken by first-encountered (``Counter.most_common`` order).
        """
        cats = [
            tactical_profile(code)["category"]
            for code in self.codes
            if code is not None
        ]
        if not cats:
            return None
        return Counter(cats).most_common(1)[0][0]

    def tactical_field(self) -> list:
        """Per-cell tactical profile dicts, row-major (parallel to ``codes``).

        A NoData cell yields the neutral profile — the same safe default a
        costmap would apply, so the field is always dense.
        """
        return [tactical_profile(code) for code in self.codes]

    def to_dict(self) -> dict:
        """JSON-serializable dict; round-trips exactly through ``from_dict``."""
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
            "ncols": self.ncols,
            "nrows": self.nrows,
            "codes": list(self.codes),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LandCoverGrid":
        """Rebuild from ``to_dict`` output; tolerates extra keys (``fixture``, ``bbox``)."""
        return cls(
            west=data["west"],
            south=data["south"],
            east=data["east"],
            north=data["north"],
            ncols=data["ncols"],
            nrows=data["nrows"],
            codes=list(data.get("codes", [])),
            source=data.get("source", "nlcd"),
        )
