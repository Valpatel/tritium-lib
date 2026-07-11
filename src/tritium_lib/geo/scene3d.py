# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared map data -> a neutral 3D scene (digital twin of a Tritium AO).

The reusable pipeline that turns Tritium's shared GIS layers — DEM elevation,
building footprints, roads, water — into a framework-neutral 3D scene
(:class:`Scene3D`: terrain heightfield + extruded building meshes + flat
features), in AO-local metres, serializable to JSON.  A downstream writer
(``examples/isaac-scene/usd_scene_builder.py``) turns that JSON into a USD
stage so NVIDIA Isaac Sim renders a faithful 3D model of the real map area —
the ground truth the Isaac camera and robot dogs then operate in.

Separation (the standing rule): this module is PURE geometry — math + stdlib
(+ the already-declared numpy).  No USD, no isaacsim, no framework deps.  It
imports clean on aarch64 (Jetson).  The USD/pxr writer lives Isaac-side; the
HTTP exposure lives in tritium-sc.  A Scene3D is the neutral contract between
them.

Coordinate convention (single source of truth):
  * **X = east, Y = north, Z = up (elevation)**, metres — Isaac Z-up and the
    tracker's east/north agree, so no axis flips downstream.
  * The scene is **AO-local**: its own origin is the AO bbox centre, and it
    carries that origin's (lat, lng) so it can be geo-referenced back onto the
    tactical map.  This is deliberately independent of any runtime geo
    reference (which re-homes with the demo) — a scene is self-contained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

# Metres per degree at the equator (equirectangular AO-local projection).
_M_PER_DEG_LAT = 110540.0
_M_PER_DEG_LON = 111320.0

Vec3 = tuple[float, float, float]


# --------------------------------------------------------------------------- #
# AO-local projection.
# --------------------------------------------------------------------------- #

@dataclass
class LocalProjection:
    """Equirectangular lat/lng <-> AO-local metres, origin at the AO centre."""

    origin_lat: float
    origin_lng: float

    def to_local(self, lat: float, lng: float) -> tuple[float, float]:
        """(lat, lng) -> (east_m, north_m) relative to the AO origin."""
        east = (lng - self.origin_lng) * _M_PER_DEG_LON * math.cos(
            math.radians(self.origin_lat)
        )
        north = (lat - self.origin_lat) * _M_PER_DEG_LAT
        return east, north

    def to_latlng(self, east: float, north: float) -> tuple[float, float]:
        lat = self.origin_lat + north / _M_PER_DEG_LAT
        lng = self.origin_lng + east / (
            _M_PER_DEG_LON * math.cos(math.radians(self.origin_lat))
        )
        return lat, lng


# --------------------------------------------------------------------------- #
# Neutral 3D primitives.
# --------------------------------------------------------------------------- #

@dataclass
class Mesh3D:
    """A triangle mesh with a semantic type (building/terrain/road/water)."""

    name: str
    kind: str  # "building" | "terrain" | "road" | "water"
    vertices: list[Vec3] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    # Optional semantic hints carried to the USD writer / map.
    height_m: float = 0.0
    category: str = ""
    color: Optional[tuple[float, float, float]] = None

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "vertices": [list(v) for v in self.vertices],
            "faces": [list(f) for f in self.faces],
            "height_m": self.height_m,
            "category": self.category,
            "color": list(self.color) if self.color else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Mesh3D":
        return cls(
            name=d.get("name", ""),
            kind=d.get("kind", "building"),
            vertices=[tuple(v) for v in d.get("vertices", [])],
            faces=[tuple(f) for f in d.get("faces", [])],
            height_m=d.get("height_m", 0.0),
            category=d.get("category", ""),
            color=tuple(d["color"]) if d.get("color") else None,
        )


@dataclass
class Scene3D:
    """A neutral 3D scene of an AO: origin + meshes + provenance."""

    ao: str
    origin_lat: float
    origin_lng: float
    up_axis: str = "Z"
    meshes: list[Mesh3D] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add(self, mesh: Mesh3D) -> None:
        self.meshes.append(mesh)

    def by_kind(self, kind: str) -> list[Mesh3D]:
        return [m for m in self.meshes if m.kind == kind]

    def bounds(self) -> dict:
        """AABB over all vertices: {min:[x,y,z], max:[x,y,z]} (empty -> zeros)."""
        xs, ys, zs = [], [], []
        for m in self.meshes:
            for x, y, z in m.vertices:
                xs.append(x); ys.append(y); zs.append(z)
        if not xs:
            return {"min": [0, 0, 0], "max": [0, 0, 0]}
        return {"min": [min(xs), min(ys), min(zs)],
                "max": [max(xs), max(ys), max(zs)]}

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        verts = faces = 0
        for m in self.meshes:
            kinds[m.kind] = kinds.get(m.kind, 0) + 1
            verts += m.vertex_count
            faces += m.face_count
        return {"meshes": len(self.meshes), "by_kind": kinds,
                "vertices": verts, "faces": faces, "bounds": self.bounds()}

    def to_dict(self) -> dict:
        return {
            "ao": self.ao,
            "origin_lat": self.origin_lat,
            "origin_lng": self.origin_lng,
            "up_axis": self.up_axis,
            "meshes": [m.to_dict() for m in self.meshes],
            "metadata": {**self.metadata, "stats": self.stats()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scene3D":
        return cls(
            ao=d.get("ao", ""),
            origin_lat=d["origin_lat"],
            origin_lng=d["origin_lng"],
            up_axis=d.get("up_axis", "Z"),
            meshes=[Mesh3D.from_dict(m) for m in d.get("meshes", [])],
            metadata=d.get("metadata", {}),
        )

    def to_obj(self) -> str:
        """Serialize to a Wavefront OBJ string (universal 3D interchange).

        Groups meshes by name so building/terrain/road are distinguishable in
        any viewer.  1-based vertex indices per the OBJ spec.
        """
        lines = [f"# Tritium Scene3D AO={self.ao} origin=({self.origin_lat},"
                 f"{self.origin_lng}) up={self.up_axis}"]
        offset = 1
        for m in self.meshes:
            lines.append(f"g {m.kind}_{m.name}".replace(" ", "_"))
            for x, y, z in m.vertices:
                lines.append(f"v {x:.3f} {y:.3f} {z:.3f}")
            for a, b, c in m.faces:
                lines.append(f"f {a + offset} {b + offset} {c + offset}")
            offset += m.vertex_count
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Building extrusion.
# --------------------------------------------------------------------------- #

def extrude_footprint(
    polygon_en: list[tuple[float, float]],
    base_z: float,
    height: float,
    name: str = "building",
    category: str = "",
    color: Optional[tuple[float, float, float]] = None,
) -> Optional[Mesh3D]:
    """Extrude a (east, north) footprint into a prism from base_z to base_z+h.

    Produces a closed mesh: a triangulated roof, a triangulated floor, and a
    quad (two-triangle) wall per footprint edge.  Returns None for a
    degenerate footprint (<3 points).  Roof/floor use a simple fan
    triangulation — correct for the convex-ish footprints OSM emits; concave
    footprints still render as solid walls + a fan cap (good enough for a
    render twin, not a CAD model).
    """
    ring = _dedupe_ring(polygon_en)
    if len(ring) < 3:
        return None
    top_z = base_z + max(0.5, height)
    n = len(ring)
    verts: list[Vec3] = []
    for (e, nn) in ring:  # floor 0..n-1
        verts.append((e, nn, base_z))
    for (e, nn) in ring:  # roof n..2n-1
        verts.append((e, nn, top_z))
    faces: list[tuple[int, int, int]] = []
    # Roof fan (upward).
    for i in range(1, n - 1):
        faces.append((n + 0, n + i, n + i + 1))
    # Floor fan (downward, reversed winding).
    for i in range(1, n - 1):
        faces.append((0, i + 1, i))
    # Walls: for edge i -> i+1, quad (i, i+1, n+i+1, n+i).
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
    return Mesh3D(name=name, kind="building", vertices=verts, faces=faces,
                  height_m=height, category=category, color=color)


def _dedupe_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop a repeated closing vertex and consecutive duplicates."""
    out: list[tuple[float, float]] = []
    for p in ring:
        p = (float(p[0]), float(p[1]))
        if not out or (abs(out[-1][0] - p[0]) > 1e-6 or abs(out[-1][1] - p[1]) > 1e-6):
            out.append(p)
    if len(out) >= 2 and abs(out[0][0] - out[-1][0]) < 1e-6 and abs(out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
    return out


# --------------------------------------------------------------------------- #
# Terrain heightfield.
# --------------------------------------------------------------------------- #

def terrain_heightfield_mesh(
    grid,
    proj: LocalProjection,
    subsample: int = 1,
    name: str = "terrain",
) -> Optional[Mesh3D]:
    """Build a triangulated ground mesh from an ElevationGrid.

    ``grid`` is a :class:`tritium_lib.geo.gis.models.ElevationGrid` (row 0 =
    north).  Each cell centre becomes a vertex at (east, north, elevation) in
    AO-local metres; NoData cells fall back to the grid mean so the sheet stays
    watertight.  ``subsample`` thins a dense DEM (take every Nth cell).
    """
    ncols = getattr(grid, "ncols", 0)
    nrows = getattr(grid, "nrows", 0)
    if ncols < 2 or nrows < 2:
        return None
    step = max(1, int(subsample))
    present = [v for v in getattr(grid, "values", []) if v is not None]
    fill = (sum(present) / len(present)) if present else 0.0

    cols = list(range(0, ncols, step))
    rows = list(range(0, nrows, step))
    if cols[-1] != ncols - 1:
        cols.append(ncols - 1)
    if rows[-1] != nrows - 1:
        rows.append(nrows - 1)

    verts: list[Vec3] = []
    for iy in rows:
        lat = grid.cell_lat(iy)
        for ix in cols:
            lng = grid.cell_lon(ix)
            z = grid.value_at(ix, iy)
            if z is None:
                z = fill
            e, nn = proj.to_local(lat, lng)
            verts.append((e, nn, float(z)))

    w = len(cols)
    faces: list[tuple[int, int, int]] = []
    for r in range(len(rows) - 1):
        for c in range(w - 1):
            a = r * w + c
            b = a + 1
            d = a + w
            e2 = d + 1
            faces.append((a, d, b))
            faces.append((b, d, e2))
    return Mesh3D(name=name, kind="terrain", vertices=verts, faces=faces)


def make_elevation_sampler(grid, proj: LocalProjection):
    """Return f(east, north) -> elevation (nearest DEM cell; 0.0 if no grid)."""
    if grid is None or getattr(grid, "ncols", 0) < 1:
        return lambda e, n: 0.0
    present = [v for v in getattr(grid, "values", []) if v is not None]
    fill = (sum(present) / len(present)) if present else 0.0

    def _sample(east: float, north: float) -> float:
        lat, lng = proj.to_latlng(east, north)
        if grid.ncols > 1:
            fx = (lng - grid.west) / (grid.east - grid.west) * (grid.ncols - 1)
        else:
            fx = 0.0
        if grid.nrows > 1:
            fy = (grid.north - lat) / (grid.north - grid.south) * (grid.nrows - 1)
        else:
            fy = 0.0
        ix = min(grid.ncols - 1, max(0, round(fx)))
        iy = min(grid.nrows - 1, max(0, round(fy)))
        v = grid.value_at(int(ix), int(iy))
        return float(v) if v is not None else fill

    return _sample


# --------------------------------------------------------------------------- #
# GeoJSON buildings.
# --------------------------------------------------------------------------- #

def buildings_from_geojson(
    geojson: dict,
    proj: LocalProjection,
    elevation_sampler: Optional[Callable[[float, float], float]] = None,
    default_height: float = 8.0,
    max_buildings: int = 5000,
) -> list[Mesh3D]:
    """Extrude OSM-style GeoJSON building polygons into 3D meshes.

    Each ``Polygon`` feature's outer ring (lat/lng) is projected to AO-local
    metres and extruded from its ground elevation (sampled from the DEM) to
    ``height_m`` (or ``levels*3`` or ``default_height``).
    """
    meshes: list[Mesh3D] = []
    feats = (geojson or {}).get("features", []) if isinstance(geojson, dict) else []
    sample = elevation_sampler or (lambda e, n: 0.0)
    for feat in feats[:max_buildings]:
        geom = (feat or {}).get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates") or []
        if not coords:
            continue
        outer = coords[0]  # [ [lng, lat], ... ]
        ring_en: list[tuple[float, float]] = []
        for pt in outer:
            if len(pt) < 2:
                continue
            lng, lat = float(pt[0]), float(pt[1])
            ring_en.append(proj.to_local(lat, lng))
        if len(ring_en) < 3:
            continue
        props = feat.get("properties") or {}
        height = props.get("height_m")
        if not height and props.get("levels"):
            try:
                height = float(props["levels"]) * 3.0
            except (TypeError, ValueError):
                height = None
        height = float(height) if height else default_height
        # Ground elevation at the footprint centroid.
        cx = sum(p[0] for p in ring_en) / len(ring_en)
        cy = sum(p[1] for p in ring_en) / len(ring_en)
        base_z = sample(cx, cy)
        mesh = extrude_footprint(
            ring_en, base_z, height,
            name=(props.get("name") or f"b{len(meshes)}"),
            category=props.get("kind") or props.get("category") or "building",
        )
        if mesh is not None:
            meshes.append(mesh)
    return meshes


# --------------------------------------------------------------------------- #
# Top-level assembly.
# --------------------------------------------------------------------------- #

def build_scene3d(
    ao: str,
    bbox: tuple[float, float, float, float],  # (west, south, east, north)
    elevation_grid=None,
    buildings_geojson: Optional[dict] = None,
    terrain_subsample: int = 2,
    default_building_height: float = 8.0,
) -> Scene3D:
    """Assemble a Scene3D for an AO from shared GIS layers.

    Args:
        ao: AO id (e.g. "dublin").
        bbox: (west, south, east, north) — the origin is its centre.
        elevation_grid: an ElevationGrid (or None -> flat ground at z=0).
        buildings_geojson: OSM-style FeatureCollection (or None).
        terrain_subsample: thin the DEM by this factor for the terrain sheet.
        default_building_height: fallback height when a footprint has none.
    """
    west, south, east, north = bbox
    proj = LocalProjection((south + north) / 2.0, (west + east) / 2.0)
    scene = Scene3D(ao=ao, origin_lat=proj.origin_lat, origin_lng=proj.origin_lng)

    if elevation_grid is not None:
        terrain = terrain_heightfield_mesh(elevation_grid, proj, terrain_subsample)
        if terrain is not None:
            scene.add(terrain)
    sampler = make_elevation_sampler(elevation_grid, proj)

    if buildings_geojson is not None:
        for m in buildings_from_geojson(
            buildings_geojson, proj, sampler, default_building_height
        ):
            scene.add(m)

    scene.metadata["bbox"] = [west, south, east, north]
    scene.metadata["projection"] = "equirectangular_ao_local"
    return scene
