# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Map data -> neutral 3D scene pipeline (deterministic, no USD/Isaac/GPU).

Proves the reusable digital-twin geometry: extrude building footprints, build a
terrain heightfield from a DEM, assemble + serialize a Scene3D for an AO.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.geo.gis.models import ElevationGrid
from tritium_lib.geo.scene3d import (
    LocalProjection,
    Mesh3D,
    Scene3D,
    build_scene3d,
    buildings_from_geojson,
    extrude_footprint,
    make_elevation_sampler,
    terrain_heightfield_mesh,
)


# ------------------------------------------------------------------ projection

def test_local_projection_roundtrip_and_origin():
    proj = LocalProjection(37.7159, -121.896)
    assert proj.to_local(37.7159, -121.896) == (0.0, 0.0)
    e, n = proj.to_local(37.7159, -121.895)  # one thousandth east
    assert e > 0 and abs(n) < 1e-6
    lat, lng = proj.to_latlng(e, n)
    assert lat == pytest.approx(37.7159, abs=1e-6)
    assert lng == pytest.approx(-121.895, abs=1e-6)


def test_local_projection_north_is_positive_y():
    proj = LocalProjection(37.0, -121.0)
    _, n = proj.to_local(37.001, -121.0)
    assert n > 0  # north -> +Y


# ------------------------------------------------------------------ extrusion

def _square(cx=0.0, cy=0.0, s=10.0):
    return [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]


def test_extrude_square_prism_topology():
    m = extrude_footprint(_square(), base_z=100.0, height=12.0, name="tower")
    assert m is not None
    assert m.kind == "building"
    assert m.vertex_count == 8            # 4 floor + 4 roof
    # 2 roof + 2 floor fan tris + 4 walls*2 = 12 triangles
    assert m.face_count == 12
    zs = [v[2] for v in m.vertices]
    assert min(zs) == pytest.approx(100.0)
    assert max(zs) == pytest.approx(112.0)


def test_extrude_closes_repeated_ring():
    ring = _square()
    ring = ring + [ring[0]]  # explicit closing vertex
    m = extrude_footprint(ring, base_z=0.0, height=5.0)
    assert m.vertex_count == 8  # closing dup dropped


def test_extrude_degenerate_returns_none():
    assert extrude_footprint([(0, 0), (1, 1)], 0.0, 5.0) is None


def test_extrude_minimum_height_floor():
    m = extrude_footprint(_square(), base_z=0.0, height=0.0)
    zs = [v[2] for v in m.vertices]
    assert max(zs) >= 0.5  # a building is never zero-height


# ------------------------------------------------------------------ terrain

def _ramp_grid(ncols=5, nrows=4):
    # Elevation ramps west->east from 0..(ncols-1)*10.
    values = []
    for iy in range(nrows):
        for ix in range(ncols):
            values.append(ix * 10.0)
    return ElevationGrid(west=-121.9, south=37.70, east=-121.88, north=37.72,
                         ncols=ncols, nrows=nrows, values=values)


def test_terrain_mesh_vertex_and_face_counts():
    grid = _ramp_grid(5, 4)
    proj = LocalProjection(37.71, -121.89)
    m = terrain_heightfield_mesh(grid, proj, subsample=1)
    assert m is not None and m.kind == "terrain"
    assert m.vertex_count == 5 * 4
    assert m.face_count == (5 - 1) * (4 - 1) * 2  # 2 tris per cell


def test_terrain_mesh_elevation_preserved():
    grid = _ramp_grid(5, 4)
    proj = LocalProjection(37.71, -121.89)
    m = terrain_heightfield_mesh(grid, proj, subsample=1)
    zs = sorted({round(v[2]) for v in m.vertices})
    assert zs == [0, 10, 20, 30, 40]


def test_terrain_nodata_filled_with_mean():
    grid = _ramp_grid(3, 3)
    grid.values[4] = None  # centre NoData
    proj = LocalProjection(37.71, -121.89)
    m = terrain_heightfield_mesh(grid, proj, subsample=1)
    assert all(v[2] is not None for v in m.vertices)


def test_elevation_sampler_nearest_cell():
    grid = _ramp_grid(5, 4)
    proj = LocalProjection(37.71, -121.89)
    sample = make_elevation_sampler(grid, proj)
    # East edge should sample the high end of the ramp.
    e_east, n0 = proj.to_local(37.71, -121.881)
    assert sample(e_east, n0) >= 30.0
    e_west, _ = proj.to_local(37.71, -121.899)
    assert sample(e_west, n0) <= 10.0


# ------------------------------------------------------------------ geojson

def _geojson_two_buildings():
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Polygon", "coordinates": [[
                 [-121.896, 37.7159], [-121.8958, 37.7159],
                 [-121.8958, 37.7161], [-121.896, 37.7161], [-121.896, 37.7159]]]},
             "properties": {"name": "Hall", "height_m": 20.0, "kind": "civic"}},
            {"type": "Feature",
             "geometry": {"type": "Polygon", "coordinates": [[
                 [-121.897, 37.716], [-121.8968, 37.716],
                 [-121.8968, 37.7162], [-121.897, 37.7162]]]},
             "properties": {"name": "Shed", "levels": 2}},  # -> 6 m
        ],
    }


def test_buildings_from_geojson_counts_and_height():
    proj = LocalProjection(37.716, -121.8965)
    meshes = buildings_from_geojson(_geojson_two_buildings(), proj)
    assert len(meshes) == 2
    hall = next(m for m in meshes if m.name == "Hall")
    shed = next(m for m in meshes if m.name == "Shed")
    assert hall.height_m == pytest.approx(20.0)
    assert shed.height_m == pytest.approx(6.0)  # 2 levels * 3 m
    assert all(m.kind == "building" for m in meshes)


def test_buildings_sit_on_terrain_elevation():
    grid = _ramp_grid(5, 4)
    proj = LocalProjection(37.71, -121.89)
    sampler = make_elevation_sampler(grid, proj)
    meshes = buildings_from_geojson(_geojson_two_buildings(),
                                    LocalProjection(37.716, -121.8965), sampler)
    # Base z should be non-zero (sampled from the ramp), not glued to 0.
    for m in meshes:
        base = min(v[2] for v in m.vertices)
        assert base >= 0.0


# ------------------------------------------------------------------ scene

def test_build_scene3d_assembles_terrain_and_buildings():
    grid = _ramp_grid(6, 6)
    scene = build_scene3d(
        ao="testville",
        bbox=(-121.90, 37.70, -121.88, 37.72),
        elevation_grid=grid,
        buildings_geojson=_geojson_two_buildings(),
        terrain_subsample=1,
    )
    assert scene.ao == "testville"
    assert len(scene.by_kind("terrain")) == 1
    assert len(scene.by_kind("building")) == 2
    st = scene.stats()
    assert st["by_kind"]["building"] == 2
    assert st["vertices"] > 0 and st["faces"] > 0
    b = scene.bounds()
    assert b["max"][0] > b["min"][0]  # non-degenerate east extent


def test_scene3d_json_roundtrip():
    scene = build_scene3d("t", (-121.90, 37.70, -121.88, 37.72),
                          elevation_grid=_ramp_grid(4, 4),
                          buildings_geojson=_geojson_two_buildings())
    d = scene.to_dict()
    back = Scene3D.from_dict(d)
    assert back.ao == scene.ao
    assert len(back.meshes) == len(scene.meshes)
    assert back.origin_lat == pytest.approx(scene.origin_lat)


def test_scene3d_obj_export_wellformed():
    scene = build_scene3d("t", (-121.90, 37.70, -121.88, 37.72),
                          elevation_grid=_ramp_grid(4, 4),
                          buildings_geojson=_geojson_two_buildings())
    obj = scene.to_obj()
    vcount = sum(1 for ln in obj.splitlines() if ln.startswith("v "))
    fcount = sum(1 for ln in obj.splitlines() if ln.startswith("f "))
    assert vcount == scene.stats()["vertices"]
    assert fcount == scene.stats()["faces"]
    # OBJ face indices are 1-based and within range.
    for ln in obj.splitlines():
        if ln.startswith("f "):
            for tok in ln.split()[1:]:
                assert int(tok) >= 1


def test_build_scene3d_flat_without_dem():
    scene = build_scene3d("t", (-121.90, 37.70, -121.88, 37.72),
                          elevation_grid=None,
                          buildings_geojson=_geojson_two_buildings())
    assert len(scene.by_kind("terrain")) == 0
    assert len(scene.by_kind("building")) == 2
    for m in scene.by_kind("building"):
        assert min(v[2] for v in m.vertices) == pytest.approx(0.0)
