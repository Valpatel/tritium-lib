# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for water-crossing chokepoints (roads x NHD hydrography -> bridges).

Pure geometry, no network. Synthetic fixtures pin the exact crossing points and
kind inference; the packaged Boulder TIGER + NHD fixtures prove real roads
crossing the real drainage network produce bridge chokepoints.
"""

import json
from pathlib import Path

from tritium_lib.geo.gis.chokepoints import (
    chokepoint_tactical_object,
    find_water_crossings,
    infer_crossing_kind,
    meters_between,
    segment_intersection,
)

_FIX = Path(__file__).resolve().parents[2] / (
    "src/tritium_lib/geo/gis/fixtures"
)


def _road(coords, name="Test Rd", kind="S1400"):
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {"source": "tiger", "kind": kind, "name": name},
    }


def _flow(coords, name="Test Creek", kind="river"):
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {"source": "nhd", "kind": kind, "name": name},
    }


def _fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


# ---------------------------------------------------------------------------
# segment_intersection — the geometric core
# ---------------------------------------------------------------------------
class TestSegmentIntersection:
    def test_perpendicular_cross_at_origin(self):
        pt = segment_intersection((-1, 0), (1, 0), (0, -1), (0, 1))
        assert pt is not None
        assert abs(pt[0]) < 1e-12 and abs(pt[1]) < 1e-12

    def test_x_cross_midpoint(self):
        pt = segment_intersection((0, 0), (2, 2), (0, 2), (2, 0))
        assert pt == (1.0, 1.0)

    def test_parallel_returns_none(self):
        assert segment_intersection((0, 0), (1, 0), (0, 1), (1, 1)) is None

    def test_collinear_returns_none(self):
        assert segment_intersection((0, 0), (1, 0), (2, 0), (3, 0)) is None

    def test_non_overlapping_returns_none(self):
        # Lines would cross if extended, but the segments do not reach.
        assert segment_intersection((0, 0), (1, 1), (5, 0), (5, 10)) is None

    def test_touch_at_endpoint_counts(self):
        pt = segment_intersection((0, 0), (1, 0), (1, 0), (1, 1))
        assert pt == (1.0, 0.0)


# ---------------------------------------------------------------------------
# infer_crossing_kind — table-driven classification
# ---------------------------------------------------------------------------
class TestInferKind:
    def test_real_road_over_river_is_bridge(self):
        assert infer_crossing_kind("S1400", "river") == "bridge"
        assert infer_crossing_kind("S1100", "stream") == "bridge"

    def test_real_road_over_canal_is_culvert(self):
        assert infer_crossing_kind("S1400", "canal") == "culvert"
        assert infer_crossing_kind("S1200", "artificial") == "culvert"

    def test_trail_over_river_is_ford(self):
        assert infer_crossing_kind("S1500", "river") == "ford"
        assert infer_crossing_kind("S1830", "stream") == "ford"

    def test_trail_over_canal_is_ford(self):
        assert infer_crossing_kind("S1710", "canal") == "ford"

    def test_waterbody_defaults_bridge(self):
        assert infer_crossing_kind("S1400", "waterbody") == "bridge"

    def test_none_inputs_default_bridge(self):
        assert infer_crossing_kind(None, None) == "bridge"

    def test_case_insensitive(self):
        assert infer_crossing_kind("s1500", "RIVER") == "ford"


# ---------------------------------------------------------------------------
# find_water_crossings — the public API on synthetic geometry
# ---------------------------------------------------------------------------
class TestFindCrossings:
    def test_single_crossing_point_and_props(self):
        roads = _fc(_road([[-1, 0], [1, 0]], name="Main St", kind="S1400"))
        hydro = _fc(_flow([[0, -1], [0, 1]], name="Boulder Creek", kind="river"))
        fc = find_water_crossings(roads, hydro)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 1
        feat = fc["features"][0]
        assert feat["geometry"]["type"] == "Point"
        lon, lat = feat["geometry"]["coordinates"]
        assert abs(lon) < 1e-9 and abs(lat) < 1e-9
        p = feat["properties"]
        assert p["source"] == "chokepoint"
        assert p["kind"] == "bridge"
        assert p["road_name"] == "Main St"
        assert p["water_name"] == "Boulder Creek"
        assert p["name"] == "Main St @ Boulder Creek"
        assert p["id"].startswith("chk_")

    def test_no_crossing_returns_empty(self):
        roads = _fc(_road([[-1, 0], [1, 0]]))
        hydro = _fc(_flow([[0, 5], [0, 6]]))  # water far north — no crossing
        fc = find_water_crossings(roads, hydro)
        assert fc["features"] == []

    def test_two_distinct_roads_two_bridges(self):
        roads = _fc(
            _road([[-1, 0], [1, 0]], name="A St", kind="S1400"),
            _road([[-1, 0.5], [1, 0.5]], name="B St", kind="S1400"),
        )
        hydro = _fc(_flow([[0, -1], [0, 2]], name="River", kind="river"))
        fc = find_water_crossings(roads, hydro)
        assert len(fc["features"]) == 2
        names = sorted(f["properties"]["road_name"] for f in fc["features"])
        assert names == ["A St", "B St"]

    def test_dedupe_collapses_near_duplicate(self):
        # A road that weaves across the same stream twice within ~5 m collapses
        # to a single bridge; without dedupe it would be two.
        roads = _fc(_road(
            [[-0.0001, -0.00002], [0.0, 0.00002], [0.0001, -0.00002]],
            name="Wiggle Rd",
        ))
        hydro = _fc(_flow([[-0.001, 0.0], [0.001, 0.0]], name="Brook"))
        merged = find_water_crossings(roads, hydro, dedupe_m=50.0)
        assert len(merged["features"]) == 1
        split = find_water_crossings(roads, hydro, dedupe_m=0.0)
        assert len(split["features"]) == 2

    def test_bbox_clip(self):
        roads = _fc(
            _road([[-1, 0], [1, 0]], name="In St"),
            _road([[9, 0], [11, 0]], name="Out St"),
        )
        hydro = _fc(
            _flow([[0, -1], [0, 1]], name="Near"),
            _flow([[10, -1], [10, 1]], name="Far"),
        )
        fc = find_water_crossings(roads, hydro, bbox=(-2, -2, 2, 2))
        assert len(fc["features"]) == 1
        assert fc["features"][0]["properties"]["road_name"] == "In St"

    def test_waterbody_polygon_edge_crossing(self):
        # Road crossing a lake polygon crosses its boundary -> bridge(s).
        roads = _fc(_road([[-2, 0], [2, 0]], name="Causeway", kind="S1400"))
        lake = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
            },
            "properties": {"source": "nhd", "kind": "waterbody", "name": "Lake"},
        }
        fc = find_water_crossings(roads, _fc(lake))
        assert len(fc["features"]) >= 1
        assert all(f["properties"]["kind"] == "bridge" for f in fc["features"])
        assert fc["features"][0]["properties"]["water_name"] == "Lake"

    def test_deterministic_ordering(self):
        roads = _fc(
            _road([[-1, 0.5], [1, 0.5]], name="B St"),
            _road([[-1, 0], [1, 0]], name="A St"),
        )
        hydro = _fc(_flow([[0, -1], [0, 2]], name="River"))
        a = find_water_crossings(roads, hydro)
        b = find_water_crossings(roads, hydro)
        assert a == b
        # sorted by lon then lat -> both at lon 0, lat 0.0 before 0.5
        lats = [f["geometry"]["coordinates"][1] for f in a["features"]]
        assert lats == sorted(lats)

    def test_empty_and_malformed_inputs(self):
        assert find_water_crossings(None, None)["features"] == []
        assert find_water_crossings({}, {})["features"] == []
        assert find_water_crossings(
            {"features": "nope"}, {"features": None}
        )["features"] == []


# ---------------------------------------------------------------------------
# meters_between — dedupe distance sanity
# ---------------------------------------------------------------------------
def test_meters_between_scale():
    # ~0.001 deg lat ~= 110 m.
    d = meters_between((-105.0, 40.0), (-105.0, 40.001))
    assert 105.0 < d < 115.0


# ---------------------------------------------------------------------------
# Real packaged fixtures — Boulder roads x NHD drainage network
# ---------------------------------------------------------------------------
class TestBoulderFixtures:
    def _load(self, name):
        return json.loads((_FIX / name).read_text())

    def test_boulder_roads_cross_nhd_network(self):
        roads = self._load("tiger_roads_boulder.json")
        hydro = self._load("nhd_hydro_boulder.json")
        fc = find_water_crossings(roads, hydro)
        feats = fc["features"]
        # The Boulder foothills drainage is crossed by the street grid many
        # times — there must be a healthy set of real bridges.
        assert len(feats) >= 3, f"expected real crossings, got {len(feats)}"
        for f in feats:
            p = f["properties"]
            assert p["source"] == "chokepoint"
            assert p["kind"] in ("bridge", "ford", "culvert")
            assert p["id"].startswith("chk_")
            # Crossings sit in/around the Boulder AO (fixtures extend a little
            # past the nominal bbox where a road/stream runs off-edge).
            lon, lat = f["geometry"]["coordinates"]
            assert -105.32 <= lon <= -105.24
            assert 39.96 <= lat <= 40.04
        # Deterministic across runs.
        assert find_water_crossings(roads, hydro) == fc

    def test_boulder_ids_unique(self):
        roads = self._load("tiger_roads_boulder.json")
        hydro = self._load("nhd_hydro_boulder.json")
        feats = find_water_crossings(roads, hydro)["features"]
        ids = [f["properties"]["id"] for f in feats]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# chokepoint_tactical_object — the production/costmap contract
# ---------------------------------------------------------------------------
class TestTacticalObject:
    def _feat(self, **props):
        base = {
            "source": "chokepoint",
            "id": "chk_abc",
            "kind": "bridge",
            "name": "x",
            "road_name": "Main",
            "road_kind": "S1100",
            "water_name": "River",
            "water_kind": "river",
        }
        base.update(props)
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [-105.1, 40.0]},
            "properties": base,
        }

    def test_primary_river_bridge_is_key_terrain(self):
        obj = chokepoint_tactical_object(self._feat())
        assert obj["kind"] == "bridge"
        assert obj["sever"] is True
        assert obj["hold_value"] >= 6
        assert "key_terrain" in obj["tags"]
        assert "severable" in obj["tags"]
        assert obj["position"] == {"lon": -105.1, "lat": 40.0}

    def test_culvert_over_ditch_not_severable(self):
        obj = chokepoint_tactical_object(
            self._feat(kind="culvert", road_kind="S1400", water_kind="canal")
        )
        assert obj["sever"] is False
        assert "severable" not in obj["tags"]

    def test_hold_value_clamped(self):
        obj = chokepoint_tactical_object(
            self._feat(road_kind="S1740", water_kind="artificial", kind="culvert")
        )
        assert 1 <= obj["hold_value"] <= 10
