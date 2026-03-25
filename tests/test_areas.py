# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.areas — geographic area management."""

from __future__ import annotations

import json
import time

import pytest

from tritium_lib.areas import (
    Area,
    AreaClassifier,
    AreaHierarchy,
    AreaManager,
    AreaStats,
    AreaType,
)


# ---------------------------------------------------------------------------
# Helpers — reusable polygon fixtures
# ---------------------------------------------------------------------------

def _square_area(
    area_id: str = "sq1",
    name: str = "Square",
    center_lat: float = 40.0,
    center_lng: float = -74.0,
    half_deg: float = 0.001,
    area_type: str = "zone",
) -> Area:
    """Create a small square area centred on (center_lat, center_lng)."""
    return Area(
        area_id=area_id,
        name=name,
        polygon=[
            (center_lat - half_deg, center_lng - half_deg),
            (center_lat + half_deg, center_lng - half_deg),
            (center_lat + half_deg, center_lng + half_deg),
            (center_lat - half_deg, center_lng + half_deg),
        ],
        area_type=area_type,
    )


def _triangle_area(area_id: str = "tri1", name: str = "Triangle") -> Area:
    return Area(
        area_id=area_id,
        name=name,
        polygon=[
            (40.0, -74.0),
            (40.002, -74.0),
            (40.001, -73.998),
        ],
        area_type="zone",
    )


# ===================================================================
# Area dataclass tests
# ===================================================================

class TestArea:
    def test_auto_id(self) -> None:
        a = Area(name="Test")
        assert a.area_id.startswith("area_")
        assert len(a.area_id) > 5

    def test_explicit_id(self) -> None:
        a = Area(area_id="my_id", name="Test")
        assert a.area_id == "my_id"

    def test_centroid(self) -> None:
        a = _square_area()
        lat, lng = a.centroid
        assert abs(lat - 40.0) < 1e-6
        assert abs(lng - (-74.0)) < 1e-6

    def test_centroid_empty(self) -> None:
        a = Area(name="Empty")
        assert a.centroid == (0.0, 0.0)

    def test_area_sq_meters(self) -> None:
        a = _square_area(half_deg=0.001)
        # ~0.002 deg side => ~222m x ~170m at lat 40 => ~37,000 m^2
        sq = a.area_sq_meters
        assert 10_000 < sq < 100_000

    def test_bbox(self) -> None:
        a = _square_area()
        mn_lat, mn_lng, mx_lat, mx_lng = a.bbox
        assert mn_lat < mx_lat
        assert mn_lng < mx_lng

    def test_bbox_empty(self) -> None:
        a = Area(name="Empty")
        assert a.bbox == (0.0, 0.0, 0.0, 0.0)

    def test_contains_point_inside(self) -> None:
        a = _square_area()
        assert a.contains_point(40.0, -74.0) is True

    def test_contains_point_outside(self) -> None:
        a = _square_area()
        assert a.contains_point(41.0, -74.0) is False

    def test_overlaps_true(self) -> None:
        a = _square_area(area_id="a")
        b = _square_area(area_id="b", center_lat=40.0005)
        assert a.overlaps(b) is True

    def test_overlaps_false(self) -> None:
        a = _square_area(area_id="a")
        b = _square_area(area_id="b", center_lat=41.0)
        assert a.overlaps(b) is False

    def test_perimeter_meters(self) -> None:
        a = _square_area()
        perim = a.perimeter_meters
        # Square with ~0.002 deg sides => ~880m total perimeter
        assert 400 < perim < 2000

    def test_perimeter_empty(self) -> None:
        a = Area(name="Empty")
        assert a.perimeter_meters == 0.0

    def test_to_dict_roundtrip(self) -> None:
        a = _square_area(area_id="rt1", name="Roundtrip")
        a.tags = ["test", "zone"]
        a.properties = {"color": "red"}
        d = a.to_dict()
        b = Area.from_dict(d)
        assert b.area_id == "rt1"
        assert b.name == "Roundtrip"
        assert len(b.polygon) == 4
        assert b.tags == ["test", "zone"]
        assert b.properties == {"color": "red"}

    def test_geojson_roundtrip(self) -> None:
        a = _square_area(area_id="gj1", name="GeoJSON Test")
        a.tags = ["demo"]
        feat = a.to_geojson_feature()
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Polygon"
        # GeoJSON coords are [lng, lat]
        first_coord = feat["geometry"]["coordinates"][0][0]
        assert first_coord[0] == a.polygon[0][1]  # lng
        assert first_coord[1] == a.polygon[0][0]  # lat

        b = Area.from_geojson_feature(feat)
        assert b.area_id == "gj1"
        assert b.name == "GeoJSON Test"
        assert len(b.polygon) == 4
        # Verify round-trip accuracy
        for orig, restored in zip(a.polygon, b.polygon):
            assert abs(orig[0] - restored[0]) < 1e-10
            assert abs(orig[1] - restored[1]) < 1e-10


# ===================================================================
# AreaHierarchy tests
# ===================================================================

class TestAreaHierarchy:
    def test_set_parent(self) -> None:
        h = AreaHierarchy()
        h.set_parent("bldg_a", "campus")
        assert h.parent("bldg_a") == "campus"
        assert "bldg_a" in h.children("campus")

    def test_ancestors(self) -> None:
        h = AreaHierarchy()
        h.set_parent("room1", "floor1")
        h.set_parent("floor1", "bldg_a")
        h.set_parent("bldg_a", "campus")
        anc = h.ancestors("room1")
        assert anc == ["floor1", "bldg_a", "campus"]

    def test_descendants(self) -> None:
        h = AreaHierarchy()
        h.set_parent("floor1", "campus")
        h.set_parent("room1", "floor1")
        h.set_parent("room2", "floor1")
        desc = h.descendants("campus")
        assert "floor1" in desc
        assert "room1" in desc
        assert "room2" in desc
        assert len(desc) == 3

    def test_roots(self) -> None:
        h = AreaHierarchy()
        h.set_parent("bldg_a", "campus")
        h.set_parent("floor1", "bldg_a")
        roots = h.roots()
        assert roots == ["campus"]

    def test_depth(self) -> None:
        h = AreaHierarchy()
        h.set_parent("room1", "floor1")
        h.set_parent("floor1", "bldg_a")
        assert h.depth("bldg_a") == 0
        assert h.depth("floor1") == 1
        assert h.depth("room1") == 2

    def test_cycle_detection(self) -> None:
        h = AreaHierarchy()
        h.set_parent("b", "a")
        with pytest.raises(ValueError, match="cycle"):
            h.set_parent("a", "b")

    def test_self_parent_rejected(self) -> None:
        h = AreaHierarchy()
        with pytest.raises(ValueError, match="own parent"):
            h.set_parent("a", "a")

    def test_remove(self) -> None:
        h = AreaHierarchy()
        h.set_parent("floor1", "campus")
        h.set_parent("room1", "floor1")
        h.remove("floor1")
        assert h.parent("floor1") is None
        assert h.parent("room1") is None
        assert "floor1" not in h.children("campus")

    def test_reparent(self) -> None:
        h = AreaHierarchy()
        h.set_parent("room1", "floor_a")
        h.set_parent("room1", "floor_b")
        assert h.parent("room1") == "floor_b"
        assert "room1" not in h.children("floor_a")
        assert "room1" in h.children("floor_b")

    def test_serialization_roundtrip(self) -> None:
        h = AreaHierarchy()
        h.set_parent("floor1", "campus")
        h.set_parent("room1", "floor1")
        d = h.to_dict()
        h2 = AreaHierarchy.from_dict(d)
        assert h2.parent("floor1") == "campus"
        assert h2.parent("room1") == "floor1"
        assert "room1" in h2.descendants("campus")


# ===================================================================
# AreaStats tests
# ===================================================================

class TestAreaStats:
    def test_entry_exit(self) -> None:
        s = AreaStats(area_id="z1")
        s.record_entry("tgt_1", timestamp=1000.0)
        assert s.target_count == 1
        assert s.total_entries == 1
        assert "tgt_1" in s.active_target_ids
        assert s.peak_occupancy == 1

        s.record_exit("tgt_1", timestamp=1060.0)
        assert s.target_count == 0
        assert s.total_exits == 1
        assert s.avg_dwell_seconds == 60.0

    def test_peak_occupancy(self) -> None:
        s = AreaStats(area_id="z2")
        s.record_entry("a", timestamp=100.0)
        s.record_entry("b", timestamp=101.0)
        s.record_entry("c", timestamp=102.0)
        assert s.peak_occupancy == 3
        s.record_exit("b", timestamp=103.0)
        assert s.peak_occupancy == 3  # peak preserved

    def test_to_dict(self) -> None:
        s = AreaStats(area_id="z3")
        s.record_entry("tgt_1", timestamp=100.0)
        d = s.to_dict()
        assert d["area_id"] == "z3"
        assert d["target_count"] == 1
        assert "tgt_1" in d["active_target_ids"]


# ===================================================================
# AreaClassifier tests
# ===================================================================

class TestAreaClassifier:
    def test_name_keyword_residential(self) -> None:
        c = AreaClassifier()
        a = Area(name="Apartment Complex", polygon=[(0, 0), (0, 1), (1, 1)])
        assert c.classify(a) == "residential"

    def test_name_keyword_commercial(self) -> None:
        c = AreaClassifier()
        a = Area(name="Shopping Mall", polygon=[(0, 0), (0, 1), (1, 1)])
        assert c.classify(a) == "commercial"

    def test_name_keyword_military(self) -> None:
        c = AreaClassifier()
        a = Area(name="Fort Hamilton", polygon=[(0, 0), (0, 1), (1, 1)])
        assert c.classify(a) == "military"

    def test_tag_match(self) -> None:
        c = AreaClassifier()
        a = Area(name="Sector 7", polygon=[(0, 0), (0, 1), (1, 1)],
                 tags=["industrial"])
        assert c.classify(a) == "industrial"

    def test_size_heuristic_room(self) -> None:
        c = AreaClassifier()
        # Very small polygon — should be classified as "room"
        a = Area(
            name="Zone X",
            polygon=[
                (40.0, -74.0),
                (40.00005, -74.0),
                (40.00005, -73.99995),
                (40.0, -73.99995),
            ],
        )
        result = c.classify(a)
        assert result in ("room", "zone")  # depends on exact area calc

    def test_custom_rule(self) -> None:
        c = AreaClassifier()
        c.add_rule("always_water", lambda a: "water" if "wet" in a.name.lower() else None)
        a = Area(name="Wet Marsh", polygon=[(0, 0), (0, 1), (1, 1)])
        # Custom rule added after defaults — but name keywords take priority
        # "Wet Marsh" doesn't match any default keyword, so custom fires
        assert c.classify(a) == "water"

    def test_fallback_other(self) -> None:
        c = AreaClassifier()
        a = Area(name="??", polygon=[])
        assert c.classify(a) == "other"


# ===================================================================
# AreaManager tests
# ===================================================================

class TestAreaManager:
    def test_create_and_get(self) -> None:
        mgr = AreaManager()
        a = mgr.create(_square_area(area_id="test1"))
        assert mgr.get("test1") is a
        assert mgr.count() == 1

    def test_update(self) -> None:
        mgr = AreaManager()
        a = mgr.create(_square_area(area_id="u1", name="Original"))
        a.name = "Updated"
        result = mgr.update(a)
        assert result is not None
        assert mgr.get("u1").name == "Updated"

    def test_update_nonexistent(self) -> None:
        mgr = AreaManager()
        a = Area(area_id="ghost", name="Ghost")
        assert mgr.update(a) is None

    def test_delete(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="d1"))
        assert mgr.delete("d1") is True
        assert mgr.get("d1") is None
        assert mgr.count() == 0

    def test_delete_nonexistent(self) -> None:
        mgr = AreaManager()
        assert mgr.delete("nope") is False

    def test_list_filter_type(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="r1", area_type="residential"))
        mgr.create(_square_area(area_id="c1", area_type="commercial"))
        res = mgr.list_areas(area_type="residential")
        assert len(res) == 1
        assert res[0].area_id == "r1"

    def test_list_filter_tag(self) -> None:
        mgr = AreaManager()
        a = _square_area(area_id="t1")
        a.tags = ["important"]
        mgr.create(a)
        mgr.create(_square_area(area_id="t2"))
        res = mgr.list_areas(tag="important")
        assert len(res) == 1
        assert res[0].area_id == "t1"

    def test_areas_containing(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="sq1"))
        mgr.create(_square_area(area_id="sq2", center_lat=41.0))
        results = mgr.areas_containing(40.0, -74.0)
        ids = [a.area_id for a in results]
        assert "sq1" in ids
        assert "sq2" not in ids

    def test_areas_near(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="near1", center_lat=40.0))
        mgr.create(_square_area(area_id="far1", center_lat=41.0))
        results = mgr.areas_near(40.0, -74.0, 1000.0)
        ids = [a.area_id for a, _ in results]
        assert "near1" in ids
        assert "far1" not in ids

    def test_find_overlaps(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="o1", center_lat=40.0))
        mgr.create(_square_area(area_id="o2", center_lat=40.0005))
        mgr.create(_square_area(area_id="o3", center_lat=42.0))
        overlaps = mgr.find_overlaps()
        pair_ids = [(a, b) for a, b in overlaps]
        assert ("o1", "o2") in pair_ids or ("o2", "o1") in pair_ids
        # o3 should not overlap with either
        for a, b in pair_ids:
            assert "o3" not in (a, b)

    def test_classify(self) -> None:
        mgr = AreaManager()
        a = _square_area(area_id="cl1", name="Shopping District")
        mgr.create(a)
        assert mgr.classify("cl1") == "commercial"

    def test_auto_classify_all(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="ac1", name="Fort Knox"))
        mgr.create(_square_area(area_id="ac2", name="City Park"))
        results = mgr.auto_classify_all()
        assert results["ac1"] == "military"
        assert results["ac2"] == "park"
        assert mgr.get("ac1").area_type == "military"
        assert mgr.get("ac2").area_type == "park"

    def test_stats_tracking(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="st1"))
        # Target enters the area
        inside = mgr.record_target_position("tgt_a", 40.0, -74.0, timestamp=1000.0)
        assert "st1" in inside
        st = mgr.get_stats("st1")
        assert st.target_count == 1
        assert st.total_entries == 1

        # Target exits
        inside = mgr.record_target_position("tgt_a", 41.0, -74.0, timestamp=1060.0)
        assert "st1" not in inside
        assert st.total_exits == 1
        assert st.avg_dwell_seconds == 60.0

    def test_geojson_roundtrip(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="gj1", name="Zone A"))
        mgr.create(_triangle_area(area_id="gj2", name="Zone B"))
        geojson = mgr.to_geojson()
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 2

        mgr2 = AreaManager()
        imported = mgr2.from_geojson(geojson)
        assert len(imported) == 2
        assert mgr2.get("gj1") is not None
        assert mgr2.get("gj2") is not None

    def test_json_export_import(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="ji1", name="Alpha"))
        mgr.create(_square_area(area_id="ji2", name="Beta"))
        mgr.hierarchy.set_parent("ji2", "ji1")
        json_str = mgr.export_json()

        mgr2 = AreaManager()
        count = mgr2.import_json(json_str)
        assert count == 2
        assert mgr2.get("ji1") is not None
        assert mgr2.hierarchy.parent("ji2") == "ji1"

    def test_hierarchy_integration(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="campus", name="Campus", half_deg=0.01))
        mgr.create(_square_area(area_id="bldg_a", name="Building A", half_deg=0.003))
        mgr.create(_square_area(area_id="room_101", name="Room 101", half_deg=0.0005))
        mgr.hierarchy.set_parent("bldg_a", "campus")
        mgr.hierarchy.set_parent("room_101", "bldg_a")

        assert mgr.hierarchy.ancestors("room_101") == ["bldg_a", "campus"]
        assert "room_101" in mgr.hierarchy.descendants("campus")

    def test_delete_cleans_hierarchy(self) -> None:
        mgr = AreaManager()
        mgr.create(_square_area(area_id="p"))
        mgr.create(_square_area(area_id="c"))
        mgr.hierarchy.set_parent("c", "p")
        mgr.delete("c")
        assert "c" not in mgr.hierarchy.children("p")


# ===================================================================
# AreaType enum tests
# ===================================================================

class TestAreaType:
    def test_values(self) -> None:
        assert AreaType.RESIDENTIAL.value == "residential"
        assert AreaType.MILITARY.value == "military"
        assert AreaType.CAMPUS.value == "campus"

    def test_string_comparison(self) -> None:
        assert AreaType.COMMERCIAL == "commercial"
