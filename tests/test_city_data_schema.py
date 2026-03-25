# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Phase 1.5 quality gate tests for CityDataSchema.

Validates data integrity, height estimation accuracy, building
classification, and schema validation against malformed data.
"""

import math

import pytest
from pydantic import ValidationError

from tritium_lib.models.city import (
    BUILDING_CATEGORIES,
    BUILDING_TYPE_HEIGHTS,
    CITY_DATA_SCHEMA_VERSION,
    ROAD_WIDTHS,
    CityBarrier,
    CityBuilding,
    CityData,
    CityDataStats,
    CityEntrance,
    CityLanduse,
    CityPOI,
    CityRoad,
    CityTree,
    CityWater,
    classify_building,
    estimate_building_height,
    get_road_width,
)


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_version_is_integer(self):
        assert isinstance(CITY_DATA_SCHEMA_VERSION, int)

    def test_version_positive(self):
        assert CITY_DATA_SCHEMA_VERSION >= 1

    def test_city_data_includes_version(self):
        cd = CityData(center={"lat": 0, "lng": 0})
        assert cd.schema_version == CITY_DATA_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Building validation
# ---------------------------------------------------------------------------


class TestBuildingValidation:
    def test_valid_building(self):
        b = CityBuilding(
            id=1,
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            height=12.0,
            type="office",
            category="commercial",
        )
        assert b.height == 12.0
        assert b.category == "commercial"

    def test_reject_1_point_polygon(self):
        with pytest.raises(ValidationError):
            CityBuilding(id=1, polygon=[[0, 0]])

    def test_reject_2_point_polygon(self):
        with pytest.raises(ValidationError):
            CityBuilding(id=1, polygon=[[0, 0], [1, 1]])

    def test_accept_3_point_polygon(self):
        b = CityBuilding(id=1, polygon=[[0, 0], [1, 0], [0, 1]])
        assert len(b.polygon) == 3

    def test_reject_nan_coordinate(self):
        with pytest.raises(ValidationError):
            CityBuilding(id=1, polygon=[[float("nan"), 0], [1, 1], [2, 2]])

    def test_reject_inf_coordinate(self):
        with pytest.raises(ValidationError):
            CityBuilding(id=1, polygon=[[float("inf"), 0], [1, 1], [2, 2]])

    def test_reject_negative_inf(self):
        with pytest.raises(ValidationError):
            CityBuilding(id=1, polygon=[[float("-inf"), 0], [1, 1], [2, 2]])

    def test_reject_single_coord_point(self):
        with pytest.raises(ValidationError):
            CityBuilding(id=1, polygon=[[0], [1, 1], [2, 2]])

    def test_reject_zero_height(self):
        with pytest.raises(ValidationError):
            CityBuilding(id=1, polygon=[[0, 0], [1, 0], [0, 1]], height=0.0)

    def test_reject_negative_height(self):
        with pytest.raises(ValidationError):
            CityBuilding(id=1, polygon=[[0, 0], [1, 0], [0, 1]], height=-5.0)

    def test_default_values(self):
        b = CityBuilding(id=1, polygon=[[0, 0], [1, 0], [0, 1]])
        assert b.height == 8.0
        assert b.type == "yes"
        assert b.category == "residential"
        assert b.name == ""
        assert b.levels is None


# ---------------------------------------------------------------------------
# Road validation
# ---------------------------------------------------------------------------


class TestRoadValidation:
    def test_valid_road(self):
        r = CityRoad(id=1, points=[[0, 0], [100, 0]], width=8.0, **{"class": "primary"})
        assert r.road_class == "primary"
        assert r.width == 8.0

    def test_reject_1_point_road(self):
        with pytest.raises(ValidationError):
            CityRoad(id=1, points=[[0, 0]])

    def test_accept_2_point_road(self):
        r = CityRoad(id=1, points=[[0, 0], [10, 10]])
        assert len(r.points) == 2

    def test_reject_nan_in_road(self):
        with pytest.raises(ValidationError):
            CityRoad(id=1, points=[[float("nan"), 0], [1, 1]])

    def test_default_values(self):
        r = CityRoad(id=1, points=[[0, 0], [1, 1]])
        assert r.road_class == "residential"
        assert r.width == 6.0
        assert r.lanes == 2
        assert r.oneway is False


# ---------------------------------------------------------------------------
# Tree validation
# ---------------------------------------------------------------------------


class TestTreeValidation:
    def test_valid_tree(self):
        t = CityTree(pos=[10.0, 20.0], height=8.0, species="oak")
        assert t.height == 8.0

    def test_reject_nan_position(self):
        with pytest.raises(ValidationError):
            CityTree(pos=[float("nan"), 0])

    def test_reject_single_coord(self):
        with pytest.raises(ValidationError):
            CityTree(pos=[10.0])

    def test_default_values(self):
        t = CityTree(pos=[0, 0])
        assert t.height == 6.0
        assert t.leaf_type == "broadleaved"


# ---------------------------------------------------------------------------
# Height estimation
# ---------------------------------------------------------------------------


class TestHeightEstimation:
    def test_explicit_height_tag(self):
        assert estimate_building_height({"height": "12.5"}) == 12.5

    def test_height_with_m_suffix(self):
        assert estimate_building_height({"height": "10m"}) == 10.0

    def test_height_with_space_m(self):
        assert estimate_building_height({"height": "15 m"}) == 15.0

    def test_levels_to_height(self):
        h = estimate_building_height({"building:levels": "5"})
        assert h == pytest.approx(16.0)  # 5*3 + 1

    def test_levels_1(self):
        h = estimate_building_height({"building:levels": "1"})
        assert h == pytest.approx(4.0)  # 1*3 + 1

    def test_levels_10(self):
        h = estimate_building_height({"building:levels": "10"})
        assert h == pytest.approx(31.0)  # 10*3 + 1

    def test_explicit_height_overrides_levels(self):
        h = estimate_building_height({"height": "20", "building:levels": "3"})
        assert h == 20.0

    def test_type_apartments(self):
        assert estimate_building_height({"building": "apartments"}) == 15.0

    def test_type_garage(self):
        assert estimate_building_height({"building": "garage"}) == 3.0

    def test_type_office(self):
        assert estimate_building_height({"building": "office"}) == 18.0

    def test_type_cathedral(self):
        assert estimate_building_height({"building": "cathedral"}) == 25.0

    def test_type_shed(self):
        assert estimate_building_height({"building": "shed"}) == 3.0

    def test_unknown_type_defaults_8m(self):
        assert estimate_building_height({"building": "yes"}) == 8.0

    def test_no_tags_defaults_8m(self):
        assert estimate_building_height({}) == 8.0

    def test_invalid_height_falls_through(self):
        h = estimate_building_height({"height": "not_a_number", "building": "office"})
        assert h == 18.0  # Falls through to type default

    def test_invalid_levels_falls_through(self):
        h = estimate_building_height({"building:levels": "many", "building": "house"})
        assert h == 7.0

    # Known building accuracy tests
    def test_grand_hyatt_94m(self):
        """Grand Hyatt SF has building:levels=36 in OSM."""
        h = estimate_building_height({"building:levels": "36"})
        assert abs(h - 94.0) < 20  # Within 20m of real 94m

    def test_macys_45m(self):
        """Macy's SF has height=45 in OSM."""
        h = estimate_building_height({"height": "45"})
        assert h == 45.0


# ---------------------------------------------------------------------------
# Building classification
# ---------------------------------------------------------------------------


class TestBuildingClassification:
    def test_residential_types(self):
        for btype in ["apartments", "house", "detached", "terrace", "farm"]:
            assert classify_building({"building": btype}) == "residential", f"{btype} should be residential"

    def test_commercial_types(self):
        for btype in ["commercial", "retail", "office", "hotel", "supermarket"]:
            assert classify_building({"building": btype}) == "commercial", f"{btype} should be commercial"

    def test_industrial_types(self):
        for btype in ["industrial", "warehouse", "manufacture"]:
            assert classify_building({"building": btype}) == "industrial", f"{btype} should be industrial"

    def test_civic_types(self):
        for btype in ["hospital", "school", "university", "government", "prison"]:
            assert classify_building({"building": btype}) == "civic", f"{btype} should be civic"

    def test_religious_types(self):
        for btype in ["church", "cathedral", "mosque", "synagogue", "temple"]:
            assert classify_building({"building": btype}) == "religious", f"{btype} should be religious"

    def test_utility_types(self):
        for btype in ["garage", "shed", "parking", "roof"]:
            assert classify_building({"building": btype}) == "utility", f"{btype} should be utility"

    def test_unknown_defaults_residential(self):
        assert classify_building({"building": "yes"}) == "residential"
        assert classify_building({"building": "unknown_type"}) == "residential"

    def test_all_categories_covered(self):
        categories = set(BUILDING_CATEGORIES.values())
        expected = {"residential", "commercial", "industrial", "civic", "religious", "utility"}
        assert categories == expected


# ---------------------------------------------------------------------------
# Road width estimation
# ---------------------------------------------------------------------------


class TestRoadWidth:
    def test_motorway(self):
        assert get_road_width("motorway") == 14.0

    def test_residential(self):
        assert get_road_width("residential") == 6.0

    def test_footway(self):
        assert get_road_width("footway") == 2.0

    def test_explicit_width_tag(self):
        assert get_road_width("residential", {"width": "8"}) == 8.0

    def test_explicit_width_with_m(self):
        assert get_road_width("residential", {"width": "12m"}) == 12.0

    def test_invalid_width_falls_back(self):
        assert get_road_width("primary", {"width": "wide"}) == 10.0

    def test_unknown_type_defaults_6m(self):
        assert get_road_width("unknown_road_type") == 6.0

    def test_all_standard_types_have_widths(self):
        standard = ["motorway", "trunk", "primary", "secondary", "tertiary",
                     "residential", "service", "footway", "cycleway", "path"]
        for rt in standard:
            assert rt in ROAD_WIDTHS, f"{rt} missing from ROAD_WIDTHS"


# ---------------------------------------------------------------------------
# Full CityData assembly
# ---------------------------------------------------------------------------


class TestCityDataAssembly:
    def test_empty_city(self):
        cd = CityData(center={"lat": 37.78, "lng": -122.41})
        assert len(cd.buildings) == 0
        assert len(cd.roads) == 0

    def test_city_with_all_features(self):
        cd = CityData(
            center={"lat": 37.78, "lng": -122.41},
            radius=300,
            buildings=[CityBuilding(id=1, polygon=[[0, 0], [10, 0], [10, 10]])],
            roads=[CityRoad(id=2, points=[[0, 0], [100, 0]], **{"class": "primary"})],
            trees=[CityTree(pos=[5, 5])],
            barriers=[CityBarrier(id=3, points=[[0, 0], [10, 0]])],
            water=[CityWater(id=4, polygon=[[0, 0], [10, 0], [10, 10]])],
            entrances=[CityEntrance(pos=[5, 0])],
            pois=[CityPOI(pos=[5, 5], type="restaurant")],
            landuse=[CityLanduse(id=5, polygon=[[0, 0], [100, 0], [100, 100]], type="park")],
            stats=CityDataStats(buildings=1, roads=1, trees=1, barriers=1, water=1, entrances=1, pois=1),
        )
        assert len(cd.buildings) == 1
        assert len(cd.roads) == 1
        assert len(cd.trees) == 1
        assert cd.stats.buildings == 1

    def test_schema_version_in_output(self):
        cd = CityData(center={"lat": 0, "lng": 0})
        d = cd.model_dump()
        assert "schema_version" in d
        assert d["schema_version"] == CITY_DATA_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Lookup table completeness
# ---------------------------------------------------------------------------


class TestLookupCompleteness:
    def test_height_defaults_count(self):
        assert len(BUILDING_TYPE_HEIGHTS) >= 30

    def test_categories_count(self):
        assert len(BUILDING_CATEGORIES) >= 30

    def test_road_widths_count(self):
        assert len(ROAD_WIDTHS) >= 15

    def test_all_categorized_types_have_heights(self):
        """Every type in BUILDING_CATEGORIES should have a height default."""
        for btype in BUILDING_CATEGORIES:
            assert btype in BUILDING_TYPE_HEIGHTS, f"{btype} in categories but missing from heights"

    def test_heights_are_reasonable(self):
        for btype, h in BUILDING_TYPE_HEIGHTS.items():
            assert 2.0 <= h <= 100.0, f"{btype} height {h}m is unreasonable"

    def test_widths_are_reasonable(self):
        for rtype, w in ROAD_WIDTHS.items():
            assert 1.0 <= w <= 20.0, f"{rtype} width {w}m is unreasonable"
