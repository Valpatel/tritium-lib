# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine.world.procedural_city — demo city generator."""

import pytest

from tritium_lib.sim_engine.world.procedural_city import generate_demo_city


class TestGenerateDemoCity:
    """Tests for procedural city generation."""

    def test_returns_schema_v2(self):
        city = generate_demo_city()
        assert city["schema_version"] == 2

    def test_has_required_keys(self):
        city = generate_demo_city()
        for key in ("buildings", "roads", "trees", "landuse", "barriers",
                     "water", "entrances", "pois", "stats", "center"):
            assert key in city, f"Missing key: {key}"

    def test_has_buildings(self):
        city = generate_demo_city(radius=300)
        assert len(city["buildings"]) > 0

    def test_has_roads(self):
        city = generate_demo_city(radius=300)
        assert len(city["roads"]) > 0

    def test_buildings_have_polygon(self):
        city = generate_demo_city(radius=100)
        for b in city["buildings"]:
            assert "polygon" in b
            assert len(b["polygon"]) == 4  # rectangular
            assert "height" in b
            assert b["height"] > 0

    def test_roads_have_points(self):
        city = generate_demo_city(radius=100)
        for r in city["roads"]:
            assert "points" in r
            assert len(r["points"]) == 2
            assert "class" in r
            assert "width" in r

    def test_deterministic_with_seed(self):
        city1 = generate_demo_city(seed=42)
        city2 = generate_demo_city(seed=42)
        assert len(city1["buildings"]) == len(city2["buildings"])
        assert len(city1["roads"]) == len(city2["roads"])
        assert city1["buildings"][0]["polygon"] == city2["buildings"][0]["polygon"]

    def test_different_seed_different_result(self):
        city1 = generate_demo_city(seed=42)
        city2 = generate_demo_city(seed=99)
        # Different seeds should produce different layouts
        # (buildings count might differ due to random park placement)
        assert city1["buildings"] != city2["buildings"]

    def test_stats_match(self):
        city = generate_demo_city(radius=200)
        stats = city["stats"]
        assert stats["buildings"] == len(city["buildings"])
        assert stats["roads"] == len(city["roads"])
        assert stats["trees"] == len(city["trees"])
        assert stats["landuse"] == len(city["landuse"])

    def test_procedural_flag(self):
        city = generate_demo_city()
        assert city["_procedural"] is True

    def test_center(self):
        city = generate_demo_city()
        assert city["center"] == {"lat": 0, "lng": 0}

    def test_small_radius(self):
        city = generate_demo_city(radius=50, block_size=20)
        assert len(city["roads"]) > 0
        assert len(city["buildings"]) >= 0

    def test_zero_radius(self):
        city = generate_demo_city(radius=0)
        # Should still return valid structure even if empty
        assert city["schema_version"] == 2
        assert isinstance(city["buildings"], list)
        assert isinstance(city["roads"], list)

    def test_road_classes(self):
        city = generate_demo_city(radius=300)
        classes = {r["class"] for r in city["roads"]}
        # Should have at least residential and one main road type
        assert "residential" in classes

    def test_building_zones(self):
        city = generate_demo_city(radius=300, seed=42)
        types = {b["type"] for b in city["buildings"]}
        # Should have variety of zone types
        assert len(types) >= 2
