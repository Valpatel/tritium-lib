# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.mapgen — procedural map generation."""

from __future__ import annotations

import math

import pytest

from tritium_lib.sim_engine.mapgen import (
    GeneratedMap,
    MAP_PRESETS,
    MapFeature,
    MapGenerator,
    TerrainType,
    _clamp,
    _clamp_int,
    _dist,
    _lerp,
    _meander,
    _point_in_bounds,
    _uid,
    generate_preset,
)
from tritium_lib.sim_engine.terrain import HeightMap


# ===================================================================
# TerrainType enum
# ===================================================================

class TestTerrainType:
    def test_all_values_exist(self):
        expected = {"grass", "dirt", "sand", "rock", "water", "swamp",
                    "forest", "urban", "road", "bridge"}
        actual = {t.value for t in TerrainType}
        assert actual == expected

    def test_enum_count(self):
        assert len(TerrainType) == 10

    def test_by_name(self):
        assert TerrainType.GRASS.value == "grass"
        assert TerrainType.WATER.value == "water"
        assert TerrainType.BRIDGE.value == "bridge"


# ===================================================================
# MapFeature dataclass
# ===================================================================

class TestMapFeature:
    def test_creation(self):
        f = MapFeature("f1", "building", (10.0, 20.0), (5.0, 8.0))
        assert f.feature_id == "f1"
        assert f.feature_type == "building"
        assert f.position == (10.0, 20.0)
        assert f.size == (5.0, 8.0)
        assert f.rotation == 0.0
        assert f.properties == {}

    def test_with_properties(self):
        f = MapFeature("f2", "tower", (0, 0), (3, 3), rotation=45.0,
                       properties={"height": 20, "material": "steel"})
        assert f.rotation == 45.0
        assert f.properties["height"] == 20

    def test_all_feature_types(self):
        for ft in ("building", "road", "river", "hill", "forest",
                    "bridge", "wall", "tower"):
            f = MapFeature("x", ft, (0, 0), (1, 1))
            assert f.feature_type == ft


# ===================================================================
# GeneratedMap dataclass
# ===================================================================

class TestGeneratedMap:
    def test_fields(self):
        m = GeneratedMap(
            width=100, height=100,
            heightmap=[[0.0]], terrain_types=[["grass"]],
            features=[], spawn_points={}, objectives=[],
            roads=[], rivers=[], seed=42,
        )
        assert m.width == 100
        assert m.seed == 42


# ===================================================================
# Helper functions
# ===================================================================

class TestHelpers:
    def test_uid_unique(self):
        ids = {_uid() for _ in range(100)}
        assert len(ids) == 100

    def test_uid_length(self):
        assert len(_uid()) == 12

    def test_dist(self):
        assert _dist((0, 0), (3, 4)) == pytest.approx(5.0)
        assert _dist((1, 1), (1, 1)) == 0.0

    def test_lerp(self):
        assert _lerp(0, 10, 0.5) == 5.0
        assert _lerp(0, 10, 0.0) == 0.0
        assert _lerp(0, 10, 1.0) == 10.0

    def test_clamp(self):
        assert _clamp(5, 0, 10) == 5
        assert _clamp(-1, 0, 10) == 0
        assert _clamp(15, 0, 10) == 10

    def test_clamp_int(self):
        assert _clamp_int(5, 0, 10) == 5
        assert _clamp_int(-1, 0, 10) == 0

    def test_meander_endpoints(self):
        import random
        rng = random.Random(42)
        pts = _meander((0, 0), (100, 0), rng, amplitude=10, segments=10, seed=1)
        assert len(pts) == 11
        assert pts[0][0] == pytest.approx(0.0, abs=1e-6)
        assert pts[-1][0] == pytest.approx(100.0, abs=1e-6)

    def test_meander_zero_length(self):
        import random
        rng = random.Random(1)
        pts = _meander((5, 5), (5, 5), rng)
        assert len(pts) == 2

    def test_point_in_bounds(self):
        assert _point_in_bounds((50, 50), 100, 100) is True
        assert _point_in_bounds((-1, 50), 100, 100) is False
        assert _point_in_bounds((50, 50), 100, 100, margin=10) is True
        assert _point_in_bounds((5, 50), 100, 100, margin=10) is False


# ===================================================================
# MapGenerator — construction
# ===================================================================

class TestMapGeneratorInit:
    def test_defaults(self):
        g = MapGenerator()
        assert g.width == 500.0
        assert g.height == 500.0
        assert g.cell_size == 5.0

    def test_custom_size(self):
        g = MapGenerator(200, 300, cell_size=10, seed=7)
        assert g.width == 200.0
        assert g.height == 300.0
        assert g.seed == 7
        assert g._cols == 20
        assert g._rows == 30

    def test_deterministic_seed(self):
        g1 = MapGenerator(100, 100, seed=99)
        g2 = MapGenerator(100, 100, seed=99)
        g1.generate_terrain("hilly")
        g2.generate_terrain("hilly")
        assert g1._hm == g2._hm

    def test_random_seed_when_none(self):
        g = MapGenerator(100, 100, seed=None)
        assert isinstance(g.seed, int)


# ===================================================================
# Terrain generation styles
# ===================================================================

class TestGenerateTerrain:
    @pytest.mark.parametrize("style", [
        "flat", "hilly", "mountainous", "coastal", "island", "valley", "mixed",
    ])
    def test_style_produces_heightmap(self, style):
        g = MapGenerator(100, 100, cell_size=5, seed=42)
        ret = g.generate_terrain(style)
        assert ret is g  # chaining
        assert len(g._hm) == g._rows
        assert len(g._hm[0]) == g._cols

    def test_flat_low_variance(self):
        g = MapGenerator(100, 100, cell_size=5, seed=42)
        g.generate_terrain("flat")
        vals = [g._hm[r][c] for r in range(g._rows) for c in range(g._cols)]
        assert max(vals) - min(vals) < 10.0

    def test_mountainous_high_variance(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("mountainous")
        vals = [g._hm[r][c] for r in range(g._rows) for c in range(g._cols)]
        assert max(vals) - min(vals) > 20.0

    def test_coastal_water_on_right(self):
        g = MapGenerator(200, 100, cell_size=5, seed=42)
        g.generate_terrain("coastal")
        # Rightmost column should tend to be lower/water
        right_col = [g._hm[r][g._cols - 1] for r in range(g._rows)]
        avg_right = sum(right_col) / len(right_col)
        left_col = [g._hm[r][0] for r in range(g._rows)]
        avg_left = sum(left_col) / len(left_col)
        assert avg_left > avg_right

    def test_island_edges_low(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("island")
        # Corner cells should be low/water
        corners = [g._hm[0][0], g._hm[0][-1], g._hm[-1][0], g._hm[-1][-1]]
        center = g._hm[g._rows // 2][g._cols // 2]
        assert all(c < center for c in corners)

    def test_terrain_types_assigned(self):
        g = MapGenerator(100, 100, cell_size=5, seed=42)
        g.generate_terrain("hilly")
        types = {g._terrain[r][c] for r in range(g._rows)
                 for c in range(g._cols)}
        # Should have at least grass
        assert "grass" in types

    def test_water_terrain_for_negative_elevation(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("island")
        # Cells with elevation < -2 should be water
        for r in range(g._rows):
            for c in range(g._cols):
                if g._hm[r][c] < -2.0:
                    assert g._terrain[r][c] == "water"


# ===================================================================
# add_city
# ===================================================================

class TestAddCity:
    def test_produces_buildings(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_city((100, 100), radius=60, density=0.6)
        buildings = [f for f in g._features if f.feature_type == "building"]
        assert len(buildings) > 0

    def test_produces_roads(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_city((100, 100), radius=60, density=0.6)
        assert len(g._roads) > 0

    def test_urban_terrain_painted(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_city((100, 100), radius=60)
        # Center may be road or urban (roads cross through center)
        urban_count = sum(
            1 for r in range(g._rows) for c in range(g._cols)
            if g._terrain[r][c] == "urban"
        )
        assert urban_count > 0

    def test_density_affects_building_count(self):
        g1 = MapGenerator(200, 200, cell_size=5, seed=42)
        g1.generate_terrain("flat")
        g1.add_city((100, 100), radius=60, density=0.9)
        b1 = len([f for f in g1._features if f.feature_type == "building"])

        g2 = MapGenerator(200, 200, cell_size=5, seed=42)
        g2.generate_terrain("flat")
        g2.add_city((100, 100), radius=60, density=0.1)
        b2 = len([f for f in g2._features if f.feature_type == "building"])

        assert b1 > b2

    def test_diagonal_road_with_high_density(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_city((100, 100), radius=60, density=0.5)
        # With density > 0.3 and radius > 40 we get a diagonal
        road_count = len(g._roads)
        assert road_count >= 3  # at least H + V + diagonal

    def test_chaining(self):
        g = MapGenerator(200, 200, seed=42)
        ret = g.generate_terrain("flat").add_city((100, 100), 50)
        assert ret is g

    def test_building_properties(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_city((100, 100), radius=60, density=0.8)
        buildings = [f for f in g._features if f.feature_type == "building"]
        assert len(buildings) > 0
        b = buildings[0]
        assert "material" in b.properties
        assert "height" in b.properties
        assert b.properties["height"] > 0


# ===================================================================
# add_village
# ===================================================================

class TestAddVillage:
    def test_produces_buildings(self):
        g = MapGenerator(300, 300, cell_size=5, seed=42)
        g.generate_terrain("hilly")
        g.add_village((150, 150), radius=50)
        buildings = [f for f in g._features if f.feature_type == "building"]
        assert len(buildings) >= 5

    def test_produces_roads(self):
        g = MapGenerator(300, 300, cell_size=5, seed=42)
        g.generate_terrain("hilly")
        g.add_village((150, 150), radius=50)
        assert len(g._roads) >= 2

    def test_cottage_class(self):
        g = MapGenerator(300, 300, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_village((150, 150), radius=50)
        cottages = [f for f in g._features
                    if f.properties.get("building_class") == "cottage"]
        assert len(cottages) > 0


# ===================================================================
# add_river
# ===================================================================

class TestAddRiver:
    def test_river_polyline(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_river((0, 100), (200, 100), width=10)
        assert len(g._rivers) == 1
        assert len(g._rivers[0]) > 2

    def test_water_cells_painted(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_river((0, 100), (200, 100), width=15)
        water_count = sum(
            1 for r in range(g._rows) for c in range(g._cols)
            if g._terrain[r][c] == "water"
        )
        assert water_count > 0

    def test_heightmap_depressed(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_river((0, 100), (200, 100), width=10)
        # Some cells should be depressed to -3
        depressed = sum(
            1 for r in range(g._rows) for c in range(g._cols)
            if g._hm[r][c] <= -3.0
        )
        assert depressed > 0

    def test_bridge_at_road_crossing(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_road((100, 0), (100, 200))
        g.add_river((0, 100), (200, 100), width=10)
        bridges = [f for f in g._features if f.feature_type == "bridge"]
        assert len(bridges) > 0


# ===================================================================
# add_forest
# ===================================================================

class TestAddForest:
    def test_forest_terrain(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_forest((100, 100), radius=40, density=0.5)
        cx, cy = g._world_to_cell((100, 100))
        assert g._terrain[cy][cx] == "forest"

    def test_tree_features(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_forest((100, 100), radius=40, density=0.7)
        trees = [f for f in g._features if f.feature_type == "forest"]
        assert len(trees) > 0

    def test_density_affects_tree_count(self):
        g1 = MapGenerator(200, 200, cell_size=5, seed=42)
        g1.generate_terrain("flat")
        g1.add_forest((100, 100), radius=40, density=1.0)
        t1 = len([f for f in g1._features if f.feature_type == "forest"])

        g2 = MapGenerator(200, 200, cell_size=5, seed=42)
        g2.generate_terrain("flat")
        g2.add_forest((100, 100), radius=40, density=0.1)
        t2 = len([f for f in g2._features if f.feature_type == "forest"])

        assert t1 > t2


# ===================================================================
# add_road
# ===================================================================

class TestAddRoad:
    def test_road_polyline(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_road((0, 100), (200, 100))
        assert len(g._roads) == 1
        assert len(g._roads[0]) > 2

    def test_road_terrain_cells(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_road((0, 100), (200, 100), width=8)
        road_count = sum(
            1 for r in range(g._rows) for c in range(g._cols)
            if g._terrain[r][c] == "road"
        )
        assert road_count > 0


# ===================================================================
# place_spawn_points
# ===================================================================

class TestPlaceSpawnPoints:
    def test_factions_created(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.place_spawn_points(["blue", "red"])
        assert "blue" in g._spawn_points
        assert "red" in g._spawn_points

    def test_spawn_cluster_size(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.place_spawn_points(["alpha"])
        assert len(g._spawn_points["alpha"]) == 4  # 1 primary + 3 cluster

    def test_factions_spread_apart(self):
        g = MapGenerator(500, 500, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.place_spawn_points(["blue", "red"], min_distance=100)
        blue = g._spawn_points["blue"][0]
        red = g._spawn_points["red"][0]
        assert _dist(blue, red) > 50  # should be well apart

    def test_three_factions(self):
        g = MapGenerator(500, 500, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.place_spawn_points(["a", "b", "c"])
        assert len(g._spawn_points) == 3

    def test_spawns_within_map(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.place_spawn_points(["x", "y"])
        for faction, pts in g._spawn_points.items():
            for p in pts:
                assert 0 <= p[0] <= 200
                assert 0 <= p[1] <= 200


# ===================================================================
# place_objectives
# ===================================================================

class TestPlaceObjectives:
    def test_correct_count(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.place_objectives(3)
        assert len(g._objectives) == 3

    def test_objective_fields(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.place_objectives(1)
        obj = g._objectives[0]
        assert "id" in obj
        assert "type" in obj
        assert "position" in obj
        assert "radius" in obj
        assert "name" in obj

    def test_objective_types_alternate(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.place_objectives(4)
        types = [o["type"] for o in g._objectives]
        assert types[0] == "capture_point"
        assert types[1] == "defend_point"
        assert types[2] == "capture_point"
        assert types[3] == "defend_point"

    def test_objectives_within_map(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.place_objectives(5)
        for obj in g._objectives:
            px, py = obj["position"]
            assert 0 <= px <= 200
            assert 0 <= py <= 200


# ===================================================================
# result()
# ===================================================================

class TestResult:
    def test_returns_generated_map(self):
        g = MapGenerator(100, 100, cell_size=5, seed=7)
        g.generate_terrain("flat")
        m = g.result()
        assert isinstance(m, GeneratedMap)
        assert m.width == 100
        assert m.height == 100
        assert m.seed == 7

    def test_heightmap_dimensions(self):
        g = MapGenerator(100, 100, cell_size=5, seed=7)
        g.generate_terrain("flat")
        m = g.result()
        assert len(m.heightmap) == g._rows
        assert len(m.heightmap[0]) == g._cols

    def test_result_is_a_copy(self):
        g = MapGenerator(100, 100, cell_size=5, seed=7)
        g.generate_terrain("flat")
        m1 = g.result()
        m2 = g.result()
        assert m1.heightmap is not m2.heightmap
        assert m1.features is not m2.features


# ===================================================================
# to_three_js()
# ===================================================================

class TestToThreeJs:
    def test_returns_dict(self):
        g = MapGenerator(100, 100, cell_size=5, seed=42)
        g.generate_terrain("flat")
        d = g.to_three_js()
        assert isinstance(d, dict)
        assert d["width"] == 100
        assert d["height"] == 100
        assert d["seed"] == 42

    def test_contains_heightmap(self):
        g = MapGenerator(100, 100, cell_size=5, seed=42)
        g.generate_terrain("flat")
        d = g.to_three_js()
        assert "heightmap" in d
        assert len(d["heightmap"]) == g._rows

    def test_features_serialized(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_city((100, 100), radius=40, density=0.5)
        d = g.to_three_js()
        assert len(d["features"]) > 0
        f0 = d["features"][0]
        assert "id" in f0
        assert "position" in f0
        assert "x" in f0["position"]

    def test_spawn_points_serialized(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.place_spawn_points(["blue"])
        d = g.to_three_js()
        assert "blue" in d["spawn_points"]
        assert "x" in d["spawn_points"]["blue"][0]

    def test_roads_serialized(self):
        g = MapGenerator(200, 200, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_road((0, 50), (200, 50))
        d = g.to_three_js()
        assert len(d["roads"]) == 1
        assert "x" in d["roads"][0][0]


# ===================================================================
# to_heightmap()
# ===================================================================

class TestToHeightmap:
    def test_returns_heightmap_object(self):
        g = MapGenerator(100, 100, cell_size=5, seed=42)
        g.generate_terrain("hilly")
        hm = g.to_heightmap()
        assert isinstance(hm, HeightMap)
        assert hm.width == g._cols
        assert hm.height == g._rows
        assert hm.cell_size == g.cell_size

    def test_elevations_match(self):
        g = MapGenerator(50, 50, cell_size=5, seed=42)
        g.generate_terrain("hilly")
        hm = g.to_heightmap()
        for r in range(g._rows):
            for c in range(g._cols):
                assert hm.get_elevation(c, r) == pytest.approx(g._hm[r][c])


# ===================================================================
# MAP_PRESETS
# ===================================================================

class TestMapPresets:
    def test_all_presets_exist(self):
        expected = {"city_block", "village", "coastal_base", "mountain_pass",
                    "island", "desert_town", "forest_camp"}
        assert set(MAP_PRESETS.keys()) == expected

    @pytest.mark.parametrize("name", list(MAP_PRESETS.keys()))
    def test_preset_generates_valid_map(self, name):
        m = generate_preset(name, seed=42)
        assert isinstance(m, GeneratedMap)
        assert m.width > 0
        assert m.height > 0
        assert len(m.heightmap) > 0
        assert len(m.terrain_types) > 0
        assert m.seed == 42

    @pytest.mark.parametrize("name", list(MAP_PRESETS.keys()))
    def test_preset_has_spawn_points(self, name):
        m = generate_preset(name, seed=42)
        assert len(m.spawn_points) >= 2

    @pytest.mark.parametrize("name", list(MAP_PRESETS.keys()))
    def test_preset_has_objectives(self, name):
        m = generate_preset(name, seed=42)
        assert len(m.objectives) >= 2

    @pytest.mark.parametrize("name", list(MAP_PRESETS.keys()))
    def test_preset_deterministic(self, name):
        m1 = generate_preset(name, seed=99)
        m2 = generate_preset(name, seed=99)
        assert m1.heightmap == m2.heightmap

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            generate_preset("nonexistent")

    def test_city_block_dimensions(self):
        m = generate_preset("city_block", seed=1)
        assert m.width == 200
        assert m.height == 200

    def test_coastal_base_dimensions(self):
        m = generate_preset("coastal_base", seed=1)
        assert m.width == 500
        assert m.height == 300

    def test_desert_town_sand_terrain(self):
        m = generate_preset("desert_town", seed=42)
        # Most cells should be sand or road/urban
        sand_count = sum(
            1 for row in m.terrain_types for cell in row if cell == "sand"
        )
        total = sum(len(row) for row in m.terrain_types)
        assert sand_count / total > 0.3

    def test_island_has_water(self):
        m = generate_preset("island", seed=42)
        water_count = sum(
            1 for row in m.terrain_types for cell in row if cell == "water"
        )
        assert water_count > 0

    def test_forest_camp_has_forest(self):
        m = generate_preset("forest_camp", seed=42)
        forest_count = sum(
            1 for row in m.terrain_types for cell in row if cell == "forest"
        )
        assert forest_count > 0

    def test_village_has_buildings(self):
        m = generate_preset("village", seed=42)
        buildings = [f for f in m.features if f.feature_type == "building"]
        assert len(buildings) > 0


# ===================================================================
# Integration: full pipeline
# ===================================================================

class TestIntegration:
    def test_full_pipeline(self):
        g = MapGenerator(300, 300, cell_size=5, seed=42)
        g.generate_terrain("hilly")
        g.add_city((150, 150), radius=50, density=0.5)
        g.add_forest((50, 50), radius=30)
        g.add_river((0, 200), (300, 50), width=12)
        g.add_road((0, 150), (300, 150))
        g.place_spawn_points(["alpha", "bravo"], min_distance=100)
        g.place_objectives(3)
        m = g.result()

        assert m.width == 300
        assert len(m.features) > 0
        assert len(m.roads) > 0
        assert len(m.rivers) == 1
        assert len(m.spawn_points) == 2
        assert len(m.objectives) == 3

        hm = g.to_heightmap()
        assert isinstance(hm, HeightMap)

        js = g.to_three_js()
        assert js["width"] == 300

    def test_chaining_api(self):
        m = (
            MapGenerator(200, 200, seed=1)
            .generate_terrain("flat")
            .add_road((0, 100), (200, 100))
            .add_forest((50, 50), 30)
            .add_village((150, 150), 30)
            .place_spawn_points(["a", "b"])
            .place_objectives(2)
            .result()
        )
        assert isinstance(m, GeneratedMap)
        assert len(m.spawn_points) == 2

    def test_multiple_cities(self):
        g = MapGenerator(500, 500, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_city((100, 100), radius=40, density=0.5)
        g.add_city((400, 400), radius=40, density=0.5)
        buildings = [f for f in g._features if f.feature_type == "building"]
        assert len(buildings) > 2

    def test_multiple_rivers(self):
        g = MapGenerator(300, 300, cell_size=5, seed=42)
        g.generate_terrain("flat")
        g.add_river((0, 100), (300, 100))
        g.add_river((0, 200), (300, 200))
        assert len(g._rivers) == 2
