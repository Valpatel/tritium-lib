# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the geospatial segmentation pipeline.

All tests run without heavy dependencies (torch, rasterio, SAM).
Tests use synthetic numpy arrays and mock HTTP for tile downloads.
"""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    """Test Pydantic models import and validate correctly."""

    def test_area_of_operations(self):
        from tritium_lib.intelligence.geospatial.models import AreaOfOperations
        from tritium_lib.models.gis import TileBounds

        ao = AreaOfOperations(
            id="test_area",
            name="Test Area",
            bounds=TileBounds(min_lat=30.0, min_lon=-97.8, max_lat=30.3, max_lon=-97.5),
            zoom=17,
        )
        assert ao.id == "test_area"
        assert ao.bounds.center_lat == pytest.approx(30.15)
        assert ao.zoom == 17

    def test_segmentation_config_defaults(self):
        from tritium_lib.intelligence.geospatial.models import SegmentationConfig

        config = SegmentationConfig()
        assert config.model_name == "sam2-tiny"
        assert config.device == "auto"
        assert config.min_area_m2 == 10.0
        assert len(config.text_prompts) == 7

    def test_segmented_region(self):
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        region = SegmentedRegion(
            geometry_wkt="POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            terrain_type=TerrainType.BUILDING,
            confidence=0.85,
            area_m2=500.0,
            centroid_lat=30.15,
            centroid_lon=-97.65,
        )
        assert region.terrain_type == TerrainType.BUILDING
        assert region.confidence == 0.85

    def test_terrain_layer_metadata(self):
        from tritium_lib.intelligence.geospatial.models import TerrainLayerMetadata

        meta = TerrainLayerMetadata(ao_id="test", segment_count=42)
        assert meta.ao_id == "test"
        assert meta.model_used == "color_heuristic"

    def test_movement_cost_table(self):
        from tritium_lib.intelligence.geospatial.models import (
            PEDESTRIAN_COSTS,
            LIGHT_VEHICLE_COSTS,
            DRONE_COSTS,
        )
        assert PEDESTRIAN_COSTS.sidewalk == 0.7
        assert LIGHT_VEHICLE_COSTS.sidewalk == float("inf")
        assert DRONE_COSTS.water == 1.0


# ---------------------------------------------------------------------------
# TerrainType enum extension
# ---------------------------------------------------------------------------

class TestTerrainType:
    """Test that new segmentation-specific terrain types work."""

    def test_new_terrain_types_exist(self):
        from tritium_lib.models.terrain import TerrainType

        assert TerrainType.BUILDING == "building"
        assert TerrainType.ROAD == "road"
        assert TerrainType.SIDEWALK == "sidewalk"
        assert TerrainType.PARKING == "parking"
        assert TerrainType.VEGETATION == "vegetation"
        assert TerrainType.BRIDGE == "bridge"
        assert TerrainType.RAIL == "rail"
        assert TerrainType.BARREN == "barren"

    def test_original_types_unchanged(self):
        from tritium_lib.models.terrain import TerrainType

        assert TerrainType.URBAN == "urban"
        assert TerrainType.WATER == "water"
        assert TerrainType.FOREST == "forest"

    def test_terrain_path_loss_new_types(self):
        from tritium_lib.models.terrain import TerrainType, terrain_path_loss_db

        # New types should not raise
        for tt in [TerrainType.BUILDING, TerrainType.ROAD, TerrainType.SIDEWALK]:
            loss = terrain_path_loss_db(100.0, 2400.0, tt)
            assert loss > 0


# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------

class TestDeps:
    """Test dependency flag detection."""

    def test_flags_are_bools(self):
        from tritium_lib.intelligence.geospatial._deps import (
            HAS_NUMPY, HAS_PILLOW, HAS_TORCH, HAS_SAM,
        )
        assert isinstance(HAS_NUMPY, bool)
        assert isinstance(HAS_PILLOW, bool)
        assert isinstance(HAS_TORCH, bool)
        assert isinstance(HAS_SAM, bool)

    def test_require_raises_on_missing(self):
        from tritium_lib.intelligence.geospatial._deps import require

        with pytest.raises(ImportError, match="FakeDep"):
            require(False, "FakeDep", "geospatial")

    def test_require_passes_on_present(self):
        from tritium_lib.intelligence.geospatial._deps import require

        require(True, "RealDep")  # should not raise


# ---------------------------------------------------------------------------
# TerrainClassifier — color heuristic
# ---------------------------------------------------------------------------

class TestTerrainClassifier:
    """Test HSV-based terrain classification with synthetic images."""

    @pytest.fixture
    def classifier(self):
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
        return TerrainClassifier()

    def _make_solid_image(self, r, g, b, size=100):
        """Create a solid-color image array and full mask."""
        import numpy as np
        img = np.full((size, size, 3), [r, g, b], dtype=np.uint8)
        mask = np.ones((size, size), dtype=bool)
        return img, mask

    def test_classify_water_blue(self, classifier):
        from tritium_lib.models.terrain import TerrainType
        import numpy as np

        img, mask = self._make_solid_image(30, 80, 180)  # blue
        terrain, conf = classifier.classify_segment(img, mask)
        assert terrain == TerrainType.WATER
        assert conf > 0.3

    def test_classify_vegetation_green(self, classifier):
        from tritium_lib.models.terrain import TerrainType
        import numpy as np

        img, mask = self._make_solid_image(40, 140, 40)  # green
        terrain, conf = classifier.classify_segment(img, mask)
        assert terrain == TerrainType.VEGETATION
        assert conf > 0.3

    def test_classify_road_gray(self, classifier):
        from tritium_lib.models.terrain import TerrainType
        import numpy as np

        img, mask = self._make_solid_image(140, 140, 140)  # neutral gray
        terrain, conf = classifier.classify_segment(img, mask)
        assert terrain in (TerrainType.ROAD, TerrainType.PARKING, TerrainType.SIDEWALK)
        assert conf > 0.2

    def test_classify_empty_mask(self, classifier):
        from tritium_lib.models.terrain import TerrainType
        import numpy as np

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        terrain, conf = classifier.classify_segment(img, mask)
        assert terrain == TerrainType.UNKNOWN
        assert conf == 0.0

    def test_classify_multiple_segments(self, classifier):
        import numpy as np

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:100, :, :] = [30, 80, 180]   # blue top half
        img[100:, :, :] = [40, 140, 40]   # green bottom half

        mask_top = np.zeros((200, 200), dtype=bool)
        mask_top[:100, :] = True
        mask_bot = np.zeros((200, 200), dtype=bool)
        mask_bot[100:, :] = True

        segments = [{"mask": mask_top}, {"mask": mask_bot}]
        results = classifier.classify_segments(img, segments)
        assert len(results) == 2

    def test_rgb_to_hsv(self, classifier):
        import numpy as np

        # Pure red: H=0, S=255, V=255
        rgb = np.array([[255, 0, 0]], dtype=np.uint8)
        hsv = classifier._rgb_to_hsv(rgb)
        assert hsv[0, 0] == pytest.approx(0.0, abs=1.0)  # H ≈ 0
        assert hsv[0, 1] == pytest.approx(255.0, abs=1.0)  # S = 255
        assert hsv[0, 2] == pytest.approx(255.0, abs=1.0)  # V = 255


# ---------------------------------------------------------------------------
# VectorConverter
# ---------------------------------------------------------------------------

class TestVectorConverter:
    """Test mask-to-polygon conversion."""

    @pytest.fixture
    def converter(self):
        from tritium_lib.intelligence.geospatial.vector_converter import VectorConverter
        return VectorConverter(min_area_px=10)

    def test_bbox_fallback_simple_mask(self, converter):
        """Test that bounding box fallback works without OpenCV."""
        import numpy as np

        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 30:70] = True

        # Force bbox fallback by testing directly
        polys = converter._mask_to_polygons_bbox(mask, None)
        assert len(polys) == 1
        assert polys[0]["area_px"] == 60 * 40

    def test_bbox_with_geo_transform(self, converter):
        import numpy as np

        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 30:70] = True

        geo = (0.001, -0.001, -97.8, 30.3)  # lon_per_px, lat_per_px, origin
        polys = converter._mask_to_polygons_bbox(mask, geo)
        assert len(polys) == 1
        assert polys[0]["area_m2"] > 0

    def test_empty_mask_returns_empty(self, converter):
        import numpy as np

        mask = np.zeros((100, 100), dtype=bool)
        polys = converter.mask_to_polygons(mask, None)
        assert polys == []

    def test_tiny_mask_filtered(self, converter):
        """Masks smaller than min_area_px are filtered out."""
        import numpy as np

        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True  # 1 pixel — below min_area_px=10
        polys = converter.mask_to_polygons(mask, None)
        assert polys == []

    def test_to_geojson(self, converter):
        polys = [{
            "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
            "area_m2": 100,
            "area_px": 400,
            "centroid": (0.5, 0.5),
            "terrain_type": "building",
            "confidence": 0.8,
        }]
        geojson = converter.to_geojson(polys)
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 1
        assert geojson["features"][0]["geometry"]["type"] == "Polygon"

    def test_wkt_generation(self, converter):
        wkt = converter._coords_to_wkt([(0, 0), (1, 0), (1, 1), (0, 0)])
        assert wkt == "POLYGON ((0 0, 1 0, 1 1, 0 0))"


# ---------------------------------------------------------------------------
# TerrainLayer — cache and queries
# ---------------------------------------------------------------------------

class TestTerrainLayer:
    """Test terrain layer caching and query functionality."""

    @pytest.fixture
    def terrain_layer(self, tmp_path):
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        return TerrainLayer(cache_dir=tmp_path / "terrain")

    def test_empty_layer_returns_unknown(self, terrain_layer):
        from tritium_lib.models.terrain import TerrainType
        assert terrain_layer.terrain_at(30.0, -97.0) == TerrainType.UNKNOWN

    def test_cache_round_trip(self, terrain_layer, tmp_path):
        from tritium_lib.intelligence.geospatial.models import (
            SegmentedRegion,
            TerrainLayerMetadata,
        )
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.models.gis import TileBounds
        from tritium_lib.models.terrain import TerrainType

        # Manually populate
        terrain_layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON ((-97.7 30.2, -97.6 30.2, -97.6 30.3, -97.7 30.3, -97.7 30.2))",
                terrain_type=TerrainType.BUILDING,
                confidence=0.9,
                area_m2=1000.0,
                centroid_lat=30.25,
                centroid_lon=-97.65,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON ((-97.8 30.0, -97.7 30.0, -97.7 30.1, -97.8 30.1, -97.8 30.0))",
                terrain_type=TerrainType.WATER,
                confidence=0.95,
                area_m2=5000.0,
                centroid_lat=30.05,
                centroid_lon=-97.75,
            ),
        ]
        terrain_layer._metadata = TerrainLayerMetadata(
            ao_id="test_round_trip",
            segment_count=2,
            bounds=TileBounds(min_lat=30.0, min_lon=-97.8, max_lat=30.3, max_lon=-97.5),
        )
        terrain_layer._build_grid_index()
        terrain_layer._save_cache("test_round_trip")

        # Reload
        layer2 = TerrainLayer(cache_dir=tmp_path / "terrain")
        assert layer2.load_cached("test_round_trip") is True
        assert len(layer2.regions) == 2
        assert layer2.regions[0].terrain_type == TerrainType.BUILDING

    def test_features_by_type(self, terrain_layer):
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        terrain_layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,
                area_m2=100,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.BUILDING,
                area_m2=200,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,
                area_m2=300,
            ),
        ]
        roads = terrain_layer.features_by_type(TerrainType.ROAD)
        assert len(roads) == 2
        buildings = terrain_layer.features_by_type(TerrainType.BUILDING)
        assert len(buildings) == 1

    def test_terrain_brief(self, terrain_layer):
        from tritium_lib.intelligence.geospatial.models import (
            SegmentedRegion,
            TerrainLayerMetadata,
        )
        from tritium_lib.models.terrain import TerrainType

        terrain_layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.BUILDING,
                area_m2=500,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.WATER,
                area_m2=2000,
            ),
        ]
        terrain_layer._metadata = TerrainLayerMetadata(
            ao_id="brief_test", segment_count=2,
        )

        brief = terrain_layer.terrain_brief()
        assert "TERRAIN BRIEF" in brief
        assert "water" in brief.lower()
        assert "building" in brief.lower()

    def test_nearest_feature(self, terrain_layer):
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        terrain_layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.BRIDGE,
                area_m2=100,
                centroid_lat=30.0,
                centroid_lon=-97.0,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.BRIDGE,
                area_m2=100,
                centroid_lat=31.0,
                centroid_lon=-97.0,
            ),
        ]
        nearest = terrain_layer.nearest_feature(30.1, -97.0, TerrainType.BRIDGE)
        assert nearest is not None
        assert nearest.centroid_lat == 30.0

    def test_to_geojson(self, terrain_layer):
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        terrain_layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                terrain_type=TerrainType.ROAD,
                confidence=0.8,
                area_m2=100,
            ),
        ]
        geojson = terrain_layer.to_geojson()
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 1
        props = geojson["features"][0]["properties"]
        assert props["terrain_type"] == "road"

    def test_load_nonexistent_cache(self, terrain_layer):
        assert terrain_layer.load_cached("nonexistent") is False


# ---------------------------------------------------------------------------
# MovementProfile upgrades
# ---------------------------------------------------------------------------

class TestMovementProfile:
    """Test upgraded MovementProfile with new terrain fields."""

    def test_profiles_have_new_fields(self):
        from tritium_lib.sim_engine.world.grid_pathfinder import PROFILES

        ped = PROFILES["pedestrian"]
        assert hasattr(ped, "sidewalk")
        assert hasattr(ped, "parking")
        assert hasattr(ped, "vegetation")
        assert hasattr(ped, "bridge")
        assert hasattr(ped, "rail")
        assert hasattr(ped, "barren")

    def test_pedestrian_prefers_sidewalk(self):
        from tritium_lib.sim_engine.world.grid_pathfinder import PROFILES

        ped = PROFILES["pedestrian"]
        assert ped.sidewalk <= ped.road  # sidewalk is at least as good as road

    def test_vehicle_avoids_sidewalk(self):
        from tritium_lib.sim_engine.world.grid_pathfinder import PROFILES

        lv = PROFILES["light_vehicle"]
        assert lv.sidewalk >= 999.0  # vehicles can't use sidewalks

    def test_aerial_ignores_terrain(self):
        from tritium_lib.sim_engine.world.grid_pathfinder import PROFILES

        air = PROFILES["aerial"]
        assert air.sidewalk == 1.0
        assert air.water == 1.0
        assert air.vegetation == 1.0

    def test_terrain_to_field_mapping(self):
        from tritium_lib.sim_engine.world.grid_pathfinder import _TERRAIN_TO_FIELD

        assert "sidewalk" in _TERRAIN_TO_FIELD
        assert "parking" in _TERRAIN_TO_FIELD
        assert "vegetation" in _TERRAIN_TO_FIELD
        assert "bridge" in _TERRAIN_TO_FIELD

    def test_rover_profile_exists(self):
        from tritium_lib.sim_engine.world.grid_pathfinder import PROFILES

        assert "rover" in PROFILES
        rover = PROFILES["rover"]
        assert rover.road < 1.0
        assert rover.water >= 999.0


# ---------------------------------------------------------------------------
# TileDownloader
# ---------------------------------------------------------------------------

class TestTileDownloader:
    """Test tile downloading and stitching."""

    def test_tile_cache_path(self, tmp_path):
        from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader
        from tritium_lib.models.gis import TileCoord

        dl = TileDownloader(cache_dir=tmp_path)
        tile = TileCoord(x=100, y=200, zoom=17)
        path = dl._tile_cache_path(tile, "satellite")
        assert "satellite" in str(path)
        assert "17" in str(path)

    def test_tiles_for_ao(self, tmp_path):
        from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader
        from tritium_lib.intelligence.geospatial.models import AreaOfOperations
        from tritium_lib.models.gis import TileBounds

        dl = TileDownloader(cache_dir=tmp_path)
        ao = AreaOfOperations(
            id="test",
            name="Test",
            bounds=TileBounds(min_lat=30.26, min_lon=-97.74, max_lat=30.27, max_lon=-97.73),
            zoom=17,
        )
        tiles = dl.tiles_for_ao(ao)
        assert len(tiles) > 0

    def test_geo_transform(self, tmp_path):
        from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader
        from tritium_lib.intelligence.geospatial.models import AreaOfOperations
        from tritium_lib.models.gis import TileBounds

        dl = TileDownloader(cache_dir=tmp_path)
        ao = AreaOfOperations(
            id="test",
            name="Test",
            bounds=TileBounds(min_lat=30.26, min_lon=-97.74, max_lat=30.27, max_lon=-97.73),
            zoom=17,
        )
        gt = dl.get_geo_transform(ao, 1024, 1024)
        assert len(gt) == 4
        lon_per_px, lat_per_px, origin_lon, origin_lat = gt
        # lon should increase left-to-right
        assert lon_per_px > 0
        # lat should decrease top-to-bottom (negative)
        assert lat_per_px < 0

    def test_pixel_to_latlon(self, tmp_path):
        from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader

        dl = TileDownloader(cache_dir=tmp_path)
        geo = (0.001, -0.001, -97.8, 30.3)
        lat, lon = dl.pixel_to_latlon(100, 50, geo)
        assert lon == pytest.approx(-97.7)
        assert lat == pytest.approx(30.25)


# ---------------------------------------------------------------------------
# SegmentationEngine fallback
# ---------------------------------------------------------------------------

class TestSegmentationEngine:
    """Test segmentation engine, especially fallback paths."""

    def test_color_region_fallback(self, tmp_path):
        """Test that color-based fallback produces segments."""
        from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine
        from tritium_lib.intelligence.geospatial._deps import HAS_NUMPY, HAS_PILLOW

        if not HAS_NUMPY or not HAS_PILLOW:
            pytest.skip("numpy and Pillow required")

        import numpy as np
        from PIL import Image

        # Create a simple test image with two colors
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[:64, :, :] = [30, 80, 180]   # blue top
        img[64:, :, :] = [40, 140, 40]   # green bottom

        img_path = tmp_path / "test.png"
        Image.fromarray(img).save(img_path)

        engine = SegmentationEngine()
        segments = engine._segment_with_color_regions(img_path)
        assert len(segments) > 0
        for seg in segments:
            assert "mask" in seg
            assert "area" in seg
            assert "bbox" in seg

    def test_empty_result_without_deps(self):
        """Engine returns empty list when no backend is available."""
        from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine

        engine = SegmentationEngine()
        # Directly test — if no SAM and no numpy, should return []
        with patch(
            "tritium_lib.intelligence.geospatial.segmentation.HAS_SAM", False
        ), patch(
            "tritium_lib.intelligence.geospatial.segmentation.HAS_TORCH", False
        ), patch(
            "tritium_lib.intelligence.geospatial.segmentation.HAS_NUMPY", False
        ):
            result = engine.segment_image(Path("nonexistent.png"))
            assert result == []
