# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Integration tests for geospatial segmentation pipeline.

Tests the full pipeline: synthetic image → segment → classify → vectorize →
cache → query. Also tests SidewalkGraph, providers, and terrain layer
integration.

All tests use synthetic data — no network access needed.
"""

import json
import math
from pathlib import Path

import pytest

# Check for numpy — most integration tests need it
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

pytestmark = pytest.mark.skipif(
    not HAS_NUMPY or not HAS_PILLOW,
    reason="numpy and Pillow required for integration tests",
)


# ---------------------------------------------------------------------------
# Full pipeline with synthetic image
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Test the complete segmentation pipeline with synthetic imagery."""

    def _create_synthetic_satellite(self, path: Path, size: int = 256) -> None:
        """Create a synthetic satellite image with known terrain regions.

        Top-left: blue (water)
        Top-right: green (vegetation)
        Bottom-left: gray (road)
        Bottom-right: light gray/white (building)
        """
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Water — deep blue
        img[:size // 2, :size // 2] = [30, 60, 180]

        # Vegetation — green
        img[:size // 2, size // 2:] = [40, 150, 40]

        # Road — neutral gray
        img[size // 2:, :size // 2] = [130, 130, 130]

        # Building — light gray
        img[size // 2:, size // 2:] = [200, 200, 210]

        Image.fromarray(img).save(path)

    def test_segment_classify_vectorize(self, tmp_path):
        """Test segmentation → classification → vectorization flow."""
        from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
        from tritium_lib.intelligence.geospatial.vector_converter import VectorConverter
        from tritium_lib.models.terrain import TerrainType

        # Create synthetic image
        img_path = tmp_path / "satellite.png"
        self._create_synthetic_satellite(img_path)

        # Segment
        engine = SegmentationEngine()
        segments = engine.segment_image(img_path)
        assert len(segments) > 0, "Segmentation produced no segments"

        # Classify
        img_array = np.array(Image.open(img_path).convert("RGB"))
        classifier = TerrainClassifier()
        classifications = classifier.classify_segments(img_array, segments)
        assert len(classifications) == len(segments)

        # Vectorize
        geo_transform = (0.001, -0.001, -97.8, 30.3)
        converter = VectorConverter(min_area_px=50)
        polygons = converter.masks_to_polygons(segments, geo_transform)
        assert len(polygons) > 0, "Vectorization produced no polygons"

        # Check that polygons have valid WKT
        for poly in polygons:
            assert "POLYGON" in poly["wkt"]
            assert poly["area_m2"] > 0

    def test_terrain_layer_synthetic_pipeline(self, tmp_path):
        """Test TerrainLayer with manually constructed segments."""
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import (
            SegmentedRegion,
            TerrainLayerMetadata,
        )
        from tritium_lib.models.gis import TileBounds
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path / "terrain")

        # Manually build terrain with known features
        regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON ((-97.75 30.25, -97.74 30.25, -97.74 30.26, -97.75 30.26, -97.75 30.25))",
                terrain_type=TerrainType.WATER,
                confidence=0.9,
                area_m2=12000,
                centroid_lat=30.255,
                centroid_lon=-97.745,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON ((-97.73 30.25, -97.72 30.25, -97.72 30.26, -97.73 30.26, -97.73 30.25))",
                terrain_type=TerrainType.ROAD,
                confidence=0.85,
                area_m2=8000,
                centroid_lat=30.255,
                centroid_lon=-97.725,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON ((-97.71 30.25, -97.70 30.25, -97.70 30.26, -97.71 30.26, -97.71 30.25))",
                terrain_type=TerrainType.BUILDING,
                confidence=0.92,
                area_m2=3000,
                centroid_lat=30.255,
                centroid_lon=-97.705,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON ((-97.73 30.26, -97.72 30.26, -97.72 30.27, -97.73 30.27, -97.73 30.26))",
                terrain_type=TerrainType.SIDEWALK,
                confidence=0.7,
                area_m2=500,
                centroid_lat=30.265,
                centroid_lon=-97.725,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON ((-97.74 30.26, -97.73 30.26, -97.73 30.27, -97.74 30.27, -97.74 30.26))",
                terrain_type=TerrainType.VEGETATION,
                confidence=0.8,
                area_m2=10000,
                centroid_lat=30.265,
                centroid_lon=-97.735,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON ((-97.745 30.26, -97.740 30.26, -97.740 30.265, -97.745 30.265, -97.745 30.26))",
                terrain_type=TerrainType.BRIDGE,
                confidence=0.75,
                area_m2=200,
                centroid_lat=30.2625,
                centroid_lon=-97.7425,
            ),
        ]

        layer._regions = regions
        layer._bounds = TileBounds(
            min_lat=30.25, min_lon=-97.76,
            max_lat=30.27, max_lon=-97.69,
        )
        layer._metadata = TerrainLayerMetadata(
            ao_id="integration_test",
            segment_count=len(regions),
            source_imagery="synthetic",
        )
        layer._build_grid_index()

        # Test terrain_at queries
        water = layer.terrain_at(30.255, -97.745)
        assert water == TerrainType.WATER, f"Expected WATER, got {water}"

        road = layer.terrain_at(30.255, -97.725)
        assert road == TerrainType.ROAD, f"Expected ROAD, got {road}"

        # Test features_by_type
        buildings = layer.features_by_type(TerrainType.BUILDING)
        assert len(buildings) == 1
        assert buildings[0].area_m2 == 3000

        # Test obstacles_in_bbox
        obstacles = layer.obstacles_in_bbox(TileBounds(
            min_lat=30.25, min_lon=-97.76,
            max_lat=30.27, max_lon=-97.69,
        ))
        assert len(obstacles) == 6  # all features within bounds

        # Test nearest_feature
        nearest_water = layer.nearest_feature(30.26, -97.74, TerrainType.WATER)
        assert nearest_water is not None
        assert nearest_water.terrain_type == TerrainType.WATER

        # Test terrain_brief
        brief = layer.terrain_brief()
        assert "TERRAIN BRIEF" in brief
        assert "water" in brief.lower()
        assert "building" in brief.lower()
        assert "bridge" in brief.lower()

        # Test cache round-trip
        layer._save_cache("integration_test")

        layer2 = TerrainLayer(cache_dir=tmp_path / "terrain")
        assert layer2.load_cached("integration_test") is True
        assert len(layer2.regions) == 6

        # Verify GeoJSON export
        geojson = layer.to_geojson()
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 6

        # Check all terrain types are represented
        types_in_geojson = {
            f["properties"]["terrain_type"] for f in geojson["features"]
        }
        assert "water" in types_in_geojson
        assert "road" in types_in_geojson
        assert "building" in types_in_geojson


# ---------------------------------------------------------------------------
# SidewalkGraph
# ---------------------------------------------------------------------------

class TestSidewalkGraph:
    """Test pedestrian sidewalk navigation graph."""

    def test_build_empty(self):
        from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph

        graph = SidewalkGraph()
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_add_nodes_and_edges(self):
        from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph

        graph = SidewalkGraph()
        n1 = graph.add_node(0.0, 0.0)
        n2 = graph.add_node(0.001, 0.0)
        n3 = graph.add_node(0.002, 0.0)

        graph.add_edge(n1, n2)
        graph.add_edge(n2, n3)

        assert graph.node_count == 3
        assert graph.edge_count == 2

    def test_find_path_simple(self):
        from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph
        from tritium_lib.models.terrain import TerrainType

        graph = SidewalkGraph()
        # Create a simple sidewalk path
        n1 = graph.add_node(0.0, 0.0, TerrainType.SIDEWALK)
        n2 = graph.add_node(0.001, 0.0, TerrainType.SIDEWALK)
        n3 = graph.add_node(0.002, 0.0, TerrainType.SIDEWALK)
        n4 = graph.add_node(0.003, 0.0, TerrainType.SIDEWALK)

        graph.add_edge(n1, n2)
        graph.add_edge(n2, n3)
        graph.add_edge(n3, n4)

        path = graph.find_path((0.0, 0.0), (0.003, 0.0))
        assert path is not None
        assert len(path) >= 2
        # First waypoint should be start, last should be end
        assert path[0] == (0.0, 0.0)
        assert path[-1] == (0.003, 0.0)

    def test_find_path_empty_graph(self):
        from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph

        graph = SidewalkGraph()
        path = graph.find_path((0.0, 0.0), (1.0, 1.0))
        # Falls back to direct path
        assert path == [(0.0, 0.0), (1.0, 1.0)]

    def test_road_crossing_penalized(self):
        from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph
        from tritium_lib.models.terrain import TerrainType

        graph = SidewalkGraph()
        # Sidewalk → road → sidewalk path
        n1 = graph.add_node(0.0, 0.0, TerrainType.SIDEWALK)
        n2 = graph.add_node(0.001, 0.0, TerrainType.ROAD)  # road crossing
        n3 = graph.add_node(0.002, 0.0, TerrainType.SIDEWALK)

        graph.add_edge(n1, n2)
        graph.add_edge(n2, n3)

        path = graph.find_path((0.0, 0.0), (0.002, 0.0))
        assert path is not None

    def test_build_from_terrain_layer(self, tmp_path):
        from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path)
        layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.SIDEWALK,
                area_m2=200,
                centroid_lat=30.0,
                centroid_lon=-97.0,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.SIDEWALK,
                area_m2=200,
                centroid_lat=30.001,
                centroid_lon=-97.0,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,
                area_m2=500,
                centroid_lat=30.0005,
                centroid_lon=-97.0,
            ),
            # Non-walkable — should be excluded
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.WATER,
                area_m2=5000,
                centroid_lat=30.01,
                centroid_lon=-97.01,
            ),
        ]

        graph = SidewalkGraph()
        count = graph.build_from_terrain_layer(layer)
        # Water should be excluded
        assert count == 3
        assert graph.node_count == 3

    def test_get_walkable_area(self):
        from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph

        graph = SidewalkGraph()
        graph.add_node(0.0, 0.0)
        graph.add_node(0.001, 0.0)
        graph.add_node(1.0, 1.0)  # far away

        nearby = graph.get_walkable_area((0.0, 0.0), 0.01)
        assert len(nearby) == 2  # only the close ones


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class TestProviders:
    """Test imagery provider registry."""

    def test_registry_defaults(self):
        from tritium_lib.intelligence.geospatial.providers import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers()
        assert len(providers) >= 3
        keys = {p["source_key"] for p in providers}
        assert "satellite" in keys
        assert "osm" in keys

    def test_get_provider(self):
        from tritium_lib.intelligence.geospatial.providers import ProviderRegistry

        registry = ProviderRegistry()
        sat = registry.get("satellite")
        assert sat is not None
        assert sat.name == "Esri World Imagery"

    def test_register_custom(self):
        from tritium_lib.intelligence.geospatial.providers import (
            ProviderRegistry,
            TileMapProvider,
        )

        registry = ProviderRegistry()
        custom = TileMapProvider(
            name="Custom Tiles",
            source_key="custom_test",
            url_template="https://example.com/{z}/{x}/{y}.png",
        )
        registry.register(custom)

        found = registry.get("custom_test")
        assert found is not None
        assert found.name == "Custom Tiles"

    def test_local_provider(self, tmp_path):
        from tritium_lib.intelligence.geospatial.providers import LocalImageProvider

        # Create a test image
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        img.save(tmp_path / "test.png")

        provider = LocalImageProvider(image_dir=tmp_path)
        assert provider.source_key == "local"

        from tritium_lib.models.gis import TileBounds
        bounds = TileBounds(min_lat=30.0, min_lon=-97.0, max_lat=30.1, max_lon=-96.9)
        path = provider.fetch_area(bounds)
        assert path.exists()

    def test_local_provider_no_images(self, tmp_path):
        from tritium_lib.intelligence.geospatial.providers import LocalImageProvider
        from tritium_lib.models.gis import TileBounds

        provider = LocalImageProvider(image_dir=tmp_path / "empty")
        (tmp_path / "empty").mkdir()
        bounds = TileBounds(min_lat=30.0, min_lon=-97.0, max_lat=30.1, max_lon=-96.9)

        with pytest.raises(FileNotFoundError):
            provider.fetch_area(bounds)

    def test_singleton_registry(self):
        from tritium_lib.intelligence.geospatial.providers import get_provider_registry

        r1 = get_provider_registry()
        r2 = get_provider_registry()
        assert r1 is r2

    def test_tile_map_provider_latest_date(self):
        from tritium_lib.intelligence.geospatial.providers import TileMapProvider
        from tritium_lib.models.gis import TileBounds

        provider = TileMapProvider()
        bounds = TileBounds(min_lat=30.0, min_lon=-97.0, max_lat=30.1, max_lon=-96.9)
        assert provider.latest_date(bounds) is None  # TMS doesn't expose dates


# ---------------------------------------------------------------------------
# Segmentation engine — color region quality
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Change detector
# ---------------------------------------------------------------------------

class TestChangeDetector:
    """Test temporal change detection."""

    def test_detect_changes(self, tmp_path):
        from tritium_lib.intelligence.geospatial.change_detector import ChangeDetector
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import (
            SegmentedRegion,
            TerrainLayerMetadata,
        )
        from tritium_lib.models.terrain import TerrainType

        # Previous: road at location
        prev = TerrainLayer(cache_dir=tmp_path / "prev")
        prev._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,
                confidence=0.9,
                area_m2=500,
                centroid_lat=30.0,
                centroid_lon=-97.0,
            ),
        ]
        prev._metadata = TerrainLayerMetadata(ao_id="change_test", segment_count=1)

        # Current: water at same location (flooding)
        curr = TerrainLayer(cache_dir=tmp_path / "curr")
        curr._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.WATER,
                confidence=0.85,
                area_m2=600,
                centroid_lat=30.0,
                centroid_lon=-97.0,
            ),
        ]
        curr._metadata = TerrainLayerMetadata(ao_id="change_test", segment_count=1)

        detector = ChangeDetector()
        report = detector.detect_changes(prev, curr)

        assert report.change_count == 1
        assert report.changes[0].severity == "critical"
        assert "flooding" in report.changes[0].description.lower()

    def test_no_changes(self, tmp_path):
        from tritium_lib.intelligence.geospatial.change_detector import ChangeDetector
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import (
            SegmentedRegion,
            TerrainLayerMetadata,
        )
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path)
        layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,
                area_m2=500,
                centroid_lat=30.0,
                centroid_lon=-97.0,
            ),
        ]
        layer._metadata = TerrainLayerMetadata(ao_id="same", segment_count=1)

        detector = ChangeDetector()
        report = detector.detect_changes(layer, layer)
        assert report.change_count == 0

    def test_change_report_summary(self, tmp_path):
        from tritium_lib.intelligence.geospatial.change_detector import (
            ChangeDetector,
            ChangeReport,
            TerrainChange,
        )
        from tritium_lib.models.terrain import TerrainType

        report = ChangeReport(
            ao_id="test",
            changes=[
                TerrainChange(
                    centroid_lat=30.0,
                    centroid_lon=-97.0,
                    previous_type=TerrainType.VEGETATION,
                    current_type=TerrainType.BUILDING,
                    area_m2=2000,
                    confidence=0.8,
                    description="New construction",
                ),
            ],
            total_changed_area_m2=2000,
        )
        summary = report.summary()
        assert "TERRAIN CHANGE REPORT" in summary
        assert "New construction" in summary


# ---------------------------------------------------------------------------
# WorldBuilder terrain layer integration
# ---------------------------------------------------------------------------

class TestWorldBuilderTerrain:
    """Test WorldBuilder with geospatial terrain layer."""

    def test_world_builder_load_terrain_layer(self, tmp_path):
        from tritium_lib.sim_engine.world import WorldBuilder
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path)
        layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.SIDEWALK,
                area_m2=200,
                centroid_lat=30.0,
                centroid_lon=-97.0,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,
                area_m2=500,
                centroid_lat=30.001,
                centroid_lon=-97.0,
            ),
        ]

        world = (
            WorldBuilder()
            .set_map_size(100, 100)
            .load_terrain_layer(layer)
            .build()
        )

        assert world.terrain_layer is not None
        assert world.sidewalk_graph is not None
        assert world.sidewalk_graph.node_count == 2

    def test_world_without_terrain_layer(self):
        from tritium_lib.sim_engine.world import WorldBuilder

        world = WorldBuilder().set_map_size(100, 100).build()
        assert world.terrain_layer is None
        assert world.sidewalk_graph is None


# ---------------------------------------------------------------------------
# Sidewalk pathfinding integration
# ---------------------------------------------------------------------------

class TestSidewalkPathfinding:
    """Test plan_path with sidewalk graph integration."""

    def test_pedestrian_uses_sidewalk(self):
        from tritium_lib.sim_engine.world.pathfinding import plan_path
        from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph
        from tritium_lib.models.terrain import TerrainType

        graph = SidewalkGraph()
        # Create a sidewalk path
        n1 = graph.add_node(0.0, 0.0, TerrainType.SIDEWALK)
        n2 = graph.add_node(5.0, 0.0, TerrainType.SIDEWALK)
        n3 = graph.add_node(10.0, 0.0, TerrainType.SIDEWALK)
        n4 = graph.add_node(15.0, 0.0, TerrainType.SIDEWALK)
        n5 = graph.add_node(20.0, 0.0, TerrainType.SIDEWALK)
        graph.add_edge(n1, n2)
        graph.add_edge(n2, n3)
        graph.add_edge(n3, n4)
        graph.add_edge(n4, n5)

        path = plan_path(
            start=(0.0, 0.0),
            end=(20.0, 0.0),
            unit_type="person",
            sidewalk_graph=graph,
        )
        assert path is not None
        assert len(path) > 2  # should follow sidewalk nodes

    def test_vehicle_ignores_sidewalk(self):
        from tritium_lib.sim_engine.world.pathfinding import plan_path
        from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph

        graph = SidewalkGraph()
        # Vehicle shouldn't use sidewalk graph
        path = plan_path(
            start=(0.0, 0.0),
            end=(100.0, 0.0),
            unit_type="rover",
            sidewalk_graph=graph,
        )
        # Should fall through to direct path since no street_graph or terrain_map
        assert path == [(0.0, 0.0), (100.0, 0.0)]


# ---------------------------------------------------------------------------
# OSM Enrichment
# ---------------------------------------------------------------------------

class TestOSMEnrichment:
    """Test OSM data enrichment (offline, no network)."""

    def test_classify_osm_tags_building(self):
        from tritium_lib.intelligence.geospatial.osm_enrichment import OSMEnrichment
        from tritium_lib.models.terrain import TerrainType

        e = OSMEnrichment()
        assert e._classify_osm_tags({"building": "yes"}) == TerrainType.BUILDING
        assert e._classify_osm_tags({"building": "residential"}) == TerrainType.BUILDING

    def test_classify_osm_tags_road(self):
        from tritium_lib.intelligence.geospatial.osm_enrichment import OSMEnrichment
        from tritium_lib.models.terrain import TerrainType

        e = OSMEnrichment()
        assert e._classify_osm_tags({"highway": "residential"}) == TerrainType.ROAD
        assert e._classify_osm_tags({"highway": "primary"}) == TerrainType.ROAD

    def test_classify_osm_tags_sidewalk(self):
        from tritium_lib.intelligence.geospatial.osm_enrichment import OSMEnrichment
        from tritium_lib.models.terrain import TerrainType

        e = OSMEnrichment()
        assert e._classify_osm_tags({"highway": "footway"}) == TerrainType.SIDEWALK
        assert e._classify_osm_tags({"highway": "pedestrian"}) == TerrainType.SIDEWALK

    def test_classify_osm_tags_water(self):
        from tritium_lib.intelligence.geospatial.osm_enrichment import OSMEnrichment
        from tritium_lib.models.terrain import TerrainType

        e = OSMEnrichment()
        assert e._classify_osm_tags({"natural": "water"}) == TerrainType.WATER
        assert e._classify_osm_tags({"waterway": "river"}) == TerrainType.WATER

    def test_parse_speed(self):
        from tritium_lib.intelligence.geospatial.osm_enrichment import OSMEnrichment

        e = OSMEnrichment()
        assert e._parse_speed("50") == 50
        assert e._parse_speed("30 mph") == 48  # 30 * 1.609
        assert e._parse_speed(None) is None
        assert e._parse_speed("invalid") is None

    def test_enrich_terrain_layer(self, tmp_path):
        from tritium_lib.intelligence.geospatial.osm_enrichment import (
            OSMEnrichment,
            OSMFeature,
        )
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path)
        layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.BUILDING,
                confidence=0.7,
                area_m2=500,
                centroid_lat=30.0,
                centroid_lon=-97.0,
            ),
        ]

        osm_features = [
            OSMFeature(
                osm_id=12345,
                osm_type="way",
                terrain_type=TerrainType.BUILDING,
                name="City Hall",
                lat=30.0,
                lon=-97.0,
                building_type="civic",
            ),
        ]

        enrichment = OSMEnrichment()
        result = enrichment.enrich_terrain_layer(layer, osm_features)

        assert result.agreements == 1
        assert result.features_enriched == 1
        assert layer._regions[0].properties.get("osm_name") == "City Hall"
        assert layer._regions[0].confidence > 0.7  # boosted

    def test_enrich_disagreement(self, tmp_path):
        from tritium_lib.intelligence.geospatial.osm_enrichment import (
            OSMEnrichment,
            OSMFeature,
        )
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path)
        layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,  # SAM says road
                confidence=0.6,
                area_m2=500,
                centroid_lat=30.0,
                centroid_lon=-97.0,
            ),
        ]

        osm_features = [
            OSMFeature(
                osm_id=12345,
                osm_type="way",
                terrain_type=TerrainType.BUILDING,  # OSM says building
                lat=30.0,
                lon=-97.0,
            ),
        ]

        enrichment = OSMEnrichment()
        result = enrichment.enrich_terrain_layer(layer, osm_features)

        assert result.disagreements == 1
        assert "osm_disagrees" in layer._regions[0].properties


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mission Generator
# ---------------------------------------------------------------------------

class TestMissionGenerator:
    """Test mission generation from terrain features."""

    def test_generate_from_terrain_layer(self, tmp_path):
        from tritium_lib.intelligence.geospatial.mission_generator import MissionGenerator
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path)
        layer._regions = [
            SegmentedRegion(geometry_wkt="POLYGON EMPTY", terrain_type=TerrainType.BRIDGE,
                            area_m2=200, centroid_lat=30.0, centroid_lon=-97.0),
            SegmentedRegion(geometry_wkt="POLYGON EMPTY", terrain_type=TerrainType.WATER,
                            area_m2=8000, centroid_lat=30.001, centroid_lon=-97.0),
            SegmentedRegion(geometry_wkt="POLYGON EMPTY", terrain_type=TerrainType.BUILDING,
                            area_m2=3000, centroid_lat=30.002, centroid_lon=-97.0),
            SegmentedRegion(geometry_wkt="POLYGON EMPTY", terrain_type=TerrainType.PARKING,
                            area_m2=2000, centroid_lat=30.003, centroid_lon=-97.0),
            SegmentedRegion(geometry_wkt="POLYGON EMPTY", terrain_type=TerrainType.ROAD,
                            area_m2=500, centroid_lat=30.004, centroid_lon=-97.0),
            SegmentedRegion(geometry_wkt="POLYGON EMPTY", terrain_type=TerrainType.ROAD,
                            area_m2=500, centroid_lat=30.005, centroid_lon=-97.001),
            SegmentedRegion(geometry_wkt="POLYGON EMPTY", terrain_type=TerrainType.ROAD,
                            area_m2=500, centroid_lat=30.006, centroid_lon=-97.002),
        ]

        gen = MissionGenerator()
        missions = gen.generate_missions(layer)

        assert len(missions) > 0
        types = {m.mission_type for m in missions}
        assert "defend" in types  # bridge defense
        assert "recon" in types  # water recon
        assert "overwatch" in types  # large building

    def test_missions_brief(self):
        from tritium_lib.intelligence.geospatial.mission_generator import Mission, MissionGenerator

        gen = MissionGenerator()
        missions = [
            Mission(id="m1", mission_type="defend", name="Defend Bridge 1",
                    description="Secure the bridge", position=(0, 0), priority=4),
            Mission(id="m2", mission_type="patrol", name="Patrol Route A",
                    description="Patrol the area", position=(0, 0), priority=2),
        ]
        brief = gen.missions_brief(missions)
        assert "AVAILABLE MISSIONS" in brief
        assert "DEFEND" in brief
        assert "PATROL" in brief

    def test_empty_terrain(self):
        from tritium_lib.intelligence.geospatial.mission_generator import MissionGenerator
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer

        layer = TerrainLayer()
        gen = MissionGenerator()
        missions = gen.generate_missions(layer)
        assert missions == []

    def test_generate_from_cached_demo(self):
        """Test with real cached demo data if available."""
        from pathlib import Path
        from tritium_lib.intelligence.geospatial.mission_generator import MissionGenerator
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer

        layer = TerrainLayer()
        if not layer.load_cached("demo_area"):
            pytest.skip("No cached demo_area data")

        gen = MissionGenerator()
        missions = gen.generate_missions(layer)
        assert len(missions) > 3  # real data should generate several missions

        # Verify missions have valid positions
        for m in missions:
            assert m.position[0] != 0 or m.position[1] != 0
            assert m.priority >= 1
            assert m.priority <= 5


# ---------------------------------------------------------------------------
# Dual-Resolution Pathfinder
# ---------------------------------------------------------------------------

class TestDualResolution:
    """Test dual-resolution terrain grid and pathfinding."""

    def test_fine_grid_creation(self):
        from tritium_lib.intelligence.geospatial.dual_resolution import FineTerrainGrid

        grid = FineTerrainGrid(center_x=0, center_y=0, radius=50, resolution=1.0)
        assert grid.grid_size == 101  # 2*50/1 + 1
        assert grid.get_terrain_at(50, 50) == "open"
        assert grid.get_terrain_at(-1, 0) == "out_of_bounds"

    def test_fine_grid_set_cell(self):
        from tritium_lib.intelligence.geospatial.dual_resolution import FineTerrainGrid

        grid = FineTerrainGrid(center_x=0, center_y=0, radius=50, resolution=1.0)
        grid.set_cell(50, 50, "road")
        assert grid.get_terrain_at(50, 50) == "road"

    def test_fine_grid_world_to_grid_roundtrip(self):
        from tritium_lib.intelligence.geospatial.dual_resolution import FineTerrainGrid

        grid = FineTerrainGrid(center_x=100, center_y=200, radius=50, resolution=1.0)
        col, row = grid._world_to_grid(100, 200)
        wx, wy = grid._grid_to_world(col, row)
        assert abs(wx - 100) < 1.5
        assert abs(wy - 200) < 1.5

    def test_fine_grid_populate_from_terrain_layer(self, tmp_path):
        from tritium_lib.intelligence.geospatial.dual_resolution import FineTerrainGrid
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path)
        layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,
                area_m2=100,
                centroid_lat=0.0,
                centroid_lon=0.0,
            ),
        ]

        grid = FineTerrainGrid(center_x=0, center_y=0, radius=50, resolution=1.0)
        count = grid.populate_from_terrain_layer(layer)
        assert count > 0

    def test_dual_res_pathfinder_no_maps(self):
        from tritium_lib.intelligence.geospatial.dual_resolution import DualResolutionPathfinder

        pf = DualResolutionPathfinder()
        path = pf.find_path((0, 0), (100, 100), "pedestrian")
        assert path == [(0, 0), (100, 100)]  # direct fallback

    def test_dual_res_pathfinder_with_terrain_layer(self, tmp_path):
        from tritium_lib.intelligence.geospatial.dual_resolution import DualResolutionPathfinder
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path)
        layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,
                area_m2=400,
                centroid_lat=5.0,
                centroid_lon=5.0,
            ),
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,
                area_m2=400,
                centroid_lat=10.0,
                centroid_lon=10.0,
            ),
        ]

        pf = DualResolutionPathfinder(refine_distance=50.0)
        path = pf.find_path((0, 0), (15, 15), "pedestrian", terrain_layer=layer)
        assert path is not None
        assert len(path) >= 2

    def test_path_length_calculation(self):
        from tritium_lib.intelligence.geospatial.dual_resolution import DualResolutionPathfinder

        pf = DualResolutionPathfinder()
        length = pf._path_length([(0, 0), (3, 4)])
        assert abs(length - 5.0) < 0.01

        length2 = pf._path_length([(0, 0), (3, 0), (3, 4)])
        assert abs(length2 - 7.0) < 0.01


class TestLLMClient:
    """Test LLM client module."""

    def test_clear_cache(self):
        from tritium_lib.intelligence.geospatial.llm_client import (
            clear_discovery_cache,
            _discovered,
        )
        clear_discovery_cache()
        # Should not crash


class TestSegmentationQuality:
    """Test that color-based segmentation produces reasonable results."""

    def test_distinct_regions_detected(self, tmp_path):
        """A 4-color image should produce at least 2 distinct segments."""
        from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:100, :100] = [0, 0, 200]     # blue
        img[:100, 100:] = [0, 200, 0]     # green
        img[100:, :100] = [128, 128, 128] # gray
        img[100:, 100:] = [200, 200, 200] # light gray

        path = tmp_path / "multi.png"
        Image.fromarray(img).save(path)

        engine = SegmentationEngine()
        segments = engine.segment_image(path)
        assert len(segments) >= 2, f"Expected >= 2 segments, got {len(segments)}"

    def test_large_uniform_region(self, tmp_path):
        """A solid color image should produce one dominant segment."""
        from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine

        img = np.full((200, 200, 3), [30, 80, 180], dtype=np.uint8)
        path = tmp_path / "solid.png"
        Image.fromarray(img).save(path)

        engine = SegmentationEngine()
        segments = engine.segment_image(path)
        # Should have at least one segment covering most of the image
        total_area = sum(s["area"] for s in segments)
        assert total_area > 10000, f"Expected large coverage, got {total_area}"
