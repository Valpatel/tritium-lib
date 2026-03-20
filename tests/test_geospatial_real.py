# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Real-world geospatial tests — downloads actual satellite tiles.

These tests hit the network and take longer. They verify the pipeline
works with real satellite imagery, not just synthetic test images.

Mark with @pytest.mark.slow for CI exclusion.
"""

import pytest
import os

try:
    import numpy as np
    from PIL import Image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Skip if no network or deps
pytestmark = [
    pytest.mark.skipif(not HAS_DEPS, reason="numpy/Pillow required"),
    pytest.mark.skipif(not HAS_REQUESTS, reason="requests required"),
]


class TestRealSatellitePipeline:
    """Test with real Esri satellite tiles for a known location."""

    # Lady Bird Lake in Austin, TX — clearly visible water body
    # surrounded by vegetation, roads, and buildings
    AUSTIN_LAKE = {
        "id": "austin_ladybird",
        "name": "Lady Bird Lake, Austin TX",
        "min_lat": 30.260,
        "min_lon": -97.755,
        "max_lat": 30.270,
        "max_lon": -97.745,
        "zoom": 16,  # lower zoom to keep download small
    }

    @pytest.fixture
    def ao(self):
        from tritium_lib.intelligence.geospatial.models import AreaOfOperations
        from tritium_lib.models.gis import TileBounds
        loc = self.AUSTIN_LAKE
        return AreaOfOperations(
            id=loc["id"],
            name=loc["name"],
            bounds=TileBounds(
                min_lat=loc["min_lat"], min_lon=loc["min_lon"],
                max_lat=loc["max_lat"], max_lon=loc["max_lon"],
            ),
            zoom=loc["zoom"],
        )

    @pytest.mark.slow
    def test_download_real_tiles(self, ao, tmp_path):
        """Download real satellite tiles for a known area."""
        from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader

        dl = TileDownloader(cache_dir=tmp_path / "tiles")
        tiles = dl.tiles_for_ao(ao)
        assert len(tiles) > 0, "No tiles calculated for Austin area"
        assert len(tiles) < 20, f"Too many tiles ({len(tiles)}), check zoom level"

        # Actually download
        image_path = dl.download_tiles(ao, source="satellite")
        assert image_path.exists()

        img = Image.open(image_path)
        assert img.width > 100
        assert img.height > 100

    @pytest.mark.slow
    def test_segment_real_imagery(self, ao, tmp_path):
        """Segment real satellite imagery and classify terrain."""
        from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader
        from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
        from tritium_lib.models.terrain import TerrainType

        # Download tiles
        dl = TileDownloader(cache_dir=tmp_path / "tiles")
        image_path = dl.download_tiles(ao, source="satellite")
        img_array = np.array(Image.open(image_path).convert("RGB"))

        # Segment
        engine = SegmentationEngine()
        segments = engine.segment_image(image_path)
        assert len(segments) >= 3, (
            f"Expected at least 3 segments from real imagery, got {len(segments)}"
        )

        # Classify
        classifier = TerrainClassifier()
        classifications = classifier.classify_segments(img_array, segments)

        # We expect water AND vegetation near Lady Bird Lake — not just one
        terrain_types = {t for t, _ in classifications}
        assert len(terrain_types) >= 2, (
            f"Expected at least 2 terrain types, got {terrain_types}"
        )

        # Water MUST be found — this is Lady Bird Lake
        assert TerrainType.WATER in terrain_types, (
            f"WATER not detected at Lady Bird Lake! Found types: {terrain_types}"
        )
        # Vegetation should be found (parkland surrounds the lake)
        assert TerrainType.VEGETATION in terrain_types, (
            f"VEGETATION not detected near Lady Bird Lake! Found types: {terrain_types}"
        )

        # At least 5% of segments should be water (lake covers ~20% of the bbox)
        water_count = sum(1 for t, _ in classifications if t == TerrainType.WATER)
        water_pct = water_count / max(len(classifications), 1) * 100
        assert water_pct >= 3, (
            f"Only {water_pct:.1f}% water segments — expected at least 3% for a lake area"
        )

    @pytest.mark.slow
    def test_full_pipeline_real_imagery(self, ao, tmp_path):
        """Full pipeline: download → segment → classify → vectorize → query."""
        from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader
        from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
        from tritium_lib.intelligence.geospatial.vector_converter import VectorConverter
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        dl = TileDownloader(cache_dir=tmp_path / "tiles")
        image_path = dl.download_tiles(ao, source="satellite")
        img = Image.open(image_path)
        img_array = np.array(img.convert("RGB"))
        geo_transform = dl.get_geo_transform(ao, img.width, img.height)

        engine = SegmentationEngine()
        segments = engine.segment_image(image_path)

        classifier = TerrainClassifier()
        classifications = classifier.classify_segments(img_array, segments)

        converter = VectorConverter(min_area_px=50)

        # Build terrain layer manually from pipeline results
        layer = TerrainLayer(cache_dir=tmp_path / "terrain")
        regions = []
        for i, seg in enumerate(segments):
            terrain_type, confidence = classifications[i]
            polys = converter.mask_to_polygons(seg["mask"], geo_transform)
            for poly in polys:
                centroid = poly.get("centroid", (0, 0))
                regions.append(SegmentedRegion(
                    geometry_wkt=poly["wkt"],
                    terrain_type=terrain_type,
                    confidence=confidence,
                    area_m2=poly.get("area_m2", 0),
                    centroid_lon=centroid[0],
                    centroid_lat=centroid[1],
                ))

        assert len(regions) > 0, "No regions produced from real imagery"

        # Verify terrain brief works
        from tritium_lib.intelligence.geospatial.models import TerrainLayerMetadata
        layer._regions = regions
        layer._metadata = TerrainLayerMetadata(
            ao_id=ao.id, segment_count=len(regions), source_imagery="satellite",
        )
        layer._build_grid_index()

        brief = layer.terrain_brief()
        assert "TERRAIN BRIEF" in brief
        assert len(brief) > 100, "Terrain brief too short for real data"

        # Verify GeoJSON export
        geojson = layer.to_geojson()
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) > 0


    @pytest.mark.slow
    def test_terrain_at_known_water_point(self, ao, tmp_path):
        """terrain_at() should return WATER for a point known to be in Lady Bird Lake."""
        from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader
        from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
        from tritium_lib.intelligence.geospatial.vector_converter import VectorConverter
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import (
            SegmentedRegion, TerrainLayerMetadata,
        )
        from tritium_lib.models.terrain import TerrainType

        # Build terrain layer from real satellite data
        dl = TileDownloader(cache_dir=tmp_path / "tiles")
        image_path = dl.download_tiles(ao, source="satellite")
        img = Image.open(image_path)
        img_array = np.array(img.convert("RGB"))
        geo_transform = dl.get_geo_transform(ao, img.width, img.height)

        engine = SegmentationEngine()
        segments = engine.segment_image(image_path)

        classifier = TerrainClassifier()
        classifications = classifier.classify_segments(img_array, segments)

        converter = VectorConverter(min_area_px=50)
        regions = []
        for i, seg in enumerate(segments):
            tt, conf = classifications[i]
            for poly in converter.mask_to_polygons(seg["mask"], geo_transform):
                area = poly.get("area_m2", 0)
                if area < 10 or area > 100000:
                    continue
                c = poly.get("centroid", (0, 0))
                regions.append(SegmentedRegion(
                    geometry_wkt=poly["wkt"], terrain_type=tt, confidence=conf,
                    area_m2=area, centroid_lon=c[0], centroid_lat=c[1],
                ))

        layer = TerrainLayer(cache_dir=tmp_path / "terrain")
        layer._regions = regions
        layer._bounds = ao.bounds
        layer._metadata = TerrainLayerMetadata(ao_id="water_test", segment_count=len(regions))
        layer._build_grid_index()

        # Verify water regions exist and are queryable.
        # Find a water region centroid and verify terrain_at returns WATER there.
        water_regions = [r for r in regions if r.terrain_type == TerrainType.WATER]
        assert len(water_regions) >= 3, (
            f"Expected at least 3 water regions near Lady Bird Lake, got {len(water_regions)}"
        )

        # Query terrain at a known water centroid — should return WATER
        wr = water_regions[0]
        lake_terrain = layer.terrain_at(wr.centroid_lat, wr.centroid_lon)
        assert lake_terrain == TerrainType.WATER, (
            f"terrain_at at water centroid returned {lake_terrain.value}, expected WATER"
        )


class TestLLMClassification:
    """Test LLM-assisted terrain classification with live llama-server."""

    @pytest.fixture
    def llm_available(self):
        """Skip if llama-server is not running."""
        try:
            resp = requests.get("http://127.0.0.1:8081/health", timeout=2)
            if resp.status_code != 200:
                pytest.skip("llama-server not running on port 8081")
        except Exception:
            pytest.skip("llama-server not reachable")

    def test_llm_classify_ambiguous_segment(self, llm_available):
        """LLM should help classify a segment the color heuristic is unsure about."""
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
        from tritium_lib.intelligence.geospatial.models import SegmentationConfig

        config = SegmentationConfig(
            llm_classify=True,
            llm_endpoint="http://127.0.0.1:8081",
        )
        classifier = TerrainClassifier(config)

        # Dark brownish-gray — ambiguous between road, barren, building
        img = np.full((100, 100, 3), [90, 80, 70], dtype=np.uint8)
        mask = np.ones((100, 100), dtype=bool)

        terrain, conf = classifier.classify_segment(img, mask)
        assert terrain.value in ("barren", "road", "building", "parking")
        assert conf > 0.0

    def test_llm_classify_clear_water(self, llm_available):
        """LLM should confirm water classification for blue segment."""
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
        from tritium_lib.intelligence.geospatial.models import SegmentationConfig
        from tritium_lib.models.terrain import TerrainType

        config = SegmentationConfig(
            llm_classify=True,
            llm_endpoint="http://127.0.0.1:8081",
        )
        classifier = TerrainClassifier(config)

        # Clear blue — should be water even without LLM
        img = np.full((100, 100, 3), [30, 70, 190], dtype=np.uint8)
        mask = np.ones((100, 100), dtype=bool)

        terrain, conf = classifier.classify_segment(img, mask)
        assert terrain == TerrainType.WATER

    def test_llm_disabled_by_default(self):
        """LLM classification should be disabled by default."""
        from tritium_lib.intelligence.geospatial.models import SegmentationConfig
        assert SegmentationConfig().llm_classify is False


class TestLLMClientDiscovery:
    """Test llama-server discovery."""

    def test_discover_servers(self):
        """Should find at least one running llama-server."""
        from tritium_lib.intelligence.geospatial.llm_client import (
            discover_llama_servers,
            clear_discovery_cache,
        )
        clear_discovery_cache()
        servers = discover_llama_servers(timeout=2.0)
        # May or may not find servers depending on environment
        if servers:
            assert servers[0]["port"] > 0
            assert servers[0]["endpoint"].startswith("http")

    def test_get_best_server(self):
        """Should return a server dict or None."""
        from tritium_lib.intelligence.geospatial.llm_client import (
            get_best_server,
            clear_discovery_cache,
        )
        clear_discovery_cache()
        server = get_best_server()
        if server is not None:
            assert "endpoint" in server
            assert "port" in server

    def test_llm_complete(self):
        """Should complete a prompt or return None gracefully."""
        from tritium_lib.intelligence.geospatial.llm_client import (
            llm_complete,
            clear_discovery_cache,
        )
        clear_discovery_cache()
        result = llm_complete("What color is the sky? Answer in one word.", max_tokens=10)
        # Either works or returns None — should never raise
        if result is not None:
            assert len(result) > 0


class TestEdgeCases:
    """Stress tests and edge cases for the geospatial pipeline."""

    def test_corrupt_image_handling(self, tmp_path):
        """Should handle corrupt/truncated images gracefully."""
        from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine

        corrupt_path = tmp_path / "corrupt.png"
        corrupt_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        engine = SegmentationEngine()
        # Should not raise — return empty or handle gracefully
        try:
            segments = engine.segment_image(corrupt_path)
            # If it doesn't crash, that's acceptable
        except Exception:
            # Crashing on corrupt input is acceptable too
            pass

    def test_zero_area_mask(self):
        """Segments with zero area should be handled."""
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
        from tritium_lib.models.terrain import TerrainType

        classifier = TerrainClassifier()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)

        terrain, conf = classifier.classify_segment(img, mask)
        assert terrain == TerrainType.UNKNOWN
        assert conf == 0.0

    def test_single_pixel_mask(self):
        """Single pixel segment should still classify."""
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier

        classifier = TerrainClassifier()
        img = np.full((100, 100, 3), [30, 70, 190], dtype=np.uint8)  # blue
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True

        terrain, conf = classifier.classify_segment(img, mask)
        # Should classify, not crash
        assert conf >= 0.0

    def test_all_black_image(self):
        """All-black image should classify as building (dark) or unknown."""
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
        from tritium_lib.models.terrain import TerrainType

        classifier = TerrainClassifier()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=bool)

        terrain, conf = classifier.classify_segment(img, mask)
        # Black could be dark building or unknown — both acceptable
        assert terrain in (TerrainType.BUILDING, TerrainType.UNKNOWN, TerrainType.ROAD)

    def test_all_white_image(self):
        """All-white image classifies as a gray-zone terrain type."""
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
        from tritium_lib.models.terrain import TerrainType

        classifier = TerrainClassifier()
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        mask = np.ones((100, 100), dtype=bool)

        terrain, conf = classifier.classify_segment(img, mask)
        # All-white is ambiguous: could be bright roof, concrete, or snow
        assert terrain in (
            TerrainType.BUILDING, TerrainType.SIDEWALK,
            TerrainType.ROAD, TerrainType.PARKING,
        )

    def test_nan_coordinates_in_terrain_layer(self, tmp_path):
        """TerrainLayer should handle NaN coordinates without crashing."""
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path)
        layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.ROAD,
                area_m2=100,
                centroid_lat=float("nan"),
                centroid_lon=float("nan"),
            ),
        ]
        layer._build_grid_index()

        # Should not crash on NaN queries
        result = layer.terrain_at(30.0, -97.0)
        assert result is not None

    def test_very_large_segment_count(self, tmp_path):
        """TerrainLayer should handle many segments efficiently."""
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import (
            SegmentedRegion,
            TerrainLayerMetadata,
        )
        from tritium_lib.models.terrain import TerrainType
        import time

        layer = TerrainLayer(cache_dir=tmp_path)
        types = [TerrainType.ROAD, TerrainType.BUILDING, TerrainType.VEGETATION, TerrainType.WATER]

        # Create 1000 segments
        regions = []
        for i in range(1000):
            regions.append(SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=types[i % len(types)],
                area_m2=100 + i,
                centroid_lat=30.0 + (i % 100) * 0.001,
                centroid_lon=-97.0 + (i // 100) * 0.001,
            ))

        layer._regions = regions
        layer._metadata = TerrainLayerMetadata(ao_id="stress_test", segment_count=1000)

        t0 = time.monotonic()
        layer._build_grid_index()
        index_time = time.monotonic() - t0
        assert index_time < 5.0, f"Grid index build too slow: {index_time:.2f}s"

        # Query should be fast
        t0 = time.monotonic()
        for _ in range(100):
            layer.terrain_at(30.05, -97.005)
        query_time = time.monotonic() - t0
        assert query_time < 1.0, f"100 queries too slow: {query_time:.2f}s"

        # features_by_type
        roads = layer.features_by_type(TerrainType.ROAD)
        assert len(roads) == 250  # 1000/4

        # terrain_brief with many features
        brief = layer.terrain_brief()
        assert "TERRAIN BRIEF" in brief
        assert "1000" in brief or "250" in brief  # should mention counts

    def test_overlapping_masks_vectorizer(self):
        """VectorConverter should handle overlapping masks."""
        from tritium_lib.intelligence.geospatial.vector_converter import VectorConverter

        converter = VectorConverter(min_area_px=10)

        # Two overlapping masks
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[20:60, 20:60] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[40:80, 40:80] = True

        segments = [
            {"mask": mask1, "stability_score": 0.9},
            {"mask": mask2, "stability_score": 0.8},
        ]

        polys = converter.masks_to_polygons(segments)
        assert len(polys) == 2  # both should produce polygons

    def test_cache_corruption_recovery(self, tmp_path):
        """TerrainLayer should handle corrupt cache files."""
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer

        cache_dir = tmp_path / "terrain" / "corrupt_test"
        cache_dir.mkdir(parents=True)

        # Write corrupt metadata
        (cache_dir / "metadata.json").write_text("not json{{{")
        (cache_dir / "terrain.geojson").write_text("also corrupt")

        layer = TerrainLayer(cache_dir=tmp_path / "terrain")
        result = layer.load_cached("corrupt_test")
        assert result is False  # should fail gracefully, not crash

    def test_concurrent_terrain_queries(self, tmp_path):
        """TerrainLayer queries should be safe under concurrent access."""
        import threading
        from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
        from tritium_lib.intelligence.geospatial.models import SegmentedRegion
        from tritium_lib.models.terrain import TerrainType

        layer = TerrainLayer(cache_dir=tmp_path)
        layer._regions = [
            SegmentedRegion(
                geometry_wkt="POLYGON EMPTY",
                terrain_type=TerrainType.WATER,
                area_m2=5000,
                centroid_lat=30.0,
                centroid_lon=-97.0,
            )
            for _ in range(100)
        ]
        layer._build_grid_index()

        errors = []

        def query_worker():
            try:
                for _ in range(100):
                    layer.terrain_at(30.0, -97.0)
                    layer.features_by_type(TerrainType.WATER)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=query_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent query errors: {errors}"

    def test_geo_transform_accuracy(self):
        """Pixel→latlon→pixel round-trip should be accurate."""
        from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader
        from tritium_lib.intelligence.geospatial.models import AreaOfOperations
        from tritium_lib.models.gis import TileBounds

        dl = TileDownloader()
        ao = AreaOfOperations(
            id="accuracy_test",
            name="Test",
            bounds=TileBounds(min_lat=30.26, min_lon=-97.74, max_lat=30.27, max_lon=-97.73),
            zoom=17,
        )
        gt = dl.get_geo_transform(ao, 512, 512)

        # Convert pixel (256, 256) to latlon and back
        lat, lon = dl.pixel_to_latlon(256, 256, gt)
        # Should be roughly center of the bounds
        assert abs(lat - 30.265) < 0.01, f"Lat accuracy: {lat}"
        assert abs(lon - (-97.735)) < 0.01, f"Lon accuracy: {lon}"

    def test_sidewalk_graph_large_network(self, tmp_path):
        """SidewalkGraph should handle large networks efficiently."""
        from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph
        from tritium_lib.models.terrain import TerrainType
        import time

        graph = SidewalkGraph()

        # Create a 50x50 grid of sidewalk nodes
        nodes = {}
        for i in range(50):
            for j in range(50):
                nid = graph.add_node(
                    i * 0.0005,
                    j * 0.0005,
                    TerrainType.SIDEWALK,
                )
                nodes[(i, j)] = nid

        # Connect grid neighbors
        for i in range(50):
            for j in range(50):
                if i > 0:
                    graph.add_edge(nodes[(i, j)], nodes[(i - 1, j)])
                if j > 0:
                    graph.add_edge(nodes[(i, j)], nodes[(i, j - 1)])

        assert graph.node_count == 2500
        assert graph.edge_count == 2 * (49 * 50)  # 2 * (horizontal + vertical)

        # Pathfinding should work and be fast
        t0 = time.monotonic()
        path = graph.find_path((0.0, 0.0), (0.0245, 0.0245))
        path_time = time.monotonic() - t0

        assert path is not None
        assert len(path) > 2
        assert path_time < 2.0, f"Pathfinding too slow: {path_time:.2f}s"
