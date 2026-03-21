#!/usr/bin/env bash
# Geospatial Segmentation Demo — process a real area end-to-end
# Downloads satellite tiles, segments, classifies, caches, reports.
#
# Usage:
#   ./geo-demo.sh                    # Default: Austin downtown
#   ./geo-demo.sh 30.26 -97.75 30.28 -97.73 17  # Custom bbox + zoom
set -euo pipefail

cd "$(dirname "$0")"

# Default: downtown Austin, TX — mix of water, roads, buildings, vegetation
MIN_LAT="${1:-30.260}"
MIN_LON="${2:--97.755}"
MAX_LAT="${3:-30.275}"
MAX_LON="${4:--97.740}"
ZOOM="${5:-16}"

echo "╔══════════════════════════════════════════════════╗"
echo "║  TRITIUM GEOSPATIAL SEGMENTATION DEMO           ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Area: ($MIN_LAT, $MIN_LON) → ($MAX_LAT, $MAX_LON)"
echo "║  Zoom: $ZOOM"
echo "╚══════════════════════════════════════════════════╝"
echo ""

python3 - <<'PYTHON_SCRIPT' "$MIN_LAT" "$MIN_LON" "$MAX_LAT" "$MAX_LON" "$ZOOM"
import sys
import time
import json
from pathlib import Path

min_lat, min_lon, max_lat, max_lon, zoom = (
    float(sys.argv[1]), float(sys.argv[2]),
    float(sys.argv[3]), float(sys.argv[4]),
    int(sys.argv[5]),
)

from tritium_lib.models.gis import TileBounds
from tritium_lib.intelligence.geospatial.models import (
    AreaOfOperations, SegmentationConfig,
)
from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader
from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine
from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
from tritium_lib.intelligence.geospatial.vector_converter import VectorConverter
from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
from tritium_lib.intelligence.geospatial.models import SegmentedRegion, TerrainLayerMetadata
from tritium_lib.intelligence.geospatial.sidewalk_graph import SidewalkGraph

import numpy as np
from PIL import Image

ao = AreaOfOperations(
    id="demo_area",
    name="Demo Area",
    bounds=TileBounds(min_lat=min_lat, min_lon=min_lon, max_lat=max_lat, max_lon=max_lon),
    zoom=zoom,
)

t_total = time.monotonic()

# Step 1: Download tiles
print("▶ Step 1: Downloading satellite tiles...")
t0 = time.monotonic()
dl = TileDownloader(cache_dir=Path("data/cache/tiles"))
tiles = dl.tiles_for_ao(ao)
print(f"  Tiles needed: {len(tiles)}")
image_path = dl.download_tiles(ao, source="satellite")
print(f"  Stitched image: {image_path}")
print(f"  ✓ Download complete ({time.monotonic()-t0:.1f}s)")
print()

# Step 2: Segment
print("▶ Step 2: Segmenting image...")
t0 = time.monotonic()

# Check for llama-server for LLM-assisted classification
llm_available = False
try:
    import requests
    r = requests.get("http://127.0.0.1:8081/health", timeout=1)
    llm_available = r.status_code == 200
except Exception:
    pass

config = SegmentationConfig(
    llm_classify=llm_available,
    llm_endpoint="http://127.0.0.1:8081",
)

engine = SegmentationEngine(config)
segments = engine.segment_image(image_path)
print(f"  Segments found: {len(segments)}")
print(f"  ✓ Segmentation complete ({time.monotonic()-t0:.1f}s)")
print()

# Step 3: Classify
print("▶ Step 3: Classifying terrain...")
t0 = time.monotonic()
img = Image.open(image_path)
img_array = np.array(img.convert("RGB"))
geo_transform = dl.get_geo_transform(ao, img.width, img.height)

classifier = TerrainClassifier(config)
classifications = classifier.classify_segments(img_array, segments)

type_counts = {}
for terrain, conf in classifications:
    name = terrain.value
    type_counts[name] = type_counts.get(name, 0) + 1

print(f"  Classification (LLM: {'enabled' if llm_available else 'disabled'}):")
for name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"    {name}: {count} segments")
print(f"  ✓ Classification complete ({time.monotonic()-t0:.1f}s)")
print()

# Step 4: Vectorize + build terrain layer
print("▶ Step 4: Building terrain layer...")
t0 = time.monotonic()
converter = VectorConverter(min_area_px=50)

layer = TerrainLayer(cache_dir=Path("data/cache/terrain"))
regions = []
for i, seg in enumerate(segments):
    terrain_type, confidence = classifications[i]
    polys = converter.mask_to_polygons(seg["mask"], geo_transform)
    for poly in polys:
        area_m2 = poly.get("area_m2", 0)
        if area_m2 < config.min_area_m2 or area_m2 > config.max_area_m2:
            continue
        centroid = poly.get("centroid", (0, 0))
        regions.append(SegmentedRegion(
            geometry_wkt=poly["wkt"],
            terrain_type=terrain_type,
            confidence=confidence,
            area_m2=area_m2,
            centroid_lon=centroid[0],
            centroid_lat=centroid[1],
        ))

elapsed = time.monotonic() - t_total
layer._regions = regions
layer._bounds = ao.bounds
layer._metadata = TerrainLayerMetadata(
    ao_id=ao.id,
    segment_count=len(regions),
    processing_time_s=elapsed,
    source_imagery="satellite",
    bounds=ao.bounds,
)
layer._build_grid_index()
layer._save_cache(ao.id)
print(f"  Satellite features: {len(regions)}")

# Step 4b: Fuse with OSM data for richer terrain
print("  Fusing with OpenStreetMap...")
try:
    from tritium_lib.intelligence.geospatial.osm_enrichment import OSMEnrichment
    osm_enrichment = OSMEnrichment()
    osm_features = osm_enrichment.fetch_osm(ao.bounds)
    if osm_features:
        osm_regions = []
        osm_cells = set()
        for f in osm_features:
            if f.lat == 0 and f.lon == 0:
                continue
            props = {"osm_id": f.osm_id, "source": "osm"}
            if f.name:
                props["osm_name"] = f.name
            if f.road_type:
                props["road_type"] = f.road_type
            osm_regions.append(SegmentedRegion(
                geometry_wkt="POLYGON EMPTY", terrain_type=f.terrain_type,
                confidence=0.85, area_m2=100,
                centroid_lat=f.lat, centroid_lon=f.lon, properties=props,
            ))
            osm_cells.add((int(f.lon * 10000), int(f.lat * 10000)))
        # Add satellite gap-fill
        sat_fill = 0
        for r in regions:
            cell = (int(r.centroid_lon * 10000), int(r.centroid_lat * 10000))
            if cell not in osm_cells:
                osm_regions.append(r)
                sat_fill += 1
        regions = osm_regions
        layer._regions = regions
        layer._metadata.segment_count = len(regions)
        layer._metadata.source_imagery = "satellite+osm"
        layer._build_grid_index()
        layer._save_cache(ao.id)
        named = sum(1 for r in regions if r.properties.get("osm_name"))
        print(f"  OSM: {len(osm_features)} features, {named} named")
        print(f"  Fused: {len(regions)} total ({len(osm_features)} OSM + {sat_fill} satellite)")
except Exception as e:
    print(f"  OSM fusion skipped: {e}")

print(f"  ✓ Terrain layer built ({time.monotonic()-t0:.1f}s)")
print()

# Step 5: Build sidewalk graph
print("▶ Step 5: Building navigation graph...")
t0 = time.monotonic()
sg = SidewalkGraph()
node_count = sg.build_from_terrain_layer(layer)
print(f"  Navigation nodes: {sg.node_count}")
print(f"  Navigation edges: {sg.edge_count}")

# Test a pathfinding query
if sg.node_count >= 2:
    nodes = list(sg._nodes.values())
    start = (nodes[0].x, nodes[0].y)
    mid = (nodes[len(nodes)//2].x, nodes[len(nodes)//2].y)
    end = (nodes[-1].x, nodes[-1].y)
    path = sg.find_path(start, end)
    if path and len(path) > 2:
        print(f"  Sample path (full AO): {len(path)} waypoints")
    else:
        print(f"  Sample path: direct (nodes not connected)")
    path2 = sg.find_path(start, mid)
    if path2 and len(path2) > 2:
        print(f"  Sample path (half AO): {len(path2)} waypoints")

print(f"  ✓ Navigation graph built ({time.monotonic()-t0:.1f}s)")
print()

# Step 6: Terrain brief
print("▶ Step 6: Terrain Brief")
print("─" * 50)
print(layer.terrain_brief())
print("─" * 50)
print()

# Step 7: Mission Generation
print("▶ Step 7: Mission Generation")
from tritium_lib.intelligence.geospatial.mission_generator import MissionGenerator
gen = MissionGenerator()
missions = gen.generate_missions(layer)
print(f"  Missions generated: {len(missions)}")
for m in missions[:8]:
    print(f"    [{m.priority}] {m.mission_type.upper()}: {m.name}")
print()

# Step 8: Export GeoJSON
geojson = layer.to_geojson()
geojson_path = Path("data/cache/terrain") / ao.id / "terrain.geojson"
print(f"▶ GeoJSON exported: {geojson_path}")
print(f"  Features: {len(geojson['features'])}")
print()

# Step 9: Summary
total_time = time.monotonic() - t_total
total_area = sum(r.area_m2 for r in regions)
print("╔══════════════════════════════════════════════════╗")
print(f"║  COMPLETE — {len(regions)} terrain features in {total_time:.1f}s")
print(f"║  Total area: {total_area:,.0f} m²")
print(f"║  Image: {img.width}×{img.height} px")
print(f"║  Cache: data/cache/terrain/{ao.id}/")
if llm_available:
    print(f"║  LLM: llama-server on :8081 (assisted classification)")
else:
    print(f"║  LLM: not available (color heuristic only)")
print("╚══════════════════════════════════════════════════╝")
PYTHON_SCRIPT
