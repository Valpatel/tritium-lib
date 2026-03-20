# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""CLI entry point for geospatial operations.

Usage:
    python -m tritium_lib.intelligence.geospatial.cli process --lat 30.265 --lon -97.748 --radius 500
    python -m tritium_lib.intelligence.geospatial.cli brief demo_area
    python -m tritium_lib.intelligence.geospatial.cli missions demo_area
    python -m tritium_lib.intelligence.geospatial.cli query --lat 30.266 --lon -97.748 demo_area
    python -m tritium_lib.intelligence.geospatial.cli status
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def cmd_process(args):
    """Process an area centered on lat/lon with given radius."""
    from tritium_lib.models.gis import TileBounds
    from tritium_lib.intelligence.geospatial.models import (
        AreaOfOperations, SegmentationConfig, SegmentedRegion, TerrainLayerMetadata,
    )
    from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader
    from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine
    from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier
    from tritium_lib.intelligence.geospatial.vector_converter import VectorConverter
    from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
    import numpy as np
    from PIL import Image

    # Compute bbox from center + radius
    r_deg = args.radius / 111320.0
    bounds = TileBounds(
        min_lat=args.lat - r_deg, min_lon=args.lon - r_deg / math.cos(math.radians(args.lat)),
        max_lat=args.lat + r_deg, max_lon=args.lon + r_deg / math.cos(math.radians(args.lat)),
    )

    ao = AreaOfOperations(id=args.ao_id, name=args.ao_id, bounds=bounds, zoom=args.zoom)

    # Check llama-server
    llm_ok = False
    if args.llm:
        try:
            import requests
            llm_ok = requests.get(f"http://127.0.0.1:{args.llm_port}/health", timeout=1).status_code == 200
        except Exception:
            pass

    config = SegmentationConfig(llm_classify=llm_ok, llm_endpoint=f"http://127.0.0.1:{args.llm_port}")
    t0 = time.monotonic()

    dl = TileDownloader(cache_dir=Path("data/cache/tiles"))
    image_path = dl.download_tiles(ao, source=args.source)
    img = Image.open(image_path)
    img_array = np.array(img.convert("RGB"))
    gt = dl.get_geo_transform(ao, img.width, img.height)

    engine = SegmentationEngine(config)
    segments = engine.segment_image(image_path)

    classifier = TerrainClassifier(config)
    classifications = classifier.classify_segments(img_array, segments)

    converter = VectorConverter(min_area_px=50)
    regions = []
    for i, seg in enumerate(segments):
        tt, conf = classifications[i]
        for poly in converter.mask_to_polygons(seg["mask"], gt):
            area = poly.get("area_m2", 0)
            if area < 10 or area > 100000:
                continue
            c = poly.get("centroid", (0, 0))
            regions.append(SegmentedRegion(
                geometry_wkt=poly["wkt"], terrain_type=tt, confidence=conf,
                area_m2=area, centroid_lon=c[0], centroid_lat=c[1],
            ))

    elapsed = time.monotonic() - t0

    layer = TerrainLayer(cache_dir=Path("data/cache/terrain"))
    layer._regions = regions
    layer._bounds = bounds
    layer._metadata = TerrainLayerMetadata(
        ao_id=ao.id, segment_count=len(regions),
        processing_time_s=elapsed, source_imagery=args.source, bounds=bounds,
    )
    layer._build_grid_index()
    layer._save_cache(ao.id)

    print(f"Processed {len(regions)} features in {elapsed:.1f}s")
    print(f"Cached to data/cache/terrain/{ao.id}/")
    print()
    print(layer.terrain_brief())


def cmd_brief(args):
    """Show terrain brief for a cached AO."""
    from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
    layer = TerrainLayer()
    if not layer.load_cached(args.ao_id):
        print(f"No cached terrain for AO '{args.ao_id}'", file=sys.stderr)
        sys.exit(1)
    print(layer.terrain_brief())


def cmd_missions(args):
    """Generate missions for a cached AO."""
    from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
    from tritium_lib.intelligence.geospatial.mission_generator import MissionGenerator
    layer = TerrainLayer()
    if not layer.load_cached(args.ao_id):
        print(f"No cached terrain for AO '{args.ao_id}'", file=sys.stderr)
        sys.exit(1)
    gen = MissionGenerator()
    missions = gen.generate_missions(layer)
    print(gen.missions_brief(missions))


def cmd_query(args):
    """Query terrain at a point."""
    from tritium_lib.intelligence.geospatial.terrain_layer import TerrainLayer
    layer = TerrainLayer()
    if not layer.load_cached(args.ao_id):
        print(f"No cached terrain for AO '{args.ao_id}'", file=sys.stderr)
        sys.exit(1)
    result = layer.terrain_at(args.lat, args.lon)
    print(f"({args.lat}, {args.lon}): {result.value}")


def cmd_status(args):
    """Show cached terrain areas."""
    cache_dir = Path("data/cache/terrain")
    if not cache_dir.exists():
        print("No cached terrain data.")
        return
    for d in sorted(cache_dir.iterdir()):
        meta_path = d / "metadata.json"
        if d.is_dir() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                print(f"  {meta.get('ao_id', d.name):20s} {meta.get('segment_count', 0):5d} features  {meta.get('processing_time_s', 0):.1f}s  {meta.get('source_imagery', '')}")
            except Exception:
                print(f"  {d.name:20s} (corrupt metadata)")


def main():
    parser = argparse.ArgumentParser(description="Tritium Geospatial Segmentation CLI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("process", help="Process an area from satellite imagery")
    p.add_argument("--lat", type=float, required=True, help="Center latitude")
    p.add_argument("--lon", type=float, required=True, help="Center longitude")
    p.add_argument("--radius", type=float, default=500, help="Radius in meters (default: 500)")
    p.add_argument("--zoom", type=int, default=16, help="Tile zoom level (default: 16)")
    p.add_argument("--ao-id", default="cli_area", help="Area of operations ID")
    p.add_argument("--source", default="satellite", help="Tile source (satellite, osm)")
    p.add_argument("--llm", action="store_true", help="Use llama-server for classification")
    p.add_argument("--llm-port", type=int, default=8081, help="llama-server port")

    p = sub.add_parser("brief", help="Show terrain brief")
    p.add_argument("ao_id", help="Area of operations ID")

    p = sub.add_parser("missions", help="Generate tactical missions")
    p.add_argument("ao_id", help="Area of operations ID")

    p = sub.add_parser("query", help="Query terrain at a point")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("ao_id", help="Area of operations ID")

    sub.add_parser("status", help="Show cached terrain areas")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    {"process": cmd_process, "brief": cmd_brief, "missions": cmd_missions,
     "query": cmd_query, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
