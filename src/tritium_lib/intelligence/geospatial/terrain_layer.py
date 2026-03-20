# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""TerrainLayer — the main API for geospatial segmentation.

Orchestrates the full pipeline: download tiles → segment → classify →
vectorize → cache. Provides runtime queries (terrain_at, obstacles_in_bbox,
features_by_type) and integration hooks for pathfinding and commander AI.

Cache layout:
    data/cache/terrain/{ao_id}/
        metadata.json       — TerrainLayerMetadata
        terrain.geojson     — classified polygons
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

from tritium_lib.intelligence.geospatial._deps import (
    HAS_NUMPY,
    HAS_PILLOW,
    require,
)
from tritium_lib.intelligence.geospatial.models import (
    AreaOfOperations,
    SegmentationConfig,
    SegmentedRegion,
    TerrainLayerMetadata,
)
from tritium_lib.models.gis import TileBounds
from tritium_lib.models.terrain import TerrainType

logger = logging.getLogger(__name__)


class TerrainLayer:
    """Queryable terrain layer built from geospatial segmentation.

    Process flow:
        1. process_area() downloads imagery, segments, classifies, caches
        2. load_cached() restores from cache
        3. terrain_at() / obstacles_in_bbox() / features_by_type() query

    Grid index: divides the AO into cells for O(1) point-in-terrain queries
    without requiring rtree or spatial index dependencies.
    """

    def __init__(self, cache_dir: Path = Path("data/cache/terrain")) -> None:
        self.cache_dir = Path(cache_dir)
        self._regions: list[SegmentedRegion] = []
        self._metadata: Optional[TerrainLayerMetadata] = None
        self._bounds: Optional[TileBounds] = None
        # Grid index for fast spatial queries
        self._grid: dict[tuple[int, int], list[int]] = {}
        self._grid_resolution: float = 0.001  # ~111m at equator

    def process_area(
        self,
        ao: AreaOfOperations,
        config: Optional[SegmentationConfig] = None,
        source: str = "satellite",
    ) -> TerrainLayerMetadata:
        """Run the full segmentation pipeline on an area of operations.

        Downloads satellite tiles, segments the stitched image, classifies
        each segment, converts to vector polygons, and caches results.

        Returns metadata about the processed terrain layer.
        """
        require(HAS_NUMPY, "numpy", "geospatial")
        require(HAS_PILLOW, "Pillow", "geospatial")

        import numpy as np
        from PIL import Image

        config = config or SegmentationConfig()
        t0 = time.monotonic()

        # Step 1: Download and stitch tiles
        from tritium_lib.intelligence.geospatial.tile_downloader import TileDownloader

        downloader = TileDownloader(cache_dir=self.cache_dir.parent / "tiles")
        image_path = downloader.download_tiles(ao, source=source)

        # Get geo transform for coordinate conversion
        img = Image.open(image_path)
        geo_transform = downloader.get_geo_transform(ao, img.width, img.height)
        img_array = np.array(img.convert("RGB"))
        img.close()

        # Step 2: Segment the image
        from tritium_lib.intelligence.geospatial.segmentation import SegmentationEngine

        seg_engine = SegmentationEngine(config)
        segments = seg_engine.segment_image(image_path)

        logger.info("Segmented %d regions from %s", len(segments), image_path)

        # Step 3: Classify each segment
        from tritium_lib.intelligence.geospatial.terrain_classifier import TerrainClassifier

        classifier = TerrainClassifier(config)
        classifications = classifier.classify_segments(img_array, segments)

        # Step 4: Convert masks to vector polygons
        from tritium_lib.intelligence.geospatial.vector_converter import VectorConverter

        converter = VectorConverter(
            simplify_tolerance=config.simplify_tolerance,
            min_area_px=max(1, int(config.min_area_m2)),
            max_area_px=int(config.max_area_m2 * 10),
        )

        self._regions = []
        for i, seg in enumerate(segments):
            terrain_type, confidence = classifications[i]
            polys = converter.mask_to_polygons(seg["mask"], geo_transform)
            for poly in polys:
                # Filter by area
                area_m2 = poly.get("area_m2", 0)
                if area_m2 < config.min_area_m2 or area_m2 > config.max_area_m2:
                    continue

                centroid = poly.get("centroid", (0, 0))
                region = SegmentedRegion(
                    geometry_wkt=poly["wkt"],
                    terrain_type=terrain_type,
                    confidence=confidence,
                    area_m2=area_m2,
                    centroid_lon=centroid[0],
                    centroid_lat=centroid[1],
                )
                self._regions.append(region)

        self._bounds = ao.bounds
        self._build_grid_index()

        elapsed = time.monotonic() - t0

        # Cache results
        self._metadata = TerrainLayerMetadata(
            ao_id=ao.id,
            segment_count=len(self._regions),
            model_used=config.model_name,
            processing_time_s=elapsed,
            source_imagery=source,
            bounds=ao.bounds,
        )
        self._save_cache(ao.id)

        logger.info(
            "Processed AO '%s': %d terrain features in %.1fs",
            ao.id, len(self._regions), elapsed,
        )
        return self._metadata

    def load_cached(self, ao_id: str) -> bool:
        """Load a previously cached terrain layer.

        Returns True if cache was found and loaded, False otherwise.
        """
        cache_dir = self.cache_dir / ao_id
        meta_path = cache_dir / "metadata.json"
        geojson_path = cache_dir / "terrain.geojson"

        if not meta_path.exists() or not geojson_path.exists():
            return False

        try:
            self._metadata = TerrainLayerMetadata.model_validate_json(
                meta_path.read_text()
            )
            geojson = json.loads(geojson_path.read_text())

            self._regions = []
            for feature in geojson.get("features", []):
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})

                # Reconstruct WKT from GeoJSON coordinates
                coords = geom.get("coordinates", [[]])
                if coords and coords[0]:
                    ring = ", ".join(f"{x} {y}" for x, y in coords[0])
                    wkt = f"POLYGON (({ring}))"
                else:
                    wkt = "POLYGON EMPTY"

                terrain_str = props.get("terrain_type", "unknown")
                try:
                    terrain_type = TerrainType(terrain_str)
                except ValueError:
                    terrain_type = TerrainType.UNKNOWN

                centroid = props.get("centroid", (0, 0))
                region = SegmentedRegion(
                    geometry_wkt=wkt,
                    terrain_type=terrain_type,
                    confidence=props.get("confidence", 0.0),
                    area_m2=props.get("area_m2", 0.0),
                    centroid_lon=centroid[0] if isinstance(centroid, (list, tuple)) else 0,
                    centroid_lat=centroid[1] if isinstance(centroid, (list, tuple)) else 0,
                )
                self._regions.append(region)

            if self._metadata.bounds:
                self._bounds = self._metadata.bounds

            self._build_grid_index()
            logger.info("Loaded cached terrain for AO '%s': %d features", ao_id, len(self._regions))
            return True

        except Exception as e:
            logger.warning("Failed to load cached terrain for '%s': %s", ao_id, e)
            return False

    # --- Runtime queries ---

    def terrain_at(self, lat: float, lon: float) -> TerrainType:
        """Query terrain type at a geographic point.

        Two-pass approach:
        1. Try exact point-in-polygon (ray casting) against WKT coords
        2. If no PIP match, fall back to nearest-centroid within radius

        This handles both precise polygon boundaries AND grid-based
        segments where the query point falls between block edges.

        Returns UNKNOWN if no segment covers the point.
        """
        cell = self._geo_to_cell(lon, lat)
        candidates = self._grid.get(cell, [])

        if not candidates:
            return TerrainType.UNKNOWN

        # Pass 1: exact point-in-polygon (sorted by area, smallest first)
        candidates_sorted = sorted(
            candidates,
            key=lambda idx: self._regions[idx].area_m2,
        )

        for idx in candidates_sorted:
            region = self._regions[idx]
            coords = self._wkt_to_coords(region.geometry_wkt)
            if len(coords) >= 3 and _point_in_polygon(lon, lat, coords):
                return region.terrain_type

        # Pass 2: nearest centroid within radius (for block-based segments
        # where query point may fall between block edges)
        best_type = TerrainType.UNKNOWN
        best_dist = float("inf")

        for idx in candidates:
            region = self._regions[idx]
            dlat = lat - region.centroid_lat
            dlon = lon - region.centroid_lon
            dist_sq = dlat * dlat + dlon * dlon
            radius_deg = math.sqrt(region.area_m2) / 111_320
            if dist_sq < radius_deg * radius_deg and dist_sq < best_dist:
                best_dist = dist_sq
                best_type = region.terrain_type

        return best_type

    def terrain_at_local(self, x: float, y: float, geo_ref: Any) -> TerrainType:
        """Query terrain at local sim coordinates via a geo reference.

        geo_ref must have a to_latlon(x, y) method returning (lat, lon).
        """
        if hasattr(geo_ref, "to_latlon"):
            lat, lon = geo_ref.to_latlon(x, y)
            return self.terrain_at(lat, lon)
        return TerrainType.UNKNOWN

    def obstacles_in_bbox(self, bounds: TileBounds) -> list[SegmentedRegion]:
        """Get all terrain features within a bounding box."""
        results = []
        for region in self._regions:
            if bounds.contains(region.centroid_lat, region.centroid_lon):
                results.append(region)
        return results

    def features_by_type(self, terrain_type: TerrainType) -> list[SegmentedRegion]:
        """Get all features of a specific terrain type."""
        return [r for r in self._regions if r.terrain_type == terrain_type]

    def nearest_feature(
        self,
        lat: float,
        lon: float,
        terrain_type: TerrainType,
    ) -> Optional[SegmentedRegion]:
        """Find the nearest feature of a given type to a point."""
        best: Optional[SegmentedRegion] = None
        best_dist = float("inf")

        for region in self._regions:
            if region.terrain_type != terrain_type:
                continue
            dlat = lat - region.centroid_lat
            dlon = lon - region.centroid_lon
            dist = dlat * dlat + dlon * dlon
            if dist < best_dist:
                best_dist = dist
                best = region

        return best

    @property
    def regions(self) -> list[SegmentedRegion]:
        """All segmented regions."""
        return list(self._regions)

    @property
    def metadata(self) -> Optional[TerrainLayerMetadata]:
        return self._metadata

    def to_geojson(self) -> dict:
        """Export all regions as a GeoJSON FeatureCollection."""
        features = []
        for region in self._regions:
            # Parse WKT to coordinates
            coords = self._wkt_to_coords(region.geometry_wkt)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords] if coords else [[]],
                },
                "properties": {
                    "terrain_type": region.terrain_type.value,
                    "confidence": region.confidence,
                    "area_m2": region.area_m2,
                    "centroid": (region.centroid_lon, region.centroid_lat),
                },
            })
        return {"type": "FeatureCollection", "features": features}

    # --- Integration hooks ---

    def populate_terrain_map(self, terrain_map: Any) -> int:
        """Populate a TerrainMap (tritium-sc) with segmented terrain.

        Calls terrain_map.set_cell(x, y, terrain_type_str) for each
        segmented region. Returns number of cells set.

        The terrain_map must have:
            set_cell(x: float, y: float, terrain_type: str)
            _world_to_grid(x, y) -> (col, row)
        """
        if not hasattr(terrain_map, "set_cell"):
            logger.warning("terrain_map has no set_cell method")
            return 0

        # Map new TerrainType values to strings the existing TerrainMap understands
        type_mapping = {
            TerrainType.BUILDING: "building",
            TerrainType.ROAD: "road",
            TerrainType.WATER: "water",
            TerrainType.VEGETATION: "yard",
            TerrainType.PARKING: "road",
            TerrainType.SIDEWALK: "road",
            TerrainType.BRIDGE: "road",
            TerrainType.BARREN: "open",
            TerrainType.RAIL: "open",
        }

        count = 0
        for region in self._regions:
            cell_type = type_mapping.get(region.terrain_type)
            if cell_type is None:
                continue

            # Convert geo centroid to local coordinates if geo_to_local is available
            x, y = region.centroid_lon, region.centroid_lat

            try:
                # Set the centroid cell
                terrain_map.set_cell(x, y, cell_type)
                count += 1

                # Fill surrounding cells based on region area
                # Each cell is ~5m, so fill a radius proportional to sqrt(area)
                if hasattr(terrain_map, '_world_to_grid') and hasattr(terrain_map, '_grid_to_world'):
                    radius_m = math.sqrt(region.area_m2) / 2
                    # Convert radius to grid cells (assuming 5m resolution)
                    resolution = getattr(terrain_map, 'resolution', 5.0)
                    cell_radius = max(1, int(radius_m / resolution))
                    # Cap fill radius to avoid filling the whole map
                    cell_radius = min(cell_radius, 10)

                    col, row = terrain_map._world_to_grid(x, y)
                    for dc in range(-cell_radius, cell_radius + 1):
                        for dr in range(-cell_radius, cell_radius + 1):
                            if dc * dc + dr * dr <= cell_radius * cell_radius:
                                wx, wy = terrain_map._grid_to_world(col + dc, row + dr)
                                try:
                                    terrain_map.set_cell(wx, wy, cell_type)
                                    count += 1
                                except Exception:
                                    pass
            except Exception:
                pass

        logger.info("Populated terrain map with %d cells", count)
        return count

    def populate_graph(self, graph: Any) -> int:
        """Add terrain features as entities in a TritiumGraph.

        Each region becomes a 'terrain_feature' entity. Returns count added.
        """
        if not hasattr(graph, "create_entity"):
            return 0

        count = 0
        for i, region in enumerate(self._regions):
            feature_id = f"terrain_{region.terrain_type.value}_{i}"
            try:
                graph.create_entity(
                    entity_type="Location",
                    id=feature_id,
                    name=f"{region.terrain_type.value} #{i}",
                    properties=json.dumps({
                        "terrain_type": region.terrain_type.value,
                        "area_m2": region.area_m2,
                        "centroid_lat": region.centroid_lat,
                        "centroid_lon": region.centroid_lon,
                        "confidence": region.confidence,
                    }),
                )
                count += 1
            except Exception:
                pass

        return count

    def terrain_brief(self) -> str:
        """Generate a tactical terrain brief for commander AI.

        Returns a text summary of the terrain composition, key features,
        chokepoints, and obstacles suitable for Amy/BattleNarrator context.
        """
        if not self._regions:
            return "No terrain data available."

        # Count features by type
        type_counts: dict[str, int] = {}
        type_areas: dict[str, float] = {}
        for r in self._regions:
            t = r.terrain_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
            type_areas[t] = type_areas.get(t, 0.0) + r.area_m2

        total_area = sum(type_areas.values()) or 1.0

        lines = []
        ao_id = self._metadata.ao_id if self._metadata else "unknown"
        source = self._metadata.source_imagery if self._metadata else "unknown"
        lines.append(f"TERRAIN BRIEF — AO \"{ao_id}\"")

        if self._metadata:
            lines.append(f"Processed from {source} imagery, {self._metadata.segment_count} features")

        lines.append("")
        lines.append("Composition:")
        for t_type in sorted(type_areas, key=lambda t: type_areas[t], reverse=True):
            pct = type_areas[t_type] / total_area * 100
            count = type_counts[t_type]
            lines.append(f"  {pct:5.1f}% {t_type} ({count} features, {type_areas[t_type]:.0f} m²)")

        # Key terrain
        lines.append("")
        lines.append("Key Terrain:")
        water = self.features_by_type(TerrainType.WATER)
        if water:
            lines.append(f"  - Water obstacles: {len(water)} bodies")
        bridges = self.features_by_type(TerrainType.BRIDGE)
        if bridges:
            lines.append(f"  - Bridge crossings: {len(bridges)}")
        buildings = self.features_by_type(TerrainType.BUILDING)
        if buildings:
            large = [b for b in buildings if b.area_m2 > 2000]
            lines.append(f"  - Buildings: {len(buildings)} total, {len(large)} large (>2000 m²)")
        veg = self.features_by_type(TerrainType.VEGETATION)
        if veg:
            dense = [v for v in veg if v.area_m2 > 5000]
            lines.append(f"  - Vegetation: {len(veg)} areas, {len(dense)} dense (>5000 m²)")

        return "\n".join(lines)

    # --- Grid index ---

    def _build_grid_index(self) -> None:
        """Build spatial grid index for fast point queries."""
        self._grid.clear()
        for i, region in enumerate(self._regions):
            cell = self._geo_to_cell(region.centroid_lon, region.centroid_lat)
            if cell not in self._grid:
                self._grid[cell] = []
            self._grid[cell].append(i)

            # Also index in adjacent cells for features that span boundaries
            radius_deg = math.sqrt(region.area_m2) / 111_320
            cells_span = max(1, int(radius_deg / self._grid_resolution))
            if cells_span > 1:
                for dx in range(-cells_span, cells_span + 1):
                    for dy in range(-cells_span, cells_span + 1):
                        neighbor = (cell[0] + dx, cell[1] + dy)
                        if neighbor not in self._grid:
                            self._grid[neighbor] = []
                        if i not in self._grid[neighbor]:
                            self._grid[neighbor].append(i)

    def _geo_to_cell(self, lon: float, lat: float) -> tuple[int, int]:
        """Convert lon/lat to grid cell index."""
        # Guard against NaN/inf coordinates
        if math.isnan(lon) or math.isnan(lat) or math.isinf(lon) or math.isinf(lat):
            return (0, 0)
        return (
            int(lon / self._grid_resolution),
            int(lat / self._grid_resolution),
        )

    # --- Cache management ---

    def _save_cache(self, ao_id: str) -> None:
        """Save terrain layer to cache."""
        cache_dir = self.cache_dir / ao_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        if self._metadata:
            self._metadata.cache_path = str(cache_dir)
            meta_path = cache_dir / "metadata.json"
            meta_path.write_text(self._metadata.model_dump_json(indent=2))

        # Save GeoJSON
        geojson_path = cache_dir / "terrain.geojson"
        geojson_path.write_text(json.dumps(self.to_geojson(), indent=2))

    @staticmethod
    def _wkt_to_coords(wkt: str) -> list[tuple[float, float]]:
        """Parse WKT POLYGON to coordinate list."""
        if "EMPTY" in wkt:
            return []
        # Extract coordinate string between (( and ))
        try:
            inner = wkt.split("((")[1].split("))")[0]
            coords = []
            for pair in inner.split(","):
                parts = pair.strip().split()
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))
            return coords
        except (IndexError, ValueError):
            return []


def _point_in_polygon(
    x: float,
    y: float,
    polygon: list[tuple[float, float]],
) -> bool:
    """Ray-casting point-in-polygon test.

    Casts a horizontal ray from (x, y) to +infinity and counts how
    many polygon edges it crosses. Odd = inside, even = outside.

    Args:
        x: test point X (longitude)
        y: test point Y (latitude)
        polygon: list of (x, y) vertices forming a closed ring

    Returns:
        True if point is inside the polygon.
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if the ray crosses this edge
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside
