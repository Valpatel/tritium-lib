# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""OSM data enrichment — fuse OpenStreetMap semantics with terrain segmentation.

Downloads OSM features via the Overpass API for an AO and enriches
segmented terrain features with:
- Building names, types, heights
- Road names, types, speed limits, surface types
- POI data (shops, amenities, natural features)
- Land use classification

Confidence-weighted fusion: when SAM says "building" and OSM says
"building" → high confidence. When they disagree → flag for review.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tritium_lib.models.gis import TileBounds
from tritium_lib.models.terrain import TerrainType

logger = logging.getLogger(__name__)

# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM tag → TerrainType mapping
_OSM_TERRAIN_MAP: dict[str, TerrainType] = {
    # Buildings
    "building": TerrainType.BUILDING,
    # Roads
    "highway:motorway": TerrainType.ROAD,
    "highway:trunk": TerrainType.ROAD,
    "highway:primary": TerrainType.ROAD,
    "highway:secondary": TerrainType.ROAD,
    "highway:tertiary": TerrainType.ROAD,
    "highway:residential": TerrainType.ROAD,
    "highway:service": TerrainType.ROAD,
    "highway:unclassified": TerrainType.ROAD,
    # Sidewalks & paths
    "highway:footway": TerrainType.SIDEWALK,
    "highway:pedestrian": TerrainType.SIDEWALK,
    "highway:path": TerrainType.SIDEWALK,
    "highway:cycleway": TerrainType.SIDEWALK,
    # Water
    "natural:water": TerrainType.WATER,
    "waterway:river": TerrainType.WATER,
    "waterway:stream": TerrainType.WATER,
    "waterway:canal": TerrainType.WATER,
    # Vegetation
    "natural:wood": TerrainType.VEGETATION,
    "natural:tree": TerrainType.VEGETATION,
    "landuse:forest": TerrainType.VEGETATION,
    "landuse:grass": TerrainType.VEGETATION,
    "leisure:park": TerrainType.VEGETATION,
    "leisure:garden": TerrainType.VEGETATION,
    # Parking
    "amenity:parking": TerrainType.PARKING,
    "landuse:parking": TerrainType.PARKING,
    # Rail
    "railway:rail": TerrainType.RAIL,
    "railway:light_rail": TerrainType.RAIL,
    "railway:tram": TerrainType.RAIL,
    # Bridge
    "man_made:bridge": TerrainType.BRIDGE,
    # Barren
    "landuse:quarry": TerrainType.BARREN,
    "landuse:construction": TerrainType.BARREN,
    "landuse:brownfield": TerrainType.BARREN,
}


@dataclass
class OSMFeature:
    """A single OSM feature with its tags and geometry."""
    osm_id: int
    osm_type: str  # node, way, relation
    terrain_type: TerrainType
    name: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)
    lat: float = 0.0
    lon: float = 0.0
    # Full geometry (list of [lat, lon] for ways, None for nodes)
    geometry: Optional[list[list[float]]] = None
    # Enrichment data extracted from tags
    building_type: Optional[str] = None
    road_type: Optional[str] = None
    speed_limit: Optional[int] = None
    surface: Optional[str] = None
    height_m: Optional[float] = None
    lanes: Optional[int] = None


@dataclass
class EnrichmentResult:
    """Result of enriching a terrain layer with OSM data."""
    features_enriched: int = 0
    osm_features_found: int = 0
    agreements: int = 0  # SAM and OSM agree on type
    disagreements: int = 0  # SAM and OSM disagree
    new_features: int = 0  # OSM features not in SAM
    osm_features: list[OSMFeature] = field(default_factory=list)


class OSMEnrichment:
    """Fetches and fuses OSM data with terrain segmentation.

    Usage:
        enrichment = OSMEnrichment()
        osm_features = enrichment.fetch_osm(bounds)
        result = enrichment.enrich_terrain_layer(terrain_layer, osm_features)
    """

    def __init__(
        self,
        cache_dir: Path = Path("data/cache/osm"),
        rate_limit_s: float = 1.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.rate_limit_s = rate_limit_s
        self._last_fetch: float = 0.0

    def fetch_osm(self, bounds: TileBounds) -> list[OSMFeature]:
        """Fetch OSM features within bounds via Overpass API.

        Returns list of OSMFeature objects with terrain classifications.
        Results are cached to avoid hammering the Overpass API.
        """
        # Check cache first
        cache_key = f"{bounds.min_lat:.4f}_{bounds.min_lon:.4f}_{bounds.max_lat:.4f}_{bounds.max_lon:.4f}"
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                return self._parse_overpass_response(data)
            except Exception:
                pass

        # Build Overpass query — fetch buildings, roads, water, landuse
        # Use `out geom` to get full polygon/way geometry (not just centroids)
        bbox = f"{bounds.min_lat},{bounds.min_lon},{bounds.max_lat},{bounds.max_lon}"
        query = f"""
        [out:json][timeout:60];
        (
          way["building"]({bbox});
          way["highway"]({bbox});
          way["natural"="water"]({bbox});
          way["waterway"]({bbox});
          way["landuse"]({bbox});
          way["leisure"="park"]({bbox});
          way["amenity"="parking"]({bbox});
          way["railway"]({bbox});
          way["barrier"]({bbox});
          node["natural"="tree"]({bbox});
        );
        out geom;
        """

        try:
            import requests
        except ImportError:
            logger.warning("requests not installed — cannot fetch OSM data")
            return []

        # Rate limit
        elapsed = time.monotonic() - self._last_fetch
        if elapsed < self.rate_limit_s:
            time.sleep(self.rate_limit_s - elapsed)

        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=60,
                headers={"User-Agent": "Tritium/1.0 (geospatial enrichment)"},
            )
            resp.raise_for_status()
            self._last_fetch = time.monotonic()

            data = resp.json()

            # Cache result
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data))

            features = self._parse_overpass_response(data)
            logger.info("Fetched %d OSM features for bounds %s", len(features), cache_key)
            return features

        except Exception as e:
            logger.warning("OSM fetch failed: %s", e)
            return []

    def _parse_overpass_response(self, data: dict) -> list[OSMFeature]:
        """Parse Overpass JSON response into OSMFeature objects."""
        features = []

        for element in data.get("elements", []):
            tags = element.get("tags", {})
            osm_type = element.get("type", "node")
            osm_id = element.get("id", 0)

            # Determine terrain type from tags
            terrain_type = self._classify_osm_tags(tags)
            if terrain_type is None:
                continue

            # Get position and geometry
            geometry: list[list[float]] | None = None
            if osm_type == "node":
                lat = element.get("lat", 0.0)
                lon = element.get("lon", 0.0)
            else:
                # Extract full geometry from `out geom` response
                geom_pts = element.get("geometry", [])
                if geom_pts:
                    geometry = [[pt["lat"], pt["lon"]] for pt in geom_pts]
                    # Centroid from geometry
                    lat = sum(p[0] for p in geometry) / len(geometry)
                    lon = sum(p[1] for p in geometry) / len(geometry)
                else:
                    # Fallback to center (from `out center`)
                    center = element.get("center", {})
                    lat = center.get("lat", 0.0)
                    lon = center.get("lon", 0.0)

            feature = OSMFeature(
                osm_id=osm_id,
                osm_type=osm_type,
                terrain_type=terrain_type,
                name=tags.get("name"),
                tags=tags,
                lat=lat,
                lon=lon,
                geometry=geometry,
                building_type=tags.get("building") if "building" in tags else None,
                road_type=tags.get("highway") if "highway" in tags else None,
                speed_limit=self._parse_speed(tags.get("maxspeed")),
                surface=tags.get("surface"),
                height_m=self._parse_height(tags.get("height")),
                lanes=self._parse_int(tags.get("lanes")),
            )
            features.append(feature)

        return features

    def _classify_osm_tags(self, tags: dict[str, str]) -> Optional[TerrainType]:
        """Map OSM tags to TerrainType."""
        # Check specific tag combinations first
        if "building" in tags:
            return TerrainType.BUILDING

        highway = tags.get("highway")
        if highway:
            key = f"highway:{highway}"
            return _OSM_TERRAIN_MAP.get(key, TerrainType.ROAD)

        natural = tags.get("natural")
        if natural:
            key = f"natural:{natural}"
            return _OSM_TERRAIN_MAP.get(key)

        waterway = tags.get("waterway")
        if waterway:
            return TerrainType.WATER

        landuse = tags.get("landuse")
        if landuse:
            key = f"landuse:{landuse}"
            return _OSM_TERRAIN_MAP.get(key)

        leisure = tags.get("leisure")
        if leisure:
            key = f"leisure:{leisure}"
            return _OSM_TERRAIN_MAP.get(key)

        amenity = tags.get("amenity")
        if amenity:
            key = f"amenity:{amenity}"
            return _OSM_TERRAIN_MAP.get(key)

        railway = tags.get("railway")
        if railway:
            return TerrainType.RAIL

        barrier = tags.get("barrier")
        if barrier:
            return TerrainType.BUILDING  # Barriers act as obstacles like buildings

        return None

    def enrich_terrain_layer(
        self,
        terrain_layer: Any,
        osm_features: list[OSMFeature],
    ) -> EnrichmentResult:
        """Enrich a TerrainLayer with OSM semantic data.

        For each OSM feature, finds the nearest segmented region and:
        - If types agree: boost confidence, add OSM metadata
        - If types disagree: flag the disagreement, keep higher-confidence
        - If no segmented region nearby: add as new feature

        Returns enrichment statistics.
        """
        result = EnrichmentResult(osm_features_found=len(osm_features))

        if not hasattr(terrain_layer, '_regions'):
            return result

        result.osm_features = osm_features

        for osm_feat in osm_features:
            # Find nearest segmented region
            nearest = None
            nearest_dist = float("inf")

            for region in terrain_layer._regions:
                dlat = region.centroid_lat - osm_feat.lat
                dlon = region.centroid_lon - osm_feat.lon
                dist = dlat * dlat + dlon * dlon
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = region

            # Match threshold: ~50m at equator
            match_threshold = 0.0005 ** 2

            if nearest is not None and nearest_dist < match_threshold:
                if nearest.terrain_type == osm_feat.terrain_type:
                    # Agreement — boost confidence, add metadata
                    nearest.confidence = min(nearest.confidence + 0.1, 0.95)
                    if osm_feat.name:
                        nearest.properties["osm_name"] = osm_feat.name
                    if osm_feat.building_type:
                        nearest.properties["building_type"] = osm_feat.building_type
                    if osm_feat.road_type:
                        nearest.properties["road_type"] = osm_feat.road_type
                    if osm_feat.speed_limit:
                        nearest.properties["speed_limit"] = osm_feat.speed_limit
                    if osm_feat.surface:
                        nearest.properties["surface"] = osm_feat.surface
                    if osm_feat.height_m:
                        nearest.properties["height_m"] = osm_feat.height_m
                    nearest.properties["osm_id"] = osm_feat.osm_id
                    result.agreements += 1
                    result.features_enriched += 1
                else:
                    # Disagreement — keep both, flag
                    nearest.properties["osm_disagrees"] = osm_feat.terrain_type.value
                    nearest.properties["osm_type"] = osm_feat.terrain_type.value
                    nearest.properties["osm_id"] = osm_feat.osm_id
                    result.disagreements += 1
                    result.features_enriched += 1
            else:
                result.new_features += 1

        logger.info(
            "OSM enrichment: %d enriched, %d agree, %d disagree, %d new",
            result.features_enriched, result.agreements,
            result.disagreements, result.new_features,
        )
        return result

    @staticmethod
    def _parse_speed(value: Optional[str]) -> Optional[int]:
        """Parse OSM maxspeed tag to km/h."""
        if not value:
            return None
        try:
            # Handle "50", "30 mph", etc.
            value = value.strip()
            if value.endswith("mph"):
                return int(float(value.replace("mph", "").strip()) * 1.609)
            return int(float(value.split()[0]))
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _parse_height(value: Optional[str]) -> Optional[float]:
        """Parse OSM height tag to meters."""
        if not value:
            return None
        try:
            return float(value.replace("m", "").strip())
        except ValueError:
            return None

    @staticmethod
    def _parse_int(value: Optional[str]) -> Optional[int]:
        """Parse an integer from an OSM tag."""
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def features_to_geojson(features: list[OSMFeature]) -> dict:
        """Convert OSMFeature list to GeoJSON FeatureCollection.

        Features with geometry become Polygon/LineString features.
        Point features (nodes) become Point features.
        """
        geojson_features = []
        for f in features:
            props = {
                "osm_id": f.osm_id,
                "osm_type": f.osm_type,
                "terrain_type": f.terrain_type.value if hasattr(f.terrain_type, "value") else str(f.terrain_type),
                "name": f.name or "",
            }
            if f.building_type:
                props["building_type"] = f.building_type
            if f.road_type:
                props["road_type"] = f.road_type
            if f.speed_limit:
                props["speed_limit"] = f.speed_limit
            if f.surface:
                props["surface"] = f.surface
            if f.height_m:
                props["height"] = f.height_m
            if f.lanes:
                props["lanes"] = f.lanes

            if f.geometry and len(f.geometry) >= 3:
                # Polygon (building, landuse, water body)
                coords = [[pt[1], pt[0]] for pt in f.geometry]  # [lon, lat]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                geometry = {"type": "Polygon", "coordinates": [coords]}
            elif f.geometry and len(f.geometry) >= 2:
                # LineString (road, waterway)
                coords = [[pt[1], pt[0]] for pt in f.geometry]
                geometry = {"type": "LineString", "coordinates": coords}
            else:
                # Point (tree, node)
                geometry = {"type": "Point", "coordinates": [f.lon, f.lat]}

            geojson_features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": props,
            })

        return {"type": "FeatureCollection", "features": geojson_features}
