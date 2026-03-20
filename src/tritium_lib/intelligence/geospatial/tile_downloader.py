# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""TMS tile downloader — fetch, cache, and stitch satellite imagery.

Downloads map tiles from TMS sources (Esri World Imagery, OSM, custom),
caches them locally, and stitches into a single image for segmentation.

Uses existing GIS primitives from tritium_lib.models.gis.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from tritium_lib.intelligence.geospatial._deps import (
    HAS_NUMPY,
    HAS_PILLOW,
    require,
)
from tritium_lib.intelligence.geospatial.models import AreaOfOperations
from tritium_lib.models.gis import TileCoord, tile_to_lat_lon, tiles_in_bounds

logger = logging.getLogger(__name__)

# TMS URL templates for common providers
TILE_SOURCES: dict[str, str] = {
    "satellite": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "topo": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
    ),
}

# Rate limit between fetches (seconds) — respect tile server policies
RATE_LIMIT_S = 0.1
# User-Agent for tile requests
USER_AGENT = "Tritium/1.0 (geospatial segmentation pipeline)"


class TileDownloader:
    """Downloads TMS tiles, caches them, and stitches into images.

    Tile cache layout:
        {cache_dir}/{source}/{zoom}/{x}/{y}.png

    Stitched output:
        {cache_dir}/stitched/{ao_id}_{zoom}.png
    """

    def __init__(self, cache_dir: Path = Path("data/cache/tiles")) -> None:
        self.cache_dir = Path(cache_dir)
        self._last_fetch_time: float = 0.0

    def download_tiles(
        self,
        ao: AreaOfOperations,
        source: str = "satellite",
    ) -> Path:
        """Download and stitch tiles for an area of operations.

        Returns path to the stitched image file (PNG).
        """
        require(HAS_PILLOW, "Pillow", "geospatial")

        from PIL import Image

        # Get tile URL template
        url_template = TILE_SOURCES.get(source, source)

        # Calculate tiles needed
        bounds_tuple = (
            ao.bounds.min_lat, ao.bounds.min_lon,
            ao.bounds.max_lat, ao.bounds.max_lon,
        )
        tiles = tiles_in_bounds(bounds_tuple, ao.zoom)

        if not tiles:
            raise ValueError(f"No tiles found for bounds {ao.bounds} at zoom {ao.zoom}")

        logger.info(
            "Downloading %d tiles for AO '%s' at zoom %d from %s",
            len(tiles), ao.id, ao.zoom, source,
        )

        # Download each tile (with caching)
        tile_images: dict[tuple[int, int], Path] = {}
        for tile in tiles:
            tile_path = self._fetch_tile(tile, url_template, source)
            tile_images[(tile.x, tile.y)] = tile_path

        # Stitch tiles into single image
        return self._stitch_tiles(tile_images, tiles, ao, source)

    def _tile_cache_path(self, tile: TileCoord, source: str) -> Path:
        """Cache path for a single tile."""
        source_key = source if source in TILE_SOURCES else "custom"
        return self.cache_dir / source_key / str(tile.zoom) / str(tile.x) / f"{tile.y}.png"

    def _fetch_tile(
        self,
        tile: TileCoord,
        url_template: str,
        source: str,
    ) -> Path:
        """Fetch a single tile, using cache if available."""
        cache_path = self._tile_cache_path(tile, source)

        if cache_path.exists():
            return cache_path

        # Build URL
        url = url_template.format(z=tile.zoom, x=tile.x, y=tile.y)

        # Rate limit
        elapsed = time.monotonic() - self._last_fetch_time
        if elapsed < RATE_LIMIT_S:
            time.sleep(RATE_LIMIT_S - elapsed)

        # Fetch
        try:
            import requests
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            resp.raise_for_status()

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(resp.content)
            self._last_fetch_time = time.monotonic()

            logger.debug("Fetched tile %s", tile.url_path)
            return cache_path

        except Exception as e:
            logger.warning("Failed to fetch tile %s: %s", tile.url_path, e)
            raise

    def _stitch_tiles(
        self,
        tile_images: dict[tuple[int, int], Path],
        tiles: list[TileCoord],
        ao: AreaOfOperations,
        source: str,
    ) -> Path:
        """Stitch downloaded tiles into a single image."""
        from PIL import Image

        if not tile_images:
            raise ValueError("No tiles to stitch")

        # Find grid dimensions
        xs = sorted({t.x for t in tiles})
        ys = sorted({t.y for t in tiles})
        cols = len(xs)
        rows = len(ys)

        # Load one tile to get dimensions
        sample_path = next(iter(tile_images.values()))
        with Image.open(sample_path) as sample:
            tile_w, tile_h = sample.size

        # Create output image
        output = Image.new("RGB", (cols * tile_w, rows * tile_h))

        for tile in tiles:
            path = tile_images.get((tile.x, tile.y))
            if path is None or not path.exists():
                continue
            col_idx = xs.index(tile.x)
            row_idx = ys.index(tile.y)
            with Image.open(path) as img:
                output.paste(img, (col_idx * tile_w, row_idx * tile_h))

        # Save stitched image
        source_key = source if source in TILE_SOURCES else "custom"
        stitch_dir = self.cache_dir / "stitched"
        stitch_dir.mkdir(parents=True, exist_ok=True)
        output_path = stitch_dir / f"{ao.id}_{ao.zoom}_{source_key}.png"
        output.save(output_path, "PNG")

        logger.info(
            "Stitched %d×%d tiles → %s (%dx%d px)",
            cols, rows, output_path, output.width, output.height,
        )
        return output_path

    def get_geo_transform(
        self,
        ao: AreaOfOperations,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        """Get geo transform for pixel ↔ lat/lon conversion.

        Returns (lon_per_px, lat_per_px, origin_lon, origin_lat) where
        origin is the top-left (NW) corner.
        """
        bounds_tuple = (
            ao.bounds.min_lat, ao.bounds.min_lon,
            ao.bounds.max_lat, ao.bounds.max_lon,
        )
        tiles = tiles_in_bounds(bounds_tuple, ao.zoom)
        if not tiles:
            return (0.0, 0.0, ao.bounds.min_lon, ao.bounds.max_lat)

        xs = sorted({t.x for t in tiles})
        ys = sorted({t.y for t in tiles})

        # NW corner of first tile
        nw_lat, nw_lon = tile_to_lat_lon(xs[0], ys[0], ao.zoom)
        # SE corner (NW of tile one past the last)
        se_lat, se_lon = tile_to_lat_lon(xs[-1] + 1, ys[-1] + 1, ao.zoom)

        lon_per_px = (se_lon - nw_lon) / image_width if image_width else 0
        lat_per_px = (se_lat - nw_lat) / image_height if image_height else 0

        return (lon_per_px, lat_per_px, nw_lon, nw_lat)

    def pixel_to_latlon(
        self,
        px: int,
        py: int,
        geo_transform: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """Convert pixel coordinates to lat/lon."""
        lon_per_px, lat_per_px, origin_lon, origin_lat = geo_transform
        lon = origin_lon + px * lon_per_px
        lat = origin_lat + py * lat_per_px
        return (lat, lon)

    def tiles_for_ao(self, ao: AreaOfOperations) -> list[TileCoord]:
        """Return all tiles covering an AO."""
        bounds_tuple = (
            ao.bounds.min_lat, ao.bounds.min_lon,
            ao.bounds.max_lat, ao.bounds.max_lon,
        )
        return tiles_in_bounds(bounds_tuple, ao.zoom)

    def cached_stitch_path(
        self,
        ao: AreaOfOperations,
        source: str = "satellite",
    ) -> Optional[Path]:
        """Return path to cached stitched image, or None."""
        source_key = source if source in TILE_SOURCES else "custom"
        path = self.cache_dir / "stitched" / f"{ao.id}_{ao.zoom}_{source_key}.png"
        return path if path.exists() else None
