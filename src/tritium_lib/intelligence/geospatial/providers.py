# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Satellite imagery provider registry.

Defines the ImageryProvider protocol and built-in providers for
different satellite/aerial imagery sources. Extensible for Planet Labs,
Sentinel, Maxar, drone feeds, and custom TMS endpoints.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from tritium_lib.models.gis import TileBounds

logger = logging.getLogger(__name__)


@runtime_checkable
class ImageryProvider(Protocol):
    """Interface for satellite/aerial imagery sources."""

    @property
    def name(self) -> str:
        """Human-readable name of this provider."""
        ...

    @property
    def source_key(self) -> str:
        """Short key used in cache paths and config."""
        ...

    def fetch_area(
        self,
        bounds: TileBounds,
        zoom: int = 17,
        date: Optional[datetime] = None,
    ) -> Path:
        """Download imagery for an area, return path to stitched image."""
        ...

    def latest_date(self, bounds: TileBounds) -> Optional[datetime]:
        """When was the most recent image captured for this area?"""
        ...


class TileMapProvider:
    """Generic TMS tile provider (Esri, OSM, Mapbox, etc.).

    This is the default provider — works with any TMS endpoint
    without authentication.
    """

    def __init__(
        self,
        name: str = "Esri World Imagery",
        source_key: str = "satellite",
        url_template: str = (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attribution: str = "Esri, Maxar, Earthstar Geographics",
        max_zoom: int = 19,
    ) -> None:
        self._name = name
        self._source_key = source_key
        self._url_template = url_template
        self._attribution = attribution
        self._max_zoom = max_zoom

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_key(self) -> str:
        return self._source_key

    def fetch_area(
        self,
        bounds: TileBounds,
        zoom: int = 17,
        date: Optional[datetime] = None,
    ) -> Path:
        """Download tiles and stitch into a single image."""
        from tritium_lib.intelligence.geospatial.models import AreaOfOperations
        from tritium_lib.intelligence.geospatial.tile_downloader import (
            TileDownloader,
            TILE_SOURCES,
        )

        # Register our URL template temporarily
        TILE_SOURCES[self._source_key] = self._url_template

        ao = AreaOfOperations(
            id=f"provider_{self._source_key}_{zoom}",
            name=self._name,
            bounds=bounds,
            zoom=min(zoom, self._max_zoom),
        )
        dl = TileDownloader()
        return dl.download_tiles(ao, source=self._source_key)

    def latest_date(self, bounds: TileBounds) -> Optional[datetime]:
        """TMS providers don't expose imagery dates."""
        return None


class LocalImageProvider:
    """Serves imagery from a local directory of GeoTIFFs or PNGs.

    Useful for drone orthomosaics, pre-downloaded imagery, or
    test fixtures.
    """

    def __init__(
        self,
        name: str = "Local Imagery",
        source_key: str = "local",
        image_dir: Path = Path("data/imagery"),
    ) -> None:
        self._name = name
        self._source_key = source_key
        self._image_dir = Path(image_dir)

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_key(self) -> str:
        return self._source_key

    def fetch_area(
        self,
        bounds: TileBounds,
        zoom: int = 17,
        date: Optional[datetime] = None,
    ) -> Path:
        """Find the best matching local image for the area."""
        # Look for images in the directory
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg"):
            images = sorted(self._image_dir.glob(ext))
            if images:
                return images[0]

        raise FileNotFoundError(
            f"No imagery found in {self._image_dir}. "
            f"Place GeoTIFF or PNG files there."
        )

    def latest_date(self, bounds: TileBounds) -> Optional[datetime]:
        """Return the modification time of the newest image."""
        latest = None
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg"):
            for img in self._image_dir.glob(ext):
                mtime = datetime.fromtimestamp(img.stat().st_mtime)
                if latest is None or mtime > latest:
                    latest = mtime
        return latest


class ProviderRegistry:
    """Registry of available imagery providers.

    Manages provider instances and allows lookup by source key.
    Pre-populates with common free providers.
    """

    def __init__(self) -> None:
        self._providers: dict[str, ImageryProvider] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in providers."""
        self.register(TileMapProvider(
            name="Esri World Imagery",
            source_key="satellite",
        ))
        self.register(TileMapProvider(
            name="OpenStreetMap",
            source_key="osm",
            url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            attribution="OpenStreetMap contributors",
            max_zoom=19,
        ))
        self.register(TileMapProvider(
            name="Esri World Topo",
            source_key="topo",
            url_template=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
            ),
            attribution="Esri",
        ))

    def register(self, provider: ImageryProvider) -> None:
        """Register a new imagery provider."""
        self._providers[provider.source_key] = provider
        logger.info("Registered imagery provider: %s (%s)", provider.name, provider.source_key)

    def get(self, source_key: str) -> Optional[ImageryProvider]:
        """Get a provider by source key."""
        return self._providers.get(source_key)

    def list_providers(self) -> list[dict[str, str]]:
        """List all registered providers."""
        return [
            {"name": p.name, "source_key": p.source_key}
            for p in self._providers.values()
        ]

    @property
    def providers(self) -> dict[str, ImageryProvider]:
        """All registered providers."""
        return dict(self._providers)


# Module-level singleton
_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry (lazy singleton)."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
