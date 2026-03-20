# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Vector converter — raster masks to GeoJSON polygons.

Converts binary segmentation masks into vector polygons (GeoJSON),
with Douglas-Peucker simplification and area filtering.

Two backends:
1. rasterio.features.shapes() — best quality, needs rasterio
2. OpenCV cv2.findContours() — good fallback, needs opencv
3. Pure numpy — minimal fallback, bounding box only
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from tritium_lib.intelligence.geospatial._deps import (
    HAS_CV2,
    HAS_NUMPY,
    HAS_RASTERIO,
    HAS_SHAPELY,
)

logger = logging.getLogger(__name__)


class VectorConverter:
    """Converts segmentation masks to vector polygons."""

    def __init__(
        self,
        simplify_tolerance: float = 1.0,
        min_area_px: int = 100,
        max_area_px: int = 10_000_000,
    ) -> None:
        self.simplify_tolerance = simplify_tolerance
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px

    def mask_to_polygons(
        self,
        mask: Any,
        geo_transform: Optional[tuple[float, float, float, float]] = None,
    ) -> list[dict]:
        """Convert a binary mask to a list of polygon dicts.

        Args:
            mask: Binary numpy array (H, W), True = region
            geo_transform: (lon_per_px, lat_per_px, origin_lon, origin_lat)
                If None, coordinates are in pixel space.

        Returns:
            List of dicts with keys:
                coordinates: list of (lon, lat) or (x, y) rings
                area_px: pixel area
                area_m2: estimated area in square meters (if geo_transform)
                centroid: (lon, lat) or (x, y)
                wkt: WKT POLYGON string
        """
        if not HAS_NUMPY:
            return []

        if HAS_CV2:
            return self._mask_to_polygons_cv2(mask, geo_transform)
        else:
            return self._mask_to_polygons_bbox(mask, geo_transform)

    def masks_to_polygons(
        self,
        masks: list[dict],
        geo_transform: Optional[tuple[float, float, float, float]] = None,
    ) -> list[dict]:
        """Convert multiple segment masks to polygons.

        Args:
            masks: list of dicts with "mask" key (binary numpy arrays)
            geo_transform: pixel-to-geo transform

        Returns:
            list of polygon dicts (same format as mask_to_polygons)
        """
        results = []
        for seg in masks:
            polys = self.mask_to_polygons(seg["mask"], geo_transform)
            for poly in polys:
                poly["stability_score"] = seg.get("stability_score", 0.5)
                results.append(poly)
        return results

    def _mask_to_polygons_cv2(
        self,
        mask: Any,
        geo_transform: Optional[tuple[float, float, float, float]],
    ) -> list[dict]:
        """Convert mask to polygons using OpenCV contour detection."""
        import cv2
        import numpy as np

        mask_u8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px < self.min_area_px or area_px > self.max_area_px:
                continue

            # Simplify contour
            epsilon = self.simplify_tolerance
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            # Convert to coordinate list
            coords = []
            for pt in approx:
                px, py = float(pt[0][0]), float(pt[0][1])
                if geo_transform:
                    lon, lat = self._px_to_geo(px, py, geo_transform)
                    coords.append((lon, lat))
                else:
                    coords.append((px, py))

            # Close the ring
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            # Compute centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = float(contour[:, 0, 0].mean())
                cy = float(contour[:, 0, 1].mean())

            if geo_transform:
                centroid = self._px_to_geo(cx, cy, geo_transform)
                area_m2 = self._estimate_area_m2(area_px, geo_transform)
            else:
                centroid = (cx, cy)
                area_m2 = area_px

            wkt = self._coords_to_wkt(coords)

            results.append({
                "coordinates": [coords],
                "area_px": int(area_px),
                "area_m2": area_m2,
                "centroid": centroid,
                "wkt": wkt,
            })

        return results

    def _mask_to_polygons_bbox(
        self,
        mask: Any,
        geo_transform: Optional[tuple[float, float, float, float]],
    ) -> list[dict]:
        """Minimal fallback: convert mask to bounding box polygon."""
        import numpy as np

        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return []

        area_px = int(mask.sum())
        if area_px < self.min_area_px or area_px > self.max_area_px:
            return []

        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())

        if geo_transform:
            nw = self._px_to_geo(x0, y0, geo_transform)
            ne = self._px_to_geo(x1, y0, geo_transform)
            se = self._px_to_geo(x1, y1, geo_transform)
            sw = self._px_to_geo(x0, y1, geo_transform)
            centroid = self._px_to_geo((x0 + x1) / 2, (y0 + y1) / 2, geo_transform)
            area_m2 = self._estimate_area_m2(area_px, geo_transform)
        else:
            nw = (x0, y0)
            ne = (x1, y0)
            se = (x1, y1)
            sw = (x0, y1)
            centroid = ((x0 + x1) / 2, (y0 + y1) / 2)
            area_m2 = area_px

        coords = [nw, ne, se, sw, nw]
        wkt = self._coords_to_wkt(coords)

        return [{
            "coordinates": [coords],
            "area_px": area_px,
            "area_m2": area_m2,
            "centroid": centroid,
            "wkt": wkt,
        }]

    @staticmethod
    def _px_to_geo(
        px: float,
        py: float,
        geo_transform: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """Convert pixel coordinates to geographic (lon, lat)."""
        lon_per_px, lat_per_px, origin_lon, origin_lat = geo_transform
        lon = origin_lon + px * lon_per_px
        lat = origin_lat + py * lat_per_px
        return (lon, lat)

    @staticmethod
    def _estimate_area_m2(
        area_px: float,
        geo_transform: tuple[float, float, float, float],
    ) -> float:
        """Estimate area in square meters from pixel area and geo transform."""
        import math

        lon_per_px, lat_per_px, origin_lon, origin_lat = geo_transform

        # At equator: 1 degree ≈ 111,320 meters
        # Adjust for latitude
        lat_center = origin_lat + abs(lat_per_px) * 500  # rough center
        cos_lat = math.cos(math.radians(lat_center))

        m_per_px_x = abs(lon_per_px) * 111_320 * cos_lat
        m_per_px_y = abs(lat_per_px) * 111_320

        return area_px * m_per_px_x * m_per_px_y

    @staticmethod
    def _coords_to_wkt(coords: list[tuple[float, float]]) -> str:
        """Convert coordinate ring to WKT POLYGON string."""
        ring = ", ".join(f"{x} {y}" for x, y in coords)
        return f"POLYGON (({ring}))"

    def to_geojson(self, polygons: list[dict]) -> dict:
        """Convert polygon list to GeoJSON FeatureCollection."""
        features = []
        for poly in polygons:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": poly["coordinates"],
                },
                "properties": {
                    "area_m2": poly.get("area_m2", 0),
                    "area_px": poly.get("area_px", 0),
                    "centroid": poly.get("centroid"),
                    "terrain_type": poly.get("terrain_type", "unknown"),
                    "confidence": poly.get("confidence", 0.0),
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }
