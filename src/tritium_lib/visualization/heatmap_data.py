# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""HeatmapData — 2D grid data structure for heatmap rendering.

Pure data: a 2D float grid with bounds metadata.  Exports to Vega-Lite
rect-mark specs or simple SVG rectangles.
"""

from __future__ import annotations

import html as html_mod
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HeatmapBounds:
    """Spatial bounds for the heatmap grid."""

    min_x: float = 0.0
    max_x: float = 1.0
    min_y: float = 0.0
    max_y: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
        }


class HeatmapData:
    """2D intensity grid for heatmap visualization.

    Parameters
    ----------
    title : str
        Display title for the heatmap.
    resolution : int
        Grid size (resolution x resolution).
    bounds : HeatmapBounds or None
        Spatial bounds; defaults to 0-1 unit square.
    """

    def __init__(
        self,
        title: str = "Heatmap",
        resolution: int = 50,
        bounds: HeatmapBounds | None = None,
    ) -> None:
        self.title = title
        self.resolution = max(1, resolution)
        self.bounds = bounds or HeatmapBounds()
        self._grid: list[list[float]] = [
            [0.0] * self.resolution for _ in range(self.resolution)
        ]

    # -- Mutation -----------------------------------------------------------

    def set_cell(self, row: int, col: int, value: float) -> None:
        """Set the value of a single cell.  Clamps indices to valid range."""
        r = max(0, min(self.resolution - 1, row))
        c = max(0, min(self.resolution - 1, col))
        self._grid[r][c] = value

    def add_to_cell(self, row: int, col: int, value: float) -> None:
        """Add *value* to a cell (accumulate)."""
        r = max(0, min(self.resolution - 1, row))
        c = max(0, min(self.resolution - 1, col))
        self._grid[r][c] += value

    def set_grid(self, grid: list[list[float]]) -> None:
        """Replace the entire grid.  Must match (resolution x resolution)."""
        if len(grid) != self.resolution:
            raise ValueError(
                f"Grid has {len(grid)} rows, expected {self.resolution}"
            )
        for row in grid:
            if len(row) != self.resolution:
                raise ValueError(
                    f"Grid row has {len(row)} cols, expected {self.resolution}"
                )
        self._grid = [list(row) for row in grid]

    def clear(self) -> None:
        """Zero out all cells."""
        self._grid = [
            [0.0] * self.resolution for _ in range(self.resolution)
        ]

    # -- Query --------------------------------------------------------------

    @property
    def grid(self) -> list[list[float]]:
        """The raw 2D grid (copy)."""
        return [list(row) for row in self._grid]

    @property
    def max_value(self) -> float:
        """Maximum cell value in the grid."""
        return max(
            (self._grid[r][c]
             for r in range(self.resolution)
             for c in range(self.resolution)),
            default=0.0,
        )

    @property
    def min_value(self) -> float:
        """Minimum cell value in the grid."""
        return min(
            (self._grid[r][c]
             for r in range(self.resolution)
             for c in range(self.resolution)),
            default=0.0,
        )

    @property
    def total(self) -> float:
        """Sum of all cell values."""
        return sum(
            self._grid[r][c]
            for r in range(self.resolution)
            for c in range(self.resolution)
        )

    @property
    def nonzero_count(self) -> int:
        """Number of cells with value > 0."""
        return sum(
            1
            for r in range(self.resolution)
            for c in range(self.resolution)
            if self._grid[r][c] > 0
        )

    def get_cell(self, row: int, col: int) -> float:
        """Get the value of a single cell.  Returns 0.0 for out-of-range."""
        if 0 <= row < self.resolution and 0 <= col < self.resolution:
            return self._grid[row][col]
        return 0.0

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "title": self.title,
            "resolution": self.resolution,
            "bounds": self.bounds.to_dict(),
            "grid": self.grid,
            "max_value": self.max_value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeatmapData:
        """Deserialize from a dictionary."""
        bounds_data = data.get("bounds", {})
        bounds = HeatmapBounds(
            min_x=bounds_data.get("min_x", 0.0),
            max_x=bounds_data.get("max_x", 1.0),
            min_y=bounds_data.get("min_y", 0.0),
            max_y=bounds_data.get("max_y", 1.0),
        )
        hm = cls(
            title=data.get("title", "Heatmap"),
            resolution=data.get("resolution", 50),
            bounds=bounds,
        )
        grid = data.get("grid")
        if grid is not None:
            hm.set_grid(grid)
        return hm

    # -- Export: Vega-Lite --------------------------------------------------

    def to_vega_lite(self, width: int = 400, height: int = 400) -> dict[str, Any]:
        """Export as a Vega-Lite heatmap specification.

        Uses rect marks with a sequential color scale from dark (#0d0d1a)
        through cyan (#00f0ff) to white.
        """
        values = []
        for r in range(self.resolution):
            for c in range(self.resolution):
                v = self._grid[r][c]
                if v > 0:
                    values.append({"row": r, "col": c, "value": v})

        spec: dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": self.title,
            "width": width,
            "height": height,
            "data": {"values": values},
            "mark": "rect",
            "encoding": {
                "x": {
                    "field": "col",
                    "type": "ordinal",
                    "title": "X",
                },
                "y": {
                    "field": "row",
                    "type": "ordinal",
                    "title": "Y",
                    "sort": "descending",
                },
                "color": {
                    "field": "value",
                    "type": "quantitative",
                    "scale": {
                        "scheme": "viridis",
                    },
                    "title": "Intensity",
                },
                "tooltip": [
                    {"field": "row", "type": "ordinal"},
                    {"field": "col", "type": "ordinal"},
                    {"field": "value", "type": "quantitative"},
                ],
            },
        }
        return spec

    def to_vega_lite_json(self, **kwargs: Any) -> str:
        """Export as a Vega-Lite JSON string."""
        return json.dumps(self.to_vega_lite(**kwargs), indent=2)

    # -- Export: SVG --------------------------------------------------------

    def to_svg(
        self,
        width: int = 400,
        height: int = 400,
        margin: int = 40,
    ) -> str:
        """Generate a simple SVG heatmap using colored rectangles.

        Uses a cyan intensity gradient on a dark background.
        """
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin
        cell_w = plot_w / self.resolution
        cell_h = plot_h / self.resolution
        mx = self.max_value

        lines: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'style="background:#0d0d1a">',
            # Title
            f'<text x="{width // 2}" y="{margin - 10}" '
            f'text-anchor="middle" fill="#00f0ff" font-size="14" '
            f'font-family="monospace">{html_mod.escape(self.title)}</text>',
        ]

        for r in range(self.resolution):
            for c in range(self.resolution):
                v = self._grid[r][c]
                if v <= 0:
                    continue
                intensity = v / mx if mx > 0 else 0
                # Interpolate from dark to cyan
                red = int(0 * intensity)
                green = int(240 * intensity)
                blue = int(255 * intensity)
                color = f"rgb({red},{green},{blue})"
                x = margin + c * cell_w
                y = margin + r * cell_h
                lines.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" '
                    f'width="{cell_w:.1f}" height="{cell_h:.1f}" '
                    f'fill="{color}" opacity="0.9"/>'
                )

        lines.append("</svg>")
        return "\n".join(lines)
