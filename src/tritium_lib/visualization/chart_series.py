# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""ChartSeries — time-series / categorical data for line and bar charts.

Pure data structure.  Holds an ordered list of (x, y) data points plus
display metadata.  Exports to Vega-Lite line/bar specs or simple SVG.
"""

from __future__ import annotations

import html as html_mod
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataPoint:
    """A single (x, y) data point with optional label.

    Attributes
    ----------
    x : float
        Horizontal value (time, category index, etc.).
    y : float
        Vertical value (measurement).
    label : str
        Optional display label for the point.
    """

    x: float
    y: float
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"x": self.x, "y": self.y}
        if self.label:
            d["label"] = self.label
        return d


class ChartSeries:
    """Ordered collection of data points for a single chart series.

    Parameters
    ----------
    title : str
        Chart title.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.
    color : str
        Series color (CSS hex).
    chart_type : str
        ``"line"`` or ``"bar"``.
    """

    def __init__(
        self,
        title: str = "Chart",
        x_label: str = "X",
        y_label: str = "Y",
        color: str = "#00f0ff",
        chart_type: str = "line",
    ) -> None:
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.color = color
        self.chart_type = chart_type if chart_type in ("line", "bar") else "line"
        self._points: list[DataPoint] = []

    # -- Mutation -----------------------------------------------------------

    def add_point(
        self, x: float, y: float, label: str = ""
    ) -> DataPoint:
        """Append a data point.  Points are stored in insertion order."""
        pt = DataPoint(x=x, y=y, label=label)
        self._points.append(pt)
        return pt

    def add_points(self, points: list[tuple[float, float]]) -> None:
        """Append multiple (x, y) tuples at once."""
        for x, y in points:
            self._points.append(DataPoint(x=x, y=y))

    def clear(self) -> None:
        """Remove all data points."""
        self._points.clear()

    def sort_by_x(self) -> None:
        """Sort points by x value (ascending)."""
        self._points.sort(key=lambda p: p.x)

    # -- Query --------------------------------------------------------------

    @property
    def points(self) -> list[DataPoint]:
        """All data points (copy)."""
        return list(self._points)

    def __len__(self) -> int:
        return len(self._points)

    def __bool__(self) -> bool:
        return len(self._points) > 0

    @property
    def x_values(self) -> list[float]:
        return [p.x for p in self._points]

    @property
    def y_values(self) -> list[float]:
        return [p.y for p in self._points]

    @property
    def x_min(self) -> float:
        return min(p.x for p in self._points) if self._points else 0.0

    @property
    def x_max(self) -> float:
        return max(p.x for p in self._points) if self._points else 0.0

    @property
    def y_min(self) -> float:
        return min(p.y for p in self._points) if self._points else 0.0

    @property
    def y_max(self) -> float:
        return max(p.y for p in self._points) if self._points else 0.0

    @property
    def y_mean(self) -> float:
        """Mean of y values."""
        if not self._points:
            return 0.0
        return sum(p.y for p in self._points) / len(self._points)

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "color": self.color,
            "chart_type": self.chart_type,
            "points": [p.to_dict() for p in self._points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChartSeries:
        cs = cls(
            title=data.get("title", "Chart"),
            x_label=data.get("x_label", "X"),
            y_label=data.get("y_label", "Y"),
            color=data.get("color", "#00f0ff"),
            chart_type=data.get("chart_type", "line"),
        )
        for pt_data in data.get("points", []):
            cs._points.append(
                DataPoint(
                    x=float(pt_data.get("x", 0)),
                    y=float(pt_data.get("y", 0)),
                    label=str(pt_data.get("label", "")),
                )
            )
        return cs

    # -- Export: Vega-Lite --------------------------------------------------

    def to_vega_lite(self, width: int = 600, height: int = 300) -> dict[str, Any]:
        """Export as a Vega-Lite specification dictionary.

        Uses ``line`` or ``bar`` mark based on :attr:`chart_type`.
        """
        values = [p.to_dict() for p in self._points]
        mark = self.chart_type

        spec: dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": self.title,
            "width": width,
            "height": height,
            "data": {"values": values},
            "mark": {
                "type": mark,
                "color": self.color,
                "point": True if mark == "line" else False,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "title": self.x_label,
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "title": self.y_label,
                },
                "tooltip": [
                    {"field": "x", "type": "quantitative", "title": self.x_label},
                    {"field": "y", "type": "quantitative", "title": self.y_label},
                ],
            },
        }

        # Add label tooltip if any points have labels
        if any(p.label for p in self._points):
            spec["encoding"]["tooltip"].append(
                {"field": "label", "type": "nominal", "title": "Label"}
            )

        return spec

    def to_vega_lite_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_vega_lite(**kwargs), indent=2)

    # -- Export: SVG --------------------------------------------------------

    def to_svg(
        self,
        width: int = 600,
        height: int = 300,
        margin: int = 50,
    ) -> str:
        """Generate a simple SVG chart (line or bar).

        Returns a standalone SVG string with a dark background and
        cyberpunk-colored data.
        """
        if not self._points:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}">'
                f'<text x="{width // 2}" y="{height // 2}" '
                f'text-anchor="middle" fill="#888">No data</text></svg>'
            )

        plot_w = width - 2 * margin
        plot_h = height - 2 * margin

        x_min = self.x_min
        x_max = self.x_max
        y_min = self.y_min
        y_max = self.y_max

        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        def to_px(x: float, y: float) -> tuple[float, float]:
            px = margin + ((x - x_min) / x_range) * plot_w
            # SVG y is inverted
            py = margin + plot_h - ((y - y_min) / y_range) * plot_h
            return px, py

        lines: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'style="background:#0d0d1a">',
            # Title
            f'<text x="{width // 2}" y="{margin - 15}" '
            f'text-anchor="middle" fill="#00f0ff" font-size="14" '
            f'font-family="monospace">{html_mod.escape(self.title)}</text>',
            # X-axis
            f'<line x1="{margin}" y1="{margin + plot_h}" '
            f'x2="{margin + plot_w}" y2="{margin + plot_h}" '
            f'stroke="#444" stroke-width="1"/>',
            # Y-axis
            f'<line x1="{margin}" y1="{margin}" '
            f'x2="{margin}" y2="{margin + plot_h}" '
            f'stroke="#444" stroke-width="1"/>',
            # Axis labels
            f'<text x="{width // 2}" y="{height - 5}" '
            f'text-anchor="middle" fill="#666" font-size="10" '
            f'font-family="monospace">{html_mod.escape(self.x_label)}</text>',
            f'<text x="12" y="{height // 2}" '
            f'text-anchor="middle" fill="#666" font-size="10" '
            f'font-family="monospace" '
            f'transform="rotate(-90,12,{height // 2})">'
            f'{html_mod.escape(self.y_label)}</text>',
        ]

        if self.chart_type == "bar":
            bar_w = max(1, plot_w / len(self._points) * 0.8)
            gap = plot_w / len(self._points)
            for i, pt in enumerate(self._points):
                px, py = to_px(pt.x, pt.y)
                bar_h = (margin + plot_h) - py
                bx = margin + i * gap + (gap - bar_w) / 2
                lines.append(
                    f'<rect x="{bx:.1f}" y="{py:.1f}" '
                    f'width="{bar_w:.1f}" height="{bar_h:.1f}" '
                    f'fill="{self.color}" opacity="0.85"/>'
                )
        else:
            # Line chart
            path_parts: list[str] = []
            for i, pt in enumerate(self._points):
                px, py = to_px(pt.x, pt.y)
                cmd = "M" if i == 0 else "L"
                path_parts.append(f"{cmd}{px:.1f},{py:.1f}")

            if path_parts:
                lines.append(
                    f'<path d="{" ".join(path_parts)}" '
                    f'fill="none" stroke="{self.color}" stroke-width="2"/>'
                )
                # Draw data points
                for pt in self._points:
                    px, py = to_px(pt.x, pt.y)
                    lines.append(
                        f'<circle cx="{px:.1f}" cy="{py:.1f}" '
                        f'r="3" fill="{self.color}"/>'
                    )

        lines.append("</svg>")
        return "\n".join(lines)
