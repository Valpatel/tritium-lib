# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Timeline — ordered sequence of timestamped events for visualization.

Pure data structure.  Events are kept in chronological order and can be
exported to Vega-Lite JSON or simple SVG.
"""

from __future__ import annotations

import html as html_mod
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimelineEvent:
    """A single point on a timeline.

    Attributes
    ----------
    timestamp : float
        Unix/monotonic timestamp of the event.
    label : str
        Short description displayed on the timeline.
    category : str
        Grouping key (e.g. ``"ble"``, ``"motion"``, ``"audit"``).
    metadata : dict
        Arbitrary key-value pairs attached to this event.
    """

    timestamp: float
    label: str
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "category": self.category,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimelineEvent:
        """Deserialize from a dictionary."""
        return cls(
            timestamp=float(data.get("timestamp", 0.0)),
            label=str(data.get("label", "")),
            category=str(data.get("category", "")),
            metadata=dict(data.get("metadata", {})),
        )


# -- Color palette for categories -------------------------------------------

_CATEGORY_COLORS: dict[str, str] = {
    "ble": "#05ffa1",
    "wifi": "#00f0ff",
    "camera": "#ff2a6d",
    "motion": "#fcee0a",
    "audit": "#00a0ff",
    "combat": "#ff2a6d",
    "system": "#888888",
}

_DEFAULT_COLOR = "#00f0ff"


def _color_for(category: str) -> str:
    return _CATEGORY_COLORS.get(category, _DEFAULT_COLOR)


class Timeline:
    """Ordered collection of :class:`TimelineEvent` instances.

    Events are always sorted by timestamp (ascending).
    """

    def __init__(self, title: str = "Timeline") -> None:
        self.title = title
        self._events: list[TimelineEvent] = []

    # -- Mutation -----------------------------------------------------------

    def add_event(
        self,
        timestamp: float,
        label: str,
        category: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TimelineEvent:
        """Create and insert an event in chronological order."""
        evt = TimelineEvent(
            timestamp=timestamp,
            label=label,
            category=category,
            metadata=metadata or {},
        )
        self._events.append(evt)
        self._events.sort(key=lambda e: e.timestamp)
        return evt

    def add(self, event: TimelineEvent) -> None:
        """Insert an existing :class:`TimelineEvent`."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.timestamp)

    def clear(self) -> None:
        """Remove all events."""
        self._events.clear()

    # -- Query --------------------------------------------------------------

    @property
    def events(self) -> list[TimelineEvent]:
        """All events in chronological order (copy)."""
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __bool__(self) -> bool:
        return len(self._events) > 0

    @property
    def start(self) -> float | None:
        """Timestamp of the earliest event, or ``None`` if empty."""
        return self._events[0].timestamp if self._events else None

    @property
    def end(self) -> float | None:
        """Timestamp of the latest event, or ``None`` if empty."""
        return self._events[-1].timestamp if self._events else None

    @property
    def duration(self) -> float:
        """Time span from first to last event (0 if fewer than 2 events)."""
        if len(self._events) < 2:
            return 0.0
        return self._events[-1].timestamp - self._events[0].timestamp

    @property
    def categories(self) -> list[str]:
        """Unique categories present, sorted alphabetically."""
        return sorted({e.category for e in self._events if e.category})

    def filter(
        self,
        category: str | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> Timeline:
        """Return a new Timeline with events matching the filter criteria."""
        filtered = Timeline(title=self.title)
        for evt in self._events:
            if category is not None and evt.category != category:
                continue
            if start is not None and evt.timestamp < start:
                continue
            if end is not None and evt.timestamp > end:
                continue
            filtered._events.append(evt)
        return filtered

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full timeline to a dictionary."""
        return {
            "title": self.title,
            "events": [e.to_dict() for e in self._events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Timeline:
        """Deserialize from a dictionary."""
        tl = cls(title=data.get("title", "Timeline"))
        for evt_data in data.get("events", []):
            tl._events.append(TimelineEvent.from_dict(evt_data))
        tl._events.sort(key=lambda e: e.timestamp)
        return tl

    # -- Export: Vega-Lite --------------------------------------------------

    def to_vega_lite(self, width: int = 600, height: int = 200) -> dict[str, Any]:
        """Export as a Vega-Lite specification dictionary.

        Produces a horizontal strip-plot (tick marks) with time on the X
        axis, category on the Y axis, and tooltips for labels.
        """
        values = []
        for evt in self._events:
            values.append({
                "timestamp": evt.timestamp,
                "label": evt.label,
                "category": evt.category or "default",
            })

        spec: dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": self.title,
            "width": width,
            "height": height,
            "data": {"values": values},
            "mark": {"type": "tick", "thickness": 2},
            "encoding": {
                "x": {
                    "field": "timestamp",
                    "type": "quantitative",
                    "title": "Time",
                },
                "y": {
                    "field": "category",
                    "type": "nominal",
                    "title": "Category",
                },
                "color": {
                    "field": "category",
                    "type": "nominal",
                    "scale": {
                        "domain": list(_CATEGORY_COLORS.keys()),
                        "range": list(_CATEGORY_COLORS.values()),
                    },
                },
                "tooltip": [
                    {"field": "label", "type": "nominal"},
                    {"field": "timestamp", "type": "quantitative"},
                    {"field": "category", "type": "nominal"},
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
        width: int = 600,
        height: int = 200,
        margin: int = 40,
    ) -> str:
        """Generate a simple SVG timeline visualization.

        Returns a standalone SVG string with tick marks along a horizontal
        time axis.  Category colors are applied per event.
        """
        if not self._events:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}">'
                f'<text x="{width // 2}" y="{height // 2}" '
                f'text-anchor="middle" fill="#888">No events</text></svg>'
            )

        t_min = self._events[0].timestamp
        t_max = self._events[-1].timestamp
        t_range = t_max - t_min if t_max > t_min else 1.0

        plot_w = width - 2 * margin
        plot_h = height - 2 * margin

        lines: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'style="background:#0d0d1a">',
            # Title
            f'<text x="{width // 2}" y="{margin - 10}" '
            f'text-anchor="middle" fill="#00f0ff" font-size="14" '
            f'font-family="monospace">{html_mod.escape(self.title)}</text>',
            # Axis line
            f'<line x1="{margin}" y1="{height - margin}" '
            f'x2="{margin + plot_w}" y2="{height - margin}" '
            f'stroke="#444" stroke-width="1"/>',
        ]

        for evt in self._events:
            x = margin + ((evt.timestamp - t_min) / t_range) * plot_w
            color = _color_for(evt.category)
            y_base = height - margin
            # Tick mark
            lines.append(
                f'<line x1="{x:.1f}" y1="{y_base}" '
                f'x2="{x:.1f}" y2="{y_base - plot_h * 0.6}" '
                f'stroke="{color}" stroke-width="2" opacity="0.8"/>'
            )
            # Small dot
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y_base - plot_h * 0.6}" '
                f'r="3" fill="{color}"/>'
            )

        # Axis labels (start / end)
        lines.append(
            f'<text x="{margin}" y="{height - 5}" fill="#666" '
            f'font-size="10" font-family="monospace">{t_min:.1f}</text>'
        )
        lines.append(
            f'<text x="{margin + plot_w}" y="{height - 5}" fill="#666" '
            f'font-size="10" font-family="monospace" '
            f'text-anchor="end">{t_max:.1f}</text>'
        )

        lines.append("</svg>")
        return "\n".join(lines)
