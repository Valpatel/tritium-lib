# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""NetworkGraph — nodes + edges for relationship visualization.

Pure data structure for entity-relationship graphs.  Exports to Vega-Lite
node-link diagrams or simple SVG force-layout approximation.
"""

from __future__ import annotations

import html as html_mod
import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    """A node in the network graph.

    Attributes
    ----------
    node_id : str
        Unique identifier.
    label : str
        Display label.
    group : str
        Grouping key for coloring (e.g. ``"ble"``, ``"camera"``).
    metadata : dict
        Arbitrary attached data.
    x : float
        Optional layout x position (0.0 if not laid out).
    y : float
        Optional layout y position (0.0 if not laid out).
    """

    node_id: str
    label: str = ""
    group: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "label": self.label or self.node_id,
            "group": self.group,
            "metadata": dict(self.metadata),
            "x": self.x,
            "y": self.y,
        }


@dataclass
class GraphEdge:
    """An edge connecting two nodes.

    Attributes
    ----------
    source : str
        Source node ID.
    target : str
        Target node ID.
    label : str
        Edge label (relationship type).
    weight : float
        Edge weight (e.g. correlation strength).
    metadata : dict
        Arbitrary attached data.
    """

    source: str
    target: str
    label: str = ""
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }


# -- Group colors -----------------------------------------------------------

_GROUP_COLORS: dict[str, str] = {
    "ble": "#05ffa1",
    "wifi": "#00f0ff",
    "camera": "#ff2a6d",
    "mesh": "#fcee0a",
    "acoustic": "#a855f7",
    "person": "#ff2a6d",
    "vehicle": "#00f0ff",
    "device": "#05ffa1",
}

_DEFAULT_NODE_COLOR = "#00f0ff"


def _color_for_group(group: str) -> str:
    return _GROUP_COLORS.get(group, _DEFAULT_NODE_COLOR)


class NetworkGraph:
    """Collection of nodes and edges representing entity relationships.

    Parameters
    ----------
    title : str
        Display title.
    directed : bool
        Whether edges are directed.
    """

    def __init__(self, title: str = "Network Graph", directed: bool = False) -> None:
        self.title = title
        self.directed = directed
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    # -- Mutation -----------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        label: str = "",
        group: str = "",
        metadata: dict[str, Any] | None = None,
        x: float = 0.0,
        y: float = 0.0,
    ) -> GraphNode:
        """Add or update a node.  Returns the node."""
        node = GraphNode(
            node_id=node_id,
            label=label or node_id,
            group=group,
            metadata=metadata or {},
            x=x,
            y=y,
        )
        self._nodes[node_id] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        label: str = "",
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> GraphEdge:
        """Add an edge.  Source and target must be existing node IDs."""
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not found")
        edge = GraphEdge(
            source=source,
            target=target,
            label=label,
            weight=weight,
            metadata=metadata or {},
        )
        self._edges.append(edge)
        return edge

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        self._nodes.pop(node_id, None)
        self._edges = [
            e for e in self._edges
            if e.source != node_id and e.target != node_id
        ]

    def remove_edge(self, source: str, target: str) -> None:
        """Remove all edges between source and target."""
        self._edges = [
            e for e in self._edges
            if not (e.source == source and e.target == target)
        ]

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()

    # -- Query --------------------------------------------------------------

    @property
    def nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[GraphEdge]:
        return list(self._edges)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def __len__(self) -> int:
        return len(self._nodes)

    def __bool__(self) -> bool:
        return len(self._nodes) > 0

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str) -> list[str]:
        """Return IDs of all nodes connected to *node_id*."""
        result: set[str] = set()
        for e in self._edges:
            if e.source == node_id:
                result.add(e.target)
            if e.target == node_id and not self.directed:
                result.add(e.source)
        return sorted(result)

    def degree(self, node_id: str) -> int:
        """Number of edges connected to *node_id*."""
        count = 0
        for e in self._edges:
            if e.source == node_id:
                count += 1
            if e.target == node_id:
                count += 1
        return count

    @property
    def groups(self) -> list[str]:
        """Unique node groups, sorted."""
        return sorted({n.group for n in self._nodes.values() if n.group})

    # -- Layout -------------------------------------------------------------

    def circular_layout(self, radius: float = 100.0, cx: float = 0.0, cy: float = 0.0) -> None:
        """Assign node positions in a circular layout."""
        node_list = list(self._nodes.values())
        n = len(node_list)
        if n == 0:
            return
        for i, node in enumerate(node_list):
            angle = 2 * math.pi * i / n
            node.x = cx + radius * math.cos(angle)
            node.y = cy + radius * math.sin(angle)

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "directed": self.directed,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkGraph:
        g = cls(
            title=data.get("title", "Network Graph"),
            directed=data.get("directed", False),
        )
        for nd in data.get("nodes", []):
            g.add_node(
                node_id=nd.get("id", ""),
                label=nd.get("label", ""),
                group=nd.get("group", ""),
                metadata=nd.get("metadata", {}),
                x=nd.get("x", 0.0),
                y=nd.get("y", 0.0),
            )
        for ed in data.get("edges", []):
            try:
                g.add_edge(
                    source=ed.get("source", ""),
                    target=ed.get("target", ""),
                    label=ed.get("label", ""),
                    weight=ed.get("weight", 1.0),
                    metadata=ed.get("metadata", {}),
                )
            except ValueError:
                pass  # Skip edges with missing nodes during deserialization
        return g

    # -- Export: Vega-Lite --------------------------------------------------

    def to_vega_lite(self, width: int = 500, height: int = 500) -> dict[str, Any]:
        """Export as a Vega-Lite node-link diagram specification.

        Uses layered point + text marks for nodes and rule marks for edges.
        If nodes have no layout positions, applies circular layout first.
        """
        # Auto-layout if all nodes are at origin
        all_zero = all(
            n.x == 0.0 and n.y == 0.0 for n in self._nodes.values()
        )
        if all_zero and self._nodes:
            self.circular_layout(
                radius=min(width, height) * 0.35,
                cx=width / 2,
                cy=height / 2,
            )

        node_values = []
        for n in self._nodes.values():
            node_values.append({
                "id": n.node_id,
                "label": n.label or n.node_id,
                "group": n.group or "default",
                "x": n.x,
                "y": n.y,
            })

        edge_values = []
        for e in self._edges:
            src = self._nodes.get(e.source)
            tgt = self._nodes.get(e.target)
            if src and tgt:
                edge_values.append({
                    "x": src.x,
                    "y": src.y,
                    "x2": tgt.x,
                    "y2": tgt.y,
                    "label": e.label,
                })

        spec: dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": self.title,
            "width": width,
            "height": height,
            "layer": [
                # Edges
                {
                    "data": {"values": edge_values},
                    "mark": {"type": "rule", "color": "#444", "strokeWidth": 1},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative", "scale": {"zero": False}},
                        "y": {"field": "y", "type": "quantitative", "scale": {"zero": False}},
                        "x2": {"field": "x2"},
                        "y2": {"field": "y2"},
                    },
                },
                # Nodes
                {
                    "data": {"values": node_values},
                    "mark": {"type": "circle", "size": 100},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative", "scale": {"zero": False}, "title": ""},
                        "y": {"field": "y", "type": "quantitative", "scale": {"zero": False}, "title": ""},
                        "color": {
                            "field": "group",
                            "type": "nominal",
                            "scale": {
                                "domain": list(_GROUP_COLORS.keys()),
                                "range": list(_GROUP_COLORS.values()),
                            },
                        },
                        "tooltip": [
                            {"field": "id", "type": "nominal"},
                            {"field": "label", "type": "nominal"},
                            {"field": "group", "type": "nominal"},
                        ],
                    },
                },
                # Labels
                {
                    "data": {"values": node_values},
                    "mark": {"type": "text", "dy": -12, "fontSize": 10, "color": "#ccc"},
                    "encoding": {
                        "x": {"field": "x", "type": "quantitative", "scale": {"zero": False}},
                        "y": {"field": "y", "type": "quantitative", "scale": {"zero": False}},
                        "text": {"field": "label", "type": "nominal"},
                    },
                },
            ],
        }
        return spec

    def to_vega_lite_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_vega_lite(**kwargs), indent=2)

    # -- Export: SVG --------------------------------------------------------

    def to_svg(
        self,
        width: int = 500,
        height: int = 500,
        margin: int = 40,
    ) -> str:
        """Generate a simple SVG network graph.

        Applies circular layout if no positions are set.  Draws edges as
        lines and nodes as colored circles with labels.
        """
        if not self._nodes:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}">'
                f'<text x="{width // 2}" y="{height // 2}" '
                f'text-anchor="middle" fill="#888">No nodes</text></svg>'
            )

        # Auto-layout if needed
        all_zero = all(
            n.x == 0.0 and n.y == 0.0 for n in self._nodes.values()
        )
        if all_zero:
            self.circular_layout(
                radius=min(width, height) / 2 - margin - 20,
                cx=width / 2,
                cy=height / 2,
            )

        lines: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'style="background:#0d0d1a">',
            # Title
            f'<text x="{width // 2}" y="{margin - 10}" '
            f'text-anchor="middle" fill="#00f0ff" font-size="14" '
            f'font-family="monospace">{html_mod.escape(self.title)}</text>',
        ]

        # Draw edges
        for edge in self._edges:
            src = self._nodes.get(edge.source)
            tgt = self._nodes.get(edge.target)
            if src and tgt:
                lines.append(
                    f'<line x1="{src.x:.1f}" y1="{src.y:.1f}" '
                    f'x2="{tgt.x:.1f}" y2="{tgt.y:.1f}" '
                    f'stroke="#444" stroke-width="1" opacity="0.6"/>'
                )
                # Edge label at midpoint
                if edge.label:
                    mx = (src.x + tgt.x) / 2
                    my = (src.y + tgt.y) / 2
                    lines.append(
                        f'<text x="{mx:.1f}" y="{my:.1f}" '
                        f'text-anchor="middle" fill="#666" font-size="9" '
                        f'font-family="monospace">'
                        f'{html_mod.escape(edge.label)}</text>'
                    )

        # Draw nodes
        for node in self._nodes.values():
            color = _color_for_group(node.group)
            lines.append(
                f'<circle cx="{node.x:.1f}" cy="{node.y:.1f}" '
                f'r="8" fill="{color}" opacity="0.9"/>'
            )
            lines.append(
                f'<text x="{node.x:.1f}" y="{node.y - 12:.1f}" '
                f'text-anchor="middle" fill="#ccc" font-size="10" '
                f'font-family="monospace">'
                f'{html_mod.escape(node.label or node.node_id)}</text>'
            )

        lines.append("</svg>")
        return "\n".join(lines)
