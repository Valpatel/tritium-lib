# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Graph + Ontology demo — entity-relationship storage, querying, visualization.

Run standalone:
    python3 -m tritium_lib.graph.demos.graph_demo

Starts an HTTP server on port 8099 with:
    GET  /                  — Interactive SVG visualization
    GET  /api/entities      — All entities as JSON
    GET  /api/relationships — All relationships as JSON
    GET  /api/traverse/{id} — Subgraph from entity (query param: hops=N)
    GET  /api/search?q=text — Search entities by name/ID
    POST /api/entity        — Create entity (JSON body)
    POST /api/relationship  — Create relationship (JSON body)
    GET  /api/ontology      — Ontology schema summary
"""

from __future__ import annotations

import html
import json
import os
import shutil
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse, parse_qs

from tritium_lib.graph.store import TritiumGraph, NODE_TABLES, REL_TABLES
from tritium_lib.ontology import (
    OntologyRegistry,
    TRITIUM_ONTOLOGY,
)


# ── Demo Data ─────────────────────────────────────────────────────────


def populate_demo_graph(graph: TritiumGraph) -> None:
    """Populate the graph with a realistic surveillance scenario.

    Creates people, devices, vehicles, locations, and a network — then
    wires them together with carries, detected_with, traveled_with,
    observed_at, connected_to, and correlated_with relationships.
    """

    # -- People --
    graph.create_entity("Person", "person-alice", "Alice Chen", {
        "role": "analyst", "clearance": "secret",
    })
    graph.create_entity("Person", "person-bob", "Bob Martinez", {
        "role": "field-operator", "clearance": "top-secret",
    })
    graph.create_entity("Person", "person-carol", "Carol Davis", {
        "role": "suspect", "threat_level": "medium",
    })
    graph.create_entity("Person", "person-dave", "Dave Kim", {
        "role": "informant", "reliability": "B",
    })

    # -- Devices --
    graph.create_entity("Device", "ble-aa:bb:cc:01", "Alice's iPhone", {
        "mac": "AA:BB:CC:DD:EE:01", "type": "phone", "os": "iOS",
    })
    graph.create_entity("Device", "ble-aa:bb:cc:02", "Bob's Android", {
        "mac": "AA:BB:CC:DD:EE:02", "type": "phone", "os": "Android",
    })
    graph.create_entity("Device", "ble-aa:bb:cc:03", "Carol's Laptop", {
        "mac": "AA:BB:CC:DD:EE:03", "type": "laptop", "os": "Linux",
    })
    graph.create_entity("Device", "ble-dd:ee:ff:04", "Burner Phone", {
        "mac": "DD:EE:FF:00:11:04", "type": "phone", "os": "Android",
    })
    graph.create_entity("Device", "wifi-scanner-01", "WiFi Scanner Alpha", {
        "type": "scanner", "location": "entrance",
    })

    # -- Vehicles --
    graph.create_entity("Vehicle", "veh-honda-civic", "Honda Civic (Silver)", {
        "plate": "ABC-1234", "color": "silver", "make": "Honda",
    })
    graph.create_entity("Vehicle", "veh-ford-f150", "Ford F-150 (Black)", {
        "plate": "XYZ-9876", "color": "black", "make": "Ford",
    })

    # -- Locations --
    graph.create_entity("Location", "loc-hq", "HQ Building", {
        "lat": 40.7128, "lon": -74.0060, "type": "building",
    })
    graph.create_entity("Location", "loc-warehouse", "East Side Warehouse", {
        "lat": 40.7200, "lon": -73.9950, "type": "warehouse",
    })
    graph.create_entity("Location", "loc-cafe", "Downtown Cafe", {
        "lat": 40.7150, "lon": -74.0020, "type": "cafe",
    })

    # -- Networks --
    graph.create_entity("Network", "net-hq-wifi", "HQ-SecureNet", {
        "ssid": "HQ-SecureNet", "encryption": "WPA3",
    })
    graph.create_entity("Network", "net-cafe-wifi", "CafeGuest", {
        "ssid": "CafeGuest", "encryption": "WPA2",
    })

    # -- Cameras --
    graph.create_entity("Camera", "cam-entrance", "Entrance Camera", {
        "resolution": "1080p", "coverage": "entrance",
    })
    graph.create_entity("Camera", "cam-parking", "Parking Lot Camera", {
        "resolution": "4K", "coverage": "parking",
    })

    # ── Relationships ─────────────────────────────────────────────────

    # CARRIES — person -> device
    graph.add_relationship("person-alice", "ble-aa:bb:cc:01", "CARRIES", {
        "source": "visual-confirmation", "confidence": 0.95,
    })
    graph.add_relationship("person-bob", "ble-aa:bb:cc:02", "CARRIES", {
        "source": "badge-scan", "confidence": 0.99,
    })
    graph.add_relationship("person-carol", "ble-aa:bb:cc:03", "CARRIES", {
        "source": "wifi-correlation", "confidence": 0.80,
    })
    graph.add_relationship("person-carol", "ble-dd:ee:ff:04", "CARRIES", {
        "source": "rf-proximity", "confidence": 0.65,
    })

    # DETECTED_WITH — device <-> device seen together
    graph.add_relationship("ble-aa:bb:cc:01", "ble-aa:bb:cc:02", "DETECTED_WITH", {
        "source": "ble-scanner", "confidence": 0.90, "count": 12,
    })
    graph.add_relationship("ble-aa:bb:cc:03", "ble-dd:ee:ff:04", "DETECTED_WITH", {
        "source": "wifi-scanner", "confidence": 0.85, "count": 8,
    })

    # OBSERVED_AT — entity -> location
    graph.add_relationship("person-alice", "loc-hq", "OBSERVED_AT", {
        "source": "badge-reader", "confidence": 1.0,
    })
    graph.add_relationship("person-bob", "loc-hq", "OBSERVED_AT", {
        "source": "badge-reader", "confidence": 1.0,
    })
    graph.add_relationship("person-carol", "loc-warehouse", "OBSERVED_AT", {
        "source": "camera", "confidence": 0.75,
    })
    graph.add_relationship("person-carol", "loc-cafe", "OBSERVED_AT", {
        "source": "wifi-probe", "confidence": 0.70,
    })
    graph.add_relationship("person-dave", "loc-cafe", "OBSERVED_AT", {
        "source": "visual", "confidence": 0.60,
    })
    graph.add_relationship("veh-honda-civic", "loc-warehouse", "OBSERVED_AT", {
        "source": "lpr-camera", "confidence": 0.95,
    })
    graph.add_relationship("veh-ford-f150", "loc-hq", "OBSERVED_AT", {
        "source": "lpr-camera", "confidence": 0.98,
    })

    # CONNECTED_TO — device -> network
    graph.add_relationship("ble-aa:bb:cc:01", "net-hq-wifi", "CONNECTED_TO", {
        "source": "wifi-monitor", "confidence": 1.0,
    })
    graph.add_relationship("ble-aa:bb:cc:02", "net-hq-wifi", "CONNECTED_TO", {
        "source": "wifi-monitor", "confidence": 1.0,
    })
    graph.add_relationship("ble-aa:bb:cc:03", "net-cafe-wifi", "CONNECTED_TO", {
        "source": "wifi-monitor", "confidence": 0.90,
    })
    graph.add_relationship("ble-dd:ee:ff:04", "net-cafe-wifi", "CONNECTED_TO", {
        "source": "wifi-monitor", "confidence": 0.85,
    })

    # TRAVELED_WITH — person <-> person seen moving together
    graph.add_relationship("person-carol", "person-dave", "TRAVELED_WITH", {
        "source": "trajectory-analysis", "confidence": 0.70, "count": 3,
    })

    # CORRELATED_WITH — link burner phone to Carol via signal correlation
    graph.add_relationship("ble-dd:ee:ff:04", "ble-aa:bb:cc:03", "CORRELATED_WITH", {
        "source": "signal-correlation", "confidence": 0.75,
    })

    # DETECTED_BY — device seen by camera/scanner
    graph.add_relationship("ble-aa:bb:cc:01", "cam-entrance", "DETECTED_BY", {
        "source": "ble-proximity", "confidence": 0.92,
    })
    graph.add_relationship("veh-honda-civic", "cam-parking", "DETECTED_BY", {
        "source": "yolo-detection", "confidence": 0.88,
    })


# ── Query helpers ─────────────────────────────────────────────────────


def get_all_entities(graph: TritiumGraph) -> list[dict[str, Any]]:
    """Return every entity in the graph across all node tables."""
    entities: list[dict[str, Any]] = []
    for table in NODE_TABLES:
        rows = graph.query(
            f"MATCH (n:{table}) "
            f"RETURN n.id, n.name, n.entity_type, n.first_seen, "
            f"n.last_seen, n.confidence, n.properties"
        )
        for row in rows:
            entity = {
                "id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "first_seen": row[3],
                "last_seen": row[4],
                "confidence": row[5],
                "properties": _parse_props(row[6]),
            }
            entities.append(entity)
    return entities


def get_all_relationships(graph: TritiumGraph) -> list[dict[str, Any]]:
    """Return every relationship in the graph."""
    rels: list[dict[str, Any]] = []
    for table in NODE_TABLES:
        rows = graph.query(
            f"MATCH (a:{table})-[r]->(b) "
            f"RETURN a.id, b.id, label(r), r.timestamp, "
            f"r.confidence, r.source, r.count"
        )
        for row in rows:
            rels.append({
                "from_id": row[0],
                "to_id": row[1],
                "rel_type": row[2],
                "timestamp": row[3],
                "confidence": row[4],
                "source": row[5],
                "count": row[6],
            })
    # Deduplicate (same edge can appear from different table scans)
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for r in rels:
        key = (r["from_id"], r["to_id"], r["rel_type"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def find_devices_carried_by(
    graph: TritiumGraph, person_id: str
) -> list[dict[str, Any]]:
    """Find all devices carried by a specific person."""
    rels = graph.get_relationships(person_id, rel_type="CARRIES", direction="out")
    devices = []
    for rel in rels:
        entity = graph.get_entity(rel["to_id"])
        if entity:
            devices.append(entity)
    return devices


def find_traveled_together(graph: TritiumGraph) -> list[dict[str, Any]]:
    """Find all pairs of entities that traveled together."""
    pairs: list[dict[str, Any]] = []
    for table in NODE_TABLES:
        rows = graph.query(
            f"MATCH (a:{table})-[r:TRAVELED_WITH]->(b) "
            f"RETURN a.id, a.name, b.id, b.name, r.confidence, r.count"
        )
        for row in rows:
            pairs.append({
                "entity_a": {"id": row[0], "name": row[1]},
                "entity_b": {"id": row[2], "name": row[3]},
                "confidence": row[4],
                "count": row[5],
            })
    return pairs


def get_ontology_summary() -> dict[str, Any]:
    """Return a summary of the ontology schema."""
    registry = OntologyRegistry()
    registry.load_schema(TRITIUM_ONTOLOGY)

    entity_types = []
    for et in registry.list_entity_types():
        entity_types.append({
            "api_name": et.api_name,
            "display_name": et.display_name,
            "primary_key": et.primary_key_field,
            "property_count": len(et.properties),
            "interfaces": et.interfaces,
        })

    rel_types = []
    for rt in registry.list_relationship_types():
        rel_types.append({
            "api_name": rt.api_name,
            "display_name": rt.display_name,
            "from_type": rt.from_type,
            "to_type": rt.to_type,
            "cardinality": rt.cardinality.value,
        })

    interfaces = []
    for iface in TRITIUM_ONTOLOGY.interfaces.values():
        interfaces.append({
            "api_name": iface.api_name,
            "display_name": iface.display_name,
            "required_properties": iface.required_properties,
        })

    return {
        "version": TRITIUM_ONTOLOGY.version,
        "entity_types": entity_types,
        "relationship_types": rel_types,
        "interfaces": interfaces,
    }


def _parse_props(val: Any) -> Any:
    """Parse JSON properties string, returning raw value on failure."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


# ── HTML Visualization ────────────────────────────────────────────────

# Colors for each entity type
ENTITY_COLORS: dict[str, str] = {
    "Person": "#ff2a6d",
    "Device": "#00f0ff",
    "Vehicle": "#fcee0a",
    "Location": "#05ffa1",
    "Network": "#b967ff",
    "Camera": "#ff8c00",
    "MeshNode": "#00bfff",
    "Zone": "#32cd32",
}

# Icon labels per type
ENTITY_ICONS: dict[str, str] = {
    "Person": "P",
    "Device": "D",
    "Vehicle": "V",
    "Location": "L",
    "Network": "N",
    "Camera": "C",
    "MeshNode": "M",
    "Zone": "Z",
}


def _esc(text: str) -> str:
    return html.escape(str(text))


def generate_visualization_html(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> str:
    """Generate a self-contained HTML page with interactive SVG graph visualization."""
    entities_json = json.dumps(entities)
    rels_json = json.dumps(relationships)
    colors_json = json.dumps(ENTITY_COLORS)
    icons_json = json.dumps(ENTITY_ICONS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tritium Graph Ontology Demo</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0a0a0a;
    color: #c0c0c0;
    font-family: 'Courier New', 'Fira Code', monospace;
    overflow: hidden;
}}
#header {{
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 48px;
    background: #111;
    border-bottom: 1px solid #00f0ff44;
    display: flex;
    align-items: center;
    padding: 0 16px;
    z-index: 100;
    gap: 16px;
}}
#header h1 {{
    color: #00f0ff;
    font-size: 16px;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 0 10px #00f0ff66;
}}
#header .stats {{
    color: #666;
    font-size: 12px;
}}
#header .stats span {{
    color: #05ffa1;
}}
#sidebar {{
    position: fixed;
    top: 48px; right: 0; bottom: 0;
    width: 320px;
    background: #111;
    border-left: 1px solid #00f0ff22;
    overflow-y: auto;
    padding: 12px;
    z-index: 90;
    display: none;
}}
#sidebar.open {{ display: block; }}
#sidebar h2 {{
    color: #00f0ff;
    font-size: 14px;
    margin-bottom: 8px;
    text-transform: uppercase;
}}
#sidebar .entity-name {{
    color: #ff2a6d;
    font-size: 18px;
    margin-bottom: 4px;
}}
#sidebar .entity-type {{
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 3px;
    display: inline-block;
    margin-bottom: 12px;
}}
#sidebar .section {{
    margin-bottom: 16px;
    border-top: 1px solid #1a1a1a;
    padding-top: 8px;
}}
#sidebar .prop-key {{ color: #666; }}
#sidebar .prop-val {{ color: #05ffa1; }}
#sidebar .rel-item {{
    padding: 4px 0;
    font-size: 12px;
    border-bottom: 1px solid #1a1a1a;
}}
#sidebar .rel-type {{ color: #fcee0a; }}
#sidebar .close-btn {{
    position: absolute;
    top: 8px; right: 12px;
    background: none;
    border: 1px solid #ff2a6d44;
    color: #ff2a6d;
    padding: 2px 8px;
    cursor: pointer;
    font-family: inherit;
}}
#sidebar .close-btn:hover {{ background: #ff2a6d22; }}
#canvas-container {{
    position: fixed;
    top: 48px; left: 0; right: 0; bottom: 0;
}}
svg {{
    width: 100%;
    height: 100%;
    cursor: grab;
}}
svg:active {{ cursor: grabbing; }}
.edge-line {{
    stroke: #333;
    stroke-width: 1.5;
    fill: none;
}}
.edge-line:hover {{
    stroke: #00f0ff88;
    stroke-width: 2.5;
}}
.edge-label {{
    fill: #555;
    font-size: 9px;
    pointer-events: none;
    font-family: 'Courier New', monospace;
}}
.node-circle {{
    stroke-width: 2;
    cursor: pointer;
    transition: r 0.15s;
}}
.node-circle:hover {{
    filter: brightness(1.3);
}}
.node-label {{
    fill: #c0c0c0;
    font-size: 11px;
    pointer-events: none;
    font-family: 'Courier New', monospace;
    text-anchor: middle;
}}
.node-icon {{
    fill: #0a0a0a;
    font-size: 12px;
    font-weight: bold;
    pointer-events: none;
    font-family: 'Courier New', monospace;
    text-anchor: middle;
    dominant-baseline: central;
}}
.legend {{
    position: fixed;
    bottom: 12px; left: 12px;
    background: #111d;
    border: 1px solid #1a1a1a;
    padding: 8px 12px;
    border-radius: 4px;
    z-index: 80;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    margin: 3px 0;
}}
.legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}}
#controls {{
    position: fixed;
    bottom: 12px; right: 340px;
    display: flex;
    gap: 8px;
    z-index: 80;
}}
#controls button {{
    background: #111;
    border: 1px solid #00f0ff44;
    color: #00f0ff;
    padding: 6px 14px;
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
    border-radius: 3px;
}}
#controls button:hover {{
    background: #00f0ff22;
}}
</style>
</head>
<body>

<div id="header">
    <h1>Tritium Graph</h1>
    <div class="stats">
        Entities: <span id="stat-entities">0</span> |
        Relationships: <span id="stat-rels">0</span> |
        Ontology: <span id="stat-version">-</span>
    </div>
</div>

<div id="sidebar">
    <button class="close-btn" onclick="closeSidebar()">X</button>
    <div id="sidebar-content"></div>
</div>

<div id="canvas-container">
    <svg id="graph-svg">
        <defs>
            <marker id="arrowhead" markerWidth="8" markerHeight="6"
                    refX="8" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#444" />
            </marker>
        </defs>
        <g id="graph-group">
            <g id="edges-group"></g>
            <g id="nodes-group"></g>
        </g>
    </svg>
</div>

<div class="legend" id="legend"></div>

<div id="controls">
    <button onclick="resetZoom()">Reset Zoom</button>
    <button onclick="togglePhysics()">Pause Physics</button>
    <button onclick="refreshData()">Refresh</button>
</div>

<script>
// ── Data ──────────────────────────────────────────────────────────
const entities = {entities_json};
const relationships = {rels_json};
const COLORS = {colors_json};
const ICONS = {icons_json};

// Stats
document.getElementById('stat-entities').textContent = entities.length;
document.getElementById('stat-rels').textContent = relationships.length;
document.getElementById('stat-version').textContent = 'v1.0.0';

// ── Force-directed layout ─────────────────────────────────────────
const W = window.innerWidth;
const H = window.innerHeight - 48;
const CX = W / 2;
const CY = H / 2;

// Build node map
const nodeMap = {{}};
entities.forEach((e, i) => {{
    const angle = (2 * Math.PI * i) / entities.length;
    const r = Math.min(W, H) * 0.3;
    nodeMap[e.id] = {{
        ...e,
        x: CX + r * Math.cos(angle) + (Math.random() - 0.5) * 40,
        y: CY + r * Math.sin(angle) + (Math.random() - 0.5) * 40,
        vx: 0, vy: 0,
        radius: 16,
    }};
}});

// Edge list referencing node objects
const edges = relationships.map(r => ({{
    source: nodeMap[r.from_id],
    target: nodeMap[r.to_id],
    rel_type: r.rel_type,
    confidence: r.confidence || 0,
    src: r.source || '',
}})).filter(e => e.source && e.target);

// ── Physics simulation ────────────────────────────────────────────
let physicsRunning = true;
const REPULSION = 8000;
const ATTRACTION = 0.005;
const DAMPING = 0.85;
const REST_LENGTH = 140;
const CENTER_GRAVITY = 0.002;

function simulate() {{
    const nodes = Object.values(nodeMap);

    // Repulsion (all pairs)
    for (let i = 0; i < nodes.length; i++) {{
        for (let j = i + 1; j < nodes.length; j++) {{
            const a = nodes[i], b = nodes[j];
            let dx = b.x - a.x, dy = b.y - a.y;
            let dist = Math.sqrt(dx * dx + dy * dy) || 1;
            let force = REPULSION / (dist * dist);
            let fx = (dx / dist) * force;
            let fy = (dy / dist) * force;
            a.vx -= fx; a.vy -= fy;
            b.vx += fx; b.vy += fy;
        }}
    }}

    // Attraction (edges)
    edges.forEach(e => {{
        let dx = e.target.x - e.source.x;
        let dy = e.target.y - e.source.y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 1;
        let force = (dist - REST_LENGTH) * ATTRACTION;
        let fx = (dx / dist) * force;
        let fy = (dy / dist) * force;
        e.source.vx += fx; e.source.vy += fy;
        e.target.vx -= fx; e.target.vy -= fy;
    }});

    // Center gravity
    nodes.forEach(n => {{
        n.vx += (CX - n.x) * CENTER_GRAVITY;
        n.vy += (CY - n.y) * CENTER_GRAVITY;
    }});

    // Apply velocity
    nodes.forEach(n => {{
        if (n._dragging) return;
        n.vx *= DAMPING;
        n.vy *= DAMPING;
        n.x += n.vx;
        n.y += n.vy;
        // Clamp
        n.x = Math.max(40, Math.min(W - 40, n.x));
        n.y = Math.max(40, Math.min(H - 40, n.y));
    }});
}}

// ── SVG Rendering ─────────────────────────────────────────────────
const svg = document.getElementById('graph-svg');
const graphGroup = document.getElementById('graph-group');
const edgesGroup = document.getElementById('edges-group');
const nodesGroup = document.getElementById('nodes-group');

// Pan and zoom state
let transform = {{ x: 0, y: 0, scale: 1 }};
let isPanning = false, panStart = {{ x: 0, y: 0 }};

svg.addEventListener('mousedown', e => {{
    if (e.target === svg || e.target === graphGroup) {{
        isPanning = true;
        panStart = {{ x: e.clientX - transform.x, y: e.clientY - transform.y }};
    }}
}});
svg.addEventListener('mousemove', e => {{
    if (isPanning) {{
        transform.x = e.clientX - panStart.x;
        transform.y = e.clientY - panStart.y;
        applyTransform();
    }}
}});
svg.addEventListener('mouseup', () => {{ isPanning = false; }});
svg.addEventListener('mouseleave', () => {{ isPanning = false; }});
svg.addEventListener('wheel', e => {{
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    transform.scale *= delta;
    transform.scale = Math.max(0.2, Math.min(3, transform.scale));
    applyTransform();
}});

function applyTransform() {{
    graphGroup.setAttribute('transform',
        `translate(${{transform.x}},${{transform.y}}) scale(${{transform.scale}})`);
}}

function resetZoom() {{
    transform = {{ x: 0, y: 0, scale: 1 }};
    applyTransform();
}}

// Create edge elements
const edgeElements = [];
edges.forEach(e => {{
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.classList.add('edge-line');
    line.setAttribute('marker-end', 'url(#arrowhead)');
    edgesGroup.appendChild(line);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.classList.add('edge-label');
    label.textContent = e.rel_type.replace(/_/g, ' ');
    edgesGroup.appendChild(label);

    edgeElements.push({{ line, label, edge: e }});
}});

// Create node elements
const nodeElements = [];
Object.values(nodeMap).forEach(n => {{
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.style.cursor = 'pointer';

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.classList.add('node-circle');
    circle.setAttribute('r', n.radius);
    circle.setAttribute('fill', (COLORS[n.entity_type] || '#666') + '44');
    circle.setAttribute('stroke', COLORS[n.entity_type] || '#666');
    g.appendChild(circle);

    const icon = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    icon.classList.add('node-icon');
    icon.textContent = ICONS[n.entity_type] || '?';
    icon.setAttribute('fill', COLORS[n.entity_type] || '#666');
    g.appendChild(icon);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.classList.add('node-label');
    label.textContent = n.name || n.id;
    label.setAttribute('dy', n.radius + 14);
    g.appendChild(label);

    // Click handler
    g.addEventListener('click', () => showEntityDetails(n));

    // Drag
    let dragging = false, dragOff = {{ x: 0, y: 0 }};
    g.addEventListener('mousedown', e => {{
        e.stopPropagation();
        dragging = true;
        n._dragging = true;
        dragOff.x = e.clientX / transform.scale - n.x;
        dragOff.y = e.clientY / transform.scale - n.y;
    }});
    window.addEventListener('mousemove', e => {{
        if (dragging) {{
            n.x = e.clientX / transform.scale - dragOff.x;
            n.y = e.clientY / transform.scale - dragOff.y;
            n.vx = 0; n.vy = 0;
        }}
    }});
    window.addEventListener('mouseup', () => {{
        dragging = false;
        n._dragging = false;
    }});

    nodesGroup.appendChild(g);
    nodeElements.push({{ g, circle, icon, label, node: n }});
}});

function render() {{
    // Update edges
    edgeElements.forEach(({{ line, label, edge }}) => {{
        const s = edge.source, t = edge.target;
        // Offset endpoint to stop at circle edge
        const dx = t.x - s.x, dy = t.y - s.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const sx = s.x + (dx / dist) * s.radius;
        const sy = s.y + (dy / dist) * s.radius;
        const tx = t.x - (dx / dist) * t.radius;
        const ty = t.y - (dy / dist) * t.radius;

        line.setAttribute('x1', sx);
        line.setAttribute('y1', sy);
        line.setAttribute('x2', tx);
        line.setAttribute('y2', ty);

        label.setAttribute('x', (s.x + t.x) / 2);
        label.setAttribute('y', (s.y + t.y) / 2 - 4);
    }});

    // Update nodes
    nodeElements.forEach(({{ g, node }}) => {{
        g.setAttribute('transform', `translate(${{node.x}},${{node.y}})`);
    }});
}}

// ── Animation loop ────────────────────────────────────────────────
function tick() {{
    if (physicsRunning) simulate();
    render();
    requestAnimationFrame(tick);
}}
tick();

function togglePhysics() {{
    physicsRunning = !physicsRunning;
    event.target.textContent = physicsRunning ? 'Pause Physics' : 'Resume Physics';
}}

// ── Sidebar ───────────────────────────────────────────────────────
function showEntityDetails(entity) {{
    const sidebar = document.getElementById('sidebar');
    const content = document.getElementById('sidebar-content');
    const color = COLORS[entity.entity_type] || '#666';

    // Get relationships for this entity
    const outRels = relationships.filter(r => r.from_id === entity.id);
    const inRels = relationships.filter(r => r.to_id === entity.id);

    let propsHtml = '';
    if (entity.properties && typeof entity.properties === 'object') {{
        Object.entries(entity.properties).forEach(([k, v]) => {{
            propsHtml += `<div><span class="prop-key">${{k}}:</span> <span class="prop-val">${{v}}</span></div>`;
        }});
    }}

    let relsHtml = '';
    outRels.forEach(r => {{
        const target = nodeMap[r.to_id];
        relsHtml += `<div class="rel-item">
            <span class="rel-type">${{r.rel_type}}</span> &rarr;
            ${{target ? target.name || target.id : r.to_id}}
            <span class="prop-key">(conf: ${{(r.confidence || 0).toFixed(2)}})</span>
        </div>`;
    }});
    inRels.forEach(r => {{
        const source = nodeMap[r.from_id];
        relsHtml += `<div class="rel-item">
            ${{source ? source.name || source.id : r.from_id}} &rarr;
            <span class="rel-type">${{r.rel_type}}</span>
            <span class="prop-key">(conf: ${{(r.confidence || 0).toFixed(2)}})</span>
        </div>`;
    }});

    content.innerHTML = `
        <div class="entity-name" style="color:${{color}}">${{entity.name || entity.id}}</div>
        <span class="entity-type" style="background:${{color}}22;color:${{color}};border:1px solid ${{color}}44">${{entity.entity_type}}</span>
        <div style="font-size:11px;color:#666;margin-bottom:12px">ID: ${{entity.id}}</div>

        <div class="section">
            <h2>Properties</h2>
            ${{propsHtml || '<div style="color:#444">None</div>'}}
        </div>

        <div class="section">
            <h2>Relationships (${{outRels.length + inRels.length}})</h2>
            ${{relsHtml || '<div style="color:#444">None</div>'}}
        </div>

        <div class="section">
            <h2>Timeline</h2>
            <div><span class="prop-key">First seen:</span> <span class="prop-val">${{entity.first_seen || '-'}}</span></div>
            <div><span class="prop-key">Last seen:</span> <span class="prop-val">${{entity.last_seen || '-'}}</span></div>
            <div><span class="prop-key">Confidence:</span> <span class="prop-val">${{(entity.confidence || 0).toFixed(2)}}</span></div>
        </div>
    `;

    sidebar.classList.add('open');

    // Highlight node
    nodeElements.forEach(ne => {{
        const isSelected = ne.node.id === entity.id;
        const isConnected = outRels.some(r => r.to_id === ne.node.id) ||
                           inRels.some(r => r.from_id === ne.node.id);
        ne.circle.setAttribute('stroke-width', isSelected ? 4 : (isConnected ? 3 : 2));
        ne.circle.setAttribute('opacity', (isSelected || isConnected) ? 1 : 0.4);
        ne.label.setAttribute('opacity', (isSelected || isConnected) ? 1 : 0.3);
    }});

    edgeElements.forEach(ee => {{
        const connected = ee.edge.source.id === entity.id || ee.edge.target.id === entity.id;
        ee.line.style.stroke = connected ? '#00f0ff88' : '#222';
        ee.line.style.strokeWidth = connected ? 2.5 : 1;
        ee.label.style.fill = connected ? '#00f0ff' : '#333';
    }});
}}

function closeSidebar() {{
    document.getElementById('sidebar').classList.remove('open');
    // Reset highlighting
    nodeElements.forEach(ne => {{
        ne.circle.setAttribute('stroke-width', 2);
        ne.circle.setAttribute('opacity', 1);
        ne.label.setAttribute('opacity', 1);
    }});
    edgeElements.forEach(ee => {{
        ee.line.style.stroke = '#333';
        ee.line.style.strokeWidth = 1.5;
        ee.label.style.fill = '#555';
    }});
}}

function refreshData() {{
    window.location.reload();
}}

// ── Legend ─────────────────────────────────────────────────────────
const legendDiv = document.getElementById('legend');
const usedTypes = new Set(entities.map(e => e.entity_type));
let legendHtml = '';
usedTypes.forEach(t => {{
    const color = COLORS[t] || '#666';
    legendHtml += `<div class="legend-item">
        <span class="legend-dot" style="background:${{color}}"></span>
        ${{t}}
    </div>`;
}});
legendDiv.innerHTML = legendHtml;
</script>
</body>
</html>"""


# ── HTTP Server ───────────────────────────────────────────────────────


class GraphDemoHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the graph demo."""

    graph: TritiumGraph  # Set by the server setup

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        if path == "" or path == "/":
            self._serve_visualization()
        elif path == "/api/entities":
            self._json_response(get_all_entities(self.graph))
        elif path == "/api/relationships":
            self._json_response(get_all_relationships(self.graph))
        elif path.startswith("/api/traverse/"):
            entity_id = path[len("/api/traverse/"):]
            hops = int(params.get("hops", ["2"])[0])
            result = self.graph.traverse(entity_id, max_hops=hops)
            self._json_response(result)
        elif path == "/api/search":
            q = params.get("q", [""])[0]
            results = self.graph.search(q) if q else []
            self._json_response(results)
        elif path == "/api/ontology":
            self._json_response(get_ontology_summary())
        else:
            self._error_response(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._error_response(400, "Invalid JSON")
            return

        if path == "/api/entity":
            self._handle_create_entity(data)
        elif path == "/api/relationship":
            self._handle_create_relationship(data)
        else:
            self._error_response(404, "Not found")

    def _handle_create_entity(self, data: dict[str, Any]) -> None:
        entity_type = data.get("entity_type", "")
        entity_id = data.get("id", "")
        name = data.get("name", "")
        properties = data.get("properties", {})
        confidence = float(data.get("confidence", 1.0))

        if not entity_type or not entity_id:
            self._error_response(400, "entity_type and id are required")
            return

        try:
            self.graph.create_entity(entity_type, entity_id, name, properties, confidence)
            self._json_response({"status": "created", "id": entity_id}, status=201)
        except ValueError as e:
            self._error_response(400, str(e))

    def _handle_create_relationship(self, data: dict[str, Any]) -> None:
        from_id = data.get("from_id", "")
        to_id = data.get("to_id", "")
        rel_type = data.get("rel_type", "")
        properties = data.get("properties", {})

        if not from_id or not to_id or not rel_type:
            self._error_response(400, "from_id, to_id, and rel_type are required")
            return

        try:
            self.graph.add_relationship(from_id, to_id, rel_type, properties)
            self._json_response({"status": "created"}, status=201)
        except ValueError as e:
            self._error_response(400, str(e))

    def _serve_visualization(self) -> None:
        entities = get_all_entities(self.graph)
        rels = get_all_relationships(self.graph)
        page = generate_visualization_html(entities, rels)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode("utf-8"))

    def _json_response(
        self, data: Any, status: int = 200
    ) -> None:
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error_response(self, status: int, message: str) -> None:
        self._json_response({"error": message}, status=status)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging — print cleaner output."""
        pass


def run_demo(port: int = 8099) -> None:
    """Populate a graph and start the demo HTTP server.

    Args:
        port: HTTP port to listen on (default 8099).
    """
    tmpdir = tempfile.mkdtemp(prefix="tritium_graph_demo_")
    db_path = os.path.join(tmpdir, "demo.db")

    print(f"[GRAPH DEMO] Initializing KuzuDB at {db_path}")
    graph = TritiumGraph(db_path)

    print("[GRAPH DEMO] Populating demo data...")
    populate_demo_graph(graph)

    entities = get_all_entities(graph)
    rels = get_all_relationships(graph)
    print(f"[GRAPH DEMO] Loaded {len(entities)} entities, {len(rels)} relationships")

    # Print demo queries
    print("\n[GRAPH DEMO] === Demo Queries ===")

    devices = find_devices_carried_by(graph, "person-carol")
    print(f"\n  Devices carried by Carol: {[d['name'] for d in devices]}")

    travelers = find_traveled_together(graph)
    for t in travelers:
        print(f"  Traveled together: {t['entity_a']['name']} <-> {t['entity_b']['name']} "
              f"(count={t['count']}, conf={t['confidence']})")

    subgraph = graph.traverse("person-carol", max_hops=2)
    print(f"\n  Carol's 2-hop subgraph: {len(subgraph['nodes'])} nodes, "
          f"{len(subgraph['edges'])} edges")
    for node in subgraph["nodes"]:
        print(f"    [{node['entity_type']}] {node['name']} ({node['id']})")

    results = graph.search("Alice")
    print(f"\n  Search 'Alice': {[r['name'] for r in results]}")

    # Wire up handler
    GraphDemoHandler.graph = graph

    server = HTTPServer(("0.0.0.0", port), GraphDemoHandler)
    print(f"\n[GRAPH DEMO] Server running at http://localhost:{port}")
    print("[GRAPH DEMO] Endpoints:")
    print(f"  GET  http://localhost:{port}/                  - Interactive visualization")
    print(f"  GET  http://localhost:{port}/api/entities      - All entities")
    print(f"  GET  http://localhost:{port}/api/relationships - All relationships")
    print(f"  GET  http://localhost:{port}/api/traverse/ID   - Traverse from entity")
    print(f"  GET  http://localhost:{port}/api/search?q=text - Search entities")
    print(f"  POST http://localhost:{port}/api/entity        - Create entity")
    print(f"  POST http://localhost:{port}/api/relationship  - Create relationship")
    print(f"  GET  http://localhost:{port}/api/ontology      - Ontology schema")
    print("\nPress Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[GRAPH DEMO] Shutting down...")
    finally:
        server.server_close()
        graph.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    run_demo()
