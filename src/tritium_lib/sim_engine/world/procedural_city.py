# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Procedural city generator for demo/offline mode.

Generates a simple grid-based city layout with buildings, roads, trees,
and parks.  No OSM data or internet connection needed.  The output format
matches the /api/geo/city-data schema (schema_version=2) so the frontend
renderers can consume it directly.

Migrated from tritium-sc/plugins/city_sim/routes.py::generate_demo_city()
during sim engine unification (Wave 196).
"""

from __future__ import annotations

import math
import random
from typing import Any


def generate_demo_city(
    radius: float = 300.0,
    block_size: float = 60.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a procedural city for demo/offline mode.

    Returns city-data format JSON with buildings, roads, trees, and parks.
    No OSM data or internet connection needed.

    Args:
        radius: City radius in meters.
        block_size: Building block size in meters.
        seed: Random seed for reproducibility.

    Returns:
        Dict matching the /api/geo/city-data schema (schema_version=2).
    """
    rng = random.Random(seed)
    buildings: list[dict[str, Any]] = []
    roads: list[dict[str, Any]] = []
    trees: list[dict[str, Any]] = []
    landuse_list: list[dict[str, Any]] = []

    grid_spacing = block_size + 12  # road width = 12m
    half = radius * 0.8
    cols = int(half * 2 / grid_spacing)
    rows = int(half * 2 / grid_spacing)
    start_x = -half

    road_id = 1
    bldg_id = 1

    # Horizontal roads
    for r in range(rows + 1):
        z = start_x + r * grid_spacing
        is_main = r == rows // 2
        roads.append({
            "id": road_id, "points": [[-half, z], [half, z]],
            "class": "primary" if is_main else "residential",
            "name": f"{r + 1}th St", "width": 14.0 if is_main else 8.0,
            "lanes": 4 if is_main else 2, "surface": "asphalt",
            "oneway": False, "bridge": False, "tunnel": False, "maxspeed": "",
        })
        road_id += 1

    # Vertical roads
    for c in range(cols + 1):
        x = start_x + c * grid_spacing
        is_main = c == cols // 2
        roads.append({
            "id": road_id, "points": [[x, -half], [x, half]],
            "class": "secondary" if is_main else "residential",
            "name": f"{chr(65 + c % 26)} Ave", "width": 10.0 if is_main else 8.0,
            "lanes": 3 if is_main else 2, "surface": "asphalt",
            "oneway": False, "bridge": False, "tunnel": False, "maxspeed": "",
        })
        road_id += 1

    # Buildings and parks per block
    zone_types = ["residential", "commercial", "industrial"]
    for r in range(rows):
        for c in range(cols):
            bx = start_x + c * grid_spacing + 6
            bz = start_x + r * grid_spacing + 6
            bw = block_size
            bh = block_size

            dist = math.sqrt((bx + bw / 2) ** 2 + (bz + bh / 2) ** 2)
            zone = "commercial" if dist < radius * 0.2 else rng.choice(zone_types)

            # 15% chance of park
            if rng.random() < 0.15:
                landuse_list.append({
                    "id": bldg_id, "type": "park", "name": f"Park {bldg_id}",
                    "polygon": [[bx + 2, bz + 2], [bx + bw - 2, bz + 2],
                                [bx + bw - 2, bz + bh - 2], [bx + 2, bz + bh - 2]],
                })
                bldg_id += 1
                for _ in range(rng.randint(3, 7)):
                    trees.append({
                        "pos": [bx + 4 + rng.random() * (bw - 8), bz + 4 + rng.random() * (bh - 8)],
                        "species": rng.choice(["oak", "maple", "birch"]),
                        "height": 5 + rng.random() * 7, "leaf_type": "broadleaved",
                    })
                continue

            num_bldgs = 1 if zone == "industrial" else rng.randint(1, 3)
            for _ in range(num_bldgs):
                w = rng.uniform(10, min(35, bw * 0.7))
                d = rng.uniform(10, min(25, bh * 0.7))
                ox = (rng.random() - 0.5) * (bw - w - 4)
                oz = (rng.random() - 0.5) * (bh - d - 4)
                cx, cz = bx + bw / 2 + ox, bz + bh / 2 + oz

                h = {"commercial": 12 + rng.random() * 35,
                     "industrial": 6 + rng.random() * 8,
                     "residential": 5 + rng.random() * 15}[zone]

                cat = {"commercial": "commercial", "industrial": "industrial",
                       "residential": "residential"}[zone]

                buildings.append({
                    "id": bldg_id,
                    "polygon": [[cx - w / 2, cz - d / 2], [cx + w / 2, cz - d / 2],
                                [cx + w / 2, cz + d / 2], [cx - w / 2, cz + d / 2]],
                    "height": round(h, 1), "type": zone,
                    "category": cat, "name": "", "levels": max(1, int(h / 3)),
                    "roof_shape": "gabled" if cat == "residential" and h < 12 else "flat",
                    "colour": "", "material": "",
                    "address": str(rng.randint(100, 999)),
                    "street": f"{chr(65 + c % 26)} Ave",
                })
                bldg_id += 1

    return {
        "center": {"lat": 0, "lng": 0},
        "radius": radius, "schema_version": 2,
        "buildings": buildings, "roads": roads, "trees": trees,
        "landuse": landuse_list, "barriers": [], "water": [],
        "entrances": [], "pois": [],
        "stats": {
            "buildings": len(buildings), "roads": len(roads),
            "trees": len(trees), "landuse": len(landuse_list),
            "barriers": 0, "water": 0, "entrances": 0, "pois": 0,
        },
        "_procedural": True,
    }
