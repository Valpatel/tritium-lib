# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Mission generator — creates tactical missions from terrain features.

Analyzes terrain segmentation results to generate contextual missions:
- Bridge defense when bridges are detected
- Building clearing from building footprints
- Water crossing planning from water obstacles
- Patrol routes from road/sidewalk networks
- Overwatch positions from large buildings with clear fields of fire
- Staging areas from parking lots

Missions are tactical objectives with positions, descriptions, and
required unit types. They integrate with Amy's goal system.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from tritium_lib.models.terrain import TerrainType

logger = logging.getLogger(__name__)


@dataclass
class Mission:
    """A tactical mission generated from terrain analysis."""
    id: str
    mission_type: str  # patrol, defend, clear, recon, overwatch, staging
    name: str
    description: str
    position: tuple[float, float]  # (lon, lat)
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    priority: int = 3  # 1-5, 5 = highest
    required_units: list[str] = field(default_factory=list)  # unit types
    terrain_features: list[str] = field(default_factory=list)
    estimated_duration_s: float = 300.0


# Mission templates keyed by terrain composition
_MISSION_TEMPLATES = {
    "bridge_defense": {
        "mission_type": "defend",
        "name_fmt": "Defend {name}",
        "desc_fmt": "Secure bridge crossing at ({lon:.4f}, {lat:.4f}). {context}",
        "priority": 4,
        "required_units": ["infantry", "turret"],
        "duration": 600,
    },
    "building_clear": {
        "mission_type": "clear",
        "name_fmt": "Clear {name}",
        "desc_fmt": "Clear and secure building at ({lon:.4f}, {lat:.4f}). {context}",
        "priority": 3,
        "required_units": ["infantry"],
        "duration": 300,
    },
    "overwatch": {
        "mission_type": "overwatch",
        "name_fmt": "Overwatch from {name}",
        "desc_fmt": "Establish overwatch position at ({lon:.4f}, {lat:.4f}). {context}",
        "priority": 3,
        "required_units": ["sniper", "drone"],
        "duration": 900,
    },
    "water_recon": {
        "mission_type": "recon",
        "name_fmt": "Recon {name}",
        "desc_fmt": "Reconnoiter water obstacle at ({lon:.4f}, {lat:.4f}). {context}",
        "priority": 2,
        "required_units": ["drone", "scout_drone"],
        "duration": 300,
    },
    "patrol_route": {
        "mission_type": "patrol",
        "name_fmt": "Patrol {name}",
        "desc_fmt": "Patrol route through {name}. {context}",
        "priority": 2,
        "required_units": ["rover", "infantry"],
        "duration": 600,
    },
    "staging_area": {
        "mission_type": "staging",
        "name_fmt": "Stage at {name}",
        "desc_fmt": "Establish staging area at parking lot ({lon:.4f}, {lat:.4f}). {context}",
        "priority": 2,
        "required_units": ["vehicle", "apc"],
        "duration": 300,
    },
}


class MissionGenerator:
    """Generates tactical missions from terrain segmentation data.

    Analyzes terrain features to identify:
    - Chokepoints (bridges, narrow roads between buildings)
    - Objectives (large buildings, infrastructure)
    - Obstacles (water, dense vegetation)
    - Patrol routes (roads, sidewalks)
    - Overwatch positions (tall/large buildings with open sightlines)
    """

    def __init__(self) -> None:
        self._next_id = 0

    def generate_missions(self, terrain_layer: Any) -> list[Mission]:
        """Generate all applicable missions from a terrain layer.

        Returns missions sorted by priority (highest first).
        """
        if not hasattr(terrain_layer, 'regions'):
            return []

        missions = []
        missions.extend(self._bridge_defense_missions(terrain_layer))
        missions.extend(self._building_missions(terrain_layer))
        missions.extend(self._water_recon_missions(terrain_layer))
        missions.extend(self._patrol_missions(terrain_layer))
        missions.extend(self._staging_missions(terrain_layer))

        # Sort by priority descending
        missions.sort(key=lambda m: m.priority, reverse=True)

        logger.info("Generated %d missions from terrain features", len(missions))
        return missions

    def _next_mission_id(self) -> str:
        self._next_id += 1
        return f"mission_{self._next_id}"

    def _bridge_defense_missions(self, terrain_layer: Any) -> list[Mission]:
        """Generate bridge defense missions."""
        bridges = terrain_layer.features_by_type(TerrainType.BRIDGE)
        water = terrain_layer.features_by_type(TerrainType.WATER)

        missions = []
        for bridge in bridges:
            # Check if bridge is near water (it should be)
            near_water = any(
                _distance_deg(bridge.centroid_lat, bridge.centroid_lon,
                              w.centroid_lat, w.centroid_lon) < 0.003
                for w in water
            )

            context = "Single crossing point — chokepoint." if near_water else "Bridge crossing."
            tmpl = _MISSION_TEMPLATES["bridge_defense"]

            missions.append(Mission(
                id=self._next_mission_id(),
                mission_type=tmpl["mission_type"],
                name=tmpl["name_fmt"].format(name=f"Bridge {len(missions)+1}"),
                description=tmpl["desc_fmt"].format(
                    lon=bridge.centroid_lon, lat=bridge.centroid_lat, context=context,
                ),
                position=(bridge.centroid_lon, bridge.centroid_lat),
                priority=tmpl["priority"],
                required_units=list(tmpl["required_units"]),
                terrain_features=["bridge"],
                estimated_duration_s=tmpl["duration"],
            ))

        return missions

    def _building_missions(self, terrain_layer: Any) -> list[Mission]:
        """Generate building clearing and overwatch missions."""
        buildings = terrain_layer.features_by_type(TerrainType.BUILDING)
        missions = []

        # Overwatch positions: large buildings OR named buildings from OSM
        overwatch_candidates = [
            b for b in buildings
            if b.area_m2 > 2000 or b.properties.get("osm_name")
        ]
        # Prioritize named buildings, then sort by area
        overwatch_candidates.sort(
            key=lambda b: (0 if b.properties.get("osm_name") else 1, -b.area_m2)
        )
        for i, bldg in enumerate(overwatch_candidates[:5]):
            name = bldg.properties.get("osm_name", f"Structure {i+1}")
            tmpl = _MISSION_TEMPLATES["overwatch"]

            # Check for open sightlines (no adjacent large buildings blocking view)
            adjacent_buildings = sum(
                1 for b in buildings
                if b != bldg and _distance_deg(
                    bldg.centroid_lat, bldg.centroid_lon,
                    b.centroid_lat, b.centroid_lon,
                ) < 0.001
            )
            context = (
                f"{bldg.area_m2:.0f} m² structure with clear sightlines."
                if adjacent_buildings < 3
                else f"{bldg.area_m2:.0f} m² structure in dense urban area."
            )

            missions.append(Mission(
                id=self._next_mission_id(),
                mission_type=tmpl["mission_type"],
                name=tmpl["name_fmt"].format(name=name),
                description=tmpl["desc_fmt"].format(
                    lon=bldg.centroid_lon, lat=bldg.centroid_lat, context=context,
                ),
                position=(bldg.centroid_lon, bldg.centroid_lat),
                priority=tmpl["priority"],
                required_units=list(tmpl["required_units"]),
                terrain_features=["building"],
                estimated_duration_s=tmpl["duration"],
            ))

        # Clusters of small buildings → clearing missions
        small_buildings = [b for b in buildings if 100 < b.area_m2 <= 2000]
        if len(small_buildings) >= 3:
            # Find centroid of building cluster
            avg_lat = sum(b.centroid_lat for b in small_buildings) / len(small_buildings)
            avg_lon = sum(b.centroid_lon for b in small_buildings) / len(small_buildings)
            tmpl = _MISSION_TEMPLATES["building_clear"]

            missions.append(Mission(
                id=self._next_mission_id(),
                mission_type=tmpl["mission_type"],
                name=tmpl["name_fmt"].format(name=f"Urban Block ({len(small_buildings)} buildings)"),
                description=tmpl["desc_fmt"].format(
                    lon=avg_lon, lat=avg_lat,
                    context=f"{len(small_buildings)} buildings in cluster.",
                ),
                position=(avg_lon, avg_lat),
                priority=tmpl["priority"],
                required_units=list(tmpl["required_units"]),
                terrain_features=["building"],
                estimated_duration_s=tmpl["duration"] * min(len(small_buildings), 10),
            ))

        return missions

    def _water_recon_missions(self, terrain_layer: Any) -> list[Mission]:
        """Generate water obstacle reconnaissance missions."""
        water = terrain_layer.features_by_type(TerrainType.WATER)
        missions = []

        # Large water bodies need recon for crossing options
        large_water = [w for w in water if w.area_m2 > 5000]
        for i, body in enumerate(large_water[:3]):
            tmpl = _MISSION_TEMPLATES["water_recon"]
            missions.append(Mission(
                id=self._next_mission_id(),
                mission_type=tmpl["mission_type"],
                name=tmpl["name_fmt"].format(name=f"Water Body {i+1}"),
                description=tmpl["desc_fmt"].format(
                    lon=body.centroid_lon, lat=body.centroid_lat,
                    context=f"{body.area_m2:.0f} m² water obstacle. Identify crossing points.",
                ),
                position=(body.centroid_lon, body.centroid_lat),
                priority=tmpl["priority"],
                required_units=list(tmpl["required_units"]),
                terrain_features=["water"],
                estimated_duration_s=tmpl["duration"],
            ))

        return missions

    def _patrol_missions(self, terrain_layer: Any) -> list[Mission]:
        """Generate patrol route missions from road/sidewalk network."""
        roads = terrain_layer.features_by_type(TerrainType.ROAD)
        sidewalks = terrain_layer.features_by_type(TerrainType.SIDEWALK)

        if len(roads) < 3 and len(sidewalks) < 3:
            return []

        # Build a patrol route through road centroids
        route_features = roads + sidewalks
        if not route_features:
            return []

        # Select evenly-spaced waypoints
        step = max(1, len(route_features) // 6)
        waypoints = [
            (f.centroid_lon, f.centroid_lat)
            for f in route_features[::step]
        ][:8]  # max 8 waypoints

        if len(waypoints) < 2:
            return []

        tmpl = _MISSION_TEMPLATES["patrol_route"]
        center = waypoints[len(waypoints) // 2]

        return [Mission(
            id=self._next_mission_id(),
            mission_type=tmpl["mission_type"],
            name=tmpl["name_fmt"].format(name="Road Network"),
            description=tmpl["desc_fmt"].format(
                name="the road network",
                context=f"{len(waypoints)} waypoints covering {len(roads)} road segments.",
            ),
            position=center,
            waypoints=waypoints,
            priority=tmpl["priority"],
            required_units=list(tmpl["required_units"]),
            terrain_features=["road", "sidewalk"],
            estimated_duration_s=tmpl["duration"],
        )]

    def _staging_missions(self, terrain_layer: Any) -> list[Mission]:
        """Generate staging area missions from parking lots."""
        parking = terrain_layer.features_by_type(TerrainType.PARKING)
        missions = []

        # Large parking lots are good staging areas
        large_parking = [p for p in parking if p.area_m2 > 1000]
        for i, lot in enumerate(large_parking[:2]):
            tmpl = _MISSION_TEMPLATES["staging_area"]
            missions.append(Mission(
                id=self._next_mission_id(),
                mission_type=tmpl["mission_type"],
                name=tmpl["name_fmt"].format(name=f"Parking Lot {i+1}"),
                description=tmpl["desc_fmt"].format(
                    lon=lot.centroid_lon, lat=lot.centroid_lat,
                    context=f"{lot.area_m2:.0f} m² open area suitable for vehicle staging.",
                ),
                position=(lot.centroid_lon, lot.centroid_lat),
                priority=tmpl["priority"],
                required_units=list(tmpl["required_units"]),
                terrain_features=["parking"],
                estimated_duration_s=tmpl["duration"],
            ))

        return missions

    def missions_brief(self, missions: list[Mission]) -> str:
        """Generate a text brief of available missions for commander AI."""
        if not missions:
            return "No missions available — process terrain data first."

        lines = [f"AVAILABLE MISSIONS ({len(missions)} total)"]
        lines.append("")

        by_type = {}
        for m in missions:
            by_type.setdefault(m.mission_type, []).append(m)

        for mtype, mlist in by_type.items():
            lines.append(f"  {mtype.upper()} ({len(mlist)}):")
            for m in mlist[:3]:  # show first 3 per type
                lines.append(f"    [{m.priority}] {m.name} — {m.description[:80]}")

        return "\n".join(lines)


def _distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Simple Euclidean distance in degrees (for nearby points)."""
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
