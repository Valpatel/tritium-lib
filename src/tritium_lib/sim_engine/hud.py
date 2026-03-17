# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Minimap and HUD data generation for the Three.js sim frontend.

Produces JSON-serializable overlay data for: minimap with fog of war,
compass with threat/objective indicators, unit roster, resource bars,
toast notifications, kill feed, and score display.  :class:`HUDEngine`
composes all sub-renderers into a single ``render_frame`` call that
returns the complete HUD state for one animation frame.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALLIANCE_COLORS: dict[str, str] = {
    "friendly": "#05ffa1",
    "hostile": "#ff2a6d",
    "neutral": "#fcee0a",
    "unknown": "#00f0ff",
}

_CARDINALS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NotificationPriority(Enum):
    """Urgency levels for HUD notifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarkerShape(Enum):
    """Minimap marker geometry."""

    DOT = "dot"
    SQUARE = "square"
    RECT = "rect"
    DIAMOND = "diamond"
    TRIANGLE = "triangle"


# ---------------------------------------------------------------------------
# MinimapRenderer
# ---------------------------------------------------------------------------


@dataclass
class MinimapMarker:
    """A single marker on the minimap."""

    x: float
    y: float
    color: str
    shape: str = "dot"
    size: float = 4.0
    label: str = ""
    entity_id: str = ""


class MinimapRenderer:
    """Generate pixel-space minimap data from world-space entities.

    Parameters
    ----------
    map_width:  World width in meters.
    map_height: World height in meters.
    minimap_size: Output pixel size (square).
    """

    def __init__(
        self,
        map_width: float,
        map_height: float,
        minimap_size: int = 200,
    ) -> None:
        if map_width <= 0 or map_height <= 0:
            raise ValueError("map dimensions must be positive")
        if minimap_size <= 0:
            raise ValueError("minimap_size must be positive")
        self.map_width = map_width
        self.map_height = map_height
        self.minimap_size = minimap_size
        self._scale_x = minimap_size / map_width
        self._scale_y = minimap_size / map_height

    # -- coordinate helpers --------------------------------------------------

    def world_to_minimap(self, pos: Vec2) -> tuple[float, float]:
        """Convert world (x, y) meters to minimap pixel coordinates."""
        px = pos[0] * self._scale_x
        py = pos[1] * self._scale_y
        return (
            max(0.0, min(px, float(self.minimap_size))),
            max(0.0, min(py, float(self.minimap_size))),
        )

    # -- rendering -----------------------------------------------------------

    def render(
        self,
        units: list[dict[str, Any]],
        vehicles: list[dict[str, Any]] | None = None,
        structures: list[dict[str, Any]] | None = None,
        fog_of_war: set[tuple[int, int]] | None = None,
        alliance: str = "friendly",
    ) -> dict[str, Any]:
        """Build minimap overlay data.

        Each input dict must contain at minimum:
        - ``position``: ``(x, y)`` world-space tuple
        - ``alliance``: one of friendly / hostile / neutral / unknown

        Units are drawn as dots, vehicles as squares, structures as rects.
        Tiles listed in *fog_of_war* are marked as hidden.

        Returns a JSON-serializable dict ready for the Three.js HUD layer.
        """
        markers: list[dict[str, Any]] = []
        vehicles = vehicles or []
        structures = structures or []

        # Units -> dots
        for u in units:
            pos = u.get("position", (0.0, 0.0))
            ally = u.get("alliance", "unknown")
            alive = u.get("is_alive", True)
            if not alive:
                continue
            mx, my = self.world_to_minimap(pos)
            markers.append(
                {
                    "x": mx,
                    "y": my,
                    "color": _ALLIANCE_COLORS.get(ally, _ALLIANCE_COLORS["unknown"]),
                    "shape": "dot",
                    "size": 4.0,
                    "entity_id": u.get("unit_id", ""),
                }
            )

        # Vehicles -> squares
        for v in vehicles:
            pos = v.get("position", (0.0, 0.0))
            ally = v.get("alliance", "unknown")
            destroyed = v.get("is_destroyed", False)
            if destroyed:
                continue
            mx, my = self.world_to_minimap(pos)
            markers.append(
                {
                    "x": mx,
                    "y": my,
                    "color": _ALLIANCE_COLORS.get(ally, _ALLIANCE_COLORS["unknown"]),
                    "shape": "square",
                    "size": 6.0,
                    "entity_id": v.get("vehicle_id", ""),
                }
            )

        # Structures -> rectangles
        for s in structures:
            pos = s.get("position", (0.0, 0.0))
            ally = s.get("alliance", "neutral")
            mx, my = self.world_to_minimap(pos)
            markers.append(
                {
                    "x": mx,
                    "y": my,
                    "color": _ALLIANCE_COLORS.get(ally, _ALLIANCE_COLORS["unknown"]),
                    "shape": "rect",
                    "size": 8.0,
                    "entity_id": s.get("building_id", ""),
                }
            )

        # Fog of war — list of hidden grid cells
        fog_cells: list[dict[str, int]] = []
        if fog_of_war:
            for gx, gy in fog_of_war:
                fog_cells.append({"gx": gx, "gy": gy})

        return {
            "minimap_size": self.minimap_size,
            "map_width": self.map_width,
            "map_height": self.map_height,
            "markers": markers,
            "fog_cells": fog_cells,
            "marker_count": len(markers),
        }

    def to_three_js(
        self,
        camera_pos: Vec2,
        camera_fov: float,
        markers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Produce a Three.js-oriented minimap dict with camera viewport.

        Parameters
        ----------
        camera_pos: World-space camera position.
        camera_fov: Camera field of view in degrees.
        markers: Optional pre-rendered marker list (from :meth:`render`).
        """
        cx, cy = self.world_to_minimap(camera_pos)
        # Approximate viewport size on minimap from FOV
        fov_rad = math.radians(camera_fov)
        viewport_half = math.tan(fov_rad / 2.0) * 20.0  # rough scale factor
        vp_px = min(viewport_half * self._scale_x, self.minimap_size / 2.0)

        return {
            "minimap_size": self.minimap_size,
            "camera": {"x": cx, "y": cy},
            "viewport": {
                "x": max(0.0, cx - vp_px),
                "y": max(0.0, cy - vp_px),
                "width": vp_px * 2.0,
                "height": vp_px * 2.0,
            },
            "markers": markers or [],
        }


# ---------------------------------------------------------------------------
# CompassHUD
# ---------------------------------------------------------------------------


class CompassHUD:
    """Compass ring with bearing-based threat and objective indicators."""

    @staticmethod
    def get_bearing(from_pos: Vec2, to_pos: Vec2) -> float:
        """Bearing in degrees (0 = north, clockwise) from *from_pos* to *to_pos*."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        # atan2 gives angle from +x axis CCW; convert to compass (N=0, CW)
        angle_rad = math.atan2(dx, dy)  # note: (dx, dy) not (dy, dx) for compass
        degrees = math.degrees(angle_rad) % 360.0
        return degrees

    @staticmethod
    def get_cardinal(degrees: float) -> str:
        """Map a bearing in degrees to an 8-point cardinal direction."""
        degrees = degrees % 360.0
        idx = int((degrees + 22.5) / 45.0) % 8
        return _CARDINALS[idx]

    def render(
        self,
        player_heading: float,
        threats: list[dict[str, Any]] | None = None,
        objectives: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build compass HUD data.

        Parameters
        ----------
        player_heading: Heading in degrees (0 = north, clockwise).
        threats: List of dicts with ``position`` and optional ``label`` keys.
        objectives: List of dicts with ``position`` and optional ``label`` keys.

        Returns a dict with compass ring, threat dots, and objective diamonds.
        """
        threats = threats or []
        objectives = objectives or []

        # Normalize heading
        heading = player_heading % 360.0
        cardinal = self.get_cardinal(heading)

        # Threat indicators — bearing relative to player heading
        threat_indicators: list[dict[str, Any]] = []
        for t in threats:
            pos = t.get("position", (0.0, 0.0))
            player_pos = t.get("player_pos", (0.0, 0.0))
            bearing = self.get_bearing(player_pos, pos)
            relative = (bearing - heading) % 360.0
            threat_indicators.append(
                {
                    "bearing": bearing,
                    "relative_bearing": relative,
                    "label": t.get("label", ""),
                    "color": "#ff2a6d",
                    "shape": "dot",
                }
            )

        # Objective indicators
        objective_indicators: list[dict[str, Any]] = []
        for o in objectives:
            pos = o.get("position", (0.0, 0.0))
            player_pos = o.get("player_pos", (0.0, 0.0))
            bearing = self.get_bearing(player_pos, pos)
            relative = (bearing - heading) % 360.0
            objective_indicators.append(
                {
                    "bearing": bearing,
                    "relative_bearing": relative,
                    "label": o.get("label", ""),
                    "color": "#fcee0a",
                    "shape": "diamond",
                }
            )

        return {
            "heading": heading,
            "cardinal": cardinal,
            "threats": threat_indicators,
            "objectives": objective_indicators,
            "threat_count": len(threat_indicators),
            "objective_count": len(objective_indicators),
        }


# ---------------------------------------------------------------------------
# UnitRoster
# ---------------------------------------------------------------------------


class UnitRoster:
    """Friendly unit list with health, ammo, status, and weapon info."""

    def render(
        self,
        units: list[dict[str, Any]],
        alliance: str = "friendly",
        sort_by: str = "health",
    ) -> dict[str, Any]:
        """Build the unit roster HUD panel.

        Parameters
        ----------
        units: List of unit dicts (must have alliance, name, health,
               max_health, ammo, status, weapon).
        alliance: Which alliance to show in the roster.
        sort_by: Sort key — ``health``, ``distance``, or ``status``.

        Returns a dict with the ordered roster entries.
        """
        entries: list[dict[str, Any]] = []

        for u in units:
            if u.get("alliance", "unknown") != alliance:
                continue
            if not u.get("is_alive", True):
                continue

            hp = u.get("health", 0.0)
            max_hp = u.get("max_health", 100.0)
            hp_pct = (hp / max_hp * 100.0) if max_hp > 0 else 0.0

            # Health bar color: green > 60%, yellow > 30%, red <= 30%
            if hp_pct > 60.0:
                bar_color = "#05ffa1"
            elif hp_pct > 30.0:
                bar_color = "#fcee0a"
            else:
                bar_color = "#ff2a6d"

            ammo = u.get("ammo", -1)
            ammo_display = "INF" if ammo < 0 else str(ammo)

            entries.append(
                {
                    "unit_id": u.get("unit_id", ""),
                    "name": u.get("name", "Unknown"),
                    "health": hp,
                    "max_health": max_hp,
                    "health_pct": round(hp_pct, 1),
                    "health_bar_color": bar_color,
                    "ammo": ammo,
                    "ammo_display": ammo_display,
                    "status": u.get("status", "idle"),
                    "weapon": u.get("weapon", ""),
                    "distance": u.get("distance", 0.0),
                }
            )

        # Sort
        if sort_by == "health":
            entries.sort(key=lambda e: e["health_pct"])
        elif sort_by == "distance":
            entries.sort(key=lambda e: e["distance"])
        elif sort_by == "status":
            _status_order = {
                "attacking": 0,
                "moving": 1,
                "idle": 2,
                "retreating": 3,
                "suppressed": 4,
            }
            entries.sort(key=lambda e: _status_order.get(e["status"], 99))

        return {
            "alliance": alliance,
            "entries": entries,
            "count": len(entries),
            "sort_by": sort_by,
        }


# ---------------------------------------------------------------------------
# ResourceHUD
# ---------------------------------------------------------------------------


_RESOURCE_ICONS: dict[str, str] = {
    "ammo": "bullet",
    "fuel": "fuel_pump",
    "medical": "medkit",
    "food": "ration",
    "credits": "coin",
    "manpower": "person",
    "steel": "gear",
    "electronics": "chip",
    "water": "drop",
    "parts": "wrench",
}

# Warning thresholds as fraction of capacity
_LOW_THRESHOLD = 0.20
_CRITICAL_THRESHOLD = 0.10


class ResourceHUD:
    """Resource bars and income/expense indicators."""

    def render(
        self,
        economy_state: dict[str, float] | None = None,
        supply_state: dict[str, float] | None = None,
        capacity: dict[str, float] | None = None,
        income: dict[str, float] | None = None,
        expenses: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Build the resource HUD panel.

        Parameters
        ----------
        economy_state: Dict of resource_name -> current amount.
        supply_state:  Dict of supply_name -> current amount (merged into bars).
        capacity:      Dict of resource_name -> max capacity.
        income:        Dict of resource_name -> per-tick income.
        expenses:      Dict of resource_name -> per-tick expense.
        """
        economy_state = economy_state or {}
        supply_state = supply_state or {}
        capacity = capacity or {}
        income = income or {}
        expenses = expenses or {}

        # Merge economy + supply into one display
        all_resources: dict[str, float] = {}
        all_resources.update(economy_state)
        all_resources.update(supply_state)

        bars: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        for name, amount in sorted(all_resources.items()):
            cap = capacity.get(name, 0.0)
            pct = (amount / cap * 100.0) if cap > 0 else 100.0
            frac = (amount / cap) if cap > 0 else 1.0

            inc = income.get(name, 0.0)
            exp = expenses.get(name, 0.0)
            net = inc - exp

            # Bar color
            if frac <= _CRITICAL_THRESHOLD:
                bar_color = "#ff2a6d"
            elif frac <= _LOW_THRESHOLD:
                bar_color = "#fcee0a"
            else:
                bar_color = "#05ffa1"

            # Net flow indicator
            if net > 0:
                flow_icon = "arrow_up"
                flow_color = "#05ffa1"
            elif net < 0:
                flow_icon = "arrow_down"
                flow_color = "#ff2a6d"
            else:
                flow_icon = "dash"
                flow_color = "#888888"

            bar = {
                "name": name,
                "amount": round(amount, 1),
                "capacity": round(cap, 1),
                "pct": round(pct, 1),
                "bar_color": bar_color,
                "icon": _RESOURCE_ICONS.get(name, "box"),
                "income": round(inc, 2),
                "expense": round(exp, 2),
                "net": round(net, 2),
                "flow_icon": flow_icon,
                "flow_color": flow_color,
            }
            bars.append(bar)

            # Warnings
            if cap > 0 and frac <= _CRITICAL_THRESHOLD:
                warnings.append(
                    {
                        "resource": name,
                        "level": "critical",
                        "message": f"{name} critically low ({pct:.0f}%)",
                        "icon": "warning_critical",
                    }
                )
            elif cap > 0 and frac <= _LOW_THRESHOLD:
                warnings.append(
                    {
                        "resource": name,
                        "level": "low",
                        "message": f"{name} running low ({pct:.0f}%)",
                        "icon": "warning",
                    }
                )

        return {
            "bars": bars,
            "warnings": warnings,
            "bar_count": len(bars),
            "warning_count": len(warnings),
        }


# ---------------------------------------------------------------------------
# NotificationQueue
# ---------------------------------------------------------------------------


@dataclass
class _Notification:
    """Internal notification record."""

    notif_id: int
    text: str
    priority: NotificationPriority
    icon: str
    duration: float  # seconds before auto-dismiss
    created: float  # monotonic timestamp
    age: float = 0.0


class NotificationQueue:
    """Toast-style notification queue with priority and auto-expiry."""

    def __init__(self, max_visible: int = 5) -> None:
        self.max_visible = max_visible
        self._queue: list[_Notification] = []
        self._next_id: int = 0

    @property
    def count(self) -> int:
        """Number of active notifications."""
        return len(self._queue)

    def add(
        self,
        text: str,
        priority: str | NotificationPriority = NotificationPriority.MEDIUM,
        icon: str = "info",
        duration: float = 5.0,
    ) -> int:
        """Enqueue a notification.  Returns notification id."""
        if isinstance(priority, str):
            priority = NotificationPriority(priority)
        nid = self._next_id
        self._next_id += 1
        self._queue.append(
            _Notification(
                notif_id=nid,
                text=text,
                priority=priority,
                icon=icon,
                duration=duration,
                created=time.monotonic(),
            )
        )
        return nid

    def tick(self, dt: float) -> list[dict[str, Any]]:
        """Advance time by *dt* seconds.  Returns active notification dicts.

        Expired notifications are removed.  Output is sorted by priority
        (critical first) then by creation time (newest first), capped at
        *max_visible*.
        """
        surviving: list[_Notification] = []
        for n in self._queue:
            n.age += dt
            if n.age < n.duration:
                surviving.append(n)
        self._queue = surviving

        # Sort: critical > high > medium > low, then newest first
        _prio_rank = {
            NotificationPriority.CRITICAL: 0,
            NotificationPriority.HIGH: 1,
            NotificationPriority.MEDIUM: 2,
            NotificationPriority.LOW: 3,
        }
        surviving.sort(key=lambda n: (_prio_rank.get(n.priority, 9), -n.notif_id))

        result: list[dict[str, Any]] = []
        for n in surviving[: self.max_visible]:
            result.append(
                {
                    "id": n.notif_id,
                    "text": n.text,
                    "priority": n.priority.value,
                    "icon": n.icon,
                    "duration": n.duration,
                    "age": round(n.age, 2),
                    "remaining": round(max(n.duration - n.age, 0.0), 2),
                }
            )
        return result

    def clear(self) -> None:
        """Remove all notifications."""
        self._queue.clear()

    def dismiss(self, notif_id: int) -> bool:
        """Remove a specific notification by id.  Returns True if found."""
        for i, n in enumerate(self._queue):
            if n.notif_id == notif_id:
                self._queue.pop(i)
                return True
        return False


# ---------------------------------------------------------------------------
# KillFeed
# ---------------------------------------------------------------------------


@dataclass
class _KillEntry:
    """A single kill-feed entry."""

    entry_id: int
    killer: str
    victim: str
    weapon: str
    killer_alliance: str
    victim_alliance: str
    timestamp: float
    age: float = 0.0


class KillFeed:
    """Scrolling kill feed for combat events."""

    def __init__(self, max_entries: int = 8, entry_duration: float = 8.0) -> None:
        self.max_entries = max_entries
        self.entry_duration = entry_duration
        self._entries: list[_KillEntry] = []
        self._next_id: int = 0

    def add(
        self,
        killer: str,
        victim: str,
        weapon: str = "",
        killer_alliance: str = "unknown",
        victim_alliance: str = "unknown",
    ) -> int:
        """Record a kill.  Returns entry id."""
        eid = self._next_id
        self._next_id += 1
        self._entries.append(
            _KillEntry(
                entry_id=eid,
                killer=killer,
                victim=victim,
                weapon=weapon,
                killer_alliance=killer_alliance,
                victim_alliance=victim_alliance,
                timestamp=time.monotonic(),
            )
        )
        # Trim to max
        if len(self._entries) > self.max_entries * 2:
            self._entries = self._entries[-self.max_entries :]
        return eid

    def tick(self, dt: float) -> list[dict[str, Any]]:
        """Advance time.  Returns visible kill-feed entries."""
        surviving: list[_KillEntry] = []
        for e in self._entries:
            e.age += dt
            if e.age < self.entry_duration:
                surviving.append(e)
        self._entries = surviving

        result: list[dict[str, Any]] = []
        for e in self._entries[-self.max_entries :]:
            result.append(
                {
                    "id": e.entry_id,
                    "killer": e.killer,
                    "victim": e.victim,
                    "weapon": e.weapon,
                    "killer_color": _ALLIANCE_COLORS.get(
                        e.killer_alliance, _ALLIANCE_COLORS["unknown"]
                    ),
                    "victim_color": _ALLIANCE_COLORS.get(
                        e.victim_alliance, _ALLIANCE_COLORS["unknown"]
                    ),
                    "age": round(e.age, 2),
                }
            )
        return result


# ---------------------------------------------------------------------------
# HUDEngine
# ---------------------------------------------------------------------------


class HUDEngine:
    """Top-level HUD compositor — combines all sub-renderers into one frame.

    Usage::

        hud = HUDEngine(map_width=500, map_height=500)
        frame = hud.render_frame(world_state, player_alliance="friendly")
        # frame is a JSON-serializable dict for the Three.js HUD overlay
    """

    def __init__(
        self,
        map_width: float = 500.0,
        map_height: float = 500.0,
        minimap_size: int = 200,
    ) -> None:
        self.minimap = MinimapRenderer(map_width, map_height, minimap_size)
        self.compass = CompassHUD()
        self.roster = UnitRoster()
        self.resources = ResourceHUD()
        self.notifications = NotificationQueue()
        self.kill_feed = KillFeed()
        self._frame_count: int = 0

    def add_notification(
        self,
        text: str,
        priority: str | NotificationPriority = NotificationPriority.MEDIUM,
        icon: str = "info",
        duration: float = 5.0,
    ) -> int:
        """Convenience: add a notification through the engine."""
        return self.notifications.add(text, priority, icon, duration)

    def add_kill(
        self,
        killer: str,
        victim: str,
        weapon: str = "",
        killer_alliance: str = "unknown",
        victim_alliance: str = "unknown",
    ) -> int:
        """Convenience: record a kill through the engine."""
        return self.kill_feed.add(killer, victim, weapon, killer_alliance, victim_alliance)

    def render_frame(
        self,
        world_state: dict[str, Any],
        player_alliance: str = "friendly",
        dt: float = 0.016,
    ) -> dict[str, Any]:
        """Produce one complete HUD frame.

        Parameters
        ----------
        world_state: Dict containing any of:
            - ``units``: list of unit dicts
            - ``vehicles``: list of vehicle dicts
            - ``structures``: list of structure dicts
            - ``fog_of_war``: set of hidden grid cells
            - ``camera_pos``: (x, y) world position
            - ``camera_fov``: float degrees
            - ``player_heading``: float degrees
            - ``threats``: list of threat dicts
            - ``objectives``: list of objective dicts
            - ``economy``: dict of resource -> amount
            - ``supply``: dict of supply -> amount
            - ``capacity``: dict of resource -> max
            - ``income``: dict of resource -> rate
            - ``expenses``: dict of resource -> rate
            - ``score``: dict of team -> score
        player_alliance: Which alliance the HUD viewer belongs to.
        dt: Delta time in seconds (for notification/kill-feed expiry).

        Returns a JSON-serializable dict with all HUD layers.
        """
        self._frame_count += 1

        units = world_state.get("units", [])
        vehicles = world_state.get("vehicles", [])
        structures = world_state.get("structures", [])
        fog = world_state.get("fog_of_war")

        # Minimap
        minimap_data = self.minimap.render(
            units=units,
            vehicles=vehicles,
            structures=structures,
            fog_of_war=fog,
            alliance=player_alliance,
        )

        # Camera viewport on minimap
        camera_pos = world_state.get("camera_pos", (250.0, 250.0))
        camera_fov = world_state.get("camera_fov", 60.0)
        minimap_threejs = self.minimap.to_three_js(
            camera_pos=camera_pos,
            camera_fov=camera_fov,
            markers=minimap_data["markers"],
        )

        # Compass
        player_heading = world_state.get("player_heading", 0.0)
        compass_data = self.compass.render(
            player_heading=player_heading,
            threats=world_state.get("threats"),
            objectives=world_state.get("objectives"),
        )

        # Unit roster
        roster_data = self.roster.render(
            units=units,
            alliance=player_alliance,
        )

        # Resources
        resource_data = self.resources.render(
            economy_state=world_state.get("economy"),
            supply_state=world_state.get("supply"),
            capacity=world_state.get("capacity"),
            income=world_state.get("income"),
            expenses=world_state.get("expenses"),
        )

        # Tick time-based elements
        active_notifs = self.notifications.tick(dt)
        active_kills = self.kill_feed.tick(dt)

        return {
            "frame": self._frame_count,
            "minimap": minimap_threejs,
            "compass": compass_data,
            "roster": roster_data,
            "resources": resource_data,
            "notifications": active_notifs,
            "kill_feed": active_kills,
            "score": world_state.get("score", {}),
        }
