# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Formation movement and pathfinding integration.

Squads move together in formation along roads and around obstacles.
Pure math — no rendering or game engine dependencies.

Key components:
    - FormationType: 10 tactical formations
    - get_formation_positions(): Pure math slot computation
    - FormationMover: Moves a group along a path in formation
    - PathPlanner: Road/off-road routing with path smoothing
    - CoverMovement: Tactical advance using cover and leapfrog

Coordinate convention:
    Vec2 = tuple[float, float] in local meters.
    +X = East, +Y = North, same as tritium_lib.geo.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    arrive,
    normalize,
    magnitude,
    _sub,
    _add,
    _scale,
)
from tritium_lib.sim_engine.ai.pathfinding import RoadNetwork, WalkableArea


# ---------------------------------------------------------------------------
# Formation types & config
# ---------------------------------------------------------------------------


class FormationType(Enum):
    """Tactical formation shapes."""

    LINE = "line"
    COLUMN = "column"
    WEDGE = "wedge"
    DIAMOND = "diamond"
    STAGGERED_COLUMN = "staggered_column"
    ECHELON_LEFT = "echelon_left"
    ECHELON_RIGHT = "echelon_right"
    CIRCLE = "circle"
    SPREAD = "spread"
    FILE = "file"


@dataclass
class FormationConfig:
    """Configuration for computing formation slot positions.

    Attributes:
        formation_type: Which formation shape to use.
        spacing: Distance between adjacent slots in meters.
        facing: Formation heading in radians (0 = +X/East, CCW positive).
        leader_pos: World-space position of the formation leader.
        num_members: Total number of members including the leader.
    """

    formation_type: FormationType
    spacing: float = 3.0
    facing: float = 0.0
    leader_pos: Vec2 = (0.0, 0.0)
    num_members: int = 1


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------


def _rotate(offset: Vec2, heading: float) -> Vec2:
    """Rotate a local-space offset by heading (radians) to world space."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return (
        offset[0] * cos_h - offset[1] * sin_h,
        offset[0] * sin_h + offset[1] * cos_h,
    )


# ---------------------------------------------------------------------------
# get_formation_positions — pure math
# ---------------------------------------------------------------------------


def get_formation_positions(config: FormationConfig) -> list[Vec2]:
    """Compute world-space positions for each formation slot.

    Slot 0 is the leader position. Remaining slots are arranged
    according to the formation type, rotated to face *config.facing*,
    and translated to *config.leader_pos*.

    Returns a list of Vec2 with length == config.num_members.
    """
    n = config.num_members
    if n <= 0:
        return []

    spacing = config.spacing
    # Compute offsets in local space (leader at origin, facing +X)
    offsets: list[Vec2] = []

    if config.formation_type == FormationType.LINE:
        # Members spread perpendicular to facing direction
        for i in range(n):
            lateral = (i - (n - 1) / 2.0) * spacing
            offsets.append((0.0, lateral))

    elif config.formation_type == FormationType.COLUMN:
        # Members stacked behind leader along facing direction
        for i in range(n):
            offsets.append((-i * spacing, 0.0))

    elif config.formation_type == FormationType.WEDGE:
        # Leader at front, others fan back in a V
        offsets.append((0.0, 0.0))
        for i in range(1, n):
            row = (i + 1) // 2
            side = 1.0 if i % 2 == 1 else -1.0
            offsets.append((-row * spacing * 0.7, side * row * spacing * 0.7))

    elif config.formation_type == FormationType.DIAMOND:
        if n == 1:
            offsets.append((0.0, 0.0))
        elif n == 2:
            offsets.append((0.0, 0.0))
            offsets.append((-spacing, 0.0))
        elif n == 3:
            offsets.append((0.0, 0.0))
            offsets.append((-spacing * 0.7, -spacing * 0.7))
            offsets.append((-spacing * 0.7, spacing * 0.7))
        else:
            # Front, left, right, back
            offsets.append((0.0, 0.0))  # point
            offsets.append((-spacing, -spacing))  # left
            offsets.append((-spacing, spacing))  # right
            offsets.append((-spacing * 2, 0.0))  # tail
            # Extra members fill interior
            for i in range(4, n):
                angle = 2 * math.pi * (i - 4) / max(1, n - 4)
                offsets.append((
                    -spacing + math.cos(angle) * spacing * 0.4,
                    math.sin(angle) * spacing * 0.4,
                ))

    elif config.formation_type == FormationType.STAGGERED_COLUMN:
        # Like column but alternating left/right offset
        for i in range(n):
            lateral = spacing * 0.5 * (1.0 if i % 2 == 1 else -1.0) if i > 0 else 0.0
            offsets.append((-i * spacing, lateral))

    elif config.formation_type == FormationType.ECHELON_LEFT:
        # Diagonal line trailing to the left
        for i in range(n):
            offsets.append((-i * spacing * 0.7, -i * spacing * 0.7))

    elif config.formation_type == FormationType.ECHELON_RIGHT:
        # Diagonal line trailing to the right
        for i in range(n):
            offsets.append((-i * spacing * 0.7, i * spacing * 0.7))

    elif config.formation_type == FormationType.CIRCLE:
        if n == 1:
            offsets.append((0.0, 0.0))
        else:
            radius = spacing * n / (2 * math.pi) if n > 2 else spacing
            for i in range(n):
                angle = 2 * math.pi * i / n
                offsets.append((
                    math.cos(angle) * radius,
                    math.sin(angle) * radius,
                ))

    elif config.formation_type == FormationType.SPREAD:
        # Wide line with extra spacing
        wide = spacing * 2.0
        for i in range(n):
            lateral = (i - (n - 1) / 2.0) * wide
            offsets.append((0.0, lateral))

    elif config.formation_type == FormationType.FILE:
        # Single file, tighter than column
        tight = spacing * 0.6
        for i in range(n):
            offsets.append((-i * tight, 0.0))

    else:
        # Fallback: cluster at leader
        for _ in range(n):
            offsets.append((0.0, 0.0))

    # Rotate offsets by facing and translate to leader_pos
    result: list[Vec2] = []
    for off in offsets:
        rotated = _rotate(off, config.facing)
        result.append((
            config.leader_pos[0] + rotated[0],
            config.leader_pos[1] + rotated[1],
        ))
    return result


# ---------------------------------------------------------------------------
# FormationMover — move group along path in formation
# ---------------------------------------------------------------------------


class FormationMover:
    """Moves a group of units along waypoints while maintaining formation.

    The leader follows the waypoint path using steering behaviors.
    Other members steer toward their formation slot positions relative
    to the leader. The formation automatically rotates to face the
    direction of movement.

    Usage:
        mover = FormationMover(
            waypoints=[(0, 0), (100, 0), (100, 100)],
            formation=FormationType.WEDGE,
            spacing=3.0,
        )
        while not mover.is_complete():
            targets = mover.tick(0.1, current_positions)
            # Apply targets to units
    """

    def __init__(
        self,
        waypoints: list[Vec2],
        formation: FormationType,
        spacing: float = 3.0,
        max_speed: float = 5.0,
        arrival_threshold: float = 1.5,
    ) -> None:
        self.waypoints = list(waypoints)
        self.formation = formation
        self.spacing = spacing
        self.max_speed = max_speed
        self.arrival_threshold = arrival_threshold

        # Start targeting the second waypoint (first is the start position)
        self._current_wp_idx = 1 if len(waypoints) > 1 else 0
        self._leader_pos: Vec2 = waypoints[0] if waypoints else (0.0, 0.0)
        self._facing: float = 0.0
        self._complete = len(waypoints) < 2
        # Initialize facing toward first target waypoint
        if len(waypoints) >= 2:
            dx = waypoints[1][0] - waypoints[0][0]
            dy = waypoints[1][1] - waypoints[0][1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                self._facing = math.atan2(dy, dx)
        self._total_dist = self._compute_total_distance()
        self._traveled_dist = 0.0
        self._member_ids: list[str] = []

    def _compute_total_distance(self) -> float:
        total = 0.0
        for i in range(len(self.waypoints) - 1):
            total += distance(self.waypoints[i], self.waypoints[i + 1])
        return max(total, 1e-6)

    def tick(
        self,
        dt: float,
        member_positions: dict[str, Vec2],
    ) -> dict[str, Vec2]:
        """Advance the formation one time step.

        Args:
            dt: Time delta in seconds.
            member_positions: Current positions keyed by member ID.
                The first key (by insertion order) is treated as the leader.

        Returns:
            Target positions for each member to steer toward.
        """
        if self._complete or not member_positions or not self.waypoints:
            return dict(member_positions)

        ids = list(member_positions.keys())
        self._member_ids = ids

        # Use internal leader position for steering continuity
        # (not the member's actual position, which may be offset by formation)
        leader_current = self._leader_pos

        # Update leader position toward current waypoint
        if self._current_wp_idx < len(self.waypoints):
            target_wp = self.waypoints[self._current_wp_idx]
            dist_to_wp = distance(leader_current, target_wp)

            if dist_to_wp < self.arrival_threshold:
                self._traveled_dist += dist_to_wp
                self._current_wp_idx += 1
                if self._current_wp_idx >= len(self.waypoints):
                    self._complete = True
                    # Final formation at last waypoint
                    self._leader_pos = self.waypoints[-1]

            if not self._complete and self._current_wp_idx < len(self.waypoints):
                target_wp = self.waypoints[self._current_wp_idx]
                dist_to_wp = distance(leader_current, target_wp)

                # Compute leader steering (arrive at waypoint)
                steering = arrive(
                    leader_current, target_wp, self.max_speed,
                    self.arrival_threshold * 3,
                )
                speed = magnitude(steering)
                move_dist = min(speed * dt, dist_to_wp)

                if speed > 1e-6:
                    direction = normalize(steering)
                    self._leader_pos = (
                        leader_current[0] + direction[0] * move_dist,
                        leader_current[1] + direction[1] * move_dist,
                    )
                    # Update facing to movement direction
                    self._facing = math.atan2(direction[1], direction[0])

        # Compute formation slots
        config = FormationConfig(
            formation_type=self.formation,
            spacing=self.spacing,
            facing=self._facing,
            leader_pos=self._leader_pos,
            num_members=len(ids),
        )
        slots = get_formation_positions(config)

        # Build target dict
        targets: dict[str, Vec2] = {}
        for i, uid in enumerate(ids):
            if i < len(slots):
                targets[uid] = slots[i]
            else:
                targets[uid] = self._leader_pos
        return targets

    def is_complete(self) -> bool:
        """True if the leader has reached the final waypoint."""
        return self._complete

    def progress(self) -> float:
        """Fraction of path completed (0.0 to 1.0)."""
        if self._complete:
            return 1.0
        if self._total_dist < 1e-6:
            return 1.0
        # Sum distances of fully completed segments
        # _current_wp_idx is the waypoint we're heading toward,
        # so completed segments are 0..1, 1..2, ..., (idx-2)..(idx-1)
        completed = 0.0
        # We started at waypoints[0] targeting waypoints[1], so
        # segments before _current_wp_idx - 1 are done
        for i in range(min(self._current_wp_idx - 1, len(self.waypoints) - 1)):
            completed += distance(self.waypoints[i], self.waypoints[i + 1])
        # Add partial progress on current segment
        if self._current_wp_idx < len(self.waypoints) and self._current_wp_idx >= 1:
            seg_start = self.waypoints[self._current_wp_idx - 1]
            seg_len = distance(seg_start, self.waypoints[self._current_wp_idx])
            if seg_len > 1e-6:
                done = distance(seg_start, self._leader_pos)
                completed += min(done, seg_len)
        return min(1.0, completed / self._total_dist)


# ---------------------------------------------------------------------------
# PathPlanner — road and off-road routing with smoothing
# ---------------------------------------------------------------------------


class PathPlanner:
    """Plan routes on road networks or off-road grids with path smoothing.

    Wraps RoadNetwork and grid-based A* with Chaikin path smoothing.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def plan_road_route(
        start: Vec2,
        end: Vec2,
        road_network: RoadNetwork,
    ) -> list[Vec2]:
        """Plan a route along roads using A*.

        Delegates to RoadNetwork.find_path. Returns an empty list
        if no path exists.
        """
        return road_network.find_path(start, end)

    @staticmethod
    def plan_off_road(
        start: Vec2,
        end: Vec2,
        obstacles: Optional[list[tuple[Vec2, float]]] = None,
        heightmap: Optional[dict[tuple[int, int], float]] = None,
        grid_size: float = 2.0,
    ) -> list[Vec2]:
        """Plan an off-road path using grid-based A* with obstacle/terrain cost.

        Args:
            start: Start position.
            end: End position.
            obstacles: List of (center, radius) circular obstacles.
            heightmap: Optional mapping of (grid_x, grid_y) -> elevation.
                       Elevation differences add movement cost.
            grid_size: Grid cell size in meters.

        Returns:
            List of waypoints from start to end, or empty if no path.
        """
        if obstacles is None:
            obstacles = []

        # Convert to grid coordinates
        def to_grid(p: Vec2) -> tuple[int, int]:
            return (int(round(p[0] / grid_size)), int(round(p[1] / grid_size)))

        def to_world(g: tuple[int, int]) -> Vec2:
            return (g[0] * grid_size, g[1] * grid_size)

        def is_blocked(gx: int, gy: int) -> bool:
            wx, wy = gx * grid_size, gy * grid_size
            for (ox, oy), radius in obstacles:
                if math.hypot(wx - ox, wy - oy) < radius + grid_size * 0.5:
                    return True
            return False

        def terrain_cost(gx: int, gy: int) -> float:
            if heightmap is None:
                return 0.0
            return abs(heightmap.get((gx, gy), 0.0)) * 0.5

        start_g = to_grid(start)
        end_g = to_grid(end)

        if is_blocked(end_g[0], end_g[1]):
            return []

        # A* on grid
        counter = 0
        open_set: list[tuple[float, int, tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, counter, start_g))
        g_score: dict[tuple[int, int], float] = {start_g: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        # 8-directional movement
        neighbors_offsets = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]

        # Limit search to a reasonable bounding box
        min_gx = min(start_g[0], end_g[0]) - 50
        max_gx = max(start_g[0], end_g[0]) + 50
        min_gy = min(start_g[1], end_g[1]) - 50
        max_gy = max(start_g[1], end_g[1]) + 50

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end_g:
                # Reconstruct
                path_grid: list[tuple[int, int]] = []
                node = current
                while node in came_from:
                    path_grid.append(node)
                    node = came_from[node]
                path_grid.append(start_g)
                path_grid.reverse()
                # Convert to world, prepend exact start, append exact end
                result = [start]
                for g in path_grid[1:-1]:
                    result.append(to_world(g))
                result.append(end)
                return result

            for dx, dy in neighbors_offsets:
                nx, ny = current[0] + dx, current[1] + dy
                if nx < min_gx or nx > max_gx or ny < min_gy or ny > max_gy:
                    continue
                if is_blocked(nx, ny):
                    continue
                move_cost = math.hypot(dx, dy) * grid_size
                t_cost = terrain_cost(nx, ny)
                tentative_g = g_score.get(current, float("inf")) + move_cost + t_cost
                if tentative_g < g_score.get((nx, ny), float("inf")):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    h = math.hypot(
                        (nx - end_g[0]) * grid_size,
                        (ny - end_g[1]) * grid_size,
                    )
                    counter += 1
                    heapq.heappush(open_set, (tentative_g + h, counter, (nx, ny)))

        return []  # No path found

    @staticmethod
    def smooth_path(
        waypoints: list[Vec2],
        iterations: int = 3,
    ) -> list[Vec2]:
        """Smooth a path using Chaikin corner-cutting.

        Preserves start and end points. Each iteration doubles the
        point count (minus endpoints) and rounds corners for natural
        movement.

        Args:
            waypoints: Input path.
            iterations: Number of smoothing passes.

        Returns:
            Smoothed path.
        """
        if len(waypoints) < 3:
            return list(waypoints)

        path = list(waypoints)
        for _ in range(iterations):
            if len(path) < 3:
                break
            new_path: list[Vec2] = [path[0]]
            for i in range(len(path) - 1):
                p0 = path[i]
                p1 = path[i + 1]
                # Q = 3/4 * P0 + 1/4 * P1
                q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
                # R = 1/4 * P0 + 3/4 * P1
                r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
                new_path.append(q)
                new_path.append(r)
            new_path.append(path[-1])
            path = new_path

        return path


# ---------------------------------------------------------------------------
# CoverMovement — tactical advance using cover positions
# ---------------------------------------------------------------------------


class CoverMovement:
    """Plan movement that uses cover and avoids open ground under fire."""

    @staticmethod
    def plan_covered_advance(
        start: Vec2,
        end: Vec2,
        cover_positions: list[Vec2],
        threat_positions: list[Vec2],
        max_exposure: float = 15.0,
    ) -> list[Vec2]:
        """Plan a path from start to end that stays near cover.

        Uses A* over cover positions, preferring positions that are
        shielded from threats (cover is between the unit and the threat).

        Args:
            start: Starting position.
            end: Destination.
            cover_positions: Available cover points in the area.
            threat_positions: Known enemy positions.
            max_exposure: Maximum distance to travel in open ground
                between cover positions.

        Returns:
            Path waypoints through cover positions, or direct path
            if no cover route exists.
        """
        if not cover_positions:
            return [start, end]

        # Score each cover position: lower = better
        def exposure_score(pos: Vec2) -> float:
            """How exposed is this position to known threats?"""
            if not threat_positions:
                return 0.0
            score = 0.0
            for threat in threat_positions:
                d = distance(pos, threat)
                if d < 1.0:
                    score += 100.0
                else:
                    score += 1.0 / d
            return score

        # Build graph: start + cover_positions + end
        nodes: list[Vec2] = [start] + list(cover_positions) + [end]
        n = len(nodes)
        start_idx = 0
        end_idx = n - 1

        # A* on cover graph
        counter = 0
        open_set: list[tuple[float, int, int]] = []
        heapq.heappush(open_set, (0.0, counter, start_idx))
        g_score: dict[int, float] = {start_idx: 0.0}
        came_from: dict[int, int] = {}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end_idx:
                path: list[Vec2] = []
                node = current
                while node in came_from:
                    path.append(nodes[node])
                    node = came_from[node]
                path.append(nodes[start_idx])
                path.reverse()
                return path

            for neighbor in range(n):
                if neighbor == current:
                    continue
                d = distance(nodes[current], nodes[neighbor])
                if d > max_exposure and neighbor != end_idx:
                    continue  # Too far to traverse in the open

                # Cost = distance + exposure penalty
                exp = exposure_score(nodes[neighbor])
                cost = d + exp * 10.0
                tentative_g = g_score.get(current, float("inf")) + cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = distance(nodes[neighbor], end)
                    counter += 1
                    heapq.heappush(open_set, (tentative_g + h, counter, neighbor))

        # No cover route found — go direct
        return [start, end]

    @staticmethod
    def leapfrog_advance(
        squads: list[dict[str, Vec2]],
        direction: Vec2,
        bound_distance: float = 15.0,
    ) -> list[dict]:
        """Plan alternating move/cover phases for leapfrog advance.

        Two or more groups alternate: one moves forward while the
        others provide cover. Each bound moves *bound_distance*
        meters in *direction*.

        Args:
            squads: List of squad position dicts (member_id -> position).
            direction: Direction of advance.
            bound_distance: Distance each bound covers.

        Returns:
            List of phase dicts with keys:
                - 'moving': index of squad that moves
                - 'covering': indices of squads providing cover
                - 'targets': target positions for the moving squad
                - 'phase': phase number (0-indexed)
        """
        if not squads:
            return []

        d_norm = normalize(direction)
        if magnitude(d_norm) < 1e-6:
            return []

        phases: list[dict] = []
        num_squads = len(squads)

        # Generate enough phases for all squads to move once per cycle
        for cycle in range(num_squads):
            moving_idx = cycle % num_squads
            covering_indices = [i for i in range(num_squads) if i != moving_idx]

            # Compute advance targets for the moving squad
            move_vec = _scale(d_norm, bound_distance * (cycle + 1))
            targets: dict[str, Vec2] = {}
            for uid, pos in squads[moving_idx].items():
                targets[uid] = _add(pos, _scale(d_norm, bound_distance))

            phases.append({
                "moving": moving_idx,
                "covering": covering_indices,
                "targets": targets,
                "phase": cycle,
            })

        return phases


# ---------------------------------------------------------------------------
# FormationManager — multi-formation coordinator
# ---------------------------------------------------------------------------


class FormationManager:
    """Manage multiple named formations and update all member positions each tick.

    Central coordinator that tracks which units belong to which formations,
    handles assignment/removal, and computes target positions for every
    unit in a single ``tick()`` call.

    Usage::

        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE, spacing=3.0)
        mgr.assign("alpha", "unit_1", is_leader=True)
        mgr.assign("alpha", "unit_2")
        mgr.assign("alpha", "unit_3")

        # Each tick:
        positions = {"unit_1": (10, 0), "unit_2": (8, 2), "unit_3": (8, -2)}
        targets = mgr.tick(0.1, positions)
        # targets == {"unit_1": ..., "unit_2": ..., "unit_3": ...}
    """

    def __init__(self) -> None:
        # formation_id -> dict with config fields + member list
        self._formations: dict[str, dict] = {}
        # unit_id -> formation_id (reverse lookup)
        self._unit_to_formation: dict[str, str] = {}

    # -- Formation lifecycle ---------------------------------------------------

    def create_formation(
        self,
        formation_id: str,
        formation_type: FormationType,
        spacing: float = 3.0,
    ) -> None:
        """Create a new named formation.

        Args:
            formation_id: Unique name for this formation.
            formation_type: Shape of the formation.
            spacing: Distance between adjacent slots in meters.

        Raises:
            ValueError: If formation_id already exists.
        """
        if formation_id in self._formations:
            raise ValueError(f"Formation '{formation_id}' already exists")
        self._formations[formation_id] = {
            "type": formation_type,
            "spacing": spacing,
            "leader": None,
            "members": [],  # ordered list of unit IDs (leader first when set)
        }

    def remove_formation(self, formation_id: str) -> bool:
        """Remove a formation and unassign all its members.

        Returns True if the formation existed and was removed.
        """
        info = self._formations.pop(formation_id, None)
        if info is None:
            return False
        for uid in info["members"]:
            self._unit_to_formation.pop(uid, None)
        return True

    def set_formation_type(
        self,
        formation_id: str,
        formation_type: FormationType,
    ) -> None:
        """Change the shape of an existing formation."""
        info = self._formations.get(formation_id)
        if info is None:
            raise KeyError(f"Formation '{formation_id}' not found")
        info["type"] = formation_type

    def set_spacing(self, formation_id: str, spacing: float) -> None:
        """Change the spacing of an existing formation."""
        info = self._formations.get(formation_id)
        if info is None:
            raise KeyError(f"Formation '{formation_id}' not found")
        info["spacing"] = spacing

    # -- Unit assignment -------------------------------------------------------

    def assign(
        self,
        formation_id: str,
        unit_id: str,
        is_leader: bool = False,
    ) -> None:
        """Add a unit to a formation.

        If the unit is already in another formation it is removed from
        the old one first.

        Args:
            formation_id: Target formation.
            unit_id: Unit to assign.
            is_leader: If True, this unit becomes the formation leader
                (slot 0). There can only be one leader; setting a new
                leader demotes the previous one to a regular member.

        Raises:
            KeyError: If formation_id does not exist.
        """
        info = self._formations.get(formation_id)
        if info is None:
            raise KeyError(f"Formation '{formation_id}' not found")

        # Remove from previous formation if assigned elsewhere
        old_fid = self._unit_to_formation.get(unit_id)
        if old_fid is not None and old_fid != formation_id:
            self._remove_from_formation(old_fid, unit_id)

        # Add to new formation
        if unit_id not in info["members"]:
            if is_leader:
                info["members"].insert(0, unit_id)
                info["leader"] = unit_id
            else:
                info["members"].append(unit_id)
        else:
            # Already in this formation — just update leader flag
            if is_leader:
                info["members"].remove(unit_id)
                info["members"].insert(0, unit_id)
                info["leader"] = unit_id

        self._unit_to_formation[unit_id] = formation_id

    def unassign(self, unit_id: str) -> bool:
        """Remove a unit from whatever formation it belongs to.

        Returns True if the unit was in a formation.
        """
        fid = self._unit_to_formation.pop(unit_id, None)
        if fid is None:
            return False
        self._remove_from_formation(fid, unit_id)
        return True

    def _remove_from_formation(self, formation_id: str, unit_id: str) -> None:
        info = self._formations.get(formation_id)
        if info is None:
            return
        if unit_id in info["members"]:
            info["members"].remove(unit_id)
        if info["leader"] == unit_id:
            # Promote next member to leader, or clear
            info["leader"] = info["members"][0] if info["members"] else None

    # -- Queries ---------------------------------------------------------------

    def get_formation(self, unit_id: str) -> str | None:
        """Return the formation ID a unit belongs to, or None."""
        return self._unit_to_formation.get(unit_id)

    def get_members(self, formation_id: str) -> list[str]:
        """Return the member list for a formation (leader first)."""
        info = self._formations.get(formation_id)
        if info is None:
            return []
        return list(info["members"])

    def get_leader(self, formation_id: str) -> str | None:
        """Return the leader unit ID for a formation, or None."""
        info = self._formations.get(formation_id)
        if info is None:
            return None
        return info["leader"]

    def list_formations(self) -> list[str]:
        """Return all formation IDs."""
        return list(self._formations.keys())

    @property
    def formation_count(self) -> int:
        """Number of active formations."""
        return len(self._formations)

    @property
    def unit_count(self) -> int:
        """Total number of assigned units across all formations."""
        return len(self._unit_to_formation)

    def formation_info(self, formation_id: str) -> dict | None:
        """Return a summary dict for a formation, or None if not found.

        Keys: formation_id, type, spacing, leader, members, member_count.
        """
        info = self._formations.get(formation_id)
        if info is None:
            return None
        return {
            "formation_id": formation_id,
            "type": info["type"].value,
            "spacing": info["spacing"],
            "leader": info["leader"],
            "members": list(info["members"]),
            "member_count": len(info["members"]),
        }

    # -- Tick (update all formations) ------------------------------------------

    def tick(
        self,
        dt: float,
        unit_positions: dict[str, Vec2],
    ) -> dict[str, Vec2]:
        """Compute formation target positions for all managed units.

        For each formation, the leader's current position and heading
        (derived from their movement since last tick) are used to place
        all members at their formation slots.

        Args:
            dt: Time step in seconds (reserved for future smoothing).
            unit_positions: Current world positions keyed by unit ID.

        Returns:
            Target positions for every managed unit. Units whose
            current position is not in *unit_positions* are skipped.
        """
        targets: dict[str, Vec2] = {}

        for fid, info in self._formations.items():
            members = info["members"]
            if not members:
                continue

            # Determine leader position
            leader_id = info["leader"] or members[0]
            leader_pos = unit_positions.get(leader_id)
            if leader_pos is None:
                continue

            # Compute facing from leader's movement direction
            # Use centroid of all members as a reference for heading.
            # If we have at least one follower with a position, face
            # away from the average follower (i.e., forward).
            facing = 0.0
            follower_positions: list[Vec2] = []
            for uid in members:
                if uid != leader_id:
                    pos = unit_positions.get(uid)
                    if pos is not None:
                        follower_positions.append(pos)

            if follower_positions:
                # Face away from the centroid of followers
                cx = sum(p[0] for p in follower_positions) / len(follower_positions)
                cy = sum(p[1] for p in follower_positions) / len(follower_positions)
                dx = leader_pos[0] - cx
                dy = leader_pos[1] - cy
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    facing = math.atan2(dy, dx)

            # Build formation config
            config = FormationConfig(
                formation_type=info["type"],
                spacing=info["spacing"],
                facing=facing,
                leader_pos=leader_pos,
                num_members=len(members),
            )

            slots = get_formation_positions(config)

            # Map slots to member IDs
            for i, uid in enumerate(members):
                if uid in unit_positions and i < len(slots):
                    targets[uid] = slots[i]

        return targets

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the full manager state to a JSON-friendly dict."""
        formations = {}
        for fid, info in self._formations.items():
            formations[fid] = {
                "type": info["type"].value,
                "spacing": info["spacing"],
                "leader": info["leader"],
                "members": list(info["members"]),
            }
        return {
            "formations": formations,
            "unit_assignments": dict(self._unit_to_formation),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FormationManager":
        """Reconstruct a FormationManager from a serialized dict."""
        mgr = cls()
        for fid, fdata in data.get("formations", {}).items():
            ftype = FormationType(fdata["type"])
            mgr.create_formation(fid, ftype, spacing=fdata.get("spacing", 3.0))
            leader = fdata.get("leader")
            for uid in fdata.get("members", []):
                mgr.assign(fid, uid, is_leader=(uid == leader))
        return mgr


# ---------------------------------------------------------------------------
# to_three_js helpers — serialise formation state for the Three.js viewer
# ---------------------------------------------------------------------------


def formation_to_three_js(
    config: FormationConfig,
    member_ids: list[str] | None = None,
) -> dict:
    """Serialise a formation's slot positions for the Three.js viewer.

    Returns a dict with:
        - ``formation_type``: string name of the formation
        - ``leader_pos``: [x, y] leader world position
        - ``facing``: heading in radians
        - ``slots``: list of {id, x, y} — one per member slot
        - ``lines``: list of [[x0,y0],[x1,y1]] pairs for drawing formation lines

    The ``lines`` array connects the leader to each follower, which is
    enough for Three.js to draw the formation overlay without needing
    full mesh geometry.
    """
    slots = get_formation_positions(config)
    slot_data: list[dict] = []
    for i, pos in enumerate(slots):
        entry: dict = {"slot": i, "x": pos[0], "y": pos[1]}
        if member_ids and i < len(member_ids):
            entry["id"] = member_ids[i]
        slot_data.append(entry)

    # Lines: leader (slot 0) to each follower, plus adjacent followers
    lines: list[list[list[float]]] = []
    if len(slots) >= 2:
        leader = slots[0]
        for follower in slots[1:]:
            lines.append([[leader[0], leader[1]], [follower[0], follower[1]]])

    return {
        "formation_type": config.formation_type.value,
        "leader_pos": [config.leader_pos[0], config.leader_pos[1]],
        "facing": config.facing,
        "spacing": config.spacing,
        "num_members": config.num_members,
        "slots": slot_data,
        "lines": lines,
    }


def formation_mover_to_three_js(mover: "FormationMover") -> dict:
    """Serialise a FormationMover's current state for the Three.js viewer.

    Returns the current formation snapshot including progress along the
    path, so the viewer can animate formation movement.
    """
    config = FormationConfig(
        formation_type=mover.formation,
        spacing=mover.spacing,
        facing=mover._facing,
        leader_pos=mover._leader_pos,
        num_members=len(mover._member_ids) if mover._member_ids else 1,
    )
    base = formation_to_three_js(config, mover._member_ids)
    base["progress"] = mover.progress()
    base["complete"] = mover.is_complete()
    # Waypoints for path visualisation
    base["waypoints"] = [[wp[0], wp[1]] for wp in mover.waypoints]
    return base
