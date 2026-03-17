# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Intelligence, reconnaissance, and fog of war for the Tritium sim engine.

Simulates fog of war, intelligence gathering, reconnaissance missions, and
information sharing between alliances. Supports SIGINT, HUMINT, IMINT, ELINT,
and OSINT intelligence types with realistic decay, fusion, and confidence
modeling.

Usage::

    from tritium_lib.sim_engine.intel import IntelEngine, FogOfWar, IntelType

    engine = IntelEngine(grid_size=(200, 200), cell_size=5.0)
    reports = engine.gather_imint(
        observer_pos=(100.0, 100.0),
        observer_range=50.0,
        entities={"enemy-1": (120.0, 130.0), "enemy-2": (200.0, 200.0)},
        alliance="blue",
    )
    engine.tick(1.0, observer_data={"blue": [((100.0, 100.0), 50.0)]},
                entities={"enemy-1": (120.0, 130.0)})
    picture = engine.get_intel_picture("blue")
"""

from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IntelType(Enum):
    """Intelligence discipline classification."""
    SIGINT = "sigint"   # Signals intelligence (radio intercepts)
    HUMINT = "humint"   # Human intelligence (agents, interrogations)
    IMINT = "imint"     # Imagery intelligence (visual observation)
    ELINT = "elint"     # Electronic intelligence (radar/RF emissions)
    OSINT = "osint"     # Open-source intelligence (public data)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntelReport:
    """A single intelligence report from any source."""
    report_id: str
    intel_type: IntelType
    source_id: str              # unit/sensor that gathered it
    subject_id: str | None      # what was observed (None = general area intel)
    position: Vec2              # reported position
    accuracy: float             # meters of position error
    confidence: float           # 0-1 confidence in the report
    content: str                # human-readable description
    timestamp: float            # when gathered
    expires: float              # when intel goes stale (absolute time)
    alliance: str               # who has this intel


@dataclass
class ReconMission:
    """A reconnaissance mission assigned to a unit."""
    mission_id: str
    mission_type: str           # area_recon, route_recon, point_recon, surveillance
    assigned_unit_id: str
    waypoints: list[Vec2]
    gathered_intel: list[IntelReport] = field(default_factory=list)
    status: str = "planning"    # planning, active, complete, aborted
    start_time: float = 0.0
    end_time: float | None = None
    _current_wp_index: int = field(default=0, repr=False)


# ---------------------------------------------------------------------------
# Fog of War
# ---------------------------------------------------------------------------

class FogOfWar:
    """Per-alliance fog of war on a discretized grid.

    Cells transition through three states:
    - **hidden**: never seen (full fog)
    - **explored**: previously seen but not currently visible (partial fog)
    - **visible**: currently observed by a friendly unit (no fog)
    """

    def __init__(self, grid_size: tuple[int, int], cell_size: float = 5.0) -> None:
        self.grid_width = grid_size[0]
        self.grid_height = grid_size[1]
        self.cell_size = cell_size
        # Per-alliance sets of (col, row) tuples
        self.visibility: dict[str, set[tuple[int, int]]] = {}
        self.explored: dict[str, set[tuple[int, int]]] = {}

    def _pos_to_cell(self, pos: Vec2) -> tuple[int, int]:
        """Convert world position to grid cell coordinates."""
        col = int(pos[0] / self.cell_size)
        row = int(pos[1] / self.cell_size)
        col = max(0, min(col, self.grid_width - 1))
        row = max(0, min(row, self.grid_height - 1))
        return (col, row)

    def _cells_in_radius(self, center: Vec2, radius: float) -> set[tuple[int, int]]:
        """Return all grid cells within radius of a world position."""
        cells: set[tuple[int, int]] = set()
        cx, cy = center
        r_cells = int(math.ceil(radius / self.cell_size))
        center_col, center_row = self._pos_to_cell(center)

        for dc in range(-r_cells, r_cells + 1):
            for dr in range(-r_cells, r_cells + 1):
                c = center_col + dc
                r = center_row + dr
                if 0 <= c < self.grid_width and 0 <= r < self.grid_height:
                    # Check actual distance from center to cell center
                    cell_cx = (c + 0.5) * self.cell_size
                    cell_cy = (r + 0.5) * self.cell_size
                    if distance(center, (cell_cx, cell_cy)) <= radius:
                        cells.add((c, r))
        return cells

    def update_visibility(
        self, alliance: str, observer_positions: list[tuple[Vec2, float]]
    ) -> None:
        """Recompute visibility for an alliance from scratch.

        Args:
            alliance: Alliance identifier (e.g., "blue", "red").
            observer_positions: List of (position, detection_range) tuples.
        """
        if alliance not in self.explored:
            self.explored[alliance] = set()

        # Previous visible cells become explored
        old_visible = self.visibility.get(alliance, set())
        self.explored[alliance].update(old_visible)

        # Compute new visibility
        new_visible: set[tuple[int, int]] = set()
        for pos, det_range in observer_positions:
            new_visible.update(self._cells_in_radius(pos, det_range))

        self.visibility[alliance] = new_visible
        # Currently visible cells are also explored
        self.explored[alliance].update(new_visible)

    def is_visible(self, alliance: str, position: Vec2) -> bool:
        """Check if a world position is currently visible to an alliance."""
        cell = self._pos_to_cell(position)
        return cell in self.visibility.get(alliance, set())

    def is_explored(self, alliance: str, position: Vec2) -> bool:
        """Check if a world position has ever been seen by an alliance."""
        cell = self._pos_to_cell(position)
        return cell in self.explored.get(alliance, set())

    def get_visible_entities(
        self, alliance: str, all_entities: dict[str, Vec2]
    ) -> dict[str, Vec2]:
        """Return only entities that are in visible cells for an alliance."""
        visible = self.visibility.get(alliance, set())
        result: dict[str, Vec2] = {}
        for eid, pos in all_entities.items():
            cell = self._pos_to_cell(pos)
            if cell in visible:
                result[eid] = pos
        return result

    def to_three_js(self, alliance: str) -> dict[str, Any]:
        """Export fog state for Three.js rendering."""
        visible = self.visibility.get(alliance, set())
        explored_only = self.explored.get(alliance, set()) - visible
        return {
            "fog_grid": {
                "width": self.grid_width,
                "height": self.grid_height,
                "cell_size": self.cell_size,
                "visible": sorted(list(visible)),
                "explored": sorted(list(explored_only)),
                "fog_color": "#0a0a0f",
                "explored_alpha": 0.5,
                "hidden_alpha": 0.9,
            }
        }


# ---------------------------------------------------------------------------
# Intel Fusion
# ---------------------------------------------------------------------------

class IntelFusion:
    """Fuses multiple intelligence reports about the same subject."""

    @staticmethod
    def fuse_reports(reports: list[IntelReport]) -> IntelReport:
        """Merge multiple reports about the same subject into one.

        Uses confidence-weighted position averaging and takes the highest
        confidence and latest timestamp. Reports must share the same
        subject_id and alliance.

        Args:
            reports: Non-empty list of reports to fuse.

        Returns:
            A single fused IntelReport with improved accuracy and confidence.

        Raises:
            ValueError: If reports list is empty.
        """
        if not reports:
            raise ValueError("Cannot fuse empty report list")

        if len(reports) == 1:
            return reports[0]

        # Confidence-weighted position average
        total_weight = 0.0
        wx, wy = 0.0, 0.0
        best_confidence = 0.0
        latest_timestamp = 0.0
        latest_expires = 0.0
        contents: list[str] = []

        for r in reports:
            w = max(r.confidence, 0.01)  # avoid zero weight
            wx += r.position[0] * w
            wy += r.position[1] * w
            total_weight += w
            best_confidence = max(best_confidence, r.confidence)
            latest_timestamp = max(latest_timestamp, r.timestamp)
            latest_expires = max(latest_expires, r.expires)
            contents.append(r.content)

        fused_pos: Vec2 = (wx / total_weight, wy / total_weight)

        # Fused accuracy improves with more sources (1/sqrt(n) scaling)
        avg_accuracy = sum(r.accuracy for r in reports) / len(reports)
        fused_accuracy = avg_accuracy / math.sqrt(len(reports))

        # Confidence boost from multiple corroborating sources (caps at 0.99)
        fused_confidence = min(0.99, best_confidence + 0.05 * (len(reports) - 1))

        # Determine dominant intel type
        type_counts: dict[IntelType, int] = {}
        for r in reports:
            type_counts[r.intel_type] = type_counts.get(r.intel_type, 0) + 1
        dominant_type = max(type_counts, key=lambda t: type_counts[t])

        return IntelReport(
            report_id=f"fused-{uuid.uuid4().hex[:8]}",
            intel_type=dominant_type,
            source_id="fusion",
            subject_id=reports[0].subject_id,
            position=fused_pos,
            accuracy=fused_accuracy,
            confidence=fused_confidence,
            content=f"Fused from {len(reports)} reports: {'; '.join(contents[:3])}",
            timestamp=latest_timestamp,
            expires=latest_expires,
            alliance=reports[0].alliance,
        )


# ---------------------------------------------------------------------------
# Intel Engine
# ---------------------------------------------------------------------------

class IntelEngine:
    """Central intelligence engine managing fog of war, reports, and recon.

    Coordinates all intelligence gathering, manages the fog of war grid,
    tracks reconnaissance missions, and provides fused intelligence pictures
    to requesting alliances.
    """

    def __init__(
        self, grid_size: tuple[int, int] = (100, 100), cell_size: float = 5.0
    ) -> None:
        self.reports: list[IntelReport] = []
        self.fog = FogOfWar(grid_size, cell_size)
        self.missions: dict[str, ReconMission] = {}
        self._fusion = IntelFusion()
        self._current_time: float = time.time()

    def _make_report_id(self) -> str:
        return f"intel-{uuid.uuid4().hex[:8]}"

    def gather_sigint(
        self,
        listener_pos: Vec2,
        radio_messages: list[dict[str, Any]],
        alliance: str,
        listen_range: float = 500.0,
        intercept_time: float | None = None,
    ) -> list[IntelReport]:
        """Gather signals intelligence by intercepting radio messages.

        Each radio message dict should have:
        - ``position``: Vec2 — sender position
        - ``content``: str — message text
        - ``sender_id``: str — sender identifier
        - ``encrypted``: bool — if True, content is not readable

        Args:
            listener_pos: Position of the SIGINT listener.
            radio_messages: Radio traffic to scan.
            alliance: Alliance performing the intercept.
            listen_range: Maximum intercept range in meters.
            intercept_time: Override timestamp (defaults to current engine time).

        Returns:
            List of new IntelReport objects.
        """
        now = intercept_time if intercept_time is not None else self._current_time
        new_reports: list[IntelReport] = []

        for msg in radio_messages:
            msg_pos: Vec2 = msg["position"]
            dist = distance(listener_pos, msg_pos)
            if dist > listen_range:
                continue

            encrypted = msg.get("encrypted", False)
            sender = msg.get("sender_id", "unknown")
            content = msg.get("content", "")

            # Accuracy degrades with distance
            accuracy = max(5.0, dist * 0.1)  # 10% of distance, min 5m

            if encrypted:
                report_content = (
                    f"Encrypted transmission detected from bearing "
                    f"{math.degrees(math.atan2(msg_pos[1] - listener_pos[1], msg_pos[0] - listener_pos[0])):.0f} deg"
                )
                confidence = 0.3
            else:
                report_content = f"Intercepted from {sender}: {content}"
                confidence = 0.7

            # Confidence decreases with distance
            confidence *= max(0.3, 1.0 - dist / listen_range)

            report = IntelReport(
                report_id=self._make_report_id(),
                intel_type=IntelType.SIGINT,
                source_id="sigint_listener",
                subject_id=sender,
                position=msg_pos,
                accuracy=accuracy,
                confidence=confidence,
                content=report_content,
                timestamp=now,
                expires=now + 300.0,  # SIGINT valid for 5 min
                alliance=alliance,
            )
            new_reports.append(report)
            self.reports.append(report)

        return new_reports

    def gather_imint(
        self,
        observer_pos: Vec2,
        observer_range: float,
        entities: dict[str, Vec2],
        alliance: str,
        observation_time: float | None = None,
    ) -> list[IntelReport]:
        """Gather imagery intelligence through visual observation.

        Args:
            observer_pos: Position of the observer.
            observer_range: Maximum observation range in meters.
            entities: Map of entity_id to position for all entities in the world.
            alliance: Alliance performing the observation.
            observation_time: Override timestamp.

        Returns:
            List of new IntelReport objects for entities within range.
        """
        now = observation_time if observation_time is not None else self._current_time
        new_reports: list[IntelReport] = []

        for eid, epos in entities.items():
            dist = distance(observer_pos, epos)
            if dist > observer_range:
                continue

            # Closer observation = better accuracy and confidence
            range_ratio = dist / observer_range
            accuracy = max(1.0, dist * 0.05)  # 5% of distance, min 1m
            confidence = max(0.2, 1.0 - range_ratio * 0.6)

            report = IntelReport(
                report_id=self._make_report_id(),
                intel_type=IntelType.IMINT,
                source_id="visual_observer",
                subject_id=eid,
                position=epos,
                accuracy=accuracy,
                confidence=confidence,
                content=f"Visual contact with {eid} at range {dist:.0f}m",
                timestamp=now,
                expires=now + 120.0,  # visual intel valid for 2 min
                alliance=alliance,
            )
            new_reports.append(report)
            self.reports.append(report)

        return new_reports

    def gather_elint(
        self,
        sensor_pos: Vec2,
        rf_emitters: list[dict[str, Any]],
        alliance: str,
        sensor_range: float = 1000.0,
        detection_time: float | None = None,
    ) -> list[IntelReport]:
        """Gather electronic intelligence by detecting RF emissions.

        Each rf_emitter dict should have:
        - ``position``: Vec2 — emitter position
        - ``emitter_id``: str — emitter identifier
        - ``frequency_mhz``: float — emission frequency
        - ``power_w``: float — emission power (optional)

        Args:
            sensor_pos: Position of the ELINT sensor.
            rf_emitters: RF emitters in the world.
            alliance: Alliance performing the detection.
            sensor_range: Maximum detection range in meters.
            detection_time: Override timestamp.

        Returns:
            List of new IntelReport objects.
        """
        now = detection_time if detection_time is not None else self._current_time
        new_reports: list[IntelReport] = []

        for emitter in rf_emitters:
            epos: Vec2 = emitter["position"]
            dist = distance(sensor_pos, epos)
            if dist > sensor_range:
                continue

            emitter_id = emitter.get("emitter_id", "unknown")
            freq = emitter.get("frequency_mhz", 0.0)
            power = emitter.get("power_w", 0.0)

            # ELINT accuracy depends on distance and signal strength
            accuracy = max(10.0, dist * 0.15)
            confidence = max(0.2, 0.8 - (dist / sensor_range) * 0.5)

            content_parts = [f"RF emission detected from {emitter_id}"]
            if freq > 0:
                content_parts.append(f"freq={freq:.1f}MHz")
            if power > 0:
                content_parts.append(f"power={power:.1f}W")

            report = IntelReport(
                report_id=self._make_report_id(),
                intel_type=IntelType.ELINT,
                source_id="elint_sensor",
                subject_id=emitter_id,
                position=epos,
                accuracy=accuracy,
                confidence=confidence,
                content=", ".join(content_parts),
                timestamp=now,
                expires=now + 600.0,  # ELINT valid for 10 min
                alliance=alliance,
            )
            new_reports.append(report)
            self.reports.append(report)

        return new_reports

    def create_recon_mission(
        self,
        mission_type: str,
        unit_id: str,
        waypoints: list[Vec2],
        start_time: float | None = None,
    ) -> ReconMission:
        """Create a new reconnaissance mission.

        Args:
            mission_type: One of area_recon, route_recon, point_recon, surveillance.
            unit_id: ID of the unit assigned to the mission.
            waypoints: Ordered list of positions defining the recon path.
            start_time: When the mission starts (defaults to current engine time).

        Returns:
            The created ReconMission.
        """
        mission = ReconMission(
            mission_id=f"recon-{uuid.uuid4().hex[:8]}",
            mission_type=mission_type,
            assigned_unit_id=unit_id,
            waypoints=list(waypoints),
            status="planning",
            start_time=start_time if start_time is not None else self._current_time,
        )
        self.missions[mission.mission_id] = mission
        return mission

    def _advance_recon_mission(
        self,
        mission: ReconMission,
        dt: float,
        entities: dict[str, Vec2],
        unit_speed: float = 5.0,
        unit_range: float = 50.0,
    ) -> None:
        """Advance a recon mission by one tick, gathering intel along the way."""
        if mission.status != "active":
            return

        if mission._current_wp_index >= len(mission.waypoints):
            mission.status = "complete"
            mission.end_time = self._current_time
            return

        # Simulate unit movement toward next waypoint
        target_wp = mission.waypoints[mission._current_wp_index]
        # Assume unit is approximately at its current waypoint for intel gathering
        unit_pos = target_wp

        # Gather IMINT at current position
        reports = self.gather_imint(
            observer_pos=unit_pos,
            observer_range=unit_range,
            entities=entities,
            alliance="recon",
            observation_time=self._current_time,
        )
        # Tag reports with mission alliance info
        for r in reports:
            mission.gathered_intel.append(r)

        # Advance waypoint index (simplified: one WP per tick)
        mission._current_wp_index += 1

    def tick(
        self,
        dt: float,
        observer_data: dict[str, list[tuple[Vec2, float]]],
        entities: dict[str, Vec2],
    ) -> None:
        """Advance the intelligence engine by one time step.

        Args:
            dt: Time step in seconds.
            observer_data: Per-alliance list of (position, detection_range) tuples.
            entities: All entity positions in the world (id -> position).
        """
        self._current_time += dt

        # Update fog of war for each alliance
        for alliance, observers in observer_data.items():
            self.fog.update_visibility(alliance, observers)

        # Process active recon missions
        for mission in self.missions.values():
            if mission.status == "planning":
                # Auto-activate if start time has passed
                if self._current_time >= mission.start_time:
                    mission.status = "active"
            if mission.status == "active":
                self._advance_recon_mission(mission, dt, entities)

        # Expire stale intel
        self.reports = [
            r for r in self.reports if r.expires > self._current_time
        ]

        # Merge conflicting reports about the same subject
        self._merge_conflicting_reports()

    def _merge_conflicting_reports(self) -> None:
        """Find reports about the same subject and fuse them if beneficial."""
        by_subject: dict[str, list[IntelReport]] = {}
        no_subject: list[IntelReport] = []

        for r in self.reports:
            if r.subject_id is None:
                no_subject.append(r)
            else:
                key = f"{r.alliance}:{r.subject_id}"
                by_subject.setdefault(key, []).append(r)

        merged: list[IntelReport] = list(no_subject)
        for key, group in by_subject.items():
            if len(group) <= 1:
                merged.extend(group)
            else:
                # Keep the most recent two and fuse
                group.sort(key=lambda r: r.timestamp, reverse=True)
                # Fuse up to 3 most recent
                to_fuse = group[:3]
                remainder = group[3:]
                fused = self._fusion.fuse_reports(to_fuse)
                merged.append(fused)
                merged.extend(remainder)

        self.reports = merged

    def get_intel_picture(self, alliance: str) -> dict[str, Any]:
        """Get the complete intelligence picture for an alliance.

        Returns:
            Dict with keys: reports (list), fog (FogOfWar state),
            active_missions (list), threat_count, coverage_pct.
        """
        alliance_reports = [r for r in self.reports if r.alliance == alliance]
        active_missions = [
            m for m in self.missions.values()
            if m.status == "active"
        ]
        visible_cells = len(self.fog.visibility.get(alliance, set()))
        total_cells = self.fog.grid_width * self.fog.grid_height
        coverage_pct = (visible_cells / total_cells * 100.0) if total_cells > 0 else 0.0

        return {
            "alliance": alliance,
            "reports": [
                {
                    "report_id": r.report_id,
                    "type": r.intel_type.value,
                    "subject_id": r.subject_id,
                    "position": list(r.position),
                    "accuracy": r.accuracy,
                    "confidence": r.confidence,
                    "content": r.content,
                    "timestamp": r.timestamp,
                    "expires": r.expires,
                }
                for r in alliance_reports
            ],
            "active_missions": [m.mission_id for m in active_missions],
            "threat_count": sum(
                1 for r in alliance_reports if r.subject_id is not None
            ),
            "coverage_pct": round(coverage_pct, 1),
        }

    def get_threat_estimate(
        self, alliance: str, area: tuple[Vec2, float]
    ) -> dict[str, Any]:
        """Estimate enemy forces in a given area based on gathered intel.

        Args:
            alliance: Alliance requesting the estimate.
            area: Tuple of (center_position, radius) defining the area.

        Returns:
            Dict with estimated_contacts, confidence, reports, and assessed_threat.
        """
        center, radius = area
        alliance_reports = [r for r in self.reports if r.alliance == alliance]

        # Find reports whose subjects are in the area
        area_reports: list[IntelReport] = []
        seen_subjects: set[str] = set()
        for r in alliance_reports:
            if r.subject_id and distance(center, r.position) <= radius:
                area_reports.append(r)
                seen_subjects.add(r.subject_id)

        if not area_reports:
            return {
                "area_center": list(center),
                "area_radius": radius,
                "estimated_contacts": 0,
                "confidence": 0.0,
                "reports": [],
                "assessed_threat": "none",
            }

        avg_confidence = sum(r.confidence for r in area_reports) / len(area_reports)
        n_contacts = len(seen_subjects)

        # Threat level based on number of contacts
        if n_contacts >= 10:
            threat = "critical"
        elif n_contacts >= 5:
            threat = "high"
        elif n_contacts >= 2:
            threat = "moderate"
        else:
            threat = "low"

        return {
            "area_center": list(center),
            "area_radius": radius,
            "estimated_contacts": n_contacts,
            "confidence": round(avg_confidence, 2),
            "reports": [r.report_id for r in area_reports],
            "assessed_threat": threat,
        }

    def to_three_js(self, alliance: str) -> dict[str, Any]:
        """Export full intelligence layer for Three.js rendering.

        Includes fog of war grid, intel markers, and active recon paths.
        """
        fog_data = self.fog.to_three_js(alliance)
        alliance_reports = [r for r in self.reports if r.alliance == alliance]

        intel_markers = []
        for r in alliance_reports:
            intel_markers.append({
                "id": r.report_id,
                "type": r.intel_type.value,
                "position": list(r.position),
                "accuracy": r.accuracy,
                "confidence": r.confidence,
                "content": r.content,
            })

        recon_paths = []
        for m in self.missions.values():
            if m.status in ("active", "planning"):
                recon_paths.append({
                    "mission_id": m.mission_id,
                    "type": m.mission_type,
                    "waypoints": [list(wp) for wp in m.waypoints],
                    "status": m.status,
                    "progress": m._current_wp_index / max(len(m.waypoints), 1),
                })

        return {
            **fog_data,
            "intel_markers": intel_markers,
            "recon_paths": recon_paths,
        }
