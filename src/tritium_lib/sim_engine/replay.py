# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Replay recording, playback, analysis, and export for the sim engine.

Records every tick of a simulation for later playback, analysis, and sharing.
Supports compressed JSON save/load, variable-speed playback, kill timelines,
movement heatmaps, damage graphs, and CSV/GeoJSON export.

Usage::

    from tritium_lib.sim_engine.replay import ReplayRecorder, ReplayPlayer, ReplayAnalyzer

    recorder = ReplayRecorder(metadata={"preset": "urban_combat"})
    world = WORLD_PRESETS["urban_combat"]()
    for _ in range(100):
        frame_data = world.tick()
        recorder.record_from_world(world, world.events)

    recorder.save("/tmp/replay.json.gz")

    player = ReplayPlayer()
    player.load("/tmp/replay.json.gz")
    player.play()
    while (frame := player.next_frame()) is not None:
        print(frame.tick, len(frame.units))
"""

from __future__ import annotations

import gzip
import json
import math
import os
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tritium_lib.sim_engine.world import World


# ---------------------------------------------------------------------------
# ReplayFrame
# ---------------------------------------------------------------------------


@dataclass
class ReplayFrame:
    """Snapshot of a single simulation tick."""

    tick: int
    time: float
    units: list[dict[str, Any]]
    events: list[dict[str, Any]]
    render_data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "tick": self.tick,
            "time": self.time,
            "units": self.units,
            "events": self.events,
            "render_data": self.render_data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReplayFrame:
        """Deserialize from a plain dict."""
        return cls(
            tick=d["tick"],
            time=d["time"],
            units=d.get("units", []),
            events=d.get("events", []),
            render_data=d.get("render_data", {}),
        )


# ---------------------------------------------------------------------------
# ReplayRecorder
# ---------------------------------------------------------------------------


class ReplayRecorder:
    """Records simulation frames for later playback."""

    def __init__(
        self,
        metadata: dict[str, Any] | None = None,
        max_frames: int = 36000,
    ) -> None:
        self.frames: list[ReplayFrame] = []
        self.metadata: dict[str, Any] = metadata or {}
        self.max_frames: int = max_frames
        if "start_time" not in self.metadata:
            self.metadata["start_time"] = _time.time()

    def record_frame(
        self,
        tick: int,
        time: float,
        units: list[dict[str, Any]],
        events: list[dict[str, Any]],
        render_data: dict[str, Any],
    ) -> None:
        """Append a frame to the recording.

        If max_frames is reached, the oldest frame is dropped.
        """
        frame = ReplayFrame(
            tick=tick,
            time=time,
            units=units,
            events=events,
            render_data=render_data,
        )
        self.frames.append(frame)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

    def record_from_world(
        self,
        world: World,
        events: list[dict[str, Any]] | None = None,
    ) -> None:
        """Convenience: extract state from a World object and record it.

        Pulls unit snapshots from ``world.units``, events from
        ``world.events``, and render data from ``world.render()``.
        """
        units_snapshot: list[dict[str, Any]] = []
        for uid, u in world.units.items():
            units_snapshot.append({
                "id": uid,
                "name": u.name,
                "unit_type": u.unit_type.value,
                "alliance": u.alliance.value,
                "x": u.position[0],
                "y": u.position[1],
                "heading": u.heading,
                "health": u.state.health,
                "max_health": u.stats.max_health,
                "is_alive": u.state.is_alive,
                "status": u.state.status,
                "morale": u.state.morale,
                "suppression": u.state.suppression,
                "weapon": u.weapon,
                "squad_id": u.squad_id,
                "kill_count": u.state.kill_count,
                "damage_dealt": u.state.damage_dealt,
                "damage_taken": u.state.damage_taken,
            })

        frame_events = list(events) if events is not None else list(world.events)
        render_data = world.render()

        self.record_frame(
            tick=world.tick_count,
            time=world.sim_time,
            units=units_snapshot,
            events=frame_events,
            render_data=render_data,
        )

    def save(self, filepath: str) -> None:
        """Write the recording to a gzip-compressed JSON file."""
        data = {
            "version": 1,
            "metadata": self.metadata,
            "frame_count": len(self.frames),
            "frames": [f.to_dict() for f in self.frames],
        }
        raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with gzip.open(filepath, "wb") as fh:
            fh.write(raw)

    def get_size_mb(self) -> float:
        """Estimate the in-memory size of the recording in megabytes."""
        raw = json.dumps(
            [f.to_dict() for f in self.frames], separators=(",", ":")
        ).encode("utf-8")
        return len(raw) / (1024 * 1024)


# ---------------------------------------------------------------------------
# ReplayPlayer
# ---------------------------------------------------------------------------


class ReplayPlayer:
    """Plays back a recorded simulation."""

    def __init__(self) -> None:
        self.frames: list[ReplayFrame] = []
        self.metadata: dict[str, Any] = {}
        self.current_frame: int = 0
        self.playback_speed: float = 1.0
        self.is_playing: bool = False

    def load(self, filepath: str) -> None:
        """Read a gzip-compressed JSON replay file."""
        with gzip.open(filepath, "rb") as fh:
            raw = fh.read()
        data = json.loads(raw.decode("utf-8"))
        self.metadata = data.get("metadata", {})
        self.frames = [ReplayFrame.from_dict(f) for f in data.get("frames", [])]
        self.current_frame = 0
        self.is_playing = False

    def load_from_recorder(self, recorder: ReplayRecorder) -> None:
        """Load frames directly from a ReplayRecorder (no file I/O)."""
        self.frames = list(recorder.frames)
        self.metadata = dict(recorder.metadata)
        self.current_frame = 0
        self.is_playing = False

    def play(self) -> None:
        """Start or resume playback."""
        self.is_playing = True

    def pause(self) -> None:
        """Pause playback."""
        self.is_playing = False

    def stop(self) -> None:
        """Stop playback and reset to the beginning."""
        self.is_playing = False
        self.current_frame = 0

    def seek(self, tick: int) -> None:
        """Jump to a specific tick number.

        Finds the frame whose tick is closest to the requested tick.
        """
        if not self.frames:
            return
        best_idx = 0
        best_dist = abs(self.frames[0].tick - tick)
        for i, f in enumerate(self.frames):
            d = abs(f.tick - tick)
            if d < best_dist:
                best_dist = d
                best_idx = i
        self.current_frame = best_idx

    def seek_time(self, time: float) -> None:
        """Jump to a specific simulation time.

        Finds the frame whose time is closest to the requested time.
        """
        if not self.frames:
            return
        best_idx = 0
        best_dist = abs(self.frames[0].time - time)
        for i, f in enumerate(self.frames):
            d = abs(f.time - time)
            if d < best_dist:
                best_dist = d
                best_idx = i
        self.current_frame = best_idx

    def next_frame(self) -> ReplayFrame | None:
        """Return the next frame and advance the cursor, or None if done."""
        if not self.is_playing:
            return None
        if self.current_frame >= len(self.frames):
            self.is_playing = False
            return None
        frame = self.frames[self.current_frame]
        self.current_frame += 1
        return frame

    def prev_frame(self) -> ReplayFrame | None:
        """Return the previous frame and move the cursor back, or None."""
        if not self.frames:
            return None
        if self.current_frame <= 0:
            return None
        self.current_frame -= 1
        return self.frames[self.current_frame]

    def get_frame(self, tick: int) -> ReplayFrame | None:
        """Get a frame by exact tick number, or None if not found."""
        for f in self.frames:
            if f.tick == tick:
                return f
        return None

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier (e.g. 0.25, 0.5, 1.0, 2.0, 4.0)."""
        self.playback_speed = max(0.1, min(speed, 16.0))

    def total_duration(self) -> float:
        """Total duration in simulation seconds."""
        if not self.frames:
            return 0.0
        return self.frames[-1].time - self.frames[0].time

    def total_frames(self) -> int:
        """Total number of recorded frames."""
        return len(self.frames)

    def progress(self) -> float:
        """Current playback progress as a float in [0, 1]."""
        if not self.frames:
            return 0.0
        return self.current_frame / len(self.frames)


# ---------------------------------------------------------------------------
# ReplayAnalyzer
# ---------------------------------------------------------------------------


class ReplayAnalyzer:
    """Analyzes a recorded replay for insights and statistics."""

    def __init__(self, frames: list[ReplayFrame]) -> None:
        self.frames = frames

    def kill_timeline(self) -> list[dict[str, Any]]:
        """Return a list of kill events with tick, time, and details."""
        kills: list[dict[str, Any]] = []
        for frame in self.frames:
            for event in frame.events:
                if event.get("type") == "unit_killed":
                    kills.append({
                        "tick": frame.tick,
                        "time": frame.time,
                        "target_id": event.get("target_id"),
                        "source_id": event.get("source_id"),
                    })
        return kills

    def movement_heatmap(
        self, unit_id: str, grid_size: float = 10.0
    ) -> dict[str, Any]:
        """Build a grid heatmap of where a unit spent time.

        Returns a dict with ``grid_size``, ``cells`` (list of
        ``{"gx", "gy", "count"}``), and ``total_samples``.
        """
        cell_counts: dict[tuple[int, int], int] = defaultdict(int)
        total = 0
        for frame in self.frames:
            for u in frame.units:
                if u.get("id") == unit_id:
                    gx = int(u.get("x", 0.0) // grid_size)
                    gy = int(u.get("y", 0.0) // grid_size)
                    cell_counts[(gx, gy)] += 1
                    total += 1
                    break

        cells = [
            {"gx": gx, "gy": gy, "count": count}
            for (gx, gy), count in sorted(cell_counts.items())
        ]
        return {
            "unit_id": unit_id,
            "grid_size": grid_size,
            "cells": cells,
            "total_samples": total,
        }

    def damage_graph(self) -> dict[str, Any]:
        """Build per-unit damage dealt/received over time.

        Returns ``{"units": {unit_id: {"ticks": [...], "damage_dealt": [...],
        "damage_taken": [...]}}}``
        """
        result: dict[str, dict[str, list]] = {}

        for frame in self.frames:
            for u in frame.units:
                uid = u.get("id", "")
                if uid not in result:
                    result[uid] = {
                        "ticks": [],
                        "damage_dealt": [],
                        "damage_taken": [],
                    }
                result[uid]["ticks"].append(frame.tick)
                result[uid]["damage_dealt"].append(u.get("damage_dealt", 0.0))
                result[uid]["damage_taken"].append(u.get("damage_taken", 0.0))

        return {"units": result}

    def decisive_moment(self) -> ReplayFrame | None:
        """Return the frame with the most events (the decisive moment).

        If there are no frames, returns None.
        """
        if not self.frames:
            return None
        return max(self.frames, key=lambda f: len(f.events))

    def unit_path(self, unit_id: str) -> list[tuple[float, float]]:
        """Extract position path for a specific unit over all frames."""
        path: list[tuple[float, float]] = []
        for frame in self.frames:
            for u in frame.units:
                if u.get("id") == unit_id:
                    path.append((u.get("x", 0.0), u.get("y", 0.0)))
                    break
        return path

    def summary(self) -> dict[str, Any]:
        """High-level replay statistics."""
        if not self.frames:
            return {
                "total_frames": 0,
                "duration": 0.0,
                "total_events": 0,
                "total_kills": 0,
                "unique_units": 0,
                "factions": [],
            }

        total_events = sum(len(f.events) for f in self.frames)
        kills = self.kill_timeline()
        unit_ids: set[str] = set()
        factions: set[str] = set()
        for frame in self.frames:
            for u in frame.units:
                uid = u.get("id", "")
                if uid:
                    unit_ids.add(uid)
                alliance = u.get("alliance", "")
                if alliance:
                    factions.add(alliance)

        return {
            "total_frames": len(self.frames),
            "duration": self.frames[-1].time - self.frames[0].time,
            "total_events": total_events,
            "total_kills": len(kills),
            "unique_units": len(unit_ids),
            "factions": sorted(factions),
            "first_tick": self.frames[0].tick,
            "last_tick": self.frames[-1].tick,
        }


# ---------------------------------------------------------------------------
# ReplayExporter
# ---------------------------------------------------------------------------


class ReplayExporter:
    """Export replay data to external formats."""

    @staticmethod
    def to_csv(frames: list[ReplayFrame], filepath: str) -> None:
        """Export unit positions as CSV for analysis.

        Columns: tick, time, unit_id, name, alliance, x, y, health, status
        """
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w") as fh:
            fh.write("tick,time,unit_id,name,alliance,x,y,health,status\n")
            for frame in frames:
                for u in frame.units:
                    fh.write(
                        f"{frame.tick},{frame.time:.3f},"
                        f"{u.get('id', '')},"
                        f"{u.get('name', '')},"
                        f"{u.get('alliance', '')},"
                        f"{u.get('x', 0.0):.2f},"
                        f"{u.get('y', 0.0):.2f},"
                        f"{u.get('health', 0.0):.1f},"
                        f"{u.get('status', '')}\n"
                    )

    @staticmethod
    def to_geojson(frames: list[ReplayFrame]) -> dict[str, Any]:
        """Convert unit movement paths to GeoJSON LineString features.

        Each unit gets a Feature with a LineString geometry of its path.
        Note: coordinates are sim-space (x, y), not lat/lng.
        """
        # Gather paths per unit
        paths: dict[str, dict[str, Any]] = {}
        for frame in frames:
            for u in frame.units:
                uid = u.get("id", "")
                if uid not in paths:
                    paths[uid] = {
                        "name": u.get("name", uid),
                        "alliance": u.get("alliance", "unknown"),
                        "coordinates": [],
                    }
                paths[uid]["coordinates"].append(
                    [u.get("x", 0.0), u.get("y", 0.0)]
                )

        features: list[dict[str, Any]] = []
        for uid, info in paths.items():
            coords = info["coordinates"]
            # Deduplicate consecutive identical points
            deduped: list[list[float]] = []
            for pt in coords:
                if not deduped or pt != deduped[-1]:
                    deduped.append(pt)
            # GeoJSON LineString needs at least 2 points
            if len(deduped) < 2:
                continue
            features.append({
                "type": "Feature",
                "properties": {
                    "unit_id": uid,
                    "name": info["name"],
                    "alliance": info["alliance"],
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": deduped,
                },
            })

        return {
            "type": "FeatureCollection",
            "features": features,
        }
