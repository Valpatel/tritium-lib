# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for swarm coordination models."""
import math

import pytest

from tritium_lib.models.swarm import (
    SwarmCommand,
    SwarmCommandType,
    SwarmFormation,
    SwarmFormationType,
    SwarmMember,
    SwarmMemberStatus,
    SwarmRole,
    SwarmStatus,
)


class TestSwarmMember:
    """Tests for SwarmMember model."""

    def test_create_default(self):
        m = SwarmMember(member_id="r1")
        assert m.member_id == "r1"
        assert m.status == SwarmMemberStatus.ACTIVE
        assert m.role == SwarmRole.SUPPORT
        assert m.battery == 1.0

    def test_create_with_role(self):
        m = SwarmMember(
            member_id="d1",
            device_id="drone-001",
            asset_type="drone",
            role=SwarmRole.LEAD,
            has_camera=True,
        )
        assert m.role == SwarmRole.LEAD
        assert m.has_camera is True
        assert m.asset_type == "drone"


class TestSwarmFormation:
    """Tests for SwarmFormation offset computation."""

    def test_line_formation_offsets(self):
        f = SwarmFormation(formation_type=SwarmFormationType.LINE, spacing=5.0, heading=0.0)
        ids = ["a", "b", "c"]
        offsets = f.compute_offsets(ids)
        assert len(offsets) == 3
        # Line perpendicular to heading=0 means spread along X axis
        assert offsets["a"] != offsets["c"]

    def test_wedge_formation(self):
        f = SwarmFormation(formation_type=SwarmFormationType.WEDGE, spacing=5.0, heading=0.0)
        ids = ["lead", "left", "right"]
        offsets = f.compute_offsets(ids)
        assert offsets["lead"] == (0.0, 0.0)
        assert len(offsets) == 3

    def test_circle_formation(self):
        f = SwarmFormation(formation_type=SwarmFormationType.CIRCLE, spacing=5.0, heading=0.0)
        ids = ["a", "b", "c", "d"]
        offsets = f.compute_offsets(ids)
        assert len(offsets) == 4
        # All should be roughly equidistant from center
        dists = [math.sqrt(ox**2 + oy**2) for ox, oy in offsets.values()]
        # Circle radius should be consistent
        assert max(dists) - min(dists) < 0.1

    def test_diamond_formation(self):
        f = SwarmFormation(formation_type=SwarmFormationType.DIAMOND, spacing=5.0, heading=0.0)
        ids = ["front", "left", "right", "rear"]
        offsets = f.compute_offsets(ids)
        assert len(offsets) == 4

    def test_column_formation(self):
        f = SwarmFormation(formation_type=SwarmFormationType.COLUMN, spacing=3.0, heading=90.0)
        ids = ["a", "b", "c"]
        offsets = f.compute_offsets(ids)
        assert len(offsets) == 3
        # First unit at (0,0)
        assert offsets["a"] == (0.0, 0.0)

    def test_staggered_formation(self):
        f = SwarmFormation(formation_type=SwarmFormationType.STAGGERED, spacing=4.0, heading=0.0)
        ids = ["a", "b", "c", "d"]
        offsets = f.compute_offsets(ids)
        assert len(offsets) == 4

    def test_empty_members(self):
        f = SwarmFormation(formation_type=SwarmFormationType.LINE, spacing=5.0)
        offsets = f.compute_offsets([])
        assert offsets == {}

    def test_single_member(self):
        f = SwarmFormation(formation_type=SwarmFormationType.LINE, spacing=5.0)
        offsets = f.compute_offsets(["only"])
        assert len(offsets) == 1


class TestSwarmCommand:
    """Tests for SwarmCommand model."""

    def test_create_hold(self):
        cmd = SwarmCommand(
            command_id="cmd-1",
            swarm_id="alpha",
            command_type=SwarmCommandType.HOLD,
        )
        assert cmd.command_type == SwarmCommandType.HOLD

    def test_create_advance(self):
        cmd = SwarmCommand(
            command_id="cmd-2",
            swarm_id="alpha",
            command_type=SwarmCommandType.ADVANCE,
            waypoint_x=100.0,
            waypoint_y=200.0,
            max_speed=1.5,
        )
        assert cmd.waypoint_x == 100.0
        assert cmd.max_speed == 1.5

    def test_patrol_with_waypoints(self):
        cmd = SwarmCommand(
            command_id="cmd-3",
            swarm_id="bravo",
            command_type=SwarmCommandType.PATROL,
            waypoints=[(0, 0), (10, 0), (10, 10), (0, 10)],
        )
        assert len(cmd.waypoints) == 4

    def test_all_command_types(self):
        for ct in SwarmCommandType:
            cmd = SwarmCommand(command_type=ct)
            assert cmd.command_type == ct


class TestSwarmStatus:
    """Tests for SwarmStatus model."""

    def test_create_status(self):
        status = SwarmStatus(
            swarm_id="alpha",
            name="Alpha Squad",
            formation=SwarmFormationType.WEDGE,
            member_count=4,
            active_members=3,
        )
        assert status.swarm_id == "alpha"
        assert status.member_count == 4
        assert status.active_members == 3
