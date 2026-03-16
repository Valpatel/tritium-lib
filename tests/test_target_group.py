# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for TargetGroup model."""

import pytest
from tritium_lib.models.target_group import TargetGroup, TargetGroupSummary


class TestTargetGroup:
    def test_create_default(self):
        g = TargetGroup(group_id="grp_test", name="Test Group")
        assert g.group_id == "grp_test"
        assert g.name == "Test Group"
        assert g.target_ids == []
        assert g.color == "#00f0ff"
        assert g.icon == "group"
        assert g.created_by == "operator"
        assert g.created_at is not None
        assert g.updated_at is not None

    def test_add_target(self):
        g = TargetGroup(group_id="grp_1", name="G1")
        assert g.add_target("ble_aa:bb:cc:dd:ee:ff")
        assert g.target_count == 1
        # Duplicate returns False
        assert not g.add_target("ble_aa:bb:cc:dd:ee:ff")
        assert g.target_count == 1

    def test_remove_target(self):
        g = TargetGroup(group_id="grp_1", name="G1", target_ids=["t1", "t2"])
        assert g.remove_target("t1")
        assert g.target_count == 1
        # Not found returns False
        assert not g.remove_target("t1")

    def test_has_target(self):
        g = TargetGroup(group_id="grp_1", name="G1", target_ids=["t1"])
        assert g.has_target("t1")
        assert not g.has_target("t2")

    def test_target_count(self):
        g = TargetGroup(group_id="grp_1", name="G1", target_ids=["a", "b", "c"])
        assert g.target_count == 3

    def test_serialization(self):
        g = TargetGroup(
            group_id="grp_1",
            name="Building A",
            description="Devices in building A",
            target_ids=["t1", "t2"],
            color="#ff2a6d",
            icon="building",
        )
        d = g.model_dump()
        assert d["group_id"] == "grp_1"
        assert d["name"] == "Building A"
        assert d["target_ids"] == ["t1", "t2"]
        assert d["color"] == "#ff2a6d"

        # Round-trip
        g2 = TargetGroup(**d)
        assert g2.group_id == g.group_id
        assert g2.target_ids == g.target_ids


class TestTargetGroupSummary:
    def test_from_group(self):
        g = TargetGroup(
            group_id="grp_1",
            name="Patrol suspects",
            target_ids=["t1", "t2", "t3"],
            color="#05ffa1",
        )
        s = TargetGroupSummary.from_group(g)
        assert s.group_id == "grp_1"
        assert s.name == "Patrol suspects"
        assert s.target_count == 3
        assert s.color == "#05ffa1"
