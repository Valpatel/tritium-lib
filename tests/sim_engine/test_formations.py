# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for FormationManager — multi-formation coordinator.

Covers creation, assignment, removal, tick updates, queries, serialization,
edge cases, and integration with the existing formation position system.
"""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.ai.steering import Vec2, distance
from tritium_lib.sim_engine.ai.formations import (
    FormationConfig,
    FormationManager,
    FormationMover,
    FormationType,
    get_formation_positions,
)


# ===========================================================================
# FormationManager — Creation & Lifecycle
# ===========================================================================


class TestFormationManagerCreation:
    def test_empty_manager(self):
        mgr = FormationManager()
        assert mgr.formation_count == 0
        assert mgr.unit_count == 0
        assert mgr.list_formations() == []

    def test_create_formation(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE, spacing=4.0)
        assert mgr.formation_count == 1
        assert "alpha" in mgr.list_formations()

    def test_create_multiple_formations(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.create_formation("bravo", FormationType.LINE)
        mgr.create_formation("charlie", FormationType.COLUMN)
        assert mgr.formation_count == 3
        assert set(mgr.list_formations()) == {"alpha", "bravo", "charlie"}

    def test_create_duplicate_raises(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_formation("alpha", FormationType.LINE)

    def test_remove_formation(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "u1")
        assert mgr.remove_formation("alpha")
        assert mgr.formation_count == 0
        assert mgr.unit_count == 0
        assert mgr.get_formation("u1") is None

    def test_remove_nonexistent_returns_false(self):
        mgr = FormationManager()
        assert not mgr.remove_formation("ghost")


class TestFormationManagerConfig:
    def test_set_formation_type(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.set_formation_type("alpha", FormationType.CIRCLE)
        info = mgr.formation_info("alpha")
        assert info["type"] == "circle"

    def test_set_formation_type_nonexistent_raises(self):
        mgr = FormationManager()
        with pytest.raises(KeyError):
            mgr.set_formation_type("ghost", FormationType.LINE)

    def test_set_spacing(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.LINE, spacing=3.0)
        mgr.set_spacing("alpha", 5.0)
        info = mgr.formation_info("alpha")
        assert info["spacing"] == 5.0

    def test_set_spacing_nonexistent_raises(self):
        mgr = FormationManager()
        with pytest.raises(KeyError):
            mgr.set_spacing("ghost", 5.0)


# ===========================================================================
# FormationManager — Unit Assignment
# ===========================================================================


class TestFormationManagerAssignment:
    def test_assign_unit(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "u1")
        assert mgr.unit_count == 1
        assert mgr.get_formation("u1") == "alpha"
        assert "u1" in mgr.get_members("alpha")

    def test_assign_multiple_units(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        for i in range(5):
            mgr.assign("alpha", f"u{i}")
        assert mgr.unit_count == 5
        assert len(mgr.get_members("alpha")) == 5

    def test_assign_leader(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "u1")
        mgr.assign("alpha", "u2", is_leader=True)
        assert mgr.get_leader("alpha") == "u2"
        # Leader should be first in member list
        members = mgr.get_members("alpha")
        assert members[0] == "u2"

    def test_reassign_leader(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "u1", is_leader=True)
        mgr.assign("alpha", "u2", is_leader=True)
        assert mgr.get_leader("alpha") == "u2"
        members = mgr.get_members("alpha")
        assert members[0] == "u2"
        assert "u1" in members

    def test_assign_to_nonexistent_formation_raises(self):
        mgr = FormationManager()
        with pytest.raises(KeyError):
            mgr.assign("ghost", "u1")

    def test_unassign_unit(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "u1")
        assert mgr.unassign("u1")
        assert mgr.unit_count == 0
        assert mgr.get_formation("u1") is None
        assert "u1" not in mgr.get_members("alpha")

    def test_unassign_nonexistent_returns_false(self):
        mgr = FormationManager()
        assert not mgr.unassign("ghost")

    def test_unassign_leader_promotes_next(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "u1", is_leader=True)
        mgr.assign("alpha", "u2")
        mgr.assign("alpha", "u3")
        mgr.unassign("u1")
        # u2 should now be leader (first in list)
        assert mgr.get_leader("alpha") == "u2"

    def test_reassign_to_different_formation(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.create_formation("bravo", FormationType.LINE)
        mgr.assign("alpha", "u1")
        assert mgr.get_formation("u1") == "alpha"
        mgr.assign("bravo", "u1")
        assert mgr.get_formation("u1") == "bravo"
        assert "u1" not in mgr.get_members("alpha")
        assert "u1" in mgr.get_members("bravo")
        assert mgr.unit_count == 1

    def test_assign_same_unit_twice_no_duplicate(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "u1")
        mgr.assign("alpha", "u1")
        assert mgr.get_members("alpha").count("u1") == 1


# ===========================================================================
# FormationManager — Queries
# ===========================================================================


class TestFormationManagerQueries:
    def test_get_formation_unassigned(self):
        mgr = FormationManager()
        assert mgr.get_formation("nobody") is None

    def test_get_members_nonexistent_formation(self):
        mgr = FormationManager()
        assert mgr.get_members("ghost") == []

    def test_get_leader_empty_formation(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        assert mgr.get_leader("alpha") is None

    def test_get_leader_nonexistent(self):
        mgr = FormationManager()
        assert mgr.get_leader("ghost") is None

    def test_formation_info(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE, spacing=5.0)
        mgr.assign("alpha", "u1", is_leader=True)
        mgr.assign("alpha", "u2")
        info = mgr.formation_info("alpha")
        assert info is not None
        assert info["formation_id"] == "alpha"
        assert info["type"] == "wedge"
        assert info["spacing"] == 5.0
        assert info["leader"] == "u1"
        assert info["member_count"] == 2
        assert info["members"] == ["u1", "u2"]

    def test_formation_info_nonexistent(self):
        mgr = FormationManager()
        assert mgr.formation_info("ghost") is None


# ===========================================================================
# FormationManager — Tick
# ===========================================================================


class TestFormationManagerTick:
    def test_tick_empty_manager(self):
        mgr = FormationManager()
        targets = mgr.tick(0.1, {"u1": (10.0, 0.0)})
        assert targets == {}

    def test_tick_empty_formation(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        targets = mgr.tick(0.1, {})
        assert targets == {}

    def test_tick_single_unit(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.LINE)
        mgr.assign("alpha", "u1", is_leader=True)
        positions = {"u1": (10.0, 5.0)}
        targets = mgr.tick(0.1, positions)
        assert "u1" in targets
        # Single unit should be at leader position
        assert distance(targets["u1"], (10.0, 5.0)) < 1.0

    def test_tick_multiple_units_returns_all(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.LINE, spacing=5.0)
        mgr.assign("alpha", "lead", is_leader=True)
        mgr.assign("alpha", "f1")
        mgr.assign("alpha", "f2")
        positions = {
            "lead": (0.0, 0.0),
            "f1": (-3.0, 3.0),
            "f2": (-3.0, -3.0),
        }
        targets = mgr.tick(0.1, positions)
        assert len(targets) == 3
        assert "lead" in targets
        assert "f1" in targets
        assert "f2" in targets

    def test_tick_slots_distinct(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE, spacing=4.0)
        mgr.assign("alpha", "lead", is_leader=True)
        mgr.assign("alpha", "f1")
        mgr.assign("alpha", "f2")
        mgr.assign("alpha", "f3")
        positions = {
            "lead": (50.0, 50.0),
            "f1": (48.0, 52.0),
            "f2": (48.0, 48.0),
            "f3": (46.0, 50.0),
        }
        targets = mgr.tick(0.1, positions)
        # All target positions should be distinct
        pos_list = list(targets.values())
        for i in range(len(pos_list)):
            for j in range(i + 1, len(pos_list)):
                assert distance(pos_list[i], pos_list[j]) > 0.1

    def test_tick_leader_at_slot_zero(self):
        """Leader target should be near the leader's actual position.

        In a LINE formation with 2 members at spacing=3.0, slot 0 is
        offset by half the spacing from leader_pos, so we use a larger
        tolerance that accounts for the formation geometry.
        """
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.LINE, spacing=3.0)
        mgr.assign("alpha", "lead", is_leader=True)
        mgr.assign("alpha", "f1")
        positions = {"lead": (20.0, 10.0), "f1": (18.0, 10.0)}
        targets = mgr.tick(0.1, positions)
        # Leader target should be within one spacing distance of leader pos
        assert distance(targets["lead"], (20.0, 10.0)) < 3.0

    def test_tick_missing_unit_position_skipped(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "lead", is_leader=True)
        mgr.assign("alpha", "ghost")  # no position provided
        positions = {"lead": (0.0, 0.0)}
        targets = mgr.tick(0.1, positions)
        assert "lead" in targets
        assert "ghost" not in targets

    def test_tick_missing_leader_position_skips_formation(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "lead", is_leader=True)
        mgr.assign("alpha", "f1")
        positions = {"f1": (5.0, 0.0)}  # leader position missing
        targets = mgr.tick(0.1, positions)
        assert targets == {}

    def test_tick_multiple_formations(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE, spacing=3.0)
        mgr.create_formation("bravo", FormationType.LINE, spacing=4.0)

        mgr.assign("alpha", "a1", is_leader=True)
        mgr.assign("alpha", "a2")
        mgr.assign("bravo", "b1", is_leader=True)
        mgr.assign("bravo", "b2")

        positions = {
            "a1": (0.0, 0.0),
            "a2": (-2.0, 2.0),
            "b1": (100.0, 0.0),
            "b2": (98.0, 2.0),
        }
        targets = mgr.tick(0.1, positions)
        assert len(targets) == 4
        # Alpha and bravo should be in different areas
        a_center = ((targets["a1"][0] + targets["a2"][0]) / 2,
                     (targets["a1"][1] + targets["a2"][1]) / 2)
        b_center = ((targets["b1"][0] + targets["b2"][0]) / 2,
                     (targets["b1"][1] + targets["b2"][1]) / 2)
        assert distance(a_center, b_center) > 50.0

    def test_tick_with_all_formation_types(self):
        """Every formation type should produce valid targets."""
        for ft in FormationType:
            mgr = FormationManager()
            mgr.create_formation("test", ft, spacing=3.0)
            mgr.assign("test", "lead", is_leader=True)
            mgr.assign("test", "f1")
            mgr.assign("test", "f2")
            positions = {
                "lead": (10.0, 10.0),
                "f1": (8.0, 12.0),
                "f2": (8.0, 8.0),
            }
            targets = mgr.tick(0.1, positions)
            assert len(targets) == 3, f"Failed for {ft}"
            # All targets should be finite numbers
            for uid, pos in targets.items():
                assert math.isfinite(pos[0]), f"Non-finite x for {uid} in {ft}"
                assert math.isfinite(pos[1]), f"Non-finite y for {uid} in {ft}"

    def test_tick_formation_without_explicit_leader(self):
        """If no explicit leader, first assigned member is used."""
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.COLUMN)
        mgr.assign("alpha", "u1")
        mgr.assign("alpha", "u2")
        positions = {"u1": (0.0, 0.0), "u2": (-3.0, 0.0)}
        targets = mgr.tick(0.1, positions)
        assert "u1" in targets
        assert "u2" in targets


# ===========================================================================
# FormationManager — Serialization
# ===========================================================================


class TestFormationManagerSerialization:
    def test_to_dict_empty(self):
        mgr = FormationManager()
        d = mgr.to_dict()
        assert d["formations"] == {}
        assert d["unit_assignments"] == {}

    def test_to_dict_with_data(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE, spacing=4.0)
        mgr.assign("alpha", "u1", is_leader=True)
        mgr.assign("alpha", "u2")
        d = mgr.to_dict()
        assert "alpha" in d["formations"]
        assert d["formations"]["alpha"]["type"] == "wedge"
        assert d["formations"]["alpha"]["spacing"] == 4.0
        assert d["formations"]["alpha"]["leader"] == "u1"
        assert d["formations"]["alpha"]["members"] == ["u1", "u2"]
        assert d["unit_assignments"]["u1"] == "alpha"
        assert d["unit_assignments"]["u2"] == "alpha"

    def test_from_dict_roundtrip(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE, spacing=4.0)
        mgr.create_formation("bravo", FormationType.LINE, spacing=2.0)
        mgr.assign("alpha", "u1", is_leader=True)
        mgr.assign("alpha", "u2")
        mgr.assign("bravo", "u3", is_leader=True)

        d = mgr.to_dict()
        mgr2 = FormationManager.from_dict(d)

        assert mgr2.formation_count == 2
        assert mgr2.unit_count == 3
        assert mgr2.get_leader("alpha") == "u1"
        assert mgr2.get_leader("bravo") == "u3"
        assert mgr2.get_formation("u2") == "alpha"

    def test_from_dict_empty(self):
        mgr = FormationManager.from_dict({})
        assert mgr.formation_count == 0
        assert mgr.unit_count == 0


# ===========================================================================
# FormationManager — Edge Cases
# ===========================================================================


class TestFormationManagerEdgeCases:
    def test_remove_all_members_then_tick(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "u1")
        mgr.unassign("u1")
        targets = mgr.tick(0.1, {"u1": (0.0, 0.0)})
        assert targets == {}

    def test_remove_formation_cleans_up_units(self):
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "u1")
        mgr.assign("alpha", "u2")
        mgr.remove_formation("alpha")
        assert mgr.get_formation("u1") is None
        assert mgr.get_formation("u2") is None

    def test_large_formation(self):
        mgr = FormationManager()
        mgr.create_formation("big", FormationType.CIRCLE, spacing=2.0)
        for i in range(50):
            mgr.assign("big", f"u{i}", is_leader=(i == 0))
        assert mgr.unit_count == 50
        assert len(mgr.get_members("big")) == 50

        positions = {f"u{i}": (float(i), 0.0) for i in range(50)}
        targets = mgr.tick(0.1, positions)
        assert len(targets) == 50

    def test_tick_unmanaged_units_ignored(self):
        """Units not in any formation are not in the output."""
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.LINE)
        mgr.assign("alpha", "u1", is_leader=True)
        positions = {"u1": (0.0, 0.0), "rogue": (99.0, 99.0)}
        targets = mgr.tick(0.1, positions)
        assert "u1" in targets
        assert "rogue" not in targets

    def test_assign_promote_existing_to_leader(self):
        """Calling assign with is_leader on existing member promotes them."""
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE)
        mgr.assign("alpha", "u1")
        mgr.assign("alpha", "u2")
        mgr.assign("alpha", "u3")
        mgr.assign("alpha", "u3", is_leader=True)
        assert mgr.get_leader("alpha") == "u3"
        assert mgr.get_members("alpha")[0] == "u3"

    def test_zero_spacing(self):
        """All slots collapse to leader position with spacing=0."""
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.LINE, spacing=0.0)
        mgr.assign("alpha", "u1", is_leader=True)
        mgr.assign("alpha", "u2")
        mgr.assign("alpha", "u3")
        positions = {"u1": (10.0, 20.0), "u2": (8.0, 18.0), "u3": (12.0, 22.0)}
        targets = mgr.tick(0.1, positions)
        # All should be at or very near leader position
        for uid in ["u1", "u2", "u3"]:
            assert distance(targets[uid], (10.0, 20.0)) < 1.0


# ===========================================================================
# Integration — FormationManager + FormationMover
# ===========================================================================


class TestFormationManagerIntegration:
    def test_manager_and_mover_produce_same_slot_count(self):
        """Both systems should agree on slot count for the same input."""
        config = FormationConfig(
            formation_type=FormationType.WEDGE,
            spacing=3.0,
            leader_pos=(0.0, 0.0),
            num_members=4,
        )
        raw_slots = get_formation_positions(config)

        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.WEDGE, spacing=3.0)
        mgr.assign("alpha", "u0", is_leader=True)
        mgr.assign("alpha", "u1")
        mgr.assign("alpha", "u2")
        mgr.assign("alpha", "u3")

        positions = {
            "u0": (0.0, 0.0),
            "u1": (-2.0, 2.0),
            "u2": (-2.0, -2.0),
            "u3": (-4.0, 0.0),
        }
        targets = mgr.tick(0.1, positions)

        assert len(raw_slots) == len(targets) == 4

    def test_manager_preserves_formation_shape(self):
        """Relative distances between manager targets should match raw formation."""
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.LINE, spacing=5.0)
        mgr.assign("alpha", "u0", is_leader=True)
        mgr.assign("alpha", "u1")
        mgr.assign("alpha", "u2")

        positions = {
            "u0": (50.0, 50.0),
            "u1": (48.0, 55.0),
            "u2": (48.0, 45.0),
        }
        targets = mgr.tick(0.1, positions)

        # In a LINE formation, members should be equally spaced
        t_list = [targets["u0"], targets["u1"], targets["u2"]]
        d01 = distance(t_list[0], t_list[1])
        d12 = distance(t_list[1], t_list[2])
        # Both inter-slot distances should be the same
        assert abs(d01 - d12) < 1.0

    def test_change_type_affects_tick(self):
        """Changing formation type should produce different slot positions."""
        mgr = FormationManager()
        mgr.create_formation("alpha", FormationType.LINE, spacing=5.0)
        mgr.assign("alpha", "u0", is_leader=True)
        mgr.assign("alpha", "u1")
        mgr.assign("alpha", "u2")

        positions = {
            "u0": (0.0, 0.0),
            "u1": (0.0, 5.0),
            "u2": (0.0, -5.0),
        }
        targets_line = mgr.tick(0.1, positions)

        mgr.set_formation_type("alpha", FormationType.COLUMN)
        targets_col = mgr.tick(0.1, positions)

        # At least one follower target should be different
        assert targets_line["u1"] != targets_col["u1"] or targets_line["u2"] != targets_col["u2"]
