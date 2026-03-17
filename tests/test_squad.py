"""Tests for squad coordination AI.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.ai.squad import (
    MoralePropagation,
    Order,
    Squad,
    SquadRole,
    SquadState,
    SquadTactics,
)
from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_squad(n: int = 4, name: str = "Alpha") -> tuple[Squad, dict[str, Vec2]]:
    """Create a squad with *n* members at grid positions."""
    sq = Squad("sq1", name, "friendly")
    positions: dict[str, Vec2] = {}
    for i in range(n):
        uid = f"u{i}"
        role = SquadRole.LEADER if i == 0 else SquadRole.RIFLEMAN
        sq.add_member(uid, role)
        positions[uid] = (float(i * 3), 0.0)
    return sq, positions


# ---------------------------------------------------------------------------
# Squad basics
# ---------------------------------------------------------------------------


class TestSquadMembership:
    def test_add_member(self):
        sq = Squad("s1", "Bravo", "friendly")
        sq.add_member("a", SquadRole.LEADER)
        assert "a" in sq.members
        assert sq.roles["a"] == SquadRole.LEADER
        assert sq.leader_id == "a"

    def test_first_member_auto_leader(self):
        sq = Squad("s1", "Bravo", "friendly")
        sq.add_member("a", SquadRole.RIFLEMAN)
        # First member should be auto-promoted to leader
        assert sq.leader_id == "a"
        assert sq.roles["a"] == SquadRole.LEADER

    def test_duplicate_add_ignored(self):
        sq = Squad("s1", "Bravo", "friendly")
        sq.add_member("a")
        sq.add_member("a")
        assert sq.members.count("a") == 1

    def test_remove_member(self):
        sq, _ = make_squad(3)
        sq.remove_member("u1")
        assert "u1" not in sq.members
        assert "u1" not in sq.roles

    def test_remove_increments_casualties(self):
        sq, _ = make_squad(3)
        sq.remove_member("u2")
        assert sq.state.casualties == 1

    def test_remove_nonexistent_noop(self):
        sq, _ = make_squad(2)
        sq.remove_member("ghost")
        assert sq.state.casualties == 0

    def test_remove_all_members(self):
        sq, _ = make_squad(2)
        sq.remove_member("u0")
        sq.remove_member("u1")
        assert len(sq.members) == 0
        assert sq.leader_id is None


class TestLeaderAutoPromotion:
    def test_promote_on_leader_death(self):
        sq = Squad("s1", "Bravo", "friendly")
        sq.add_member("lead", SquadRole.LEADER)
        sq.add_member("sup", SquadRole.SUPPORT)
        sq.add_member("rfl", SquadRole.RIFLEMAN)
        sq.remove_member("lead")
        # Support has higher promotion priority
        assert sq.leader_id == "sup"
        assert sq.roles["sup"] == SquadRole.LEADER

    def test_promote_engineer_over_rifleman(self):
        sq = Squad("s1", "Bravo", "friendly")
        sq.add_member("lead", SquadRole.LEADER)
        sq.add_member("eng", SquadRole.ENGINEER)
        sq.add_member("rfl", SquadRole.RIFLEMAN)
        sq.remove_member("lead")
        assert sq.leader_id == "eng"

    def test_promote_fallback_first_member(self):
        sq = Squad("s1", "Bravo", "friendly")
        sq.add_member("lead", SquadRole.LEADER)
        # Second member also leader role, but stored as leader
        sq.add_member("scout", SquadRole.SCOUT)
        sq.add_member("medic", SquadRole.MEDIC)
        sq.remove_member("lead")
        # Scout and medic are lowest priority; scout comes first by list order
        # Actually medic has lower priority than scout, so scout promoted
        assert sq.leader_id == "scout"

    def test_no_promote_empty_squad(self):
        sq = Squad("s1", "Bravo", "friendly")
        sq.add_member("a", SquadRole.LEADER)
        sq.remove_member("a")
        assert sq.leader_id is None


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


class TestOrders:
    def test_issue_order(self):
        sq, _ = make_squad(2)
        o = Order(order_type="advance", target_pos=(10.0, 20.0))
        sq.issue_order(o)
        assert sq.current_order is o

    def test_order_history(self):
        sq, _ = make_squad(2)
        o1 = Order(order_type="advance")
        o2 = Order(order_type="hold")
        o3 = Order(order_type="retreat")
        sq.issue_order(o1)
        sq.issue_order(o2)
        sq.issue_order(o3)
        assert sq.current_order is o3
        assert len(sq.order_history) == 2
        assert sq.order_history[0] is o1
        assert sq.order_history[1] is o2

    def test_order_fields(self):
        o = Order(
            order_type="flank_left",
            target_pos=(5.0, 5.0),
            target_id="enemy1",
            priority=7,
            issued_at=100.0,
        )
        assert o.order_type == "flank_left"
        assert o.target_id == "enemy1"
        assert o.priority == 7
        assert o.issued_at == 100.0


# ---------------------------------------------------------------------------
# Spatial queries
# ---------------------------------------------------------------------------


class TestSpatialQueries:
    def test_center_of_mass(self):
        sq, pos = make_squad(3)
        com = sq.center_of_mass(pos)
        assert abs(com[0] - 3.0) < 0.01  # (0+3+6)/3
        assert abs(com[1] - 0.0) < 0.01

    def test_center_of_mass_empty(self):
        sq = Squad("s1", "X", "friendly")
        com = sq.center_of_mass({})
        assert com == (0.0, 0.0)

    def test_center_of_mass_missing_positions(self):
        sq, pos = make_squad(3)
        # Only provide position for one member
        com = sq.center_of_mass({"u0": (10.0, 20.0)})
        assert abs(com[0] - 10.0) < 0.01
        assert abs(com[1] - 20.0) < 0.01

    def test_spread(self):
        sq, pos = make_squad(3)
        s = sq.spread(pos)
        assert abs(s - 6.0) < 0.01  # u0 at 0, u2 at 6

    def test_spread_single_member(self):
        sq, pos = make_squad(1)
        assert sq.spread(pos) == 0.0

    def test_spread_empty(self):
        sq = Squad("s1", "X", "friendly")
        assert sq.spread({}) == 0.0

    def test_update_cohesion_tight(self):
        sq, _ = make_squad(3)
        # All at same spot
        pos = {"u0": (5.0, 5.0), "u1": (5.0, 5.0), "u2": (5.0, 5.0)}
        sq.update_cohesion(pos)
        assert sq.state.cohesion == 1.0

    def test_update_cohesion_spread_out(self):
        sq, _ = make_squad(3)
        # Very spread out
        pos = {"u0": (0.0, 0.0), "u1": (100.0, 0.0), "u2": (50.0, 100.0)}
        sq.update_cohesion(pos)
        assert sq.state.cohesion < 0.5

    def test_update_cohesion_single(self):
        sq, pos = make_squad(1)
        sq.update_cohesion(pos)
        assert sq.state.cohesion == 1.0


# ---------------------------------------------------------------------------
# Morale
# ---------------------------------------------------------------------------


class TestMorale:
    def test_update_morale_high(self):
        sq, _ = make_squad(3)
        states = {"u0": 0.9, "u1": 0.8, "u2": 1.0}
        sq.update_morale(states)
        assert sq.state.morale > 0.7

    def test_update_morale_low_with_casualties(self):
        sq, _ = make_squad(3)
        sq.state.casualties = 3  # Heavy losses
        states = {"u0": 0.5, "u1": 0.5, "u2": 0.5}
        sq.update_morale(states)
        assert sq.state.morale < 0.5

    def test_update_morale_no_living(self):
        sq, _ = make_squad(2)
        sq.update_morale({})  # No states for anyone
        assert sq.state.morale == 0.0

    def test_should_retreat_low_morale(self):
        sq, _ = make_squad(3)
        sq.state.morale = 0.2
        assert sq.should_retreat() is True

    def test_should_not_retreat_ok_morale(self):
        sq, _ = make_squad(3)
        sq.state.morale = 0.5
        assert sq.should_retreat() is False

    def test_should_retreat_high_casualties(self):
        sq = Squad("s1", "X", "friendly")
        sq.add_member("u0", SquadRole.LEADER)
        sq.state.casualties = 3  # 3 dead out of 4 total
        sq.state.morale = 0.8  # Morale still ok
        assert sq.should_retreat() is True

    def test_should_retreat_boundary(self):
        sq, _ = make_squad(3)
        sq.state.morale = 0.3
        # Exactly at boundary: morale < 0.3 is False (0.3 is NOT < 0.3)
        assert sq.should_retreat() is False


# ---------------------------------------------------------------------------
# Threat sharing
# ---------------------------------------------------------------------------


class TestThreatSharing:
    def test_share_threat(self):
        sq, _ = make_squad(2)
        sq.share_threat((10.0, 20.0), "e1")
        assert len(sq.state.known_threats) == 1
        assert sq.state.known_threats[0] == ((10.0, 20.0), "e1")

    def test_share_threat_updates_position(self):
        sq, _ = make_squad(2)
        sq.share_threat((10.0, 20.0), "e1")
        sq.share_threat((15.0, 25.0), "e1")
        assert len(sq.state.known_threats) == 1
        assert sq.state.known_threats[0][0] == (15.0, 25.0)

    def test_share_multiple_threats(self):
        sq, _ = make_squad(2)
        sq.share_threat((10.0, 10.0), "e1")
        sq.share_threat((20.0, 20.0), "e2")
        assert len(sq.state.known_threats) == 2

    def test_share_threat_raises_alert(self):
        sq, _ = make_squad(2)
        assert sq.state.alert_level == 0.0
        sq.share_threat((10.0, 10.0), "e1")
        assert sq.state.alert_level == pytest.approx(0.2)
        sq.share_threat((20.0, 20.0), "e2")
        assert sq.state.alert_level == pytest.approx(0.4)

    def test_alert_level_caps_at_1(self):
        sq, _ = make_squad(2)
        for i in range(10):
            sq.share_threat((float(i), 0.0), f"e{i}")
        assert sq.state.alert_level <= 1.0


# ---------------------------------------------------------------------------
# Formations
# ---------------------------------------------------------------------------


class TestFormations:
    def test_line_formation_count(self):
        sq, pos = make_squad(4)
        targets = SquadTactics.compute_formation(sq, pos, "line", spacing=5.0)
        assert len(targets) == 4

    def test_line_formation_spread(self):
        sq, pos = make_squad(4)
        targets = SquadTactics.compute_formation(sq, pos, "line", spacing=5.0)
        xs = sorted(t[0] for t in targets.values())
        # Should be evenly spaced at 5.0 apart
        for i in range(1, len(xs)):
            assert abs(xs[i] - xs[i - 1] - 5.0) < 0.01

    def test_line_formation_same_y(self):
        sq, pos = make_squad(4)
        targets = SquadTactics.compute_formation(sq, pos, "line")
        ys = [t[1] for t in targets.values()]
        assert all(abs(y - ys[0]) < 0.01 for y in ys)

    def test_column_formation_same_x(self):
        sq, pos = make_squad(4)
        targets = SquadTactics.compute_formation(sq, pos, "column")
        xs = [t[0] for t in targets.values()]
        assert all(abs(x - xs[0]) < 0.01 for x in xs)

    def test_wedge_leader_forward(self):
        sq, pos = make_squad(5)
        targets = SquadTactics.compute_formation(sq, pos, "wedge", spacing=4.0)
        # First member (leader) should have highest y
        leader_y = targets["u0"][1]
        for uid, tgt in targets.items():
            if uid != "u0":
                assert tgt[1] <= leader_y + 0.01

    def test_diamond_four_members(self):
        sq, pos = make_squad(4)
        targets = SquadTactics.compute_formation(sq, pos, "diamond", spacing=5.0)
        assert len(targets) == 4
        # Front, left, right, back distinct positions
        unique = set()
        for p in targets.values():
            unique.add((round(p[0], 2), round(p[1], 2)))
        assert len(unique) == 4

    def test_circle_formation_radius(self):
        sq, pos = make_squad(6)
        targets = SquadTactics.compute_formation(sq, pos, "circle", spacing=10.0)
        com = sq.center_of_mass(pos)
        for tgt in targets.values():
            d = distance(com, tgt)
            assert abs(d - 10.0) < 0.01

    def test_circle_formation_even_spacing(self):
        sq, pos = make_squad(4)
        targets = SquadTactics.compute_formation(sq, pos, "circle", spacing=5.0)
        pts = list(targets.values())
        # Adjacent units should be roughly equidistant
        dists = []
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            dists.append(distance(pts[i], pts[j]))
        for d in dists:
            assert abs(d - dists[0]) < 0.1

    def test_empty_squad_formation(self):
        sq = Squad("s1", "X", "friendly")
        targets = SquadTactics.compute_formation(sq, {}, "line")
        assert targets == {}

    def test_single_member_formation(self):
        sq, pos = make_squad(1)
        targets = SquadTactics.compute_formation(sq, pos, "line")
        assert len(targets) == 1

    def test_unknown_formation_stays_put(self):
        sq, pos = make_squad(3)
        targets = SquadTactics.compute_formation(sq, pos, "nonsense")
        for uid in sq.members:
            assert targets[uid] == pos[uid]


# ---------------------------------------------------------------------------
# Fire sectors
# ---------------------------------------------------------------------------


class TestFireSectors:
    def test_sectors_with_threats(self):
        sq, pos = make_squad(3)
        threats = [((50.0, 0.0), "e1")]
        sectors = SquadTactics.assign_fire_sectors(sq, pos, threats)
        assert len(sectors) == 3
        # All should point roughly toward the threat (positive x)
        for uid, direction in sectors.items():
            assert direction[0] > 0.5

    def test_sectors_no_threats(self):
        sq, pos = make_squad(4)
        sectors = SquadTactics.assign_fire_sectors(sq, pos, [])
        assert len(sectors) == 4
        # Should be unit vectors
        for d in sectors.values():
            mag = math.hypot(d[0], d[1])
            assert abs(mag - 1.0) < 0.01

    def test_sectors_multiple_threats(self):
        sq, pos = make_squad(4)
        threats = [((50.0, 0.0), "e1"), ((0.0, 50.0), "e2")]
        sectors = SquadTactics.assign_fire_sectors(sq, pos, threats)
        assert len(sectors) == 4

    def test_sectors_empty_squad(self):
        sq = Squad("s1", "X", "friendly")
        sectors = SquadTactics.assign_fire_sectors(sq, {}, [])
        assert sectors == {}


# ---------------------------------------------------------------------------
# Bounding overwatch
# ---------------------------------------------------------------------------


class TestBoundingOverwatch:
    def test_split_even(self):
        sq, pos = make_squad(4)
        moving, covering = SquadTactics.bounding_overwatch(sq, pos, (1.0, 0.0))
        assert len(moving) == 2
        assert len(covering) == 2
        assert set(moving + covering) == {"u0", "u1", "u2", "u3"}

    def test_split_odd(self):
        sq, pos = make_squad(5)
        moving, covering = SquadTactics.bounding_overwatch(sq, pos, (1.0, 0.0))
        assert len(moving) + len(covering) == 5

    def test_single_member(self):
        sq, pos = make_squad(1)
        moving, covering = SquadTactics.bounding_overwatch(sq, pos, (1.0, 0.0))
        assert len(moving) == 1
        assert len(covering) == 0

    def test_empty_squad(self):
        sq = Squad("s1", "X", "friendly")
        moving, covering = SquadTactics.bounding_overwatch(sq, {}, (1.0, 0.0))
        assert moving == []
        assert covering == []

    def test_forward_units_move(self):
        sq, _ = make_squad(4)
        # u3 at x=9 is furthest forward in +x direction
        pos = {"u0": (0.0, 0.0), "u1": (3.0, 0.0), "u2": (6.0, 0.0), "u3": (9.0, 0.0)}
        moving, covering = SquadTactics.bounding_overwatch(sq, pos, (1.0, 0.0))
        # Forward units (u2, u3) should be in moving group
        assert "u3" in moving
        assert "u2" in moving


# ---------------------------------------------------------------------------
# Order recommendation
# ---------------------------------------------------------------------------


class TestRecommendOrder:
    def test_retreat_when_broken(self):
        sq, pos = make_squad(3)
        sq.state.morale = 0.1
        order = SquadTactics.recommend_order(sq, pos, [])
        assert order.order_type == "retreat"

    def test_patrol_when_relaxed(self):
        sq, pos = make_squad(4)
        sq.state.alert_level = 0.0
        order = SquadTactics.recommend_order(sq, pos, [])
        assert order.order_type == "patrol"

    def test_guard_when_alert_no_threats(self):
        sq, pos = make_squad(4)
        sq.state.alert_level = 0.5
        order = SquadTactics.recommend_order(sq, pos, [])
        assert order.order_type == "guard"

    def test_advance_close_threat(self):
        sq, pos = make_squad(4)
        sq.state.morale = 0.9
        threats = [((10.0, 0.0), "e1")]
        order = SquadTactics.recommend_order(sq, pos, threats)
        assert order.order_type == "advance"
        assert order.target_id == "e1"

    def test_suppress_medium_range_low_morale(self):
        sq, pos = make_squad(2)
        sq.state.morale = 0.4
        threats = [((40.0, 0.0), "e1")]
        order = SquadTactics.recommend_order(sq, pos, threats)
        assert order.order_type == "suppress"

    def test_hold_low_ammo(self):
        sq, pos = make_squad(4)
        sq.state.ammo_status = 0.1
        sq.state.morale = 0.8
        order = SquadTactics.recommend_order(sq, pos, [((30.0, 0.0), "e1")])
        assert order.order_type == "hold"

    def test_flank_medium_range_good_morale(self):
        sq, pos = make_squad(4)
        sq.state.morale = 0.8
        threats = [((35.0, 0.0), "e1")]
        order = SquadTactics.recommend_order(sq, pos, threats)
        assert order.order_type == "flank_left"

    def test_uses_known_threats(self):
        sq, pos = make_squad(4)
        sq.state.morale = 0.9
        sq.share_threat((10.0, 0.0), "e1")
        order = SquadTactics.recommend_order(sq, pos, [])
        assert order.target_id == "e1"


# ---------------------------------------------------------------------------
# Morale propagation
# ---------------------------------------------------------------------------


class TestMoralePropagation:
    def test_basic_propagation(self):
        sq, _ = make_squad(3)
        sq.state.morale = 0.5
        states = {"u0": 0.9, "u1": 0.9, "u2": 0.9}
        MoralePropagation.propagate([sq], states, dt=1.0)
        # Morale should increase toward unit average
        assert sq.state.morale > 0.5

    def test_casualties_reduce_morale(self):
        sq, _ = make_squad(3)
        sq.state.morale = 0.8
        sq.state.casualties = 5
        states = {"u0": 0.8, "u1": 0.8, "u2": 0.8}
        MoralePropagation.propagate([sq], states, dt=1.0)
        assert sq.state.morale < 0.8

    def test_suppression_reduces_morale(self):
        sq, _ = make_squad(3)
        sq.state.morale = 0.8
        sq.state.alert_level = 1.0
        states = {"u0": 0.9, "u1": 0.9, "u2": 0.9}
        MoralePropagation.propagate([sq], states, dt=1.0)
        # Alert level acts as suppression penalty
        assert sq.state.morale < 0.9

    def test_leader_rally_bonus(self):
        sq, _ = make_squad(3)
        sq.state.morale = 0.5
        sq.state.cohesion = 1.0
        states_with_leader = {"u0": 0.7, "u1": 0.7, "u2": 0.7}
        sq2, _ = make_squad(3)
        sq2.state.morale = 0.5
        sq2.state.cohesion = 1.0
        sq2.leader_id = None  # No leader
        states_no_leader = {"u0": 0.7, "u1": 0.7, "u2": 0.7}
        MoralePropagation.propagate([sq], states_with_leader, dt=1.0)
        MoralePropagation.propagate([sq2], states_no_leader, dt=1.0)
        # Squad with leader should have slightly higher morale
        assert sq.state.morale >= sq2.state.morale

    def test_low_cohesion_slows_recovery(self):
        sq1, _ = make_squad(3)
        sq1.state.morale = 0.3
        sq1.state.cohesion = 1.0
        sq2, _ = make_squad(3)
        sq2.state.morale = 0.3
        sq2.state.cohesion = 0.1
        states = {"u0": 0.9, "u1": 0.9, "u2": 0.9}
        MoralePropagation.propagate([sq1], dict(states), dt=1.0)
        MoralePropagation.propagate([sq2], dict(states), dt=1.0)
        assert sq1.state.morale > sq2.state.morale

    def test_empty_squad_zero_morale(self):
        sq = Squad("s1", "X", "friendly")
        sq.state.morale = 0.5
        MoralePropagation.propagate([sq], {}, dt=1.0)
        assert sq.state.morale == 0.0

    def test_multiple_squads(self):
        sq1, _ = make_squad(2)
        sq2 = Squad("s2", "Bravo", "friendly")
        sq2.add_member("b0", SquadRole.LEADER)
        sq2.add_member("b1", SquadRole.RIFLEMAN)
        sq1.state.morale = 0.5
        sq2.state.morale = 0.5
        states = {"u0": 0.9, "u1": 0.9, "b0": 0.3, "b1": 0.3}
        MoralePropagation.propagate([sq1, sq2], states, dt=1.0)
        assert sq1.state.morale > sq2.state.morale

    def test_morale_clamped_0_1(self):
        sq, _ = make_squad(2)
        sq.state.morale = 0.99
        states = {"u0": 1.0, "u1": 1.0}
        MoralePropagation.propagate([sq], states, dt=10.0)
        assert sq.state.morale <= 1.0
        assert sq.state.morale >= 0.0


# ---------------------------------------------------------------------------
# SquadState dataclass
# ---------------------------------------------------------------------------


class TestSquadState:
    def test_defaults(self):
        s = SquadState()
        assert s.cohesion == 1.0
        assert s.morale == 1.0
        assert s.alert_level == 0.0
        assert s.known_threats == []
        assert s.casualties == 0
        assert s.ammo_status == 1.0

    def test_custom(self):
        s = SquadState(morale=0.5, casualties=2)
        assert s.morale == 0.5
        assert s.casualties == 2


# ---------------------------------------------------------------------------
# SquadRole enum
# ---------------------------------------------------------------------------


class TestSquadRole:
    def test_all_roles(self):
        roles = set(SquadRole)
        assert len(roles) == 6
        assert SquadRole.LEADER in roles
        assert SquadRole.MEDIC in roles

    def test_role_values(self):
        assert SquadRole.LEADER.value == "leader"
        assert SquadRole.SCOUT.value == "scout"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_formation_single_member_all_types(self):
        sq, pos = make_squad(1)
        for fmt in ("line", "column", "wedge", "diamond", "circle"):
            targets = SquadTactics.compute_formation(sq, pos, fmt)
            assert len(targets) == 1

    def test_overwatch_two_members(self):
        sq, pos = make_squad(2)
        moving, covering = SquadTactics.bounding_overwatch(sq, pos, (0.0, 1.0))
        assert len(moving) == 1
        assert len(covering) == 1

    def test_recommend_order_retreat_high_casualties(self):
        sq = Squad("s1", "X", "friendly")
        sq.add_member("u0", SquadRole.LEADER)
        sq.state.casualties = 5
        sq.state.morale = 0.8
        pos = {"u0": (0.0, 0.0)}
        order = SquadTactics.recommend_order(sq, pos, [])
        assert order.order_type == "retreat"

    def test_fire_sectors_unit_vectors(self):
        sq, pos = make_squad(4)
        threats = [((100.0, 0.0), "e1")]
        sectors = SquadTactics.assign_fire_sectors(sq, pos, threats)
        for d in sectors.values():
            mag = math.hypot(d[0], d[1])
            assert abs(mag - 1.0) < 0.01

    def test_center_of_mass_single(self):
        sq, pos = make_squad(1)
        com = sq.center_of_mass(pos)
        assert com == (0.0, 0.0)
