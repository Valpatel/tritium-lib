# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.units — all unit types."""

from tritium_lib.sim_engine.unit_types import (
    UnitType,
    CombatStats,
    MovementCategory,
    get_type,
    all_types,
    mobile_type_ids,
    static_type_ids,
    flying_type_ids,
    ground_type_ids,
    foot_type_ids,
    dispatchable_type_ids,
    get_cot_type,
    cot_type_for_target,
)


def test_all_types_discovered():
    """All unit types are discovered and registered."""
    types = all_types()
    assert len(types) >= 16  # 7 people + 7 robots + 2 sensors


def test_all_types_have_required_fields():
    """Every unit type has name, health, and speed."""
    for cls in all_types():
        assert hasattr(cls, "type_id"), f"{cls} missing type_id"
        assert hasattr(cls, "display_name"), f"{cls} missing display_name"
        assert hasattr(cls, "speed"), f"{cls} missing speed"
        assert hasattr(cls, "combat"), f"{cls} missing combat"
        assert isinstance(cls.combat, CombatStats), f"{cls} combat is not CombatStats"
        assert cls.combat.health > 0, f"{cls} has 0 health"


def test_get_type():
    """get_type returns correct unit types."""
    rover = get_type("rover")
    assert rover is not None
    assert rover.type_id == "rover"
    assert rover.speed == 2.0


def test_get_type_alias():
    """get_type resolves aliases."""
    hp = get_type("person_hostile")
    assert hp is not None
    assert hp.type_id == "hostile_person"


def test_get_type_unknown():
    """get_type returns None for unknown types."""
    assert get_type("nonexistent") is None


def test_mobile_vs_static():
    """Mobile and static sets are disjoint and complete."""
    mobile = mobile_type_ids()
    static = static_type_ids()
    assert len(mobile & static) == 0  # No overlap
    all_ids = {cls.type_id for cls in all_types()}
    assert mobile | static == all_ids


def test_flying_types():
    """Flying types include drone and swarm_drone."""
    flying = flying_type_ids()
    assert "drone" in flying
    assert "swarm_drone" in flying


def test_ground_types():
    """Ground types include rover, tank, apc."""
    ground = ground_type_ids()
    assert "rover" in ground
    assert "tank" in ground
    assert "apc" in ground


def test_foot_types():
    """Foot types include person, hostile_person."""
    foot = foot_type_ids()
    assert "person" in foot
    assert "hostile_person" in foot


def test_dispatchable_types():
    """Dispatchable types are mobile + placeable."""
    disp = dispatchable_type_ids()
    assert "rover" in disp
    assert "drone" in disp
    # People are not dispatchable
    assert "person" not in disp
    assert "hostile_person" not in disp


def test_cot_type():
    """get_cot_type returns CoT codes."""
    code = get_cot_type("turret")
    assert code is not None
    assert code.startswith("a-")


def test_cot_type_for_target():
    """cot_type_for_target swaps affiliation."""
    code = cot_type_for_target("person", "hostile")
    assert code is not None
    assert "h" in code  # hostile affiliation


def test_unit_type_helpers():
    """UnitType helper methods work."""
    rover = get_type("rover")
    assert rover.is_mobile()
    assert rover.is_ground()
    assert not rover.is_flying()
    assert not rover.is_stationary()

    turret = get_type("turret")
    assert not turret.is_mobile()
    assert turret.is_stationary()
