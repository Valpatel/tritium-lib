# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.core.inventory."""

from tritium_lib.sim_engine.core.inventory import (
    ITEM_CATALOG,
    InventoryItem,
    UnitInventory,
    build_loadout,
    select_best_weapon,
)


class TestInventoryItem:
    def test_weapon_creation(self):
        w = InventoryItem(
            item_id="w1", item_type="weapon", name="Pistol",
            damage=10.0, range=15.0, cooldown=1.0, ammo=30, max_ammo=30,
        )
        assert w.has_ammo()
        assert w.weapon_range == 15.0

    def test_weapon_range_alias(self):
        w = InventoryItem(item_id="w1", item_type="weapon")
        w.weapon_range = 25.0
        assert w.range == 25.0

    def test_armor_functional(self):
        a = InventoryItem(
            item_id="a1", item_type="armor",
            damage_reduction=0.2, durability=50,
        )
        assert a.is_functional()
        # Deplete all durability
        for _ in range(50):
            a.take_hit()
        assert not a.is_functional()
        # Extra hit doesn't go negative
        a.take_hit()
        assert a.durability == 0

    def test_consumable_use(self):
        c = InventoryItem(
            item_id="c1", item_type="consumable",
            effect_type="heal", effect_value=25.0, uses=2, max_uses=2,
        )
        assert c.can_use()
        val = c.use()
        assert val == 25.0
        assert c.uses == 1
        c.use()
        assert not c.can_use()
        assert c.use() == 0.0

    def test_unlimited_ammo(self):
        w = InventoryItem(item_id="w1", item_type="weapon", ammo=-1)
        assert w.has_ammo()  # -1 = unlimited

    def test_empty_ammo(self):
        w = InventoryItem(item_id="w1", item_type="weapon", ammo=0)
        assert not w.has_ammo()

    def test_to_dict_weapon(self):
        w = InventoryItem(
            item_id="w1", item_type="weapon", name="Gun",
            damage=10.0, range=20.0, weapon_class="projectile",
        )
        d = w.to_dict()
        assert d["item_type"] == "weapon"
        assert d["damage"] == 10.0
        assert "weapon_class" in d

    def test_to_dict_armor(self):
        a = InventoryItem(
            item_id="a1", item_type="armor",
            damage_reduction=0.3, durability=100,
        )
        d = a.to_dict()
        assert "damage_reduction" in d
        assert "durability" in d

    def test_to_dict_consumable(self):
        c = InventoryItem(
            item_id="c1", item_type="consumable",
            effect_type="heal", effect_value=50.0,
        )
        d = c.to_dict()
        assert "effect_type" in d

    def test_to_dict_device(self):
        dev = InventoryItem(
            item_id="d1", item_type="device",
            ble_mac="AA:BB:CC:DD:EE:FF", device_class="phone",
        )
        d = dev.to_dict()
        assert d["ble_mac"] == "AA:BB:CC:DD:EE:FF"
        assert d["device_class"] == "phone"

    def test_to_fog_dict(self):
        w = InventoryItem(item_id="w1", item_type="weapon", damage=100.0)
        fog = w.to_fog_dict()
        assert fog["status"] == "unknown"
        assert "damage" not in fog

    def test_grenade_to_dict(self):
        g = InventoryItem(
            item_id="g1", item_type="grenade",
            damage=40.0, blast_radius=5.0, count=2,
        )
        d = g.to_dict()
        assert d["blast_radius"] == 5.0
        assert d["count"] == 2


class TestUnitInventory:
    def test_add_item(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="sword", item_type="weapon"))
        assert len(inv.items) == 1

    def test_auto_equip_first_weapon(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="gun1", item_type="weapon"))
        assert inv.active_weapon_id == "gun1"

    def test_get_item(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="item1"))
        assert inv.get_item("item1") is not None
        assert inv.get_item("nonexistent") is None

    def test_remove_item(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="gun1", item_type="weapon"))
        removed = inv.remove_item("gun1")
        assert removed is not None
        assert len(inv.items) == 0
        assert inv.active_weapon_id is None

    def test_remove_nonexistent(self):
        inv = UnitInventory(owner_id="u1")
        assert inv.remove_item("nothing") is None

    def test_total_damage_reduction(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(
            item_id="vest", item_type="armor",
            damage_reduction=0.2, durability=50,
        ))
        assert inv.total_damage_reduction() == 0.2

    def test_damage_reduction_cap(self):
        """Multiple armor pieces stack but cap at 0.8."""
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(
            item_id="v1", item_type="armor", damage_reduction=0.5, durability=50,
        ))
        inv.add_item(InventoryItem(
            item_id="v2", item_type="armor", damage_reduction=0.5, durability=50,
        ))
        assert inv.total_damage_reduction() == 0.8

    def test_damage_armor(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(
            item_id="vest", item_type="armor",
            damage_reduction=0.3, durability=10,
        ))
        reduction = inv.damage_armor(hits=3)
        assert reduction == 0.3
        assert inv.get_item("vest").durability == 7

    def test_switch_weapon(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="g1", item_type="weapon"))
        inv.add_item(InventoryItem(item_id="g2", item_type="weapon"))
        assert inv.switch_weapon("g2")
        assert inv.active_weapon_id == "g2"

    def test_switch_weapon_invalid(self):
        inv = UnitInventory(owner_id="u1")
        assert inv.switch_weapon("nonexistent") is False

    def test_auto_switch_weapon(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="g1", item_type="weapon", ammo=0))
        inv.add_item(InventoryItem(item_id="g2", item_type="weapon", ammo=10))
        inv.active_weapon_id = "g1"
        assert inv.auto_switch_weapon()
        assert inv.active_weapon_id == "g2"

    def test_has_ammo(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="g1", item_type="weapon", ammo=0))
        assert not inv.has_ammo()
        inv.add_item(InventoryItem(item_id="g2", item_type="weapon", ammo=10))
        assert inv.has_ammo()

    def test_consume_grenade(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(
            item_id="frag", item_type="grenade",
            damage=40.0, blast_radius=5.0, count=2,
        ))
        g = inv.consume_grenade("frag")
        assert g is not None
        assert g.count == 1

    def test_consume_grenade_empty(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(
            item_id="frag", item_type="grenade", count=0,
        ))
        assert inv.consume_grenade("frag") is None

    def test_use_consumable(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(
            item_id="medkit", item_type="consumable",
            effect_value=50.0, uses=1,
        ))
        val = inv.use_consumable("medkit")
        assert val == 50.0
        assert inv.use_consumable("medkit") == 0.0

    def test_get_devices(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="phone", item_type="device"))
        inv.add_item(InventoryItem(item_id="gun", item_type="weapon"))
        devices = inv.get_devices()
        assert len(devices) == 1
        assert devices[0].item_id == "phone"

    def test_to_dict(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="g1", item_type="weapon"))
        d = inv.to_dict()
        assert d["owner_id"] == "u1"
        assert len(d["items"]) == 1

    def test_to_fog_dict(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(item_id="g1", item_type="weapon"))
        fog = inv.to_fog_dict()
        assert fog["status"] == "unknown"
        assert fog["item_count"] == 1


class TestItemCatalog:
    def test_catalog_has_weapons(self):
        weapon_keys = [k for k, v in ITEM_CATALOG.items() if v.get("item_type") == "weapon"]
        assert len(weapon_keys) >= 10

    def test_catalog_has_armor(self):
        armor_keys = [k for k, v in ITEM_CATALOG.items() if v.get("item_type") == "armor"]
        assert len(armor_keys) >= 5

    def test_catalog_has_grenades(self):
        grenade_keys = [k for k, v in ITEM_CATALOG.items() if v.get("item_type") == "grenade"]
        assert len(grenade_keys) >= 2


class TestBuildLoadout:
    def test_friendly_rover(self):
        inv = build_loadout("rover_1", "rover", "friendly")
        assert inv.owner_id == "rover_1"
        weapons = inv.get_weapons()
        assert len(weapons) >= 1

    def test_hostile_person(self):
        inv = build_loadout("h1", "person", "hostile")
        weapons = inv.get_weapons()
        assert len(weapons) >= 1

    def test_neutral_person_no_weapons(self):
        inv = build_loadout("civ1", "person", "neutral")
        weapons = inv.get_weapons()
        assert len(weapons) == 0

    def test_deterministic_loadout(self):
        """Same target_id always produces the same loadout."""
        inv1 = build_loadout("test_unit_42", "rover", "friendly")
        inv2 = build_loadout("test_unit_42", "rover", "friendly")
        assert len(inv1.items) == len(inv2.items)
        for a, b in zip(inv1.items, inv2.items):
            assert a.item_id == b.item_id

    def test_friendly_tank_loadout(self):
        inv = build_loadout("tank_1", "tank", "friendly")
        weapons = inv.get_weapons()
        assert len(weapons) >= 2  # RPG + SMG

    def test_hostile_leader_loadout(self):
        inv = build_loadout("leader_1", "hostile_leader", "hostile")
        weapons = inv.get_weapons()
        assert len(weapons) >= 2  # Rifle + pistol


class TestSelectBestWeapon:
    def test_rpg_for_vehicle(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(
            item_id="rpg", item_type="weapon",
            weapon_class="missile", damage=60.0, range=50.0, ammo=3,
        ))
        inv.add_item(InventoryItem(
            item_id="smg", item_type="weapon",
            weapon_class="projectile", damage=6.0, range=20.0, ammo=50,
        ))
        best = select_best_weapon(inv, target_asset_type="vehicle", distance=30.0)
        assert best is not None
        assert best.item_id == "rpg"

    def test_grenade_for_groups(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(
            item_id="frag", item_type="grenade",
            damage=40.0, blast_radius=5.0, count=2,
        ))
        inv.add_item(InventoryItem(
            item_id="gun", item_type="weapon",
            damage=10.0, range=20.0, ammo=30,
        ))
        best = select_best_weapon(inv, enemies_nearby=5, distance=5.0)
        assert best is not None
        assert best.item_type == "grenade"

    def test_no_weapons_returns_none(self):
        inv = UnitInventory(owner_id="u1")
        assert select_best_weapon(inv) is None

    def test_fallback_to_any_weapon(self):
        inv = UnitInventory(owner_id="u1")
        inv.add_item(InventoryItem(
            item_id="gun", item_type="weapon",
            damage=10.0, range=5.0, ammo=10,
        ))
        best = select_best_weapon(inv, distance=100.0)
        assert best is not None
        assert best.item_id == "gun"
