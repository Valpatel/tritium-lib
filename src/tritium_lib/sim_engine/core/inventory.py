# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""UnitInventory -- per-unit item inventory for armor, weapons, consumables.

Architecture
------------
Each combatant SimulationTarget may carry a UnitInventory containing:

  - ArmorItem: damage reduction (0.0-1.0) with durability that depletes on hits
  - WeaponItem: firearm with damage/range/cooldown/accuracy/ammo stats
  - ConsumableItem: one-shot items (medkits, ammo packs, etc.)

Inventory is created at spawn via build_loadout() and persists for the
unit's lifetime.  The combat pipeline queries total_damage_reduction()
each hit to compute armor mitigation.  When a weapon runs dry,
auto_switch_weapon() cycles to the next loaded weapon.

Fog-of-war: to_dict() reveals full stats; to_fog_dict() hides specifics
so hostile/neutral inventories appear as opaque counts to the viewer.

Thread safety: Inventory is owned by SimulationTarget and accessed only
from the engine tick loop (single-threaded).  No internal locking.

Determinism: build_loadout() uses the same _seed_rng pattern from
entity.py for reproducible loadouts per target_id.
"""

from __future__ import annotations

import hashlib
import random as _random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Deterministic RNG (same pattern as entity.py)
# ---------------------------------------------------------------------------

def _seed_rng(target_id: str, salt: str = "") -> _random.Random:
    """Create a deterministic RNG seeded from target_id + salt."""
    h = hashlib.sha256(f"{target_id}:{salt}".encode()).digest()
    seed = int.from_bytes(h[:8], "big")
    return _random.Random(seed)


# ---------------------------------------------------------------------------
# Item base class and subtypes
# ---------------------------------------------------------------------------

@dataclass
class InventoryItem:
    """Universal inventory item -- flat dataclass with all possible fields.

    The item_type field determines which fields are relevant:
      - "weapon": damage, range, cooldown, ammo, max_ammo, weapon_class, blast_radius
      - "armor": damage_reduction, durability, max_durability
      - "grenade": damage, blast_radius, count
      - "consumable": effect_type, effect_value, uses, max_uses
      - "device": ble_mac, wifi_mac, device_class, device_model (RF-emitting items)
    """

    item_id: str
    item_type: str = "generic"
    name: str = ""
    weight: float = 1.0

    # Weapon fields
    weapon_class: str = ""
    damage: float = 0.0
    range: float = 0.0
    cooldown: float = 0.0
    accuracy: float = 0.8
    ammo: int = -1      # -1 = unlimited
    max_ammo: int = -1

    # Armor fields
    damage_reduction: float = 0.0
    durability: int = 100
    max_durability: int = 100

    # Grenade / AoE fields
    blast_radius: float = 0.0
    count: int = 1

    # Consumable fields
    effect_type: str = ""
    effect_value: float = 0.0
    uses: int = 1
    max_uses: int = 1

    # Device fields (items that emit detectable RF signals)
    ble_mac: str = ""
    wifi_mac: str = ""
    ble_service_uuid: str = ""
    tx_power_dbm: int = -40
    device_model: str = ""
    device_class: str = ""  # "smartwatch", "phone", "laptop", "radio", "tpms"
    always_on: bool = True
    adv_interval_ms: int = 1000

    # Alias: weapon_range -> range for typed API compat
    @property
    def weapon_range(self) -> float:
        return self.range

    @weapon_range.setter
    def weapon_range(self, value: float) -> None:
        self.range = value

    def is_functional(self) -> bool:
        """Return True if armor still has durability left."""
        return self.durability > 0

    def take_hit(self) -> None:
        """Deplete one durability point."""
        if self.durability > 0:
            self.durability -= 1

    def has_ammo(self) -> bool:
        """Return True if this weapon has ammo remaining (-1 = unlimited)."""
        return self.ammo != 0  # -1=unlimited, >0=has ammo, 0=empty

    def can_use(self) -> bool:
        """Return True if consumable has uses remaining."""
        return self.uses > 0

    def use(self) -> float:
        """Use one consumable charge. Returns effect_value or 0."""
        if self.uses <= 0:
            return 0.0
        self.uses -= 1
        return self.effect_value

    def to_dict(self) -> dict:
        """Full serialization with all relevant stats."""
        d: dict = {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "name": self.name,
        }
        if self.item_type == "weapon":
            d["weapon_class"] = self.weapon_class
            d["damage"] = self.damage
            d["range"] = self.range
            d["weapon_range"] = self.range
            d["cooldown"] = self.cooldown
            d["accuracy"] = self.accuracy
            d["ammo"] = self.ammo
            d["max_ammo"] = self.max_ammo
            d["blast_radius"] = self.blast_radius
        elif self.item_type == "armor":
            d["damage_reduction"] = self.damage_reduction
            d["durability"] = self.durability
            d["max_durability"] = self.max_durability
        elif self.item_type == "grenade":
            d["damage"] = self.damage
            d["blast_radius"] = self.blast_radius
            d["count"] = self.count
        elif self.item_type == "consumable":
            d["effect_type"] = self.effect_type
            d["effect_value"] = self.effect_value
            d["uses"] = self.uses
            d["max_uses"] = self.max_uses
        elif self.item_type == "device":
            d["ble_mac"] = self.ble_mac
            d["wifi_mac"] = self.wifi_mac
            d["ble_service_uuid"] = self.ble_service_uuid
            d["tx_power_dbm"] = self.tx_power_dbm
            d["device_model"] = self.device_model
            d["device_class"] = self.device_class
            d["always_on"] = self.always_on
            d["adv_interval_ms"] = self.adv_interval_ms
        d["weight"] = self.weight
        return d

    def to_fog_dict(self) -> dict:
        """Fog-of-war serialization -- hides specific stats."""
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "status": "unknown",
        }


# Typed aliases for backward compatibility with typed subclass API
ArmorItem = InventoryItem
WeaponItem = InventoryItem
ConsumableItem = InventoryItem


# ---------------------------------------------------------------------------
# UnitInventory
# ---------------------------------------------------------------------------

@dataclass
class UnitInventory:
    """Container for all items carried by a unit.

    Provides:
      - Item storage and lookup
      - Active weapon management (select, auto-switch)
      - Aggregate damage reduction from armor
      - Grenade management
      - Serialization with fog-of-war support
    """

    owner_id: str = ""
    items: list[InventoryItem] = field(default_factory=list)
    active_weapon_id: str | None = None

    def add_item(self, item: InventoryItem) -> None:
        """Add an item to inventory."""
        self.items.append(item)
        # Auto-equip first weapon added
        if item.item_type == "weapon" and self.active_weapon_id is None:
            self.active_weapon_id = item.item_id

    def remove_item(self, item_id: str) -> InventoryItem | None:
        """Remove and return an item by ID, or None if not found."""
        for i, item in enumerate(self.items):
            if item.item_id == item_id:
                removed = self.items.pop(i)
                if self.active_weapon_id == item_id:
                    self.active_weapon_id = None
                    self.auto_switch_weapon()
                return removed
        return None

    def get_item(self, item_id: str) -> InventoryItem | None:
        """Return an item by ID."""
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None

    # -- Armor methods -------------------------------------------------------

    def get_armor(self) -> InventoryItem | None:
        """Return the first armor item, or None."""
        for item in self.items:
            if item.item_type == "armor":
                return item
        return None

    def total_damage_reduction(self) -> float:
        """Return total damage reduction from all functional armor items.

        Multiple armor pieces stack additively, capped at 0.8.
        """
        total = 0.0
        for item in self.items:
            if item.item_type == "armor" and item.is_functional():
                total += item.damage_reduction
        return min(total, 0.8)

    def damage_armor(self, hits: int = 1) -> float:
        """Apply *hits* durability damage to all equipped armor items.

        Returns the damage_reduction of the armor that was hit, or 0.0 if none.
        """
        for item in self.items:
            if item.item_type == "armor" and item.is_functional():
                reduction = item.damage_reduction
                for _ in range(hits):
                    item.take_hit()
                return reduction
        return 0.0

    # -- Weapon methods ------------------------------------------------------

    def get_active_weapon(self) -> InventoryItem | None:
        """Return the currently active weapon, or None."""
        if self.active_weapon_id is None:
            return None
        item = self.get_item(self.active_weapon_id)
        if item is not None and item.item_type == "weapon":
            return item
        return None

    def set_active_weapon(self, item_id: str) -> bool:
        """Set the active weapon by item ID. Returns True on success."""
        item = self.get_item(item_id)
        if item is not None and item.item_type == "weapon":
            self.active_weapon_id = item_id
            return True
        return False

    def switch_weapon(self, item_id: str) -> bool:
        """Switch active weapon to item_id. Returns True on success, False if invalid."""
        item = self.get_item(item_id)
        if item is None or item.item_type != "weapon":
            return False
        self.active_weapon_id = item_id
        return True

    def auto_switch_weapon(self) -> bool:
        """Switch to the next weapon with ammo. Returns True if switched."""
        active = self.get_active_weapon()
        if active is not None and active.has_ammo():
            return False

        for item in self.items:
            if item.item_type == "weapon" and item.has_ammo():
                if active is None or item.item_id != active.item_id:
                    self.active_weapon_id = item.item_id
                    return True
        return False

    def has_ammo(self) -> bool:
        """Return True if any weapon has ammo remaining."""
        for item in self.items:
            if item.item_type == "weapon" and item.has_ammo():
                return True
        return False

    def get_weapons(self) -> list[InventoryItem]:
        """Return all weapon items."""
        return [i for i in self.items if i.item_type == "weapon"]

    # -- Grenade methods -----------------------------------------------------

    def get_grenades(self) -> list[InventoryItem]:
        """Return all grenade items."""
        return [i for i in self.items if i.item_type == "grenade"]

    def consume_grenade(self, item_id: str) -> InventoryItem | None:
        """Consume one grenade of the given item_id.

        Returns the item if consumed, None if not found or empty.
        """
        item = self.get_item(item_id)
        if item is None or item.item_type != "grenade":
            return None
        if item.count <= 0:
            return None
        item.count -= 1
        return item

    # -- Consumable methods --------------------------------------------------

    def get_consumables(self) -> list[InventoryItem]:
        """Return all consumable items."""
        return [i for i in self.items if i.item_type == "consumable"]

    def use_consumable(self, item_id: str) -> float:
        """Use a consumable item. Returns effect_value or 0."""
        item = self.get_item(item_id)
        if item is not None and item.item_type == "consumable":
            return item.use()
        return 0.0

    # -- Device methods ------------------------------------------------------

    def get_devices(self) -> list[InventoryItem]:
        """Return all device items (RF-emitting)."""
        return [i for i in self.items if i.item_type == "device"]

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Full serialization -- reveals all item stats."""
        return {
            "owner_id": self.owner_id,
            "active_weapon_id": self.active_weapon_id,
            "items": [item.to_dict() for item in self.items],
        }

    def to_fog_dict(self) -> dict:
        """Fog-of-war serialization -- hides detailed stats."""
        return {
            "status": "unknown",
            "item_count": len(self.items),
        }


# ---------------------------------------------------------------------------
# ITEM_CATALOG -- reference data for all item templates
# ---------------------------------------------------------------------------

ITEM_CATALOG: dict[str, dict] = {
    # Weapons
    "nerf_pistol": {
        "item_type": "weapon", "name": "Nerf Pistol",
        "weapon_class": "projectile", "damage": 8.0, "range": 15.0,
        "cooldown": 1.0, "ammo": 30, "max_ammo": 30, "blast_radius": 0.0,
    },
    "nerf_rifle": {
        "item_type": "weapon", "name": "Nerf Rifle",
        "weapon_class": "projectile", "damage": 12.0, "range": 40.0,
        "cooldown": 1.5, "ammo": 20, "max_ammo": 20, "blast_radius": 0.0,
    },
    "nerf_shotgun": {
        "item_type": "weapon", "name": "Nerf Shotgun",
        "weapon_class": "projectile", "damage": 25.0, "range": 8.0,
        "cooldown": 2.5, "ammo": 8, "max_ammo": 8, "blast_radius": 0.0,
    },
    "nerf_rpg": {
        "item_type": "weapon", "name": "Nerf RPG",
        "weapon_class": "missile", "damage": 60.0, "range": 50.0,
        "cooldown": 8.0, "ammo": 3, "max_ammo": 3, "blast_radius": 3.0,
    },
    "nerf_smg": {
        "item_type": "weapon", "name": "Nerf SMG",
        "weapon_class": "projectile", "damage": 6.0, "range": 20.0,
        "cooldown": 0.3, "ammo": 50, "max_ammo": 50, "blast_radius": 0.0,
    },
    "nerf_blaster": {
        "item_type": "weapon", "name": "Nerf Blaster",
        "weapon_class": "projectile", "damage": 10.0, "range": 20.0,
        "cooldown": 1.0, "ammo": 30, "max_ammo": 30, "blast_radius": 0.0,
    },
    "nerf_turret": {
        "item_type": "weapon", "name": "Nerf Turret Gun",
        "weapon_class": "projectile", "damage": 15.0, "range": 25.0,
        "cooldown": 1.5, "ammo": 100, "max_ammo": 100, "blast_radius": 0.0,
    },
    "nerf_cannon": {
        "item_type": "weapon", "name": "Nerf Cannon",
        "weapon_class": "projectile", "damage": 20.0, "range": 30.0,
        "cooldown": 2.0, "ammo": 20, "max_ammo": 20, "blast_radius": 0.0,
    },
    "tank_main_gun": {
        "item_type": "weapon", "name": "Tank Main Gun",
        "weapon_class": "aoe", "damage": 30.0, "range": 40.0,
        "cooldown": 3.0, "ammo": 15, "max_ammo": 15, "blast_radius": 3.0,
    },
    "apc_mg": {
        "item_type": "weapon", "name": "APC Machine Gun",
        "weapon_class": "projectile", "damage": 8.0, "range": 25.0,
        "cooldown": 0.5, "ammo": 80, "max_ammo": 80, "blast_radius": 0.0,
    },
    "drone_gun": {
        "item_type": "weapon", "name": "Drone Dart Gun",
        "weapon_class": "projectile", "damage": 8.0, "range": 15.0,
        "cooldown": 1.0, "ammo": 20, "max_ammo": 20, "blast_radius": 0.0,
    },
    "scout_dart": {
        "item_type": "weapon", "name": "Scout Dart Gun",
        "weapon_class": "projectile", "damage": 5.0, "range": 12.0,
        "cooldown": 1.5, "ammo": 15, "max_ammo": 15, "blast_radius": 0.0,
    },
    "hostile_rifle": {
        "item_type": "weapon", "name": "Hostile Rifle",
        "weapon_class": "projectile", "damage": 10.0, "range": 30.0,
        "cooldown": 2.0, "ammo": 20, "max_ammo": 20, "blast_radius": 0.0,
    },
    # Armor
    "light_vest": {
        "item_type": "armor", "name": "Light Tactical Vest",
        "damage_reduction": 0.15, "durability": 50, "max_durability": 50,
    },
    "medium_vest": {
        "item_type": "armor", "name": "Medium Vest",
        "damage_reduction": 0.20, "durability": 80, "max_durability": 80,
    },
    "heavy_vest": {
        "item_type": "armor", "name": "Heavy Vest",
        "damage_reduction": 0.35, "durability": 120, "max_durability": 120,
    },
    "hostile_vest": {
        "item_type": "armor", "name": "Improvised Vest",
        "damage_reduction": 0.10, "durability": 30, "max_durability": 30,
    },
    "vehicle_armor": {
        "item_type": "armor", "name": "Vehicle Armor",
        "damage_reduction": 0.35, "durability": 200, "max_durability": 200,
    },
    "tank_armor": {
        "item_type": "armor", "name": "Tank Reactive Armor",
        "damage_reduction": 0.45, "durability": 300, "max_durability": 300,
    },
    "apc_armor": {
        "item_type": "armor", "name": "APC Composite Armor",
        "damage_reduction": 0.35, "durability": 200, "max_durability": 200,
    },
    "rover_plating": {
        "item_type": "armor", "name": "Rover Armor Plating",
        "damage_reduction": 0.20, "durability": 100, "max_durability": 100,
    },
    "turret_shield": {
        "item_type": "armor", "name": "Turret Blast Shield",
        "damage_reduction": 0.30, "durability": 150, "max_durability": 150,
    },
    "drone_shell": {
        "item_type": "armor", "name": "Drone Composite Shell",
        "damage_reduction": 0.10, "durability": 30, "max_durability": 30,
    },
    # Grenades
    "frag_grenade": {
        "item_type": "grenade", "name": "Frag Grenade",
        "damage": 40.0, "blast_radius": 5.0, "count": 2,
    },
    "smoke_grenade": {
        "item_type": "grenade", "name": "Smoke Grenade",
        "damage": 0.0, "blast_radius": 8.0, "count": 1,
    },
    "flashbang": {
        "item_type": "grenade", "name": "Flashbang",
        "damage": 5.0, "blast_radius": 6.0, "count": 1,
    },
}


def _make_item_from_catalog(catalog_key: str, item_id: str) -> InventoryItem | None:
    """Create an InventoryItem from the catalog."""
    tmpl = ITEM_CATALOG.get(catalog_key)
    if tmpl is None:
        return None
    return InventoryItem(item_id=item_id, **tmpl)


def build_loadout(
    target_id: str,
    asset_type: str,
    alliance: str,
) -> UnitInventory:
    """Build a default inventory loadout for a unit.

    Returns a UnitInventory (possibly empty for non-combatants).
    Loadouts are deterministic per target_id.

    Args:
        target_id: Unique unit identifier
        asset_type: Unit type (rover, drone, turret, person, etc.)
        alliance: "friendly", "hostile", "neutral"

    Returns:
        UnitInventory with appropriate items (empty for non-combatants).
    """
    inv = UnitInventory(owner_id=target_id)

    # Non-combatants get devices only (no weapons)
    if alliance == "neutral":
        _add_civilian_devices(inv, target_id, asset_type)
        return inv
    if asset_type in ("animal", "vehicle") and alliance != "hostile":
        return inv

    # --- Hostile loadouts ---
    if alliance == "hostile":
        if asset_type == "person":
            # Hostile person: pistol + improvised vest + frag grenade
            inv.add_item(_make_item_from_catalog("nerf_pistol", f"{target_id}_pistol"))
            inv.add_item(_make_item_from_catalog("hostile_vest", f"{target_id}_vest"))
            inv.add_item(_make_item_from_catalog("frag_grenade", f"{target_id}_frag"))
        elif asset_type == "hostile_leader":
            # Leader: rifle + pistol + medium vest + frag grenade
            inv.add_item(_make_item_from_catalog("nerf_rifle", f"{target_id}_rifle"))
            inv.add_item(_make_item_from_catalog("nerf_pistol", f"{target_id}_pistol"))
            inv.add_item(_make_item_from_catalog("medium_vest", f"{target_id}_vest"))
            inv.add_item(_make_item_from_catalog("frag_grenade", f"{target_id}_frag"))
            inv.active_weapon_id = f"{target_id}_rifle"
        elif asset_type == "hostile_vehicle":
            # Vehicle: RPG + hostile rifle + vehicle armor
            inv.add_item(_make_item_from_catalog("nerf_rpg", f"{target_id}_rpg"))
            inv.add_item(_make_item_from_catalog("hostile_rifle", f"{target_id}_rifle"))
            inv.add_item(_make_item_from_catalog("vehicle_armor", f"{target_id}_armor"))
            inv.active_weapon_id = f"{target_id}_rpg"
        elif asset_type == "tank":
            # Tank: RPG + tank armor
            inv.add_item(_make_item_from_catalog("nerf_rpg", f"{target_id}_rpg"))
            inv.add_item(_make_item_from_catalog("tank_armor", f"{target_id}_armor"))
            inv.active_weapon_id = f"{target_id}_rpg"
        elif asset_type == "apc":
            inv.add_item(_make_item_from_catalog("apc_mg", f"{target_id}_mg"))
            inv.add_item(_make_item_from_catalog("apc_armor", f"{target_id}_armor"))
        elif asset_type == "rover":
            inv.add_item(_make_item_from_catalog("nerf_rifle", f"{target_id}_rifle"))
            inv.add_item(_make_item_from_catalog("medium_vest", f"{target_id}_vest"))
        elif asset_type == "drone":
            inv.add_item(_make_item_from_catalog("drone_gun", f"{target_id}_gun"))
        else:
            inv.add_item(_make_item_from_catalog("nerf_pistol", f"{target_id}_pistol"))
        _add_civilian_devices(inv, target_id, asset_type)
        return inv

    # --- Friendly loadouts ---
    if asset_type == "turret":
        # Turrets: weapon only, no armor (stationary)
        inv.add_item(_make_item_from_catalog("nerf_turret", f"{target_id}_turret"))
    elif asset_type == "heavy_turret":
        inv.add_item(_make_item_from_catalog("nerf_cannon", f"{target_id}_cannon"))
    elif asset_type == "missile_turret":
        inv.add_item(_make_item_from_catalog("nerf_rpg", f"{target_id}_rpg"))
    elif asset_type == "drone":
        # Drones: weapon only, no armor (flying)
        inv.add_item(_make_item_from_catalog("drone_gun", f"{target_id}_gun"))
    elif asset_type == "scout_drone":
        inv.add_item(_make_item_from_catalog("scout_dart", f"{target_id}_dart"))
    elif asset_type == "rover":
        # Rover: SMG + rover armor
        inv.add_item(_make_item_from_catalog("nerf_smg", f"{target_id}_smg"))
        inv.add_item(_make_item_from_catalog("rover_plating", f"{target_id}_armor"))
    elif asset_type == "tank":
        # Tank: RPG + SMG + tank armor
        inv.add_item(_make_item_from_catalog("nerf_rpg", f"{target_id}_rpg"))
        inv.add_item(_make_item_from_catalog("nerf_smg", f"{target_id}_smg"))
        inv.add_item(_make_item_from_catalog("tank_armor", f"{target_id}_armor"))
        inv.active_weapon_id = f"{target_id}_rpg"
    elif asset_type == "apc":
        # APC: SMG + APC armor + smoke grenade
        inv.add_item(_make_item_from_catalog("nerf_smg", f"{target_id}_smg"))
        inv.add_item(_make_item_from_catalog("apc_armor", f"{target_id}_armor"))
        inv.add_item(_make_item_from_catalog("smoke_grenade", f"{target_id}_smoke"))
    elif asset_type == "person":
        inv.add_item(_make_item_from_catalog("nerf_blaster", f"{target_id}_blaster"))
        inv.add_item(_make_item_from_catalog("light_vest", f"{target_id}_vest"))
    else:
        inv.add_item(_make_item_from_catalog("nerf_blaster", f"{target_id}_blaster"))

    # All entities get civilian devices (phone, watch, car sensors)
    _add_civilian_devices(inv, target_id, asset_type)

    return inv


def _add_civilian_devices(inv: UnitInventory, target_id: str, asset_type: str) -> None:
    """Add personal electronic devices to any entity's inventory.

    These devices emit detectable RF signatures (BLE, WiFi) that sensors
    can pick up — bridging the simulation into the real tracking pipeline.
    """
    import hashlib

    # Deterministic RNG from target_id.  sha256 over md5 to keep bandit
    # quiet (W200-L1) — this is RNG seeding, not a security context, but
    # cheap to do right.
    h = int(hashlib.sha256(target_id.encode()).hexdigest(), 16)
    def _rng():
        nonlocal h
        h = (h * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (h >> 33) / (2**31)

    _PHONE_MODELS = [
        "iPhone 15 Pro", "iPhone 14", "iPhone SE",
        "Samsung Galaxy S24", "Samsung Galaxy A54",
        "Google Pixel 8", "Google Pixel 7a",
        "OnePlus 12", "Motorola Edge",
    ]
    _WATCH_MODELS = [
        "Apple Watch Series 9", "Apple Watch SE",
        "Samsung Galaxy Watch 6", "Fitbit Charge 6",
        "Garmin Venu 3",
    ]

    def _mac(prefix=""):
        octets = [int(_rng() * 256) for _ in range(6)]
        # Set locally administered bit
        octets[0] = (octets[0] | 0x02) & 0xFE
        return ":".join(f"{b:02X}" for b in octets)

    if asset_type in ("person", "instigator", "rioter", "civilian"):
        # 80% have a phone
        if _rng() < 0.8:
            model = _PHONE_MODELS[int(_rng() * len(_PHONE_MODELS))]
            inv.add_item(InventoryItem(
                item_id=f"{target_id}_phone",
                item_type="device",
                name=model,
                device_class="phone",
                device_model=model,
                ble_mac=_mac(),
                wifi_mac=_mac(),
                tx_power_dbm=-40,
                always_on=True,
            ))
        # 30% have a smartwatch
        if _rng() < 0.3:
            model = _WATCH_MODELS[int(_rng() * len(_WATCH_MODELS))]
            inv.add_item(InventoryItem(
                item_id=f"{target_id}_watch",
                item_type="device",
                name=model,
                device_class="smartwatch",
                device_model=model,
                ble_mac=_mac(),
                tx_power_dbm=-50,
                always_on=True,
            ))

    elif asset_type in ("vehicle", "hostile_vehicle"):
        # Cars have TPMS sensors (4 per vehicle)
        for i in range(4):
            inv.add_item(InventoryItem(
                item_id=f"{target_id}_tpms_{i}",
                item_type="device",
                name=f"TPMS Sensor {i+1}",
                device_class="tpms",
                device_model="TPMS 315MHz",
                ble_mac=_mac(),
                tx_power_dbm=-20,
                always_on=False,
                adv_interval_ms=60000,  # transmits once per minute
            ))


# ---------------------------------------------------------------------------
# select_best_weapon -- tactical AI weapon selection
# ---------------------------------------------------------------------------

def select_best_weapon(
    inventory: UnitInventory,
    target_asset_type: str = "person",
    distance: float = 10.0,
    enemies_nearby: int = 1,
    target_distance: float | None = None,
) -> InventoryItem | None:
    """Select the best weapon for a given tactical situation.

    Priority:
      1. RPG for vehicles/tanks/APCs (if in range and has ammo)
      2. Grenades for groups (enemies_nearby >= 3, in blast range, has count)
      3. Shotgun at close range (< 8m)
      4. Rifle at long range (> 20m)
      5. Best weapon by ammo at medium range

    Args:
        inventory: The unit's inventory
        target_asset_type: Type of target (person, vehicle, tank, etc.)
        distance: Distance to target in meters
        enemies_nearby: Number of enemies in the area
        target_distance: Alias for distance (backward compat)

    Returns:
        The best InventoryItem, or None if no weapons/grenades available.
    """
    # Support backward-compat positional arg
    if target_distance is not None:
        distance = target_distance

    weapons = inventory.get_weapons()
    grenades = inventory.get_grenades()

    if not weapons and not grenades:
        return None

    # 1. RPG for vehicles/tanks/APCs
    if target_asset_type in ("vehicle", "hostile_vehicle", "tank", "apc"):
        for w in weapons:
            if w.weapon_class == "missile" and w.has_ammo() and w.range >= distance:
                return w

    # 2. Grenades for groups
    if enemies_nearby >= 3:
        for g in grenades:
            if g.count > 0:
                return g

    # 3. Filter weapons with ammo
    armed = [w for w in weapons if w.has_ammo()]
    if not armed:
        return None

    # 4. Shotgun at close range
    if distance <= 8.0:
        shotguns = [w for w in armed if w.range <= 10.0 and w.damage >= 20.0]
        if shotguns:
            return max(shotguns, key=lambda w: w.damage)

    # 5. Rifle at long range -- prefer non-missile weapons unless targeting vehicles
    if distance > 20.0:
        long_range = [w for w in armed if w.range >= distance]
        if long_range:
            # Prefer projectile/beam weapons over missiles for non-vehicle targets
            non_missile = [w for w in long_range if w.weapon_class != "missile"]
            pick_from = non_missile if non_missile else long_range
            return max(pick_from, key=lambda w: w.range)

    # 6. Default: weapon with most ammo (among those in range)
    in_range = [w for w in armed if w.range >= distance]
    if in_range:
        return max(in_range, key=lambda w: w.ammo)

    # Fallback: any weapon with ammo
    return max(armed, key=lambda w: w.ammo)
