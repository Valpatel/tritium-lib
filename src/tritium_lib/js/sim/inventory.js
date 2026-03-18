/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  UnitInventory JS mirror — exact field-name parity with Python
  tritium_lib.sim_engine.core.inventory.

  Covers:
    - InventoryItem (flat, universal item with all possible fields)
    - UnitInventory (container with weapon/armor/grenade/consumable/device management)
    - ITEM_CATALOG (reference templates)
    - build_loadout (deterministic per-unit inventory generation)
    - select_best_weapon (tactical AI weapon selection)

  Wire format: toDict()/toFogDict() produce the same JSON shapes as Python.
*/

// =========================================================================
// Seeded PRNG (Mulberry32) — same as identity.js
// =========================================================================

function _mulberry32(seed) {
    let s = seed | 0;
    return function () {
        s = (s + 0x6D2B79F5) | 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function _simpleHash(str) {
    let h = 0;
    for (let i = 0; i < str.length; i++) {
        h = ((h << 5) - h + str.charCodeAt(i)) | 0;
    }
    return h >>> 0;
}

function _seedRng(targetId, salt = '') {
    const seed = _simpleHash(targetId + ':' + salt);
    return _mulberry32(seed);
}

// =========================================================================
// ID generation
// =========================================================================

let _itemIdCounter = 0;

/**
 * Generate a unique item ID.
 * @param {string} [prefix='item'] - ID prefix
 * @returns {string}
 */
export function generateItemId(prefix = 'item') {
    return prefix + '_' + (++_itemIdCounter) + '_' + Date.now().toString(36);
}

// =========================================================================
// Device model lists and OUI prefixes for MAC generation
// =========================================================================

/** @type {string[]} */
export const PHONE_MODELS = [
    'iPhone 15 Pro', 'iPhone 15', 'iPhone 14 Pro', 'iPhone 14',
    'iPhone SE', 'Samsung Galaxy S24', 'Samsung Galaxy S23',
    'Samsung Galaxy A54', 'Google Pixel 8 Pro', 'Google Pixel 8',
    'Google Pixel 7a', 'Samsung Galaxy Z Flip5',
    'OnePlus 12', 'Motorola Edge 40',
];

/** @type {string[]} */
export const WATCH_MODELS = [
    'Apple Watch Series 9', 'Apple Watch Ultra 2', 'Apple Watch SE',
    'Samsung Galaxy Watch 6', 'Google Pixel Watch 2',
    'Garmin Venu 3', 'Fitbit Sense 2',
];

/** @type {string[]} */
export const LAPTOP_MODELS = [
    'MacBook Pro 16"', 'MacBook Air M3', 'Dell XPS 15',
    'ThinkPad X1 Carbon', 'Surface Laptop 5', 'HP Spectre x360',
];

/** @type {string[]} */
export const RADIO_MODELS = [
    'Meshtastic T-LoRa', 'Baofeng UV-5R', 'Yaesu FT-60R',
];

/** OUI prefixes by manufacturer for realistic MAC generation */
const OUI_PREFIXES = {
    apple:   ['A4:C3:F0', '3C:E0:72', '78:7B:8A', '9C:20:7B', 'F0:18:98'],
    samsung: ['8C:F5:A3', 'C0:BD:D1', '50:01:D9', 'AC:5F:3E', 'B4:3A:28'],
    google:  ['54:60:09', '94:EB:2C', 'F8:8F:CA'],
    garmin:  ['00:1D:C0', 'C8:3E:99'],
    fitbit:  ['C0:D0:12', '54:B5:6E'],
    generic: ['00:1A:2B', '02:42:AC', 'AA:BB:CC'],
    tpms:    ['E0:E5:CF', 'D4:F0:EA'],
};

/**
 * Generate a realistic MAC address from a seeded RNG.
 * @param {function} rng - Seeded PRNG returning [0,1)
 * @param {string} [deviceClass='phone'] - Device class for OUI selection
 * @returns {string} MAC address like "A4:C3:F0:1A:2B:3C"
 */
export function generateMac(rng, deviceClass = 'phone') {
    let ouis;
    if (deviceClass === 'phone') {
        const roll = rng();
        if (roll < 0.45) ouis = OUI_PREFIXES.apple;
        else if (roll < 0.75) ouis = OUI_PREFIXES.samsung;
        else ouis = OUI_PREFIXES.google;
    } else if (deviceClass === 'smartwatch' || deviceClass === 'watch') {
        const roll = rng();
        if (roll < 0.5) ouis = OUI_PREFIXES.apple;
        else if (roll < 0.7) ouis = OUI_PREFIXES.samsung;
        else if (roll < 0.85) ouis = OUI_PREFIXES.garmin;
        else ouis = OUI_PREFIXES.fitbit;
    } else if (deviceClass === 'tpms') {
        ouis = OUI_PREFIXES.tpms;
    } else {
        ouis = OUI_PREFIXES.generic;
    }

    const prefix = ouis[Math.floor(rng() * ouis.length)];
    const suffix = [
        Math.floor(rng() * 256),
        Math.floor(rng() * 256),
        Math.floor(rng() * 256),
    ].map(b => b.toString(16).toUpperCase().padStart(2, '0')).join(':');
    return prefix + ':' + suffix;
}

// =========================================================================
// InventoryItem factory (mirrors Python InventoryItem dataclass)
// =========================================================================

/**
 * Create an inventory item with all possible fields.
 * The item_type field determines which fields are relevant.
 *
 * @param {Object} [props={}]
 * @returns {Object} A plain JS object matching Python InventoryItem fields
 */
export function createItem(props = {}) {
    return {
        item_id: props.item_id || generateItemId(),
        item_type: props.item_type || 'generic',  // weapon, armor, grenade, consumable, device
        name: props.name || '',
        weight: props.weight !== undefined ? props.weight : 1.0,

        // Weapon fields
        weapon_class: props.weapon_class || '',
        damage: props.damage || 0,
        range: props.range || 0,
        cooldown: props.cooldown || 0,
        accuracy: props.accuracy !== undefined ? props.accuracy : 0.8,
        ammo: props.ammo !== undefined ? props.ammo : -1,   // -1 = unlimited
        max_ammo: props.max_ammo !== undefined ? props.max_ammo : -1,

        // Armor fields
        damage_reduction: props.damage_reduction || 0.0,
        durability: props.durability !== undefined ? props.durability : 100,
        max_durability: props.max_durability !== undefined ? props.max_durability : 100,

        // Grenade / AoE fields
        blast_radius: props.blast_radius || 0.0,
        count: props.count !== undefined ? props.count : 1,

        // Consumable fields
        effect_type: props.effect_type || '',
        effect_value: props.effect_value || 0.0,
        uses: props.uses !== undefined ? props.uses : 1,
        max_uses: props.max_uses !== undefined ? props.max_uses : 1,

        // Device fields (items that emit detectable RF signals)
        ble_mac: props.ble_mac || '',
        wifi_mac: props.wifi_mac || '',
        ble_service_uuid: props.ble_service_uuid || '',
        tx_power_dbm: props.tx_power_dbm !== undefined ? props.tx_power_dbm : -40,
        device_model: props.device_model || '',
        device_class: props.device_class || '',  // smartwatch, phone, laptop, radio, tpms
        always_on: props.always_on !== undefined ? props.always_on : true,
        adv_interval_ms: props.adv_interval_ms !== undefined ? props.adv_interval_ms : 1000,
    };
}

/**
 * Check if armor item still has durability left.
 * @param {Object} item
 * @returns {boolean}
 */
export function isItemFunctional(item) {
    return item.durability > 0;
}

/**
 * Deplete one durability point on an armor item.
 * @param {Object} item
 */
export function takeHit(item) {
    if (item.durability > 0) item.durability -= 1;
}

/**
 * Check if weapon has ammo remaining (-1 = unlimited).
 * @param {Object} item
 * @returns {boolean}
 */
export function hasAmmo(item) {
    return item.ammo !== 0;
}

/**
 * Use one consumable charge. Returns effect_value or 0.
 * @param {Object} item
 * @returns {number}
 */
export function useConsumable(item) {
    if (item.uses <= 0) return 0.0;
    item.uses -= 1;
    return item.effect_value;
}

/**
 * Serialize item to full dict (mirrors Python InventoryItem.to_dict).
 * @param {Object} item
 * @returns {Object}
 */
export function itemToDict(item) {
    const d = {
        item_id: item.item_id,
        item_type: item.item_type,
        name: item.name,
    };
    if (item.item_type === 'weapon') {
        d.weapon_class = item.weapon_class;
        d.damage = item.damage;
        d.range = item.range;
        d.weapon_range = item.range;
        d.cooldown = item.cooldown;
        d.accuracy = item.accuracy;
        d.ammo = item.ammo;
        d.max_ammo = item.max_ammo;
        d.blast_radius = item.blast_radius;
    } else if (item.item_type === 'armor') {
        d.damage_reduction = item.damage_reduction;
        d.durability = item.durability;
        d.max_durability = item.max_durability;
    } else if (item.item_type === 'grenade') {
        d.damage = item.damage;
        d.blast_radius = item.blast_radius;
        d.count = item.count;
    } else if (item.item_type === 'consumable') {
        d.effect_type = item.effect_type;
        d.effect_value = item.effect_value;
        d.uses = item.uses;
        d.max_uses = item.max_uses;
    } else if (item.item_type === 'device') {
        d.ble_mac = item.ble_mac;
        d.wifi_mac = item.wifi_mac;
        d.ble_service_uuid = item.ble_service_uuid;
        d.tx_power_dbm = item.tx_power_dbm;
        d.device_model = item.device_model;
        d.device_class = item.device_class;
        d.always_on = item.always_on;
        d.adv_interval_ms = item.adv_interval_ms;
    }
    d.weight = item.weight;
    return d;
}

/**
 * Fog-of-war serialization — hides specific stats.
 * @param {Object} item
 * @returns {Object}
 */
export function itemToFogDict(item) {
    return {
        item_id: item.item_id,
        item_type: item.item_type,
        status: 'unknown',
    };
}

// =========================================================================
// UnitInventory factory (mirrors Python UnitInventory dataclass)
// =========================================================================

/**
 * Create a unit inventory container.
 * @param {string} ownerId - Owner entity ID
 * @returns {Object} Inventory object with methods
 */
export function createInventory(ownerId) {
    const inv = {
        owner_id: ownerId,
        items: [],
        active_weapon_id: null,

        /**
         * Add an item to inventory. Auto-equips first weapon.
         * @param {Object} item
         */
        addItem(item) {
            this.items.push(item);
            if (item.item_type === 'weapon' && this.active_weapon_id === null) {
                this.active_weapon_id = item.item_id;
            }
        },

        /**
         * Remove and return an item by ID, or null if not found.
         * @param {string} itemId
         * @returns {Object|null}
         */
        removeItem(itemId) {
            for (let i = 0; i < this.items.length; i++) {
                if (this.items[i].item_id === itemId) {
                    const removed = this.items.splice(i, 1)[0];
                    if (this.active_weapon_id === itemId) {
                        this.active_weapon_id = null;
                        this.autoSwitchWeapon();
                    }
                    return removed;
                }
            }
            return null;
        },

        /**
         * Return an item by ID, or null.
         * @param {string} itemId
         * @returns {Object|null}
         */
        getItem(itemId) {
            for (const item of this.items) {
                if (item.item_id === itemId) return item;
            }
            return null;
        },

        // -- Armor methods --

        /**
         * Return the first armor item, or null.
         * @returns {Object|null}
         */
        getArmor() {
            for (const item of this.items) {
                if (item.item_type === 'armor') return item;
            }
            return null;
        },

        /**
         * Return total damage reduction from all functional armor (capped at 0.8).
         * @returns {number}
         */
        totalDamageReduction() {
            let total = 0;
            for (const item of this.items) {
                if (item.item_type === 'armor' && isItemFunctional(item)) {
                    total += item.damage_reduction;
                }
            }
            return Math.min(total, 0.8);
        },

        /**
         * Apply hits to armor. Returns damage_reduction of hit armor, or 0.
         * @param {number} [hits=1]
         * @returns {number}
         */
        damageArmor(hits = 1) {
            for (const item of this.items) {
                if (item.item_type === 'armor' && isItemFunctional(item)) {
                    const reduction = item.damage_reduction;
                    for (let h = 0; h < hits; h++) takeHit(item);
                    return reduction;
                }
            }
            return 0.0;
        },

        // -- Weapon methods --

        /**
         * Return the currently active weapon, or null.
         * @returns {Object|null}
         */
        getActiveWeapon() {
            if (this.active_weapon_id === null) return null;
            const item = this.getItem(this.active_weapon_id);
            if (item !== null && item.item_type === 'weapon') return item;
            return null;
        },

        /**
         * Set active weapon by item ID. Returns true on success.
         * @param {string} itemId
         * @returns {boolean}
         */
        setActiveWeapon(itemId) {
            const item = this.getItem(itemId);
            if (item !== null && item.item_type === 'weapon') {
                this.active_weapon_id = itemId;
                return true;
            }
            return false;
        },

        /**
         * Switch to the next weapon with ammo. Returns true if switched.
         * @returns {boolean}
         */
        autoSwitchWeapon() {
            const active = this.getActiveWeapon();
            if (active !== null && hasAmmo(active)) return false;

            for (const item of this.items) {
                if (item.item_type === 'weapon' && hasAmmo(item)) {
                    if (active === null || item.item_id !== active.item_id) {
                        this.active_weapon_id = item.item_id;
                        return true;
                    }
                }
            }
            return false;
        },

        /**
         * Return true if any weapon has ammo remaining.
         * @returns {boolean}
         */
        hasAmmo() {
            for (const item of this.items) {
                if (item.item_type === 'weapon' && hasAmmo(item)) return true;
            }
            return false;
        },

        /**
         * Return all weapon items.
         * @returns {Object[]}
         */
        getWeapons() {
            return this.items.filter(i => i.item_type === 'weapon');
        },

        // -- Grenade methods --

        /**
         * Return all grenade items.
         * @returns {Object[]}
         */
        getGrenades() {
            return this.items.filter(i => i.item_type === 'grenade');
        },

        /**
         * Consume one grenade. Returns item if consumed, null otherwise.
         * @param {string} itemId
         * @returns {Object|null}
         */
        consumeGrenade(itemId) {
            const item = this.getItem(itemId);
            if (item === null || item.item_type !== 'grenade') return null;
            if (item.count <= 0) return null;
            item.count -= 1;
            return item;
        },

        // -- Consumable methods --

        /**
         * Return all consumable items.
         * @returns {Object[]}
         */
        getConsumables() {
            return this.items.filter(i => i.item_type === 'consumable');
        },

        /**
         * Use a consumable. Returns effect_value or 0.
         * @param {string} itemId
         * @returns {number}
         */
        useConsumable(itemId) {
            const item = this.getItem(itemId);
            if (item !== null && item.item_type === 'consumable') {
                return useConsumable(item);
            }
            return 0.0;
        },

        // -- Device methods --

        /**
         * Return all device items (RF-emitting).
         * @returns {Object[]}
         */
        getDevices() {
            return this.items.filter(i => i.item_type === 'device');
        },

        // -- Serialization --

        /**
         * Full serialization — reveals all item stats.
         * @returns {Object}
         */
        toDict() {
            return {
                owner_id: this.owner_id,
                active_weapon_id: this.active_weapon_id,
                items: this.items.map(item => itemToDict(item)),
            };
        },

        /**
         * Fog-of-war serialization — hides detailed stats.
         * @returns {Object}
         */
        toFogDict() {
            return {
                status: 'unknown',
                item_count: this.items.length,
            };
        },
    };

    return inv;
}

// =========================================================================
// ITEM_CATALOG — reference templates (mirrors Python ITEM_CATALOG)
// =========================================================================

/** @type {Object<string, Object>} */
export const ITEM_CATALOG = {
    // Weapons
    nerf_pistol: {
        item_type: 'weapon', name: 'Nerf Pistol',
        weapon_class: 'projectile', damage: 8.0, range: 15.0,
        cooldown: 1.0, ammo: 30, max_ammo: 30, blast_radius: 0.0,
    },
    nerf_rifle: {
        item_type: 'weapon', name: 'Nerf Rifle',
        weapon_class: 'projectile', damage: 12.0, range: 40.0,
        cooldown: 1.5, ammo: 20, max_ammo: 20, blast_radius: 0.0,
    },
    nerf_shotgun: {
        item_type: 'weapon', name: 'Nerf Shotgun',
        weapon_class: 'projectile', damage: 25.0, range: 8.0,
        cooldown: 2.5, ammo: 8, max_ammo: 8, blast_radius: 0.0,
    },
    nerf_rpg: {
        item_type: 'weapon', name: 'Nerf RPG',
        weapon_class: 'missile', damage: 60.0, range: 50.0,
        cooldown: 8.0, ammo: 3, max_ammo: 3, blast_radius: 3.0,
    },
    nerf_smg: {
        item_type: 'weapon', name: 'Nerf SMG',
        weapon_class: 'projectile', damage: 6.0, range: 20.0,
        cooldown: 0.3, ammo: 50, max_ammo: 50, blast_radius: 0.0,
    },
    nerf_blaster: {
        item_type: 'weapon', name: 'Nerf Blaster',
        weapon_class: 'projectile', damage: 10.0, range: 20.0,
        cooldown: 1.0, ammo: 30, max_ammo: 30, blast_radius: 0.0,
    },
    nerf_turret: {
        item_type: 'weapon', name: 'Nerf Turret Gun',
        weapon_class: 'projectile', damage: 15.0, range: 25.0,
        cooldown: 1.5, ammo: 100, max_ammo: 100, blast_radius: 0.0,
    },
    nerf_cannon: {
        item_type: 'weapon', name: 'Nerf Cannon',
        weapon_class: 'projectile', damage: 20.0, range: 30.0,
        cooldown: 2.0, ammo: 20, max_ammo: 20, blast_radius: 0.0,
    },
    tank_main_gun: {
        item_type: 'weapon', name: 'Tank Main Gun',
        weapon_class: 'aoe', damage: 30.0, range: 40.0,
        cooldown: 3.0, ammo: 15, max_ammo: 15, blast_radius: 3.0,
    },
    apc_mg: {
        item_type: 'weapon', name: 'APC Machine Gun',
        weapon_class: 'projectile', damage: 8.0, range: 25.0,
        cooldown: 0.5, ammo: 80, max_ammo: 80, blast_radius: 0.0,
    },
    drone_gun: {
        item_type: 'weapon', name: 'Drone Dart Gun',
        weapon_class: 'projectile', damage: 8.0, range: 15.0,
        cooldown: 1.0, ammo: 20, max_ammo: 20, blast_radius: 0.0,
    },
    scout_dart: {
        item_type: 'weapon', name: 'Scout Dart Gun',
        weapon_class: 'projectile', damage: 5.0, range: 12.0,
        cooldown: 1.5, ammo: 15, max_ammo: 15, blast_radius: 0.0,
    },
    hostile_rifle: {
        item_type: 'weapon', name: 'Hostile Rifle',
        weapon_class: 'projectile', damage: 10.0, range: 30.0,
        cooldown: 2.0, ammo: 20, max_ammo: 20, blast_radius: 0.0,
    },
    // Armor
    light_vest: {
        item_type: 'armor', name: 'Light Tactical Vest',
        damage_reduction: 0.15, durability: 50, max_durability: 50,
    },
    medium_vest: {
        item_type: 'armor', name: 'Medium Vest',
        damage_reduction: 0.20, durability: 80, max_durability: 80,
    },
    heavy_vest: {
        item_type: 'armor', name: 'Heavy Vest',
        damage_reduction: 0.35, durability: 120, max_durability: 120,
    },
    hostile_vest: {
        item_type: 'armor', name: 'Improvised Vest',
        damage_reduction: 0.10, durability: 30, max_durability: 30,
    },
    vehicle_armor: {
        item_type: 'armor', name: 'Vehicle Armor',
        damage_reduction: 0.35, durability: 200, max_durability: 200,
    },
    tank_armor: {
        item_type: 'armor', name: 'Tank Reactive Armor',
        damage_reduction: 0.45, durability: 300, max_durability: 300,
    },
    apc_armor: {
        item_type: 'armor', name: 'APC Composite Armor',
        damage_reduction: 0.35, durability: 200, max_durability: 200,
    },
    rover_plating: {
        item_type: 'armor', name: 'Rover Armor Plating',
        damage_reduction: 0.20, durability: 100, max_durability: 100,
    },
    turret_shield: {
        item_type: 'armor', name: 'Turret Blast Shield',
        damage_reduction: 0.30, durability: 150, max_durability: 150,
    },
    drone_shell: {
        item_type: 'armor', name: 'Drone Composite Shell',
        damage_reduction: 0.10, durability: 30, max_durability: 30,
    },
    // Grenades
    frag_grenade: {
        item_type: 'grenade', name: 'Frag Grenade',
        damage: 40.0, blast_radius: 5.0, count: 2,
    },
    smoke_grenade: {
        item_type: 'grenade', name: 'Smoke Grenade',
        damage: 0.0, blast_radius: 8.0, count: 1,
    },
    flashbang: {
        item_type: 'grenade', name: 'Flashbang',
        damage: 5.0, blast_radius: 6.0, count: 1,
    },
};

// =========================================================================
// Catalog helper
// =========================================================================

/**
 * Create an InventoryItem from the catalog.
 * @param {string} catalogKey
 * @param {string} itemId
 * @returns {Object|null}
 */
function _makeItemFromCatalog(catalogKey, itemId) {
    const tmpl = ITEM_CATALOG[catalogKey];
    if (!tmpl) return null;
    return createItem({ item_id: itemId, ...tmpl });
}

// =========================================================================
// Default loadout builder (mirrors Python build_loadout)
// =========================================================================

/**
 * Build a default inventory loadout for a unit.
 * Deterministic per target_id. Matches Python build_loadout() exactly.
 *
 * @param {string} targetId - Unique unit identifier
 * @param {string} assetType - Unit type (rover, drone, turret, person, etc.)
 * @param {string} alliance - "friendly", "hostile", "neutral"
 * @returns {Object} UnitInventory
 */
export function buildLoadout(targetId, assetType, alliance) {
    const inv = createInventory(targetId);

    // Non-combatants get empty inventory
    if (alliance === 'neutral') return inv;
    if ((assetType === 'animal' || assetType === 'vehicle') && alliance !== 'hostile') return inv;

    // --- Hostile loadouts ---
    if (alliance === 'hostile') {
        if (assetType === 'person') {
            inv.addItem(_makeItemFromCatalog('nerf_pistol', targetId + '_pistol'));
            inv.addItem(_makeItemFromCatalog('hostile_vest', targetId + '_vest'));
            inv.addItem(_makeItemFromCatalog('frag_grenade', targetId + '_frag'));
        } else if (assetType === 'hostile_leader') {
            inv.addItem(_makeItemFromCatalog('nerf_rifle', targetId + '_rifle'));
            inv.addItem(_makeItemFromCatalog('nerf_pistol', targetId + '_pistol'));
            inv.addItem(_makeItemFromCatalog('medium_vest', targetId + '_vest'));
            inv.addItem(_makeItemFromCatalog('frag_grenade', targetId + '_frag'));
            inv.active_weapon_id = targetId + '_rifle';
        } else if (assetType === 'hostile_vehicle') {
            inv.addItem(_makeItemFromCatalog('nerf_rpg', targetId + '_rpg'));
            inv.addItem(_makeItemFromCatalog('hostile_rifle', targetId + '_rifle'));
            inv.addItem(_makeItemFromCatalog('vehicle_armor', targetId + '_armor'));
            inv.active_weapon_id = targetId + '_rpg';
        } else if (assetType === 'tank') {
            inv.addItem(_makeItemFromCatalog('nerf_rpg', targetId + '_rpg'));
            inv.addItem(_makeItemFromCatalog('tank_armor', targetId + '_armor'));
            inv.active_weapon_id = targetId + '_rpg';
        } else if (assetType === 'apc') {
            inv.addItem(_makeItemFromCatalog('apc_mg', targetId + '_mg'));
            inv.addItem(_makeItemFromCatalog('apc_armor', targetId + '_armor'));
        } else if (assetType === 'rover') {
            inv.addItem(_makeItemFromCatalog('nerf_rifle', targetId + '_rifle'));
            inv.addItem(_makeItemFromCatalog('medium_vest', targetId + '_vest'));
        } else if (assetType === 'drone') {
            inv.addItem(_makeItemFromCatalog('drone_gun', targetId + '_gun'));
        } else {
            inv.addItem(_makeItemFromCatalog('nerf_pistol', targetId + '_pistol'));
        }
        return inv;
    }

    // --- Friendly loadouts ---
    if (assetType === 'turret') {
        inv.addItem(_makeItemFromCatalog('nerf_turret', targetId + '_turret'));
    } else if (assetType === 'heavy_turret') {
        inv.addItem(_makeItemFromCatalog('nerf_cannon', targetId + '_cannon'));
    } else if (assetType === 'missile_turret') {
        inv.addItem(_makeItemFromCatalog('nerf_rpg', targetId + '_rpg'));
    } else if (assetType === 'drone') {
        inv.addItem(_makeItemFromCatalog('drone_gun', targetId + '_gun'));
    } else if (assetType === 'scout_drone') {
        inv.addItem(_makeItemFromCatalog('scout_dart', targetId + '_dart'));
    } else if (assetType === 'rover') {
        inv.addItem(_makeItemFromCatalog('nerf_smg', targetId + '_smg'));
        inv.addItem(_makeItemFromCatalog('rover_plating', targetId + '_armor'));
    } else if (assetType === 'tank') {
        inv.addItem(_makeItemFromCatalog('nerf_rpg', targetId + '_rpg'));
        inv.addItem(_makeItemFromCatalog('nerf_smg', targetId + '_smg'));
        inv.addItem(_makeItemFromCatalog('tank_armor', targetId + '_armor'));
        inv.active_weapon_id = targetId + '_rpg';
    } else if (assetType === 'apc') {
        inv.addItem(_makeItemFromCatalog('nerf_smg', targetId + '_smg'));
        inv.addItem(_makeItemFromCatalog('apc_armor', targetId + '_armor'));
        inv.addItem(_makeItemFromCatalog('smoke_grenade', targetId + '_smoke'));
    } else if (assetType === 'person') {
        inv.addItem(_makeItemFromCatalog('nerf_blaster', targetId + '_blaster'));
        inv.addItem(_makeItemFromCatalog('light_vest', targetId + '_vest'));
    } else {
        inv.addItem(_makeItemFromCatalog('nerf_blaster', targetId + '_blaster'));
    }

    return inv;
}

// =========================================================================
// Default device loadout builder (for RF-emitting personal devices)
// =========================================================================

/**
 * Build a default device loadout for an entity (phones, watches, TPMS).
 * Uses seeded PRNG for deterministic generation.
 *
 * @param {string} targetId - Entity ID
 * @param {string} assetType - Entity type
 * @param {function} [rng] - Optional seeded PRNG. If omitted, uses targetId-seeded RNG.
 * @returns {Object} UnitInventory containing device items
 */
export function buildDefaultLoadout(targetId, assetType, rng) {
    if (!rng) rng = _seedRng(targetId, 'devices');
    const inv = createInventory(targetId);

    if (assetType === 'person') {
        // 80% have a phone
        if (rng() < 0.8) {
            inv.addItem(createItem({
                item_id: targetId + '_phone',
                item_type: 'device',
                name: 'Smartphone',
                device_class: 'phone',
                device_model: PHONE_MODELS[Math.floor(rng() * PHONE_MODELS.length)],
                ble_mac: generateMac(rng, 'phone'),
                wifi_mac: generateMac(rng, 'phone'),
                tx_power_dbm: -30,
            }));
        }
        // 30% have a smartwatch
        if (rng() < 0.3) {
            inv.addItem(createItem({
                item_id: targetId + '_watch',
                item_type: 'device',
                name: 'Smartwatch',
                device_class: 'smartwatch',
                device_model: WATCH_MODELS[Math.floor(rng() * WATCH_MODELS.length)],
                ble_mac: generateMac(rng, 'smartwatch'),
                tx_power_dbm: -50,
            }));
        }
    }

    if (assetType === 'vehicle' || assetType === 'hostile_vehicle') {
        // TPMS sensors (4 per vehicle)
        for (let i = 0; i < 4; i++) {
            inv.addItem(createItem({
                item_id: targetId + '_tpms_' + i,
                item_type: 'device',
                name: 'TPMS Sensor',
                device_class: 'tpms',
                device_model: 'TPMS Sensor',
                ble_mac: generateMac(rng, 'tpms'),
                tx_power_dbm: -20,
                adv_interval_ms: 60000,
            }));
        }
    }

    if (assetType === 'rover' || assetType === 'drone' || assetType === 'scout_drone') {
        // Military devices have a radio
        inv.addItem(createItem({
            item_id: targetId + '_radio',
            item_type: 'device',
            name: 'Tactical Radio',
            device_class: 'radio',
            device_model: RADIO_MODELS[Math.floor(rng() * RADIO_MODELS.length)],
            ble_mac: generateMac(rng, 'radio'),
            wifi_mac: generateMac(rng, 'radio'),
            tx_power_dbm: -10,
        }));
    }

    return inv;
}

// =========================================================================
// Tactical weapon selection (mirrors Python select_best_weapon)
// =========================================================================

/**
 * Select the best weapon for a given tactical situation.
 *
 * Priority:
 *   1. RPG for vehicles/tanks/APCs (if in range and has ammo)
 *   2. Grenades for groups (enemies_nearby >= 3, has count)
 *   3. Shotgun at close range (< 8m)
 *   4. Rifle at long range (> 20m)
 *   5. Best weapon by ammo at medium range
 *
 * @param {Object} inventory - UnitInventory
 * @param {string} [targetAssetType='person']
 * @param {number} [distance=10.0]
 * @param {number} [enemiesNearby=1]
 * @returns {Object|null} Best InventoryItem or null
 */
export function selectBestWeapon(inventory, targetAssetType = 'person', distance = 10.0, enemiesNearby = 1) {
    const weapons = inventory.getWeapons();
    const grenades = inventory.getGrenades();

    if (weapons.length === 0 && grenades.length === 0) return null;

    // 1. RPG for vehicles/tanks/APCs
    if (targetAssetType === 'vehicle' || targetAssetType === 'hostile_vehicle' ||
        targetAssetType === 'tank' || targetAssetType === 'apc') {
        for (const w of weapons) {
            if (w.weapon_class === 'missile' && hasAmmo(w) && w.range >= distance) {
                return w;
            }
        }
    }

    // 2. Grenades for groups
    if (enemiesNearby >= 3) {
        for (const g of grenades) {
            if (g.count > 0) return g;
        }
    }

    // 3. Filter weapons with ammo
    const armed = weapons.filter(w => hasAmmo(w));
    if (armed.length === 0) return null;

    // 4. Shotgun at close range
    if (distance <= 8.0) {
        const shotguns = armed.filter(w => w.range <= 10.0 && w.damage >= 20.0);
        if (shotguns.length > 0) {
            return shotguns.reduce((a, b) => a.damage > b.damage ? a : b);
        }
    }

    // 5. Rifle at long range
    if (distance > 20.0) {
        const longRange = armed.filter(w => w.range >= distance);
        if (longRange.length > 0) {
            const nonMissile = longRange.filter(w => w.weapon_class !== 'missile');
            const pickFrom = nonMissile.length > 0 ? nonMissile : longRange;
            return pickFrom.reduce((a, b) => a.range > b.range ? a : b);
        }
    }

    // 6. Default: weapon with most ammo in range
    const inRange = armed.filter(w => w.range >= distance);
    if (inRange.length > 0) {
        return inRange.reduce((a, b) => a.ammo > b.ammo ? a : b);
    }

    // Fallback: any weapon with ammo
    return armed.reduce((a, b) => a.ammo > b.ammo ? a : b);
}
