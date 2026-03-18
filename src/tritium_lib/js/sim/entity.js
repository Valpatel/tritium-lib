/*
  Created by Matthew Valancy
  Copyright 2026 Valpatel Software LLC
  Licensed under AGPL-3.0 — see LICENSE for details.

  SimulationTarget JS mirror — exact field-name parity with Python
  tritium_lib.sim_engine.core.entity.SimulationTarget.

  Design: flat factory function (no class hierarchy) matching the Python
  dataclass. Every entity type shares the same fields; type-specific
  behaviour is encoded in lookup tables (COMBAT_PROFILES, DRAIN_RATES).

  Wire format: the object returned by createEntity() serializes to the
  same JSON shape as SimulationTarget.to_dict(), so Python and JS state
  is interchangeable over WebSocket.
*/

// =========================================================================
// ID generation
// =========================================================================

let _idCounter = 0;

/**
 * Generate a unique entity ID.
 * @param {string} [prefix='entity'] - ID prefix
 * @returns {string}
 */
export function generateId(prefix = 'entity') {
    return prefix + '_' + (++_idCounter) + '_' + Date.now().toString(36);
}

/**
 * Reset the ID counter (for testing).
 */
export function resetIdCounter() {
    _idCounter = 0;
}

// =========================================================================
// Battery drain rates per second by asset type
// (mirrors Python _DRAIN_RATES)
// =========================================================================

/** @type {Object<string, number>} */
export const DRAIN_RATES = {
    rover: 0.001,
    drone: 0.002,
    turret: 0.0005,
    scout_drone: 0.0025,
    person: 0.0,
    vehicle: 0.0,
    animal: 0.0,
    heavy_turret: 0.0004,
    missile_turret: 0.0003,
    tank: 0.0008,
    apc: 0.0010,
    swarm_drone: 0.003,
    instigator: 0.0,
    rioter: 0.0,
    civilian: 0.0,
    scout_swarm: 0.0,
    attack_swarm: 0.0,
    bomber_swarm: 0.0,
};

// =========================================================================
// Combat stat profiles by (asset_type, alliance)
// Format: { health, max_health, weapon_range, weapon_cooldown, weapon_damage, is_combatant }
// (mirrors Python _COMBAT_PROFILES)
// =========================================================================

/**
 * @typedef {Object} CombatProfile
 * @property {number} health
 * @property {number} max_health
 * @property {number} weapon_range
 * @property {number} weapon_cooldown
 * @property {number} weapon_damage
 * @property {boolean} is_combatant
 */

/** @type {Object<string, CombatProfile>} */
export const COMBAT_PROFILES = {
    turret:           { health: 200, max_health: 200, weapon_range:  80, weapon_cooldown: 1.5, weapon_damage: 15, is_combatant: true },
    drone:            { health:  60, max_health:  60, weapon_range:  50, weapon_cooldown: 1.0, weapon_damage:  8, is_combatant: true },
    rover:            { health: 150, max_health: 150, weapon_range:  60, weapon_cooldown: 2.0, weapon_damage: 12, is_combatant: true },
    person_hostile:   { health:  80, max_health:  80, weapon_range:  40, weapon_cooldown: 2.5, weapon_damage: 10, is_combatant: true },
    person_neutral:   { health:  50, max_health:  50, weapon_range:   0, weapon_cooldown: 0.0, weapon_damage:  0, is_combatant: false },
    vehicle:          { health: 300, max_health: 300, weapon_range:   0, weapon_cooldown: 0.0, weapon_damage:  0, is_combatant: false },
    animal:           { health:  30, max_health:  30, weapon_range:   0, weapon_cooldown: 0.0, weapon_damage:  0, is_combatant: false },
    // Heavy units
    tank:             { health: 400, max_health: 400, weapon_range: 100, weapon_cooldown: 3.0, weapon_damage: 30, is_combatant: true },
    apc:              { health: 300, max_health: 300, weapon_range:  60, weapon_cooldown: 1.0, weapon_damage:  8, is_combatant: true },
    heavy_turret:     { health: 350, max_health: 350, weapon_range: 120, weapon_cooldown: 2.5, weapon_damage: 25, is_combatant: true },
    missile_turret:   { health: 200, max_health: 200, weapon_range: 150, weapon_cooldown: 5.0, weapon_damage: 50, is_combatant: true },
    // Scout variant
    scout_drone:      { health:  40, max_health:  40, weapon_range:  40, weapon_cooldown: 1.5, weapon_damage:  5, is_combatant: true },
    // Hostile variants
    hostile_vehicle:  { health: 200, max_health: 200, weapon_range:  70, weapon_cooldown: 2.0, weapon_damage: 15, is_combatant: true },
    hostile_leader:   { health: 150, max_health: 150, weapon_range:  50, weapon_cooldown: 2.0, weapon_damage: 12, is_combatant: true },
    // Swarm drone: fast, fragile, short-range
    swarm_drone:      { health:  25, max_health:  25, weapon_range:  20, weapon_cooldown: 1.0, weapon_damage:  5, is_combatant: true },
    // Civil unrest crowd roles
    instigator:       { health:  60, max_health:  60, weapon_range:  15, weapon_cooldown: 3.0, weapon_damage:  5, is_combatant: true },
    rioter:           { health:  50, max_health:  50, weapon_range:   3, weapon_cooldown: 2.0, weapon_damage:  3, is_combatant: true },
    civilian:         { health:  50, max_health:  50, weapon_range:   0, weapon_cooldown: 0.0, weapon_damage:  0, is_combatant: false },
    // Non-combatant sensors
    camera:           { health:  50, max_health:  50, weapon_range:   0, weapon_cooldown: 0.0, weapon_damage:  0, is_combatant: false },
    sensor:           { health:  30, max_health:  30, weapon_range:   0, weapon_cooldown: 0.0, weapon_damage:  0, is_combatant: false },
    // Drone swarm variants
    scout_swarm:      { health:  15, max_health:  15, weapon_range:   0, weapon_cooldown: 0.0, weapon_damage:  0, is_combatant: false },
    attack_swarm:     { health:  30, max_health:  30, weapon_range:  25, weapon_cooldown: 1.0, weapon_damage:  8, is_combatant: true },
    bomber_swarm:     { health:  50, max_health:  50, weapon_range:   0, weapon_cooldown: 0.0, weapon_damage: 40, is_combatant: true },
    // Graphling agents
    graphling:        { health:  80, max_health:  80, weapon_range:  25, weapon_cooldown: 1.5, weapon_damage:  8, is_combatant: true },
};

// =========================================================================
// Profile key resolution (mirrors Python _profile_key)
// =========================================================================

/**
 * Return the combat profile lookup key for a target.
 * Dispatch order: crowd_role > drone_variant > person+alliance > asset_type.
 * @param {string} asset_type
 * @param {string} alliance
 * @param {string|null} [crowd_role=null]
 * @param {string|null} [drone_variant=null]
 * @returns {string}
 */
export function profileKey(asset_type, alliance, crowd_role = null, drone_variant = null) {
    if (crowd_role !== null && COMBAT_PROFILES[crowd_role] !== undefined) {
        return crowd_role;
    }
    if (drone_variant !== null && COMBAT_PROFILES[drone_variant] !== undefined) {
        return drone_variant;
    }
    if (asset_type === 'person') {
        return alliance === 'hostile' ? 'person_hostile' : 'person_neutral';
    }
    return asset_type;
}

// =========================================================================
// Entity factory
// =========================================================================

/**
 * Create a simulation entity with exact field-name parity to Python
 * SimulationTarget dataclass.
 *
 * @param {Object} [props={}] - Initial property overrides
 * @returns {Object} A plain JS object matching SimulationTarget fields
 */
export function createEntity(props = {}) {
    return {
        // Identity
        target_id: props.target_id || generateId(),
        name: props.name || '',
        alliance: props.alliance || 'neutral',       // friendly, hostile, neutral, unknown
        asset_type: props.asset_type || 'person',     // person, vehicle, drone, turret, rover, animal, robot, aircraft, ...

        // Position & motion
        position: props.position || { x: 0, y: 0 },
        heading: props.heading !== undefined ? props.heading : 0,
        altitude: props.altitude !== undefined ? props.altitude : 0,
        speed: props.speed !== undefined ? props.speed : 1.0,
        battery: props.battery !== undefined ? props.battery : 1.0,

        // Waypoints
        waypoints: props.waypoints || [],
        _waypoint_index: 0,
        loop_waypoints: props.loop_waypoints || false,

        // Status
        status: props.status || 'active',

        // Combat
        health: props.health !== undefined ? props.health : 100,
        max_health: props.max_health !== undefined ? props.max_health : 100,
        weapon_range: props.weapon_range !== undefined ? props.weapon_range : 15,
        weapon_cooldown: props.weapon_cooldown !== undefined ? props.weapon_cooldown : 2,
        weapon_damage: props.weapon_damage !== undefined ? props.weapon_damage : 10,
        last_fired: 0,
        kills: 0,
        is_combatant: props.is_combatant !== undefined ? props.is_combatant : true,
        vision_range: props.vision_range !== undefined ? props.vision_range : 15,

        // FSM state name (set by engine, null when no FSM assigned)
        fsm_state: props.fsm_state || null,

        // Extended simulation fields
        squad_id: props.squad_id || null,
        morale: props.morale !== undefined ? props.morale : 1.0,
        max_morale: props.max_morale !== undefined ? props.max_morale : 1.0,
        degradation: props.degradation !== undefined ? props.degradation : 0.0,
        detected: false,
        detected_at: 0.0,
        visible: props.visible !== undefined ? props.visible : true,
        detected_by: [],
        radio_detected: false,
        radio_signal_strength: 0.0,
        is_leader: props.is_leader || false,
        _fleeing: false,

        // Mission-type fields (civil unrest, drone swarm)
        crowd_role: props.crowd_role || null,
        drone_variant: props.drone_variant || null,
        instigator_state: props.instigator_state || 'hidden',
        instigator_timer: 0.0,
        identified: false,
        ammo_count: props.ammo_count !== undefined ? props.ammo_count : -1,
        ammo_max: props.ammo_max !== undefined ? props.ammo_max : -1,

        // Source classification
        source: props.source || 'sim',

        // Rich identity (generated once at spawn)
        identity: props.identity || null,

        // Per-unit inventory (armor, weapons, consumables)
        inventory: props.inventory || null,

        // Smooth movement controller
        movement: props.movement || null,

        // Waypoints version tracking
        _waypoints_version: 0,
    };
}

// =========================================================================
// Combat profile application
// =========================================================================

/**
 * Apply combat stats from COMBAT_PROFILES based on entity type and alliance.
 * Mutates the entity in place.
 * @param {Object} entity - Entity created by createEntity()
 */
export function applyProfile(entity) {
    const key = profileKey(
        entity.asset_type,
        entity.alliance,
        entity.crowd_role,
        entity.drone_variant,
    );
    const profile = COMBAT_PROFILES[key];
    if (!profile) return;

    entity.health = profile.health;
    entity.max_health = profile.max_health;
    entity.weapon_range = profile.weapon_range;
    entity.weapon_cooldown = profile.weapon_cooldown;
    entity.weapon_damage = profile.weapon_damage;
    entity.is_combatant = profile.is_combatant;
}

// =========================================================================
// Combat helpers
// =========================================================================

/**
 * Apply damage to an entity. Returns true if eliminated (health <= 0).
 * @param {Object} entity
 * @param {number} amount
 * @returns {boolean}
 */
export function applyDamage(entity, amount) {
    if (entity.status === 'destroyed' || entity.status === 'eliminated' || entity.status === 'neutralized') {
        return true;
    }
    entity.health = Math.max(0, entity.health - amount);
    if (entity.health <= 0) {
        entity.status = 'eliminated';
        return true;
    }
    return false;
}

/**
 * Check if an entity can fire right now (cooldown elapsed, has weapon, alive).
 * @param {Object} entity
 * @param {number} [now] - Current timestamp in seconds (defaults to Date.now()/1000)
 * @returns {boolean}
 */
export function canFire(entity, now) {
    if (entity.status !== 'active' && entity.status !== 'idle' && entity.status !== 'stationary') {
        return false;
    }
    if (entity.weapon_range <= 0 || entity.weapon_damage <= 0) {
        return false;
    }
    if (!entity.is_combatant) {
        return false;
    }
    if (now === undefined) now = Date.now() / 1000;
    return (now - entity.last_fired) >= entity.weapon_cooldown;
}

// =========================================================================
// Entity tick (simulation step)
// =========================================================================

/**
 * Advance entity simulation by dt seconds.
 * Handles battery drain and movement (via movement controller or legacy linear).
 * @param {Object} entity
 * @param {number} dt - Delta time in seconds
 */
export function tickEntity(entity, dt) {
    if (entity.status === 'destroyed' || entity.status === 'low_battery' ||
        entity.status === 'neutralized' || entity.status === 'escaped' ||
        entity.status === 'eliminated') {
        return;
    }

    // Battery drain
    const drain = (DRAIN_RATES[entity.asset_type] !== undefined ? DRAIN_RATES[entity.asset_type] : 0.001) * dt;
    entity.battery = Math.max(0, entity.battery - drain);
    if (entity.battery < 0.05) {
        entity.status = 'low_battery';
        return;
    }

    if (entity.movement !== null) {
        _tickWithController(entity, dt);
    } else {
        _tickLegacy(entity, dt);
    }
}

/**
 * Movement driven by MovementController (smooth acceleration/turns).
 * @param {Object} entity
 * @param {number} dt
 * @private
 */
function _tickWithController(entity, dt) {
    const mc = entity.movement;

    // Sync waypoints to controller if they changed externally
    // (simple reference check like Python's id())
    if (entity._waypoints_version !== entity.waypoints) {
        entity._waypoints_version = entity.waypoints;
        if (entity.waypoints.length > 0) {
            mc.setWaypoints(entity.waypoints, entity.loop_waypoints);
        } else {
            mc.stop();
        }
    }

    // No waypoints and controller has arrived => idle/stationary
    if (entity.waypoints.length === 0 && mc.arrived) {
        if (entity.status === 'active') entity.status = 'idle';
        return;
    }
    if (entity.speed <= 0) {
        if (entity.status === 'active') entity.status = 'stationary';
        return;
    }

    // Tick the controller (imported from movement.js)
    mc.tick(dt);

    // Sync position and heading from controller.
    // MovementController uses math convention: 0=east, 90=north (atan2(dy,dx)).
    // Entity uses map convention: 0=north, 90=east (atan2(dx,dy)).
    entity.position = { x: mc.x, y: mc.y };
    entity.heading = (90.0 - mc.heading) % 360.0;
    if (entity.heading < 0) entity.heading += 360.0;

    // Check arrival
    if (mc.arrived) {
        if (entity.alliance === 'neutral') {
            entity.status = 'despawned';
        } else if (entity.alliance === 'hostile') {
            entity.status = 'escaped';
        } else if (entity.loop_waypoints) {
            // Friendly patrol — controller handles looping
        } else {
            entity.status = 'arrived';
        }
    }
}

/**
 * Legacy linear movement (non-combatants: neutrals, animals, etc.).
 * @param {Object} entity
 * @param {number} dt
 * @private
 */
function _tickLegacy(entity, dt) {
    if (entity.waypoints.length === 0) {
        if (entity.status === 'active') entity.status = 'idle';
        return;
    }
    if (entity.speed <= 0) {
        if (entity.status === 'active') entity.status = 'stationary';
        return;
    }

    const wp = entity.waypoints[entity._waypoint_index];
    const tx = wp.x;
    const ty = wp.y;

    const dx = tx - entity.position.x;
    const dy = ty - entity.position.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < 1.0) {
        // Arrived at waypoint
        if (entity._waypoint_index >= entity.waypoints.length - 1) {
            if (entity.alliance === 'neutral') {
                entity.status = 'despawned';
            } else if (entity.alliance === 'hostile') {
                entity.status = 'escaped';
            } else if (entity.loop_waypoints) {
                entity._waypoint_index = 0;
            } else {
                entity.status = 'arrived';
            }
            return;
        }
        entity._waypoint_index += 1;
        return;
    }

    // Update heading: atan2(dx, dy) so 0 = north (+y)
    entity.heading = Math.atan2(dx, dy) * (180 / Math.PI);

    // Move toward waypoint
    const step = Math.min(entity.speed * dt, dist);
    entity.position = {
        x: entity.position.x + (dx / dist) * step,
        y: entity.position.y + (dy / dist) * step,
    };
}

// =========================================================================
// Serialization (mirrors SimulationTarget.to_dict)
// =========================================================================

/**
 * Serialize entity to a plain object matching Python SimulationTarget.to_dict().
 * @param {Object} entity
 * @param {string|null} [viewer_alliance='friendly'] - Viewer alliance for fog-of-war
 * @returns {Object}
 */
export function entityToDict(entity, viewer_alliance = 'friendly') {
    const d = {
        target_id: entity.target_id,
        name: entity.name,
        alliance: entity.alliance,
        asset_type: entity.asset_type,
        position: { x: entity.position.x, y: entity.position.y },
        lat: 0.0,
        lng: 0.0,
        alt: 0.0,
        heading: entity.heading,
        altitude: entity.altitude,
        speed: entity.speed,
        battery: Math.round(entity.battery * 10000) / 10000,
        status: entity.status,
        waypoints: entity.waypoints.map(w => ({ x: w.x, y: w.y })),
        loop_waypoints: entity.loop_waypoints,
        health: Math.round(entity.health * 10) / 10,
        max_health: Math.round(entity.max_health * 10) / 10,
        kills: entity.kills,
        is_combatant: entity.is_combatant,
        fsm_state: entity.fsm_state,
        squad_id: entity.squad_id,
        morale: Math.round(entity.morale * 100) / 100,
        degradation: Math.round(entity.degradation * 100) / 100,
        detected: entity.detected,
        visible: entity.visible,
        detected_by: entity.detected_by,
        radio_detected: entity.radio_detected,
        radio_signal_strength: Math.round(entity.radio_signal_strength * 1000) / 1000,
        weapon_range: Math.round(entity.weapon_range * 10) / 10,
        weapon_cooldown: Math.round(entity.weapon_cooldown * 100) / 100,
        last_fired: Math.round(entity.last_fired * 1000) / 1000,
        vision_range: Math.round(entity.vision_range * 10) / 10,
        crowd_role: entity.crowd_role,
        drone_variant: entity.drone_variant,
        instigator_state: entity.instigator_state,
        identified: entity.identified,
        ammo_count: entity.ammo_count,
        ammo_max: entity.ammo_max,
        identity: entity.identity,
        source: entity.source,
        role_name: entity.role_name || null,
    };

    // Inventory serialization with fog-of-war
    if (entity.inventory !== null) {
        if (viewer_alliance === null ||
            entity.alliance === 'friendly' ||
            entity.alliance === viewer_alliance ||
            entity.status === 'eliminated') {
            d.inventory = entity.inventory.toDict ? entity.inventory.toDict() : entity.inventory;
        } else {
            d.inventory = entity.inventory.toFogDict
                ? entity.inventory.toFogDict()
                : { status: 'unknown', item_count: 0 };
        }
    } else {
        d.inventory = null;
    }

    return d;
}
