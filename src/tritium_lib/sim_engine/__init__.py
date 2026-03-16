# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tritium Simulation Engine — combat, NPC AI, physics, effects, audio.

A comprehensive tactical simulation engine covering land, sea, and air
combat with terrain, weather, crowds, logistics, medical, intelligence,
and scoring systems. Every module is pure Python with no rendering or
game-engine dependencies.

Sub-packages:
    sim_engine.ai       — Steering, pathfinding, ambient NPCs, combat AI, behavior trees
    sim_engine.audio    — Spatial audio math
    sim_engine.debug    — Debug data streams
    sim_engine.effects  — Particle systems, explosions, weapon fire
    sim_engine.physics  — Collision detection and vehicle dynamics
    sim_engine.demos    — Standalone demo scripts

Top-level modules:
    world.py          — World container, WorldBuilder, WORLD_PRESETS
    scenario.py       — Scenario config, events, objectives, waves
    renderer.py       — Render-layer descriptions for frontend
    units.py          — Infantry/vehicle unit types and templates
    vehicles.py       — Ground and aerial vehicle simulation
    arsenal.py        — 35+ weapons, projectiles, area effects
    damage.py         — Hit resolution, explosions, burst fire
    terrain.py        — Procedural heightmaps, LOS, cover, movement cost
    environment.py    — Weather, time-of-day, environmental effects
    crowd.py          — Civilian crowd dynamics and mood propagation
    destruction.py    — Structure damage, fire spread, debris
    detection.py      — Sensor simulation, detection probability
    naval.py          — Naval combat, ship physics, torpedoes
    air_combat.py     — Air combat, missiles, anti-air
    comms.py          — Radio communications, jamming
    medical.py        — Injury, triage, casualty evacuation
    logistics.py      — Supply chains, resupply, consumption
    fortifications.py — Engineering: bunkers, barriers, mines
    asymmetric.py     — Guerrilla warfare, traps, IEDs
    civilian.py       — Civilian population, infrastructure, collateral
    intel.py          — Intelligence gathering, fog of war, fusion
    scoring.py        — Score tracking, achievements, scorecards

Usage::

    from tritium_lib.sim_engine import World, WorldBuilder, WORLD_PRESETS
    from tritium_lib.sim_engine import Unit, create_unit, UNIT_TEMPLATES
    from tritium_lib.sim_engine import ARSENAL, ProjectileSimulator
    from tritium_lib.sim_engine import HeightMap, LineOfSight, CoverMap
    from tritium_lib.sim_engine import CrowdSimulator, NavalCombatEngine
"""

# ---------------------------------------------------------------------------
# Core: World, Scenario, Renderer
# ---------------------------------------------------------------------------
try:
    from .world import World, WorldBuilder, WorldConfig, WORLD_PRESETS
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.world not available: {_e}")

try:
    from .scenario import (
        Scenario,
        ScenarioConfig,
        SimEvent,
        SimState,
        WaveConfig,
        Objective,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.scenario not available: {_e}")

try:
    from .renderer import (
        RenderLayer,
        SimRenderer,
        UnitRenderer,
        ProjectileRenderer,
        EffectRenderer,
        WeatherRenderer,
        TerrainRenderer,
        CrowdRenderer,
        alliance_color,
        mood_color,
        damage_flash,
        tracer_color,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.renderer not available: {_e}")

# ---------------------------------------------------------------------------
# AI: Steering, Pathfinding, Ambient, Combat AI, Squads, Tactics
# ---------------------------------------------------------------------------
try:
    from .ai import (
        # Steering
        Vec2, distance, magnitude, normalize, truncate, heading_to_vec,
        seek, flee, arrive, wander, pursue, evade,
        follow_path, avoid_obstacles,
        separate, align, cohere, flock, formation_offset,
        # Pathfinding
        RoadNetwork, WalkableArea, plan_patrol_route, plan_random_walk,
        # Ambient
        ActivityProfile, AmbientEntity, AmbientSimulator, EntityState, EntityType,
        # City sim
        ActivityState, Building, BuildingType, DailySchedule, ErrandType,
        ERRAND_DURATIONS, NeighborhoodSim, Resident, ResidentRole,
        ScheduleEntry, SimVehicle, VehicleType,
        state_movement_type, state_rf_emission, state_visible_on_map,
        # Combat AI
        find_cover, is_in_cover, rate_cover_position,
        compute_flank_position, is_flanking,
        optimal_engagement_range, should_engage, should_retreat,
        formation_positions, assign_targets,
        suppression_cone, is_suppressed,
        make_assault_tree, make_defender_tree, make_sniper_tree,
        make_squad_leader_tree,
        # RF Signatures
        BuildingRFProfile, PersonRFProfile, RFSignatureGenerator, VehicleRFProfile,
        # Squad
        MoralePropagation, Order, Squad, SquadRole, SquadState, SquadTactics,
        # Tactics
        AIPersonality, PERSONALITY_PRESETS, TacticalAction,
        TacticalSituation, TacticsEngine, ThreatAssessment,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.ai not available: {_e}")

# ---------------------------------------------------------------------------
# Units: Infantry and vehicle unit types
# ---------------------------------------------------------------------------
try:
    from .units import (
        Unit,
        UnitType,
        UnitStats,
        UnitState,
        Alliance,
        create_unit,
        UNIT_TEMPLATES,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.units not available: {_e}")

# ---------------------------------------------------------------------------
# Vehicles: Ground vehicles, drones, convoys
# ---------------------------------------------------------------------------
try:
    from .vehicles import (
        VehicleClass,
        VehicleState,
        VehiclePhysicsEngine,
        DroneController,
        ConvoySimulator,
        create_vehicle,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.vehicles not available: {_e}")

# ---------------------------------------------------------------------------
# Combat: Weapons, damage, projectiles
# ---------------------------------------------------------------------------
try:
    from .arsenal import (
        Weapon,
        WeaponCategory,
        Projectile,
        ProjectileType,
        ProjectileSimulator,
        AreaEffect,
        AreaEffectManager,
        ARSENAL,
        get_weapon,
        weapons_by_category,
        create_explosion_effect,
        create_smoke_effect,
        create_fire_effect,
        create_teargas_effect,
        create_flashbang_effect,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.arsenal not available: {_e}")

try:
    from .damage import (
        DamageType,
        HitResult,
        DamageTracker,
        resolve_attack,
        resolve_explosion,
        resolve_burst,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.damage not available: {_e}")

# ---------------------------------------------------------------------------
# Naval: Ship combat, torpedoes, formations
# ---------------------------------------------------------------------------
try:
    from .naval import (
        ShipClass,
        ShipState,
        Torpedo,
        ShellProjectile,
        CombatEffect,
        NavalPhysics,
        FormationType,
        NavalFormation,
        NavalCombatEngine,
        SHIP_TEMPLATES,
        create_ship,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.naval not available: {_e}")

# ---------------------------------------------------------------------------
# Air: Aircraft combat, missiles, anti-air
# ---------------------------------------------------------------------------
try:
    from .air_combat import (
        AircraftClass,
        AircraftState,
        Missile,
        AntiAir,
        AirCombatEffect,
        AirCombatEngine,
        AIRCRAFT_TEMPLATES,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.air_combat not available: {_e}")

# ---------------------------------------------------------------------------
# Terrain: Heightmaps, line of sight, cover, movement cost
# ---------------------------------------------------------------------------
try:
    from .terrain import HeightMap, LineOfSight, CoverMap, MovementCost
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.terrain not available: {_e}")

# ---------------------------------------------------------------------------
# Environment: Weather, time of day
# ---------------------------------------------------------------------------
try:
    from .environment import (
        TimeOfDay,
        Weather,
        WeatherState,
        WeatherEffects,
        WeatherSimulator,
        Environment,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.environment not available: {_e}")

# ---------------------------------------------------------------------------
# Crowd: Civilian crowd simulation
# ---------------------------------------------------------------------------
try:
    from .crowd import (
        CrowdMood,
        CrowdMember,
        CrowdEvent,
        CrowdSimulator,
        CROWD_SCENARIOS,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.crowd not available: {_e}")

# ---------------------------------------------------------------------------
# Destruction: Structures, fire, debris
# ---------------------------------------------------------------------------
try:
    from .destruction import (
        StructureType,
        DamageLevel,
        Structure,
        Fire,
        Debris,
        DestructionEngine,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.destruction not available: {_e}")

# ---------------------------------------------------------------------------
# Detection: Sensors, signatures, detection probability
# ---------------------------------------------------------------------------
try:
    from .detection import (
        SensorType,
        Sensor,
        SignatureProfile,
        Detection,
        DetectionEngine,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.detection not available: {_e}")

# ---------------------------------------------------------------------------
# Comms: Radio communications, jamming
# ---------------------------------------------------------------------------
try:
    from .comms import (
        RadioType,
        RadioChannel,
        Radio,
        RadioMessage,
        Jammer,
        CommsSimulator,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.comms not available: {_e}")

# ---------------------------------------------------------------------------
# Medical: Injuries, triage, evacuation
# ---------------------------------------------------------------------------
try:
    from .medical import (
        InjuryType,
        InjurySeverity,
        TriageCategory,
        Injury,
        CasualtyState,
        EvacRequest,
        MedicalEngine,
        BODY_PARTS,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.medical not available: {_e}")

# ---------------------------------------------------------------------------
# Logistics: Supply chains, resupply
# ---------------------------------------------------------------------------
try:
    from .logistics import (
        SupplyType,
        SupplyCache,
        SupplyRequest,
        SupplyRoute,
        LowSupplyWarning,
        LogisticsEngine,
        cache_from_preset,
        LOW_SUPPLY_THRESHOLD,
        DEFAULT_RESUPPLY_RANGE,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.logistics not available: {_e}")

# ---------------------------------------------------------------------------
# Fortifications: Engineering, mines, barriers
# ---------------------------------------------------------------------------
try:
    from .fortifications import (
        FortificationType,
        Fortification,
        Mine,
        EngineeringEngine,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.fortifications not available: {_e}")

# ---------------------------------------------------------------------------
# Asymmetric: Guerrilla warfare, traps, IEDs
# ---------------------------------------------------------------------------
try:
    from .asymmetric import (
        TrapType,
        Trap,
        GuerrillaCell,
        AsymmetricEngine,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.asymmetric not available: {_e}")

# ---------------------------------------------------------------------------
# Civilian: Population, infrastructure, collateral damage
# ---------------------------------------------------------------------------
try:
    from .civilian import (
        CivilianState,
        InfrastructureType,
        Civilian,
        Infrastructure,
        CollateralDamage,
        CivilianSimulator,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.civilian not available: {_e}")

# ---------------------------------------------------------------------------
# Intel: Intelligence gathering, fog of war, fusion
# ---------------------------------------------------------------------------
try:
    from .intel import (
        IntelType,
        IntelReport,
        ReconMission,
        FogOfWar,
        IntelFusion,
        IntelEngine,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.intel not available: {_e}")

# ---------------------------------------------------------------------------
# Scoring: Score tracking, achievements
# ---------------------------------------------------------------------------
try:
    from .scoring import (
        ScoreCategory,
        Achievement,
        UnitScorecard,
        TeamScorecard,
        ScoringEngine,
        ACHIEVEMENTS,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.scoring not available: {_e}")

# ---------------------------------------------------------------------------
# Effects: Particles, explosions, weapon fire
# ---------------------------------------------------------------------------
try:
    from .effects import (
        EffectsManager,
        Particle,
        ParticleEmitter,
        blood_splatter,
        debris as debris_effect,
        explosion,
        fire as fire_effect,
        muzzle_flash,
        smoke,
        sparks,
        tracer,
        WEAPONS as WEAPON_PROFILES,
        FireMode,
        FiredRound,
        WeaponFirer,
        WeaponProfile,
        create_firer,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.effects not available: {_e}")

# ---------------------------------------------------------------------------
# Audio: Spatial audio math
# ---------------------------------------------------------------------------
try:
    from .audio import (
        SoundEvent,
        distance_attenuation,
        doppler_factor,
        explosion_parameters,
        gunshot_layers,
        occlusion_factor,
        propagation_delay,
        reverb_level,
        stereo_pan,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.audio not available: {_e}")

# ---------------------------------------------------------------------------
# Soundtrack: Game audio event system for frontend
# ---------------------------------------------------------------------------
try:
    from .soundtrack import (
        AudioCategory,
        AudioCue,
        MusicState,
        SoundtrackEngine,
        SOUND_MAP,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.soundtrack not available: {_e}")

# ---------------------------------------------------------------------------
# Physics: Collision, rigid bodies, vehicle dynamics
# ---------------------------------------------------------------------------
try:
    from .physics import (
        CollisionEvent,
        PhysicsWorld,
        RigidBody,
        VehiclePhysics,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.physics not available: {_e}")

# ---------------------------------------------------------------------------
# Debug: Debug data streams
# ---------------------------------------------------------------------------
try:
    from .debug import DebugFrame, DebugStream, DebugOverlay
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.debug not available: {_e}")

# ---------------------------------------------------------------------------
# Morale: Unit morale and psychology
# ---------------------------------------------------------------------------
try:
    from .morale import (
        MoraleEngine,
        MoraleEvent,
        MoraleEventType,
        MoraleState,
        UnitMorale,
        COMMANDER_AURA_RADIUS,
        RECOVERY_DELAY,
        RECOVERY_RATE,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.morale not available: {_e}")

# ---------------------------------------------------------------------------
# Electronic Warfare: Jammers, cyber, EMP, spoofing
# ---------------------------------------------------------------------------
try:
    from .electronic_warfare import (
        EWEngine,
        EWJammer,
        JammerType,
        CyberAttack,
        CyberAttackType,
        EMPEvent,
        EMPScale,
        SpoofContact,
        DisruptedSystem,
        EMP_PRESETS,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.electronic_warfare not available: {_e}")

# ---------------------------------------------------------------------------
# Supply Routes: Supply lines, convoys, unit supply tracking
# ---------------------------------------------------------------------------
try:
    from .supply_routes import (
        SupplyRouteEngine,
        SupplyLine,
        SupplyConvoy,
        UnitSupplyState,
        ConvoyStatus,
        RouteStatus,
        SupplyLevel,
        DELIVERY_RANGE,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.supply_routes not available: {_e}")

# ---------------------------------------------------------------------------
# Replay: Recording, playback, analysis, export
# ---------------------------------------------------------------------------
try:
    from .replay import (
        ReplayFrame,
        ReplayRecorder,
        ReplayPlayer,
        ReplayAnalyzer,
        ReplayExporter,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.replay not available: {_e}")

# ---------------------------------------------------------------------------
# Commander: Battle narration, tactical advisor
# ---------------------------------------------------------------------------
try:
    from .commander import (
        NarrationEvent,
        CommanderPersonality,
        BattleNarrator,
        TacticalAdvisor,
        NarrationLog,
        PERSONALITIES,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.commander not available: {_e}")

# ---------------------------------------------------------------------------
# Status Effects: Buffs, debuffs, DOT/HOT, crowd control
# ---------------------------------------------------------------------------
try:
    from .status_effects import (
        EffectType,
        StatusEffect,
        StatusEffectEngine,
        EFFECTS_CATALOG,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.status_effects not available: {_e}")

# ---------------------------------------------------------------------------
# Economy: Resources, build queue, tech tree, RTS layer
# ---------------------------------------------------------------------------
try:
    from .economy import (
        ResourceType,
        ResourcePool,
        UnitCost,
        BuildQueue,
        TechTree,
        EconomyEngine,
        UNIT_COSTS as UNIT_COSTS,
        TECH_TREE as TECH_TREE,
        ECONOMY_PRESETS,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.economy not available: {_e}")


# ---------------------------------------------------------------------------
# Event Bus: Global pub/sub for all sim modules
# ---------------------------------------------------------------------------
try:
    from .event_bus import (
        SimEventType,
        SimEvent as SimBusEvent,  # Alias to avoid clash with scenario.SimEvent
        SimEventBus,
        EventFilter,
        EventListener,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.event_bus not available: {_e}")

# ---------------------------------------------------------------------------
# Abilities: Unit special powers and skills
# ---------------------------------------------------------------------------
try:
    from .abilities import (
        AbilityType,
        TargetType,
        Ability,
        AbilityEngine,
        ABILITIES,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.abilities not available: {_e}")

# ---------------------------------------------------------------------------
# Cyber Warfare: Cyber assets, capabilities, effects engine
# ---------------------------------------------------------------------------
try:
    from .cyber import (
        CyberAttackType as CyberAttackType2,  # avoid clash with electronic_warfare
        CyberCapability,
        CyberAsset,
        CyberEffect,
        CyberWarfareEngine,
        CYBER_PRESETS,
        create_asset_from_preset,
    )
except ImportError as _e:  # pragma: no cover
    import warnings as _w; _w.warn(f"sim_engine.cyber not available: {_e}")

# ---------------------------------------------------------------------------
# __all__ — comprehensive public API
# ---------------------------------------------------------------------------
__all__ = [
    # --- Core ---
    "World", "WorldBuilder", "WorldConfig", "WORLD_PRESETS",
    "Scenario", "ScenarioConfig", "SimEvent", "SimState",
    "WaveConfig", "Objective",
    "RenderLayer", "SimRenderer", "UnitRenderer", "ProjectileRenderer",
    "EffectRenderer", "WeatherRenderer", "TerrainRenderer", "CrowdRenderer",
    "alliance_color", "mood_color", "damage_flash", "tracer_color",
    # --- AI: Steering ---
    "Vec2", "distance", "magnitude", "normalize", "truncate", "heading_to_vec",
    "seek", "flee", "arrive", "wander", "pursue", "evade",
    "follow_path", "avoid_obstacles",
    "separate", "align", "cohere", "flock", "formation_offset",
    # --- AI: Pathfinding ---
    "RoadNetwork", "WalkableArea", "plan_patrol_route", "plan_random_walk",
    # --- AI: Ambient ---
    "ActivityProfile", "AmbientEntity", "AmbientSimulator",
    "EntityState", "EntityType",
    # --- AI: City Sim ---
    "ActivityState", "Building", "BuildingType", "DailySchedule",
    "ErrandType", "ERRAND_DURATIONS", "NeighborhoodSim", "Resident",
    "ResidentRole", "ScheduleEntry", "SimVehicle", "VehicleType",
    "state_movement_type", "state_rf_emission", "state_visible_on_map",
    # --- AI: Combat AI ---
    "find_cover", "is_in_cover", "rate_cover_position",
    "compute_flank_position", "is_flanking",
    "optimal_engagement_range", "should_engage", "should_retreat",
    "formation_positions", "assign_targets",
    "suppression_cone", "is_suppressed",
    "make_assault_tree", "make_defender_tree",
    "make_sniper_tree", "make_squad_leader_tree",
    # --- AI: RF Signatures ---
    "BuildingRFProfile", "PersonRFProfile",
    "RFSignatureGenerator", "VehicleRFProfile",
    # --- AI: Squad ---
    "MoralePropagation", "Order", "Squad",
    "SquadRole", "SquadState", "SquadTactics",
    # --- AI: Tactics ---
    "AIPersonality", "PERSONALITY_PRESETS", "TacticalAction",
    "TacticalSituation", "TacticsEngine", "ThreatAssessment",
    # --- Units ---
    "Unit", "UnitType", "UnitStats", "UnitState",
    "Alliance", "create_unit", "UNIT_TEMPLATES",
    # --- Vehicles ---
    "VehicleClass", "VehicleState", "VehiclePhysicsEngine",
    "DroneController", "ConvoySimulator", "create_vehicle",
    # --- Combat: Arsenal ---
    "Weapon", "WeaponCategory", "Projectile", "ProjectileType",
    "ProjectileSimulator", "AreaEffect", "AreaEffectManager", "ARSENAL",
    "get_weapon", "weapons_by_category",
    "create_explosion_effect", "create_smoke_effect",
    "create_fire_effect", "create_teargas_effect", "create_flashbang_effect",
    # --- Combat: Damage ---
    "DamageType", "HitResult", "DamageTracker",
    "resolve_attack", "resolve_explosion", "resolve_burst",
    # --- Naval ---
    "ShipClass", "ShipState", "Torpedo", "ShellProjectile",
    "CombatEffect", "NavalPhysics", "FormationType",
    "NavalFormation", "NavalCombatEngine",
    "SHIP_TEMPLATES", "create_ship",
    # --- Air ---
    "AircraftClass", "AircraftState", "Missile", "AntiAir",
    "AirCombatEffect", "AirCombatEngine", "AIRCRAFT_TEMPLATES",
    # --- Terrain ---
    "HeightMap", "LineOfSight", "CoverMap", "MovementCost",
    # --- Environment ---
    "TimeOfDay", "Weather", "WeatherState", "WeatherEffects",
    "WeatherSimulator", "Environment",
    # --- Crowd ---
    "CrowdMood", "CrowdMember", "CrowdEvent",
    "CrowdSimulator", "CROWD_SCENARIOS",
    # --- Destruction ---
    "StructureType", "DamageLevel", "Structure",
    "Fire", "Debris", "DestructionEngine",
    # --- Detection ---
    "SensorType", "Sensor", "SignatureProfile",
    "Detection", "DetectionEngine",
    # --- Comms ---
    "RadioType", "RadioChannel", "Radio",
    "RadioMessage", "Jammer", "CommsSimulator",
    # --- Medical ---
    "InjuryType", "InjurySeverity", "TriageCategory",
    "Injury", "CasualtyState", "EvacRequest",
    "MedicalEngine", "BODY_PARTS",
    # --- Logistics ---
    "SupplyType", "SupplyCache", "SupplyRequest", "SupplyRoute",
    "LowSupplyWarning", "LogisticsEngine", "cache_from_preset",
    "LOW_SUPPLY_THRESHOLD", "DEFAULT_RESUPPLY_RANGE",
    # --- Fortifications ---
    "FortificationType", "Fortification", "Mine", "EngineeringEngine",
    # --- Asymmetric ---
    "TrapType", "Trap", "GuerrillaCell", "AsymmetricEngine",
    # --- Civilian ---
    "CivilianState", "InfrastructureType", "Civilian",
    "Infrastructure", "CollateralDamage", "CivilianSimulator",
    # --- Intel ---
    "IntelType", "IntelReport", "ReconMission",
    "FogOfWar", "IntelFusion", "IntelEngine",
    # --- Scoring ---
    "ScoreCategory", "Achievement", "UnitScorecard",
    "TeamScorecard", "ScoringEngine", "ACHIEVEMENTS",
    # --- Effects ---
    "EffectsManager", "Particle", "ParticleEmitter",
    "blood_splatter", "debris_effect", "explosion",
    "fire_effect", "muzzle_flash", "smoke", "sparks", "tracer",
    "WEAPON_PROFILES", "FireMode", "FiredRound",
    "WeaponFirer", "WeaponProfile", "create_firer",
    # --- Audio ---
    "SoundEvent", "distance_attenuation", "doppler_factor",
    "explosion_parameters", "gunshot_layers", "occlusion_factor",
    "propagation_delay", "reverb_level", "stereo_pan",
    # --- Soundtrack ---
    "AudioCategory", "AudioCue", "MusicState",
    "SoundtrackEngine", "SOUND_MAP",
    # --- Physics ---
    "CollisionEvent", "PhysicsWorld", "RigidBody", "VehiclePhysics",
    # --- Debug ---
    "DebugFrame", "DebugStream", "DebugOverlay",
    # --- Replay ---
    "ReplayFrame", "ReplayRecorder", "ReplayPlayer",
    "ReplayAnalyzer", "ReplayExporter",
    # --- Morale ---
    "MoraleEngine", "MoraleEvent", "MoraleEventType",
    "MoraleState", "UnitMorale",
    "COMMANDER_AURA_RADIUS", "RECOVERY_DELAY", "RECOVERY_RATE",
    # --- Electronic Warfare ---
    "EWEngine", "EWJammer", "JammerType",
    "CyberAttack", "CyberAttackType",
    "EMPEvent", "EMPScale", "SpoofContact",
    "DisruptedSystem", "EMP_PRESETS",
    # --- Supply Routes ---
    "SupplyRouteEngine", "SupplyLine", "SupplyConvoy",
    "UnitSupplyState", "ConvoyStatus", "RouteStatus",
    "SupplyLevel", "DELIVERY_RANGE",
    # --- Commander ---
    "NarrationEvent", "CommanderPersonality", "BattleNarrator",
    "TacticalAdvisor", "NarrationLog", "PERSONALITIES",
    # --- Status Effects ---
    "EffectType", "StatusEffect", "StatusEffectEngine", "EFFECTS_CATALOG",
    # --- Economy ---
    "ResourceType", "ResourcePool", "UnitCost", "BuildQueue", "TechTree",
    "EconomyEngine", "UNIT_COSTS", "TECH_TREE", "ECONOMY_PRESETS",
    # --- Event Bus ---
    "SimEventType", "SimBusEvent", "SimEventBus", "EventFilter", "EventListener",
    # --- Abilities ---
    "AbilityType", "TargetType", "Ability", "AbilityEngine", "ABILITIES",
    # --- Cyber Warfare ---
    "CyberAttackType2", "CyberCapability", "CyberAsset",
    "CyberEffect", "CyberWarfareEngine", "CYBER_PRESETS",
    "create_asset_from_preset",
]
