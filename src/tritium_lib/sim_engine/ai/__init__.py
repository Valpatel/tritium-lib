"""Simulation AI — steering behaviors, pathfinding, ambient NPCs, combat AI.

Pure math, no rendering or game engine dependencies.
Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from .steering import (
    Vec2,
    # Utility
    distance,
    magnitude,
    normalize,
    truncate,
    heading_to_vec,
    # Basic behaviors
    seek,
    flee,
    arrive,
    wander,
    pursue,
    evade,
    # Path following
    follow_path,
    # Obstacle avoidance
    avoid_obstacles,
    # Group behaviors
    separate,
    align,
    cohere,
    flock,
    # Formation
    formation_offset,
)

from .pathfinding import (
    RoadNetwork,
    WalkableArea,
    plan_patrol_route,
    plan_random_walk,
)

from .ambient import (
    ActivityProfile,
    AmbientEntity,
    AmbientSimulator,
    EntityState,
    EntityType,
)

from .city_sim import (
    ActivityState,
    Building,
    BuildingType,
    DailySchedule,
    ErrandType,
    ERRAND_DURATIONS,
    NeighborhoodSim,
    Resident,
    ResidentRole,
    ScheduleEntry,
    SimVehicle,
    VehicleType,
    state_movement_type,
    state_rf_emission,
    state_visible_on_map,
)

from .combat_ai import (
    # Cover
    find_cover,
    is_in_cover,
    rate_cover_position,
    # Flanking
    compute_flank_position,
    is_flanking,
    # Engagement
    optimal_engagement_range,
    should_engage,
    should_retreat,
    # Squad coordination
    formation_positions,
    assign_targets,
    # Suppression
    suppression_cone,
    is_suppressed,
    # Pre-built combat trees
    make_assault_tree,
    make_defender_tree,
    make_sniper_tree,
    make_squad_leader_tree,
)

from .rf_signatures import (
    BuildingRFProfile,
    PersonRFProfile,
    RFSignatureGenerator,
    VehicleRFProfile,
)

from .squad import (
    MoralePropagation,
    Order,
    Squad,
    SquadRole,
    SquadState,
    SquadTactics,
)

from .tactics import (
    AIPersonality,
    PERSONALITY_PRESETS,
    TacticalAction,
    TacticalSituation,
    TacticsEngine,
    ThreatAssessment,
)

# NumPy-vectorized variants (optional — graceful fallback if numpy missing)
try:
    from .steering_np import SteeringSystem, SpatialHash
    from .ambient_np import AmbientSimulatorNP
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

__all__ = [
    "Vec2",
    "distance",
    "magnitude",
    "normalize",
    "truncate",
    "heading_to_vec",
    "seek",
    "flee",
    "arrive",
    "wander",
    "pursue",
    "evade",
    "follow_path",
    "avoid_obstacles",
    "separate",
    "align",
    "cohere",
    "flock",
    "formation_offset",
    "RoadNetwork",
    "WalkableArea",
    "plan_patrol_route",
    "plan_random_walk",
    "ActivityProfile",
    "AmbientEntity",
    "AmbientSimulator",
    "EntityState",
    "EntityType",
    "BuildingRFProfile",
    "PersonRFProfile",
    "RFSignatureGenerator",
    "VehicleRFProfile",
    "ActivityState",
    "Building",
    "BuildingType",
    "DailySchedule",
    "ErrandType",
    "ERRAND_DURATIONS",
    "NeighborhoodSim",
    "Resident",
    "ResidentRole",
    "ScheduleEntry",
    "SimVehicle",
    "VehicleType",
    "state_movement_type",
    "state_rf_emission",
    "state_visible_on_map",
    # Combat AI
    "find_cover",
    "is_in_cover",
    "rate_cover_position",
    "compute_flank_position",
    "is_flanking",
    "optimal_engagement_range",
    "should_engage",
    "should_retreat",
    "formation_positions",
    "assign_targets",
    "suppression_cone",
    "is_suppressed",
    "make_assault_tree",
    "make_defender_tree",
    "make_sniper_tree",
    "make_squad_leader_tree",
    # Squad coordination
    "MoralePropagation",
    "Order",
    "Squad",
    "SquadRole",
    "SquadState",
    "SquadTactics",
    # Tactics engine
    "AIPersonality",
    "PERSONALITY_PRESETS",
    "TacticalAction",
    "TacticalSituation",
    "TacticsEngine",
    "ThreatAssessment",
]
