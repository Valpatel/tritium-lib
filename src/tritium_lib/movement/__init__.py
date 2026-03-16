"""Movement module — Craig Reynolds' steering behaviors.

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
]
