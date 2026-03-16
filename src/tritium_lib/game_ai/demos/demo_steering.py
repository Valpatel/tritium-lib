# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Steering behavior visual demo.

Run: python3 -m tritium_lib.game_ai.demos.demo_steering

Shows 50 flocking agents with real-time matplotlib animation.
Falls back to ASCII terminal output if matplotlib unavailable.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

try:
    import matplotlib
    if os.environ.get("DISPLAY") or sys.platform == "darwin":
        pass  # keep default backend
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from tritium_lib.game_ai.steering_np import SteeringSystem


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

WORLD_SIZE = 200.0
NUM_AGENTS = 50
NUM_OBSTACLES = 5
NUM_TARGETS = 3
DT = 0.05  # 20 Hz physics


def create_flock(ss: SteeringSystem, n: int) -> list[int]:
    """Spawn n flocking agents in a cluster near the center."""
    indices = []
    for _ in range(n):
        x = np.random.uniform(WORLD_SIZE * 0.3, WORLD_SIZE * 0.7)
        y = np.random.uniform(WORLD_SIZE * 0.3, WORLD_SIZE * 0.7)
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-1, 1)
        idx = ss.add_agent(
            pos=(x, y),
            vel=(vx, vy),
            max_speed=np.random.uniform(1.5, 3.0),
            max_force=2.5,
            behavior=(
                SteeringSystem.WANDER
                | SteeringSystem.SEPARATE
                | SteeringSystem.ALIGN
                | SteeringSystem.COHERE
            ),
        )
        indices.append(idx)
    return indices


def create_obstacles(n: int) -> np.ndarray:
    """Random circular obstacles."""
    obs = np.zeros((n, 3))  # x, y, radius
    for i in range(n):
        obs[i, 0] = np.random.uniform(20, WORLD_SIZE - 20)
        obs[i, 1] = np.random.uniform(20, WORLD_SIZE - 20)
        obs[i, 2] = np.random.uniform(5, 15)
    return obs


def create_targets(n: int) -> np.ndarray:
    """Random seek targets."""
    tgt = np.zeros((n, 2))
    for i in range(n):
        tgt[i, 0] = np.random.uniform(10, WORLD_SIZE - 10)
        tgt[i, 1] = np.random.uniform(10, WORLD_SIZE - 10)
    return tgt


def tick_flock(
    ss: SteeringSystem,
    indices: list[int],
    targets: np.ndarray,
    obstacles: np.ndarray,
    dt: float,
) -> None:
    """Run one simulation step with obstacle avoidance and target cycling."""
    # Point a random 20% of agents toward a random target each tick
    for idx in indices:
        if not ss.active[idx]:
            continue
        if np.random.random() < 0.02:
            t = targets[np.random.randint(len(targets))]
            ss.targets[idx] = t

    # Simple obstacle avoidance: push agents away from obstacles
    n = ss.count
    pos = ss.positions[:n]
    for i in range(len(obstacles)):
        ox, oy, r = obstacles[i]
        dx = pos[:, 0] - ox
        dy = pos[:, 1] - oy
        dist = np.sqrt(dx * dx + dy * dy)
        inside = dist < (r + 3.0)
        if inside.any():
            idx_inside = np.where(inside & ss.active[:n])[0]
            for idx in idx_inside:
                d = max(dist[idx], 0.1)
                force_scale = (r + 3.0 - d) * 2.0
                ss.velocities[idx, 0] += (dx[idx] / d) * force_scale * dt
                ss.velocities[idx, 1] += (dy[idx] / d) * force_scale * dt

    ss.tick(dt)

    # Wrap around world boundaries
    pos = ss.positions[:ss.count]
    pos[:, 0] = pos[:, 0] % WORLD_SIZE
    pos[:, 1] = pos[:, 1] % WORLD_SIZE


# ---------------------------------------------------------------------------
# ASCII terminal renderer
# ---------------------------------------------------------------------------

def render_ascii(
    ss: SteeringSystem,
    obstacles: np.ndarray,
    targets: np.ndarray,
    frame: int,
    fps: float,
    cols: int = 80,
    rows: int = 30,
) -> str:
    """Render the world to an ASCII grid."""
    grid = [["." for _ in range(cols)] for _ in range(rows)]

    # Draw obstacles as 'O'
    for i in range(len(obstacles)):
        ox, oy, r = obstacles[i]
        gc = int(ox / WORLD_SIZE * cols)
        gr = int(oy / WORLD_SIZE * rows)
        if 0 <= gr < rows and 0 <= gc < cols:
            grid[gr][gc] = "O"

    # Draw targets as 'X'
    for i in range(len(targets)):
        gc = int(targets[i, 0] / WORLD_SIZE * cols)
        gr = int(targets[i, 1] / WORLD_SIZE * rows)
        if 0 <= gr < rows and 0 <= gc < cols:
            grid[gr][gc] = "X"

    # Draw agents as directional arrows
    n = ss.count
    active = ss.active[:n]
    pos = ss.positions[:n]
    vel = ss.velocities[:n]
    arrows = {0: ">", 1: "v", 2: "<", 3: "^"}
    for i in range(n):
        if not active[i]:
            continue
        gc = int(pos[i, 0] / WORLD_SIZE * cols) % cols
        gr = int(pos[i, 1] / WORLD_SIZE * rows) % rows
        # Direction from velocity
        angle = math.atan2(vel[i, 1], vel[i, 0])
        quadrant = int((angle + math.pi) / (math.pi / 2)) % 4
        grid[gr][gc] = arrows.get(quadrant, "*")

    lines = ["".join(row) for row in grid]
    header = f"  Frame {frame:5d} | FPS {fps:6.1f} | Agents {int(active.sum()):3d}"
    lines.insert(0, header)
    lines.insert(1, "-" * cols)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matplotlib renderer
# ---------------------------------------------------------------------------

def run_matplotlib(duration: float) -> None:
    """Run the demo with matplotlib animation."""
    ss = SteeringSystem(max_agents=256)
    indices = create_flock(ss, NUM_AGENTS)
    obstacles = create_obstacles(NUM_OBSTACLES)
    targets = create_targets(NUM_TARGETS)

    # Tune flocking weights
    ss.w_separate = 2.0
    ss.w_align = 1.0
    ss.w_cohere = 0.8
    ss.w_wander = 0.6

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0e0e14")
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect("equal")
    ax.set_title("Tritium Steering Demo -- 50 Flocking Agents", color="#00f0ff", fontsize=14)
    ax.tick_params(colors="#555555")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    # Draw obstacles
    for i in range(len(obstacles)):
        circle = Circle(
            (obstacles[i, 0], obstacles[i, 1]), obstacles[i, 2],
            color="#ff2a6d", alpha=0.4, linewidth=1.5, fill=True,
        )
        ax.add_patch(circle)

    # Draw targets
    target_scatter = ax.scatter(
        targets[:, 0], targets[:, 1],
        c="#05ffa1", marker="*", s=200, zorder=5, label="Targets",
    )

    # Agent scatter
    agent_scatter = ax.scatter([], [], c="#00f0ff", s=20, zorder=3, label="Agents")

    # Heading quiver
    quiver = ax.quiver(
        [], [], [], [],
        color="#00f0ff", alpha=0.6, scale=60, width=0.003, zorder=4,
    )

    fps_text = ax.text(
        5, WORLD_SIZE - 8, "", color="#fcee0a", fontsize=11,
        fontfamily="monospace",
    )
    legend = ax.legend(loc="lower right", facecolor="#12121a", edgecolor="#333333",
                       labelcolor="#cccccc", fontsize=9)

    frame_count = [0]
    t_start = [time.time()]
    t_end = time.time() + duration

    def update(frame_num):
        now = time.time()
        if now >= t_end:
            plt.close(fig)
            return (agent_scatter, quiver, fps_text)

        tick_flock(ss, indices, targets, obstacles, DT)
        frame_count[0] += 1

        elapsed = now - t_start[0]
        fps = frame_count[0] / max(elapsed, 0.001)

        # Update agent positions
        n = ss.count
        active = ss.active[:n]
        pos = ss.positions[:n][active]
        vel = ss.velocities[:n][active]

        agent_scatter.set_offsets(pos)

        # Quiver for headings
        if len(pos) > 0:
            speed = np.linalg.norm(vel, axis=1, keepdims=True)
            safe_speed = np.maximum(speed, 1e-6)
            unit = vel / safe_speed
            quiver.set_offsets(pos)
            quiver.set_UVC(unit[:, 0], unit[:, 1])

        fps_text.set_text(
            f"FPS: {fps:.1f}  Agents: {int(active.sum())}  "
            f"Time: {elapsed:.1f}s/{duration:.0f}s"
        )

        return (agent_scatter, quiver, fps_text)

    ani = FuncAnimation(fig, update, interval=int(DT * 1000), blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    elapsed = time.time() - t_start[0]
    fps = frame_count[0] / max(elapsed, 0.001)
    print(f"\nCompleted {frame_count[0]} frames in {elapsed:.1f}s ({fps:.1f} FPS)")


# ---------------------------------------------------------------------------
# Terminal renderer
# ---------------------------------------------------------------------------

def run_terminal(duration: float) -> None:
    """Run the demo with ASCII terminal output."""
    ss = SteeringSystem(max_agents=256)
    indices = create_flock(ss, NUM_AGENTS)
    obstacles = create_obstacles(NUM_OBSTACLES)
    targets = create_targets(NUM_TARGETS)

    ss.w_separate = 2.0
    ss.w_align = 1.0
    ss.w_cohere = 0.8
    ss.w_wander = 0.6

    frame = 0
    t_start = time.time()
    t_end = t_start + duration

    try:
        while time.time() < t_end:
            tick_flock(ss, indices, targets, obstacles, DT)
            frame += 1

            elapsed = time.time() - t_start
            fps = frame / max(elapsed, 0.001)

            if frame % 10 == 0:
                screen = render_ascii(ss, obstacles, targets, frame, fps)
                sys.stdout.write("\033[H\033[J")  # clear terminal
                sys.stdout.write(screen + "\n")
                sys.stdout.flush()

            # Limit to ~20 FPS visually
            time.sleep(max(0, DT - (time.time() - t_start - (frame - 1) * DT)))
    except KeyboardInterrupt:
        pass

    elapsed = time.time() - t_start
    fps = frame / max(elapsed, 0.001)
    print(f"\nCompleted {frame} frames in {elapsed:.1f}s ({fps:.1f} FPS)")


# ---------------------------------------------------------------------------
# Headless (stats only)
# ---------------------------------------------------------------------------

def run_headless(duration: float) -> None:
    """Run simulation headless, print summary stats."""
    ss = SteeringSystem(max_agents=256)
    indices = create_flock(ss, NUM_AGENTS)
    obstacles = create_obstacles(NUM_OBSTACLES)
    targets = create_targets(NUM_TARGETS)

    ss.w_separate = 2.0
    ss.w_align = 1.0
    ss.w_cohere = 0.8
    ss.w_wander = 0.6

    frame = 0
    t_start = time.time()
    t_end = t_start + duration

    while time.time() < t_end:
        tick_flock(ss, indices, targets, obstacles, DT)
        frame += 1

    elapsed = time.time() - t_start
    fps = frame / max(elapsed, 0.001)
    active = int(ss.active[:ss.count].sum())

    print("=" * 50)
    print("  Tritium Steering Demo -- Headless Results")
    print("=" * 50)
    print(f"  Agents:      {NUM_AGENTS}")
    print(f"  Obstacles:   {NUM_OBSTACLES}")
    print(f"  Targets:     {NUM_TARGETS}")
    print(f"  Duration:    {elapsed:.1f}s")
    print(f"  Frames:      {frame}")
    print(f"  FPS:         {fps:.1f}")
    print(f"  Active:      {active}")
    print(f"  Tick time:   {elapsed / max(frame, 1) * 1000:.2f} ms")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tritium steering behavior visual demo")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    parser.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
    args = parser.parse_args()

    np.random.seed(42)

    if args.headless:
        run_headless(args.duration)
    elif HAS_MPL:
        run_matplotlib(args.duration)
    else:
        print("matplotlib not available, falling back to terminal renderer")
        run_terminal(args.duration)


if __name__ == "__main__":
    main()
