# Simulation Engine Visual Demos

Standalone scripts that prove the simulation works visually and measure performance.
Each demo imports only from `tritium_lib.sim_engine` -- no tritium-sc dependencies.

## Prerequisites

```bash
pip install -e ".[full]"    # numpy required, matplotlib optional
```

## Demos

### demo_steering -- Flocking & Steering Behaviors

50 agents demonstrating separation, alignment, and cohesion (Craig Reynolds' boids).
Agents flock together while seeking randomly-placed targets and avoiding obstacles.

```bash
python3 -m tritium_lib.sim_engine.demos.demo_steering
python3 -m tritium_lib.sim_engine.demos.demo_steering --headless --duration 10
```

### demo_city -- City Life Simulation

50 residents living daily routines in a neighborhood. Office workers commute,
kids go to school, retirees walk dogs. Cars drive on roads, park at destinations.
Simulated clock runs at 10x speed (1 real second = 10 sim minutes).

```bash
python3 -m tritium_lib.sim_engine.demos.demo_city
python3 -m tritium_lib.sim_engine.demos.demo_city --headless --duration 20
```

### demo_perf -- Performance Benchmark

Tests scaling from 50 to 1000 agents. Measures tick time, FPS, and memory usage.
Outputs a formatted table.

```bash
python3 -m tritium_lib.sim_engine.demos.demo_perf
python3 -m tritium_lib.sim_engine.demos.demo_perf --duration 5
```

### demo_rf -- RF Signature Visualization

Shows a neighborhood where every person and car emits BLE/WiFi/TPMS signals.
Demonstrates MAC rotation (phone MAC changes every 15 min but company ID persists)
and persistent TPMS IDs that never change.

```bash
python3 -m tritium_lib.sim_engine.demos.demo_rf
python3 -m tritium_lib.sim_engine.demos.demo_rf --headless --duration 15
```

## Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--headless` | off | Skip visualization, just run sim and print stats |
| `--duration N` | 30 | Run for N seconds |
