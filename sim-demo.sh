#!/bin/bash
# Tritium Sim Engine Demo
# Run a tactical simulation with 3D visualization in your browser.
#
# Usage:
#   ./sim-demo.sh              # Start demo (default: urban_combat)
#   ./sim-demo.sh open_field   # Start with a specific preset
#   ./sim-demo.sh --list       # List available presets
#   ./sim-demo.sh --perf       # Run performance test
#   ./sim-demo.sh --coverage   # Run test coverage report
#
# Presets: urban_combat, open_field, riot_response, convoy_ambush, drone_strike
#
# Opens http://localhost:8888 in your browser automatically.

set -e
cd "$(dirname "$0")"

PRESET="${1:-urban_combat}"
PORT="${SIM_PORT:-9090}"

if [ "$1" = "--perf" ]; then
    echo "=== Running Performance Test ==="
    python3 -m tritium_lib.sim_engine.demos.perf_test "${@:2}"
    exit $?
fi

if [ "$1" = "--coverage" ]; then
    echo "=== Running Test Coverage Report ==="
    python3 -m tritium_lib.sim_engine.demos.test_report "${@:2}"
    exit $?
fi

if [ "$1" = "--list" ] || [ "$1" = "-l" ]; then
    echo "Available presets:"
    echo "  urban_combat   — Night raid: squads + vehicles + buildings + civilians"
    echo "  open_field     — Daylight infantry battle on flat terrain"
    echo "  riot_response  — Crowd control: police vs agitated crowd"
    echo "  convoy_ambush  — Convoy route with IED ambush + guerrilla cells"
    echo "  drone_strike   — Reaper drone orbiting ground targets"
    echo ""
    echo "Usage: ./sim-demo.sh [preset]"
    exit 0
fi

echo "=== Tritium Sim Engine Demo ==="
echo "Preset: $PRESET"
echo "URL:    http://localhost:$PORT"
echo ""

# Install deps if needed
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Installing FastAPI + uvicorn..."
    pip install fastapi uvicorn[standard] 2>/dev/null || pip3 install fastapi uvicorn[standard]
fi

# Open browser after short delay
(sleep 2 && python3 -m webbrowser "http://localhost:$PORT") &

# Run the game server
SIM_PRESET="$PRESET" SIM_PORT="$PORT" python3 -m tritium_lib.sim_engine.demos.game_server
