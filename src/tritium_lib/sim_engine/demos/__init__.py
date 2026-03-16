# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Visual demo apps for the tritium-lib sim_engine module.

Each demo is a standalone script that can be run directly:

    python3 -m tritium_lib.sim_engine.demos.demo_steering
    python3 -m tritium_lib.sim_engine.demos.demo_city
    python3 -m tritium_lib.sim_engine.demos.demo_perf
    python3 -m tritium_lib.sim_engine.demos.demo_rf

All demos gracefully degrade without matplotlib (terminal output).
All demos accept --headless and --duration flags.
"""
