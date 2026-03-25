# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Entry point for running the graph demo as a module.

Usage:
    python3 -m tritium_lib.graph.demos
"""

from tritium_lib.graph.demos.graph_demo import run_demo

if __name__ == "__main__":
    run_demo()
