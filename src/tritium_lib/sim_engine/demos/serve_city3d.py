#!/usr/bin/env python3
# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Serve the 3D city demo. Run: python3 -m tritium_lib.sim_engine.ai.demos.serve_city3d"""
import http.server
import os

os.chdir(os.path.dirname(__file__))
print("City 3D Demo: http://localhost:8888/city3d.html")
http.server.HTTPServer(("", 8888), http.server.SimpleHTTPRequestHandler).serve_forever()
