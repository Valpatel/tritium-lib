# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Communications Intelligence (COMINT) — metadata-only analysis.

Analyzes intercepted communications *metadata* (who talks to whom, when,
how often, on what medium) to build network/relationship maps.  **Never**
stores or inspects message content.

Input sources:
  - BLE pairing events (two MACs observed exchanging pairing PDUs)
  - WiFi associations (client MAC associated with AP BSSID)
  - Meshtastic mesh messages (sender node_id → recipient node_id)
  - ESP-NOW frames (peer MAC → peer MAC)

Core classes:
  - ``CommLink`` — a single observed communication between two identifiers
  - ``CommNetwork`` — graph of communication relationships
  - ``CommAnalyzer`` — high-level pattern analysis

Key analysis methods:
  - ``find_communities()`` — detect groups of frequently communicating entities
  - ``find_bridges()`` — identify entities connecting different groups
  - ``communication_timeline()`` — when and with whom an entity communicates
"""

from __future__ import annotations

from .analyzer import (
    CommAnalyzer,
    CommLink,
    CommNetwork,
    CommunityResult,
    BridgeEntity,
    TimelineEntry,
)

__all__ = [
    "CommAnalyzer",
    "CommLink",
    "CommNetwork",
    "CommunityResult",
    "BridgeEntity",
    "TimelineEntry",
]
