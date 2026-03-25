# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.analytics — real-time statistics and trend analysis for the tracking pipeline.

Provides efficient, thread-safe analytics components for computing live
statistics from tracking events.  All sliding windows use O(1) amortised
bucketed counters — no per-event list appends, no linear scans.

Components:

  - **AnalyticsEngine** — unified orchestrator: ingest detections, alerts,
    and correlations, then query live rates, trends, distributions.
  - **TimeWindow** — O(1) sliding time window with bucketed counters.
  - **Counter** — multi-horizon event counter (1min, 5min, 1hr, 24hr).
  - **Histogram** — categorical distribution within a sliding window.
  - **TrendDetector** — linear-regression trend analysis over bucketed data.
  - **TopN** — track top-N items by windowed activity count.
  - **TrendResult** — dataclass for trend analysis results.

Quick start::

    from tritium_lib.analytics import AnalyticsEngine

    engine = AnalyticsEngine()
    engine.record_detection("ble_aabbccdd", source="ble", zone="lobby")
    engine.record_alert("geofence_entry", severity="warning")
    engine.record_correlation("ble_aa", "det_person_1", success=True)
    snap = engine.snapshot()
"""

from tritium_lib.analytics.engine import (
    AnalyticsEngine,
    Counter,
    Histogram,
    TimeWindow,
    TopN,
    TrendDetector,
    TrendResult,
)

__all__ = [
    "AnalyticsEngine",
    "Counter",
    "Histogram",
    "TimeWindow",
    "TopN",
    "TrendDetector",
    "TrendResult",
]
