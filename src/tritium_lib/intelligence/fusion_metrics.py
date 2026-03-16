# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Fusion metrics — tracks correlation pipeline health and performance.

Provides FusionMetrics for monitoring cross-sensor identity fusion:
  - Correlation success rate (confirmed vs rejected by operator)
  - Strategy performance (per-strategy accuracy from feedback)
  - Fusion count by source pair (e.g., BLE+camera fusions/hour)
  - Rolling time-window statistics

Used by the fusion dashboard and for RL weight optimization of
correlation strategies.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyMetric:
    """Performance tracking for a single correlation strategy."""

    name: str
    evaluations: int = 0
    contributed: int = 0  # times score > 0 when used
    confirmed: int = 0  # operator confirmed correct
    rejected: int = 0  # operator rejected as wrong
    total_score: float = 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy from operator feedback (confirmed / total feedback)."""
        total = self.confirmed + self.rejected
        if total == 0:
            return 0.0
        return self.confirmed / total

    @property
    def contribution_rate(self) -> float:
        """How often this strategy contributes (score > 0)."""
        if self.evaluations == 0:
            return 0.0
        return self.contributed / self.evaluations

    @property
    def avg_score(self) -> float:
        """Average score when evaluated."""
        if self.evaluations == 0:
            return 0.0
        return self.total_score / self.evaluations

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evaluations": self.evaluations,
            "contributed": self.contributed,
            "confirmed": self.confirmed,
            "rejected": self.rejected,
            "accuracy": round(self.accuracy, 3),
            "contribution_rate": round(self.contribution_rate, 3),
            "avg_score": round(self.avg_score, 3),
        }


@dataclass
class FusionEvent:
    """Record of a single fusion attempt."""

    timestamp: float
    source_a: str
    source_b: str
    confidence: float
    strategies_used: list[str]
    confirmed: bool | None = None  # None = no feedback yet


class FusionMetrics:
    """Thread-safe metrics collector for the correlation/fusion pipeline.

    Tracks:
    - Total fusions attempted and completed
    - Fusions per source pair (e.g., ble+camera)
    - Strategy-level accuracy from operator feedback
    - Rolling hourly rates

    Usage::

        metrics = FusionMetrics()
        metrics.record_fusion("ble", "camera", 0.85, ["spatial", "temporal"])
        metrics.record_feedback("ble_abc123", "camera_person1", confirmed=True)
        status = metrics.get_status()
    """

    def __init__(self, window_seconds: float = 3600.0) -> None:
        self._lock = threading.RLock()
        self._window = window_seconds

        # Strategy-level metrics
        self._strategies: dict[str, StrategyMetric] = {}

        # Source pair counts: (source_a, source_b) -> count
        self._source_pair_counts: dict[tuple[str, str], int] = defaultdict(int)

        # Rolling events for time-window calculations
        self._events: list[FusionEvent] = []
        self._max_events = 50000

        # Aggregate counters
        self._total_fusions = 0
        self._total_confirmed = 0
        self._total_rejected = 0
        self._total_pending = 0

        # Pending feedback: (source_a_id, source_b_id) -> event index
        self._pending_feedback: dict[tuple[str, str], int] = {}

    def record_fusion(
        self,
        source_a: str,
        source_b: str,
        confidence: float,
        strategy_scores: list[tuple[str, float]] | None = None,
        primary_id: str = "",
        secondary_id: str = "",
    ) -> None:
        """Record a successful fusion event.

        Parameters
        ----------
        source_a:
            Source type of primary target (e.g., "ble", "camera").
        source_b:
            Source type of secondary target.
        confidence:
            Final weighted confidence score.
        strategy_scores:
            List of (strategy_name, score) tuples.
        primary_id:
            Primary target ID (for feedback tracking).
        secondary_id:
            Secondary target ID (for feedback tracking).
        """
        now = time.time()
        strategies_used = []

        with self._lock:
            self._total_fusions += 1
            self._total_pending += 1

            # Normalize pair order for consistent counting
            pair = tuple(sorted([source_a, source_b]))
            self._source_pair_counts[pair] += 1

            # Record strategy contributions
            if strategy_scores:
                for name, score in strategy_scores:
                    if name not in self._strategies:
                        self._strategies[name] = StrategyMetric(name=name)
                    sm = self._strategies[name]
                    sm.evaluations += 1
                    sm.total_score += score
                    if score > 0:
                        sm.contributed += 1
                    strategies_used.append(name)

            event = FusionEvent(
                timestamp=now,
                source_a=source_a,
                source_b=source_b,
                confidence=confidence,
                strategies_used=strategies_used,
            )
            idx = len(self._events)
            self._events.append(event)

            # Track for feedback
            if primary_id and secondary_id:
                self._pending_feedback[(primary_id, secondary_id)] = idx

            # Prune old events
            if len(self._events) > self._max_events:
                cutoff = len(self._events) - self._max_events
                self._events = self._events[cutoff:]
                # Reindex pending feedback
                new_pending = {}
                for k, v in self._pending_feedback.items():
                    new_idx = v - cutoff
                    if new_idx >= 0:
                        new_pending[k] = new_idx
                self._pending_feedback = new_pending

    def record_feedback(
        self,
        primary_id: str,
        secondary_id: str,
        confirmed: bool,
    ) -> bool:
        """Record operator feedback on a fusion decision.

        Parameters
        ----------
        primary_id:
            Primary target ID from the fusion.
        secondary_id:
            Secondary target ID from the fusion.
        confirmed:
            True if operator confirms the fusion was correct.

        Returns
        -------
        bool:
            True if the feedback was matched to a pending fusion.
        """
        with self._lock:
            key = (primary_id, secondary_id)
            idx = self._pending_feedback.pop(key, None)
            if idx is None:
                # Try reversed order
                key = (secondary_id, primary_id)
                idx = self._pending_feedback.pop(key, None)

            if idx is None or idx >= len(self._events):
                return False

            event = self._events[idx]
            event.confirmed = confirmed
            self._total_pending -= 1

            if confirmed:
                self._total_confirmed += 1
            else:
                self._total_rejected += 1

            # Update strategy-level metrics
            for name in event.strategies_used:
                sm = self._strategies.get(name)
                if sm:
                    if confirmed:
                        sm.confirmed += 1
                    else:
                        sm.rejected += 1

            return True

    def record_strategy_evaluation(
        self,
        strategy_name: str,
        score: float,
    ) -> None:
        """Record a standalone strategy evaluation (even if no fusion occurred)."""
        with self._lock:
            if strategy_name not in self._strategies:
                self._strategies[strategy_name] = StrategyMetric(
                    name=strategy_name,
                )
            sm = self._strategies[strategy_name]
            sm.evaluations += 1
            sm.total_score += score
            if score > 0:
                sm.contributed += 1

    def get_hourly_rate(self) -> float:
        """Get fusions per hour in the current time window."""
        now = time.time()
        cutoff = now - self._window
        with self._lock:
            count = sum(1 for e in self._events if e.timestamp >= cutoff)
        hours = self._window / 3600.0
        return count / max(hours, 0.001)

    def get_source_pair_stats(self) -> dict[str, int]:
        """Get fusion counts by source pair."""
        with self._lock:
            return {
                f"{a}+{b}": count
                for (a, b), count in sorted(
                    self._source_pair_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }

    def get_strategy_performance(self) -> list[dict[str, Any]]:
        """Get per-strategy performance metrics."""
        with self._lock:
            return [
                sm.to_dict()
                for sm in sorted(
                    self._strategies.values(),
                    key=lambda s: s.evaluations,
                    reverse=True,
                )
            ]

    def get_confirmation_rate(self) -> float:
        """Overall confirmation rate from operator feedback."""
        with self._lock:
            total = self._total_confirmed + self._total_rejected
            if total == 0:
                return 0.0
            return self._total_confirmed / total

    def get_status(self) -> dict[str, Any]:
        """Full status report for API/dashboard consumption."""
        with self._lock:
            now = time.time()
            cutoff = now - self._window

            # Recent events within window
            recent = [e for e in self._events if e.timestamp >= cutoff]
            recent_confirmed = sum(1 for e in recent if e.confirmed is True)
            recent_rejected = sum(1 for e in recent if e.confirmed is False)

            # Recent source pairs
            recent_pairs: dict[str, int] = defaultdict(int)
            for e in recent:
                pair = "+".join(sorted([e.source_a, e.source_b]))
                recent_pairs[pair] += 1

            hours = self._window / 3600.0
            hourly_rate = len(recent) / max(hours, 0.001)

            return {
                "total_fusions": self._total_fusions,
                "total_confirmed": self._total_confirmed,
                "total_rejected": self._total_rejected,
                "total_pending_feedback": self._total_pending,
                "confirmation_rate": round(self.get_confirmation_rate(), 3),
                "hourly_rate": round(hourly_rate, 2),
                "window_fusions": len(recent),
                "window_confirmed": recent_confirmed,
                "window_rejected": recent_rejected,
                "window_seconds": self._window,
                "source_pairs": dict(
                    sorted(recent_pairs.items(), key=lambda x: x[1], reverse=True)
                ),
                "source_pairs_total": {
                    f"{a}+{b}": c
                    for (a, b), c in sorted(
                        self._source_pair_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                },
                "strategies": [
                    sm.to_dict()
                    for sm in sorted(
                        self._strategies.values(),
                        key=lambda s: s.evaluations,
                        reverse=True,
                    )
                ],
            }

    def get_strategy_weights_recommendation(self) -> dict[str, float]:
        """Suggest strategy weights based on confirmed accuracy.

        Strategies with higher accuracy from operator feedback get higher
        weights.  This can feed into RL-based weight optimization.

        Returns
        -------
        dict[str, float]:
            Recommended weights for each strategy (sum to 1.0).
        """
        with self._lock:
            accuracies: dict[str, float] = {}
            for name, sm in self._strategies.items():
                total_fb = sm.confirmed + sm.rejected
                if total_fb < 5:
                    # Not enough feedback — use contribution rate as proxy
                    accuracies[name] = max(0.1, sm.contribution_rate)
                else:
                    accuracies[name] = max(0.05, sm.accuracy)

        if not accuracies:
            return {}

        total = sum(accuracies.values())
        if total == 0:
            n = len(accuracies)
            return {k: round(1.0 / n, 3) for k in accuracies}

        return {
            k: round(v / total, 3) for k, v in accuracies.items()
        }
