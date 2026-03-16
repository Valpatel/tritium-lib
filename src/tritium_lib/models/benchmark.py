# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Benchmark result models for standardized performance reporting.

Provides a structured way to record, compare, and report performance
benchmark results across the Tritium ecosystem (edge, SC, lib).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class BenchmarkUnit(str, Enum):
    """Standard units for benchmark measurements."""

    MILLISECONDS = "ms"
    SECONDS = "s"
    MICROSECONDS = "us"
    NANOSECONDS = "ns"
    OPS_PER_SEC = "ops/s"
    BYTES = "bytes"
    KILOBYTES = "KB"
    MEGABYTES = "MB"
    PERCENT = "%"
    COUNT = "count"
    CONNECTIONS = "connections"


class BenchmarkResult(BaseModel):
    """A single benchmark measurement with pass/fail threshold.

    Attributes:
        test_name: Identifier for the benchmark test (e.g. "target_update_10k").
        metric: What is being measured (e.g. "throughput", "latency", "memory").
        value: The measured value.
        unit: Unit of measurement.
        threshold: Maximum acceptable value (or minimum for throughput metrics).
        passed: Whether the result meets the threshold.
        timestamp: When the benchmark was run.
        metadata: Additional context (hardware, software version, etc.).
    """

    test_name: str
    metric: str
    value: float
    unit: BenchmarkUnit
    threshold: float
    passed: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}

    def summary(self) -> str:
        """One-line summary string."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.test_name}/{self.metric}: "
            f"{self.value:.2f} {self.unit.value} "
            f"(threshold: {self.threshold:.2f} {self.unit.value})"
        )


class BenchmarkSuite(BaseModel):
    """Collection of benchmark results from a single run.

    Attributes:
        suite_name: Name of the benchmark suite (e.g. "target_capacity").
        results: Individual benchmark measurements.
        run_timestamp: When the suite was executed.
        environment: System environment info.
        all_passed: True if every result passed its threshold.
    """

    suite_name: str
    results: list[BenchmarkResult] = Field(default_factory=list)
    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    environment: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": False}

    @property
    def all_passed(self) -> bool:
        """True if all results passed their thresholds."""
        return all(r.passed for r in self.results) if self.results else True

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def add(
        self,
        test_name: str,
        metric: str,
        value: float,
        unit: BenchmarkUnit,
        threshold: float,
        *,
        higher_is_better: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Add a benchmark result with automatic pass/fail evaluation.

        Args:
            test_name: Test identifier.
            metric: What is being measured.
            value: Measured value.
            unit: Unit of measurement.
            threshold: Threshold for pass/fail.
            higher_is_better: If True, value >= threshold means pass.
                If False (default), value <= threshold means pass.
            metadata: Optional extra context.

        Returns:
            The created BenchmarkResult.
        """
        if higher_is_better:
            passed = value >= threshold
        else:
            passed = value <= threshold

        result = BenchmarkResult(
            test_name=test_name,
            metric=metric,
            value=value,
            unit=unit,
            threshold=threshold,
            passed=passed,
            metadata=metadata or {},
        )
        self.results.append(result)
        return result

    def report(self) -> str:
        """Multi-line report of all results."""
        lines = [
            f"Benchmark Suite: {self.suite_name}",
            f"Run: {self.run_timestamp.isoformat()}",
            f"Results: {self.pass_count} passed, {self.fail_count} failed",
            "-" * 60,
        ]
        for r in self.results:
            lines.append(r.summary())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output or API response."""
        return {
            "suite_name": self.suite_name,
            "run_timestamp": self.run_timestamp.isoformat(),
            "all_passed": self.all_passed,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "results": [r.model_dump(mode="json") for r in self.results],
            "environment": self.environment,
        }
