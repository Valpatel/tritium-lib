# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SystemRequirements — check that a system meets minimum requirements.

All checks are local-only: Python version, disk space, memory, OS, etc.
No network calls are made.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequirementCheck:
    """Result of a single requirement check.

    Attributes
    ----------
    name:
        What was checked (e.g., "python_version", "disk_space").
    passed:
        Whether the check passed.
    required:
        The required value (human-readable).
    actual:
        The actual value found.
    message:
        Explanation of the result.
    """

    name: str
    passed: bool
    required: str
    actual: str
    message: str = ""


@dataclass
class RequirementsResult:
    """Aggregated result of all requirement checks.

    Attributes
    ----------
    checks:
        Individual check results.
    checked_at:
        Timestamp of the check run.
    """

    checks: list[RequirementCheck] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    @property
    def meets_minimum(self) -> bool:
        """True if all checks passed."""
        return all(c.passed for c in self.checks)

    @property
    def passed_count(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        """Number of checks that failed."""
        return sum(1 for c in self.checks if not c.passed)

    @property
    def failures(self) -> list[RequirementCheck]:
        """Return only the checks that failed."""
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "meets_minimum": self.meets_minimum,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "checked_at": self.checked_at,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "required": c.required,
                    "actual": c.actual,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


class SystemRequirements:
    """Check that the local system meets minimum requirements for Tritium.

    Parameters
    ----------
    min_python:
        Minimum Python version as (major, minor) tuple.
    min_disk_gb:
        Minimum free disk space in gigabytes.
    min_memory_mb:
        Minimum total system memory in megabytes.
    required_dirs:
        Directories that must exist (or be creatable).
    check_path:
        Path to check disk space against (default: root).
    """

    def __init__(
        self,
        min_python: tuple[int, int] = (3, 12),
        min_disk_gb: float = 1.0,
        min_memory_mb: int = 512,
        required_dirs: list[str] | None = None,
        check_path: str = "/",
    ) -> None:
        self.min_python = min_python
        self.min_disk_gb = min_disk_gb
        self.min_memory_mb = min_memory_mb
        self.required_dirs = required_dirs or []
        self.check_path = check_path

    def check_python_version(self) -> RequirementCheck:
        """Check that the Python version meets the minimum."""
        current = (sys.version_info.major, sys.version_info.minor)
        required_str = f"{self.min_python[0]}.{self.min_python[1]}"
        actual_str = f"{current[0]}.{current[1]}.{sys.version_info.micro}"
        passed = current >= self.min_python
        return RequirementCheck(
            name="python_version",
            passed=passed,
            required=f">= {required_str}",
            actual=actual_str,
            message="" if passed else f"Python {required_str}+ required",
        )

    def check_disk_space(self) -> RequirementCheck:
        """Check that enough free disk space is available."""
        try:
            usage = shutil.disk_usage(self.check_path)
            free_gb = usage.free / (1024 ** 3)
            passed = free_gb >= self.min_disk_gb
            return RequirementCheck(
                name="disk_space",
                passed=passed,
                required=f">= {self.min_disk_gb:.1f} GB",
                actual=f"{free_gb:.1f} GB",
                message="" if passed else "Insufficient disk space",
            )
        except OSError as exc:
            return RequirementCheck(
                name="disk_space",
                passed=False,
                required=f">= {self.min_disk_gb:.1f} GB",
                actual="unknown",
                message=f"Could not check disk space: {exc}",
            )

    def check_memory(self) -> RequirementCheck:
        """Check total system memory (Linux/macOS only via /proc/meminfo).

        Falls back to a pass with 'unknown' on unsupported platforms.
        """
        try:
            if os.path.isfile("/proc/meminfo"):
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            # Value is in kB
                            kb = int(line.split()[1])
                            total_mb = kb / 1024
                            passed = total_mb >= self.min_memory_mb
                            return RequirementCheck(
                                name="system_memory",
                                passed=passed,
                                required=f">= {self.min_memory_mb} MB",
                                actual=f"{total_mb:.0f} MB",
                                message="" if passed else "Insufficient memory",
                            )
            # Platform not supported for memory check — pass by default
            return RequirementCheck(
                name="system_memory",
                passed=True,
                required=f">= {self.min_memory_mb} MB",
                actual="unknown (not Linux)",
                message="Memory check not available on this platform",
            )
        except OSError as exc:
            return RequirementCheck(
                name="system_memory",
                passed=False,
                required=f">= {self.min_memory_mb} MB",
                actual="unknown",
                message=f"Could not check memory: {exc}",
            )

    def check_platform(self) -> RequirementCheck:
        """Report the platform (always passes — informational only)."""
        plat = platform.platform()
        return RequirementCheck(
            name="platform",
            passed=True,
            required="any",
            actual=plat,
            message="",
        )

    def check_directory(self, path: str) -> RequirementCheck:
        """Check if a directory exists."""
        exists = os.path.isdir(path)
        return RequirementCheck(
            name=f"directory:{path}",
            passed=exists,
            required="exists",
            actual="exists" if exists else "missing",
            message="" if exists else f"Directory not found: {path}",
        )

    def check_local(self) -> RequirementsResult:
        """Run all local system requirement checks.

        Returns a RequirementsResult with individual check outcomes.
        """
        result = RequirementsResult()
        result.checks.append(self.check_python_version())
        result.checks.append(self.check_disk_space())
        result.checks.append(self.check_memory())
        result.checks.append(self.check_platform())

        for d in self.required_dirs:
            result.checks.append(self.check_directory(d))

        return result
