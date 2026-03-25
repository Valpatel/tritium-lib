# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""LogCollector — collect and parse logs from all Tritium components.

Scans local log directories for log files, parses structured entries,
and provides filtering/searching capabilities. No network calls.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LogLevel(str, Enum):
    """Log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Severity ordering for comparison
_LEVEL_ORDER = {
    LogLevel.DEBUG: 0,
    LogLevel.INFO: 1,
    LogLevel.WARNING: 2,
    LogLevel.ERROR: 3,
    LogLevel.CRITICAL: 4,
}


@dataclass
class LogEntry:
    """A single structured log entry.

    Attributes
    ----------
    timestamp:
        Unix timestamp of the log entry.
    level:
        Log severity level.
    component:
        Which component produced this entry (e.g., "sc", "edge").
    message:
        The log message text.
    source_file:
        Path to the log file this entry came from.
    line_number:
        Line number in the source file.
    raw:
        The raw unparsed line.
    """

    timestamp: float = 0.0
    level: LogLevel = LogLevel.INFO
    component: str = ""
    message: str = ""
    source_file: str = ""
    line_number: int = 0
    raw: str = ""

    @property
    def level_order(self) -> int:
        """Numeric severity for sorting (higher = more severe)."""
        return _LEVEL_ORDER.get(self.level, 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "source_file": self.source_file,
            "line_number": self.line_number,
        }


# Common Python logging format: 2026-03-25 10:30:45,123 - name - LEVEL - message
_LOG_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"  # timestamp
    r"[,.]?\d*"                                      # optional ms
    r"\s*[-:]\s*"                                    # separator
    r"(\w[\w.]*)"                                    # logger name
    r"\s*[-:]\s*"                                    # separator
    r"(DEBUG|INFO|WARNING|ERROR|CRITICAL)"           # level
    r"\s*[-:]\s*"                                    # separator
    r"(.*)"                                          # message
)


def _parse_timestamp(ts_str: str) -> float:
    """Parse a timestamp string to unix time. Returns 0.0 on failure."""
    try:
        t = time.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")
        return time.mktime(t)
    except (ValueError, OverflowError):
        return 0.0


class LogCollector:
    """Collect and filter logs from Tritium component log directories.

    Parameters
    ----------
    log_dirs:
        List of directories to scan for log files.
    file_patterns:
        Glob-like suffixes to match (default: [".log", ".txt"]).
    max_lines:
        Maximum number of lines to read per file (0 = unlimited).
    """

    def __init__(
        self,
        log_dirs: list[str] | None = None,
        file_patterns: list[str] | None = None,
        max_lines: int = 10000,
    ) -> None:
        # Validate log directory paths do not contain null bytes
        clean_dirs = []
        for d in (log_dirs or []):
            if "\x00" in str(d):
                continue  # silently skip paths with null bytes
            clean_dirs.append(d)
        self.log_dirs = clean_dirs
        self.file_patterns = file_patterns or [".log", ".txt"]
        self.max_lines = max_lines

    def find_log_files(self) -> list[str]:
        """Find all log files in configured directories.

        Returns absolute paths to all matching log files.
        Skips symlinks that resolve outside the configured log directory
        to prevent directory traversal via crafted symlinks.
        """
        log_files: list[str] = []
        for log_dir in self.log_dirs:
            if not os.path.isdir(log_dir):
                continue
            real_base = os.path.realpath(log_dir)
            for root, _dirs, files in os.walk(log_dir, followlinks=False):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    # Ensure file resolves under the configured log dir
                    real_fpath = os.path.realpath(fpath)
                    if not (real_fpath == real_base or real_fpath.startswith(real_base + os.sep)):
                        continue  # skip symlinks escaping the log dir
                    if any(fname.endswith(pat) for pat in self.file_patterns):
                        log_files.append(fpath)
        return sorted(log_files)

    def parse_line(
        self,
        line: str,
        source_file: str = "",
        line_number: int = 0,
    ) -> LogEntry | None:
        """Parse a single log line into a LogEntry.

        Returns None if the line doesn't match the expected format.
        """
        match = _LOG_PATTERN.match(line.strip())
        if not match:
            return None

        ts_str, component, level_str, message = match.groups()
        try:
            level = LogLevel(level_str)
        except ValueError:
            level = LogLevel.INFO

        return LogEntry(
            timestamp=_parse_timestamp(ts_str),
            level=level,
            component=component,
            message=message.strip(),
            source_file=source_file,
            line_number=line_number,
            raw=line.rstrip("\n"),
        )

    def collect(
        self,
        since_hours: float = 24.0,
        min_level: LogLevel = LogLevel.DEBUG,
        component_filter: str = "",
    ) -> list[LogEntry]:
        """Collect log entries from all configured directories.

        Parameters
        ----------
        since_hours:
            Only include entries from the last N hours.
        min_level:
            Minimum severity level to include.
        component_filter:
            If set, only include entries from this component.

        Returns
        -------
        List of LogEntry objects sorted by timestamp (newest first).
        """
        cutoff = time.time() - (since_hours * 3600)
        min_order = _LEVEL_ORDER.get(min_level, 0)
        entries: list[LogEntry] = []

        for log_file in self.find_log_files():
            try:
                file_entries = self._read_file(
                    log_file, cutoff, min_order, component_filter
                )
                entries.extend(file_entries)
            except OSError:
                continue

        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    def _read_file(
        self,
        path: str,
        cutoff: float,
        min_order: int,
        component_filter: str,
    ) -> list[LogEntry]:
        """Read and parse a single log file with filters."""
        entries: list[LogEntry] = []
        try:
            with open(path) as f:
                for line_num, line in enumerate(f, start=1):
                    if self.max_lines and line_num > self.max_lines:
                        break
                    entry = self.parse_line(line, source_file=path, line_number=line_num)
                    if entry is None:
                        continue
                    if entry.timestamp < cutoff and entry.timestamp > 0:
                        continue
                    if entry.level_order < min_order:
                        continue
                    if component_filter and entry.component != component_filter:
                        continue
                    entries.append(entry)
        except OSError:
            pass
        return entries

    def search(
        self,
        pattern: str,
        since_hours: float = 24.0,
    ) -> list[LogEntry]:
        """Search log entries for a text pattern.

        Parameters
        ----------
        pattern:
            Substring to search for in log messages (case-insensitive).
        since_hours:
            Only include entries from the last N hours.

        Returns
        -------
        Matching LogEntry objects sorted by timestamp (newest first).
        """
        all_entries = self.collect(since_hours=since_hours)
        pat_lower = pattern.lower()
        return [e for e in all_entries if pat_lower in e.message.lower()]

    def error_summary(self, since_hours: float = 24.0) -> dict[str, int]:
        """Get a count of errors per component.

        Returns a dict mapping component name to error count.
        """
        errors = self.collect(
            since_hours=since_hours,
            min_level=LogLevel.ERROR,
        )
        counts: dict[str, int] = {}
        for entry in errors:
            key = entry.component or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts
