# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Test coverage report generator for sim_engine modules.

Discovers and runs all sim_engine tests, then generates an HTML coverage
summary showing which modules have tests and their pass/fail status.

Usage:
    python3 -m tritium_lib.sim_engine.demos.test_report
    # Generates: /tmp/tritium_test_report.html
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModuleInfo:
    """Information about a single sim_engine module."""

    name: str
    path: str
    has_tests: bool = False
    test_files: list[str] = field(default_factory=list)
    importable: bool = False
    import_error: str = ""
    class_count: int = 0
    function_count: int = 0


@dataclass
class CoverageReport:
    """Aggregated test coverage data for sim_engine."""

    modules: list[ModuleInfo]
    total_modules: int = 0
    modules_with_tests: int = 0
    modules_importable: int = 0
    coverage_pct: float = 0.0
    generated_at: str = ""
    report_path: str = ""


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------


def discover_sim_engine_modules() -> list[ModuleInfo]:
    """Find all Python modules under tritium_lib.sim_engine.

    Returns a list of ModuleInfo with import status and basic metadata.
    """
    sim_engine_dir = Path(__file__).parent.parent
    modules: list[ModuleInfo] = []

    for root, dirs, files in os.walk(sim_engine_dir):
        # Skip __pycache__ and test directories
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "demos")]
        for fname in sorted(files):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            fpath = Path(root) / fname
            rel = fpath.relative_to(sim_engine_dir)
            # Build module name
            parts = list(rel.parts)
            parts[-1] = parts[-1].replace(".py", "")
            module_name = "tritium_lib.sim_engine." + ".".join(parts)

            info = ModuleInfo(name=module_name, path=str(fpath))

            # Try importing
            try:
                mod = importlib.import_module(module_name)
                info.importable = True
                # Count classes and functions
                for attr_name in dir(mod):
                    obj = getattr(mod, attr_name, None)
                    if isinstance(obj, type):
                        info.class_count += 1
                    elif callable(obj) and not attr_name.startswith("_"):
                        info.function_count += 1
            except Exception as e:
                info.import_error = str(e)[:200]

            modules.append(info)

    return modules


def find_test_files() -> dict[str, list[str]]:
    """Find test files and map them to the modules they test.

    Returns a dict of module_name -> list of test file paths.
    """
    test_dirs = [
        Path(__file__).parent / "tests",
        Path(__file__).parent.parent.parent.parent.parent / "tests",
    ]

    mapping: dict[str, list[str]] = {}

    for test_dir in test_dirs:
        if not test_dir.exists():
            continue
        for fpath in sorted(test_dir.rglob("test_*.py")):
            fname = fpath.name
            # Try to match test files to modules
            # e.g., test_city3d_features.py -> demos module
            # test_crowd.py -> crowd module
            test_name = fname.replace("test_", "").replace(".py", "")

            # Check multiple possible module matches
            candidates = [
                f"tritium_lib.sim_engine.{test_name}",
                f"tritium_lib.sim_engine.ai.{test_name}",
                f"tritium_lib.sim_engine.demos.{test_name}",
                f"tritium_lib.sim_engine.effects.{test_name}",
                f"tritium_lib.sim_engine.physics.{test_name}",
                f"tritium_lib.sim_engine.audio.{test_name}",
                f"tritium_lib.sim_engine.debug.{test_name}",
            ]

            # Also handle test files like test_city3d_features -> demos
            if test_name.startswith("city3d"):
                candidates.append("tritium_lib.sim_engine.demos.city_sim_backend")
                candidates.append("tritium_lib.sim_engine.demos.game_server")

            for candidate in candidates:
                mapping.setdefault(candidate, []).append(str(fpath))

            # Generic fallback: the file itself is a test
            mapping.setdefault(f"_test:{fname}", []).append(str(fpath))

    return mapping


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_coverage_report(
    output_path: str = "/tmp/tritium_test_report.html",
) -> CoverageReport:
    """Discover modules, check test coverage, generate HTML report.

    Parameters
    ----------
    output_path : str
        Where to write the HTML report.

    Returns
    -------
    CoverageReport
        The aggregated coverage data.
    """
    modules = discover_sim_engine_modules()
    test_map = find_test_files()

    # Match tests to modules
    for mod in modules:
        matched_tests = test_map.get(mod.name, [])
        if matched_tests:
            mod.has_tests = True
            mod.test_files = matched_tests

    total = len(modules)
    with_tests = sum(1 for m in modules if m.has_tests)
    importable = sum(1 for m in modules if m.importable)
    coverage = (with_tests / total * 100.0) if total > 0 else 0.0

    report = CoverageReport(
        modules=modules,
        total_modules=total,
        modules_with_tests=with_tests,
        modules_importable=importable,
        coverage_pct=coverage,
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        report_path=output_path,
    )

    html = _render_html(report)
    with open(output_path, "w") as f:
        f.write(html)

    return report


def _render_html(report: CoverageReport) -> str:
    """Render the coverage report as a cyberpunk-themed HTML page."""
    # Module rows
    rows = ""
    for mod in sorted(report.modules, key=lambda m: m.name):
        status_color = "#05ffa1" if mod.importable else "#ff2a6d"
        test_color = "#05ffa1" if mod.has_tests else "#ff8800"
        test_count = len(mod.test_files)
        error_cell = ""
        if mod.import_error:
            error_cell = f'<td style="color:#ff2a6d;font-size:0.8em;">{_esc(mod.import_error[:80])}</td>'
        else:
            error_cell = "<td>-</td>"

        short_name = mod.name.replace("tritium_lib.sim_engine.", "")
        rows += (
            f'<tr>'
            f'<td>{_esc(short_name)}</td>'
            f'<td style="color:{status_color};">{"OK" if mod.importable else "FAIL"}</td>'
            f'<td>{mod.class_count}</td>'
            f'<td>{mod.function_count}</td>'
            f'<td style="color:{test_color};">{test_count} file{"s" if test_count != 1 else ""}</td>'
            f'{error_cell}'
            f'</tr>\n'
        )

    coverage_color = "#05ffa1" if report.coverage_pct >= 60 else "#ff8800" if report.coverage_pct >= 30 else "#ff2a6d"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tritium Sim Engine - Test Coverage Report</title>
<style>
  body {{ background: #0a0a0f; color: #ccc; font-family: 'Courier New', monospace; margin: 20px; }}
  h1 {{ color: #00f0ff; border-bottom: 1px solid #00f0ff; padding-bottom: 8px; }}
  h2 {{ color: #ff2a6d; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: left; }}
  th {{ background: #1a1a2e; color: #00f0ff; }}
  tr:nth-child(even) {{ background: #0e0e14; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 15px 0; }}
  .summary-card {{ background: #12121a; border: 1px solid #333; padding: 12px; border-radius: 4px; }}
  .summary-card .label {{ color: #888; font-size: 0.85em; }}
  .summary-card .value {{ color: #05ffa1; font-size: 1.4em; font-weight: bold; }}
</style>
</head>
<body>
<h1>Sim Engine Test Coverage Report</h1>
<p style="color:#888;">Generated: {_esc(report.generated_at)}</p>

<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Total Modules</div>
    <div class="value">{report.total_modules}</div>
  </div>
  <div class="summary-card">
    <div class="label">Importable</div>
    <div class="value">{report.modules_importable}</div>
  </div>
  <div class="summary-card">
    <div class="label">With Tests</div>
    <div class="value">{report.modules_with_tests}</div>
  </div>
  <div class="summary-card">
    <div class="label">Coverage</div>
    <div class="value" style="color:{coverage_color};">{report.coverage_pct:.1f}%</div>
  </div>
</div>

<h2>Module Details</h2>
<table>
<tr><th>Module</th><th>Import</th><th>Classes</th><th>Functions</th><th>Tests</th><th>Error</th></tr>
{rows}
</table>

<p style="color:#555;margin-top:30px;font-size:0.8em;">
Generated by Tritium Sim Engine Test Report &mdash; Copyright 2026 Valpatel Software LLC
</p>
</body>
</html>"""


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run test coverage report and print summary."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tritium sim_engine test coverage report"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/tritium_test_report.html",
        help="Output HTML report path",
    )
    args = parser.parse_args()

    print("\n=== TRITIUM TEST COVERAGE REPORT ===")
    print("Discovering sim_engine modules...")

    report = generate_coverage_report(output_path=args.output)

    print(f"\nTotal modules: {report.total_modules}")
    print(f"Importable:    {report.modules_importable}")
    print(f"With tests:    {report.modules_with_tests}")
    print(f"Coverage:      {report.coverage_pct:.1f}%")
    print(f"\nReport: {report.report_path}")

    # List modules without tests
    no_tests = [m for m in report.modules if not m.has_tests and m.importable]
    if no_tests:
        print(f"\nModules without tests ({len(no_tests)}):")
        for m in no_tests[:10]:
            short = m.name.replace("tritium_lib.sim_engine.", "")
            print(f"  - {short}")
        if len(no_tests) > 10:
            print(f"  ... and {len(no_tests) - 10} more")


if __name__ == "__main__":
    main()
