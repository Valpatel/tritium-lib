"""Automated UI test runner for Tritium-OS devices.

Orchestrates device connection, app-by-app screenshot testing, and
visual analysis using the shared VisualCheck suite.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .device import DeviceAPI
from .visual import VisualCheck, VisualReport, Severity


@dataclass
class AppTestResult:
    app_name: str
    app_index: int
    launched: bool
    screenshot_path: Optional[str] = None
    report: Optional[VisualReport] = None

    @property
    def passed(self) -> bool:
        if not self.launched:
            return False
        return self.report.passed if self.report else False

    @property
    def error_count(self) -> int:
        return len(self.report.errors) if self.report else (0 if self.launched else 1)


@dataclass
class SuiteResult:
    device_host: str
    app_results: list[AppTestResult] = field(default_factory=list)
    launcher_report: Optional[VisualReport] = None
    launcher_screenshot: Optional[str] = None

    @property
    def passed(self) -> bool:
        if self.launcher_report and not self.launcher_report.passed:
            return False
        return all(r.passed for r in self.app_results)

    @property
    def total_errors(self) -> int:
        n = 0
        if self.launcher_report:
            n += len(self.launcher_report.errors)
        for r in self.app_results:
            n += r.error_count
        return n

    @property
    def total_warnings(self) -> int:
        n = 0
        if self.launcher_report:
            n += len(self.launcher_report.warnings)
        for r in self.app_results:
            if r.report:
                n += len(r.report.warnings)
        return n

    def summary(self) -> str:
        lines = [f"Tritium UI Test — {self.device_host}"]
        lines.append(f"  Launcher: {'PASS' if self.launcher_report and self.launcher_report.passed else 'FAIL'}")
        for r in self.app_results:
            status = "PASS" if r.passed else "FAIL"
            errs = r.error_count
            lines.append(f"  [{status}] {r.app_name} (errors={errs})")
            if r.report:
                for issue in r.report.errors:
                    lines.append(f"         ERROR: {issue}")
                for issue in r.report.warnings:
                    lines.append(f"         WARN:  {issue}")
        passed = sum(1 for r in self.app_results if r.passed)
        lines.append(f"  Result: {passed}/{len(self.app_results)} apps passed, "
                      f"{self.total_errors} errors, {self.total_warnings} warnings")
        return "\n".join(lines)


class UITestRunner:
    """Orchestrates visual testing across all apps on a Tritium-OS device.

    Usage:
        runner = UITestRunner("http://10.42.0.237")
        result = runner.run_all()
        print(result.summary())
    """

    def __init__(self, host: str, screenshot_dir: str = "/tmp/tritium_ui_test",
                 width: int = 800, height: int = 480,
                 settle_time: float = 1.0):
        self.api = DeviceAPI(host)
        self.checker = VisualCheck(width, height)
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.settle_time = settle_time

    def _save(self, img: np.ndarray, name: str) -> str:
        path = str(self.screenshot_dir / f"{name}.png")
        cv2.imwrite(path, img)
        return path

    def _capture_and_check(self, name: str,
                           has_nav_bar: bool = True) -> tuple[Optional[str], Optional[VisualReport]]:
        """Capture screenshot, run visual checks, save image."""
        img = self.api.screenshot_raw()
        if img is None:
            return None, None
        path = self._save(img, name)
        report = self.checker.run_all(img, app_name=name, has_nav_bar=has_nav_bar)
        return path, report

    def check_launcher(self) -> tuple[Optional[str], Optional[VisualReport]]:
        """Navigate home and check the launcher screen."""
        self.api.home()
        time.sleep(self.settle_time)
        return self._capture_and_check("launcher", has_nav_bar=False)

    def check_app(self, index: int, name: str,
                  flicker_frames: int = 5,
                  flicker_interval: float = 0.3) -> AppTestResult:
        """Launch an app by index, screenshot it, run visual checks, return home.

        Also captures multiple frames for:
        - Flicker detection (frames at 300ms intervals)
        - Frame tearing detection (rapid burst at 50ms intervals)
        """
        result = AppTestResult(app_name=name, app_index=index, launched=False)

        ok = self.api.launch(index)
        if not ok:
            return result
        result.launched = True

        time.sleep(self.settle_time)
        path, report = self._capture_and_check(
            f"app_{index:02d}_{name}", has_nav_bar=True
        )
        result.screenshot_path = path
        result.report = report

        if report is not None and flicker_frames > 1:
            # Capture frames for flicker detection (slower interval)
            frames = []
            for i in range(flicker_frames):
                img = self.api.screenshot_raw()
                if img is not None:
                    frames.append(img)
                    self._save(img, f"app_{index:02d}_{name}_f{i}")
                time.sleep(flicker_interval)

            if len(frames) >= 2:
                flicker_issues = self.checker.check_flicker(frames, app_name=name)
                report.issues.extend(flicker_issues)

            # Rapid burst capture for tearing detection (fast interval).
            # Tearing is most visible during active rendering — e.g. the
            # monitor app's bar updates or right after a widget state change.
            rapid_frames = []
            for i in range(6):
                img = self.api.screenshot_raw()
                if img is not None:
                    rapid_frames.append(img)
                    self._save(img, f"app_{index:02d}_{name}_tear{i}")
                time.sleep(0.05)  # ~50ms between captures

            if len(rapid_frames) >= 3:
                tear_issues = self.checker.check_frame_tearing(rapid_frames)
                report.issues.extend(tear_issues)

        # Return to launcher
        self.api.home()
        time.sleep(0.5)
        return result

    def check_button_tearing(
        self,
        button_x: int,
        button_y: int,
        button_w: int = 80,
        button_h: int = 48,
        event_name: str = "button_tap",
        capture_count: int = 4,
        capture_interval: float = 0.05,
    ) -> list:
        """Tap a button and capture frames to detect localized tearing.

        Captures a pre-tap frame, rapidly captures during/after the tap,
        waits for settle, then captures a post-settle frame. Analyzes
        the button region for partial renders and ghosting.

        Returns list of LayoutIssue from the event tearing check.
        """
        from .visual import LayoutIssue

        # Pre-event frame
        pre = self.api.screenshot_raw()
        if pre is None:
            return []

        # Tap the button
        tap_x = button_x + button_w // 2
        tap_y = button_y + button_h // 2
        self.api.tap(tap_x, tap_y)

        # Rapidly capture frames during the transition
        event_frames = []
        for i in range(capture_count):
            img = self.api.screenshot_raw()
            if img is not None:
                event_frames.append(img)
                self._save(img, f"{event_name}_f{i}")
            time.sleep(capture_interval)

        # Post-settle frame
        time.sleep(0.3)
        post = self.api.screenshot_raw()
        if post is None:
            return []

        roi = (button_x, button_y, button_w, button_h)
        return self.checker.check_event_tearing(
            pre, event_frames, post, roi, event_name=event_name
        )

    def check_nav_button_tearing(self) -> list:
        """Test all three nav bar buttons for tearing.

        Returns combined list of LayoutIssue from all nav button tests.
        """
        issues = []
        nav_y = self.checker.height - self.checker.NAV_BAR_H
        btn_w = self.checker.width // 3
        btn_h = self.checker.NAV_BAR_H

        buttons = [
            ("nav_back", 0, nav_y, btn_w, btn_h),
            ("nav_home", btn_w, nav_y, btn_w, btn_h),
            ("nav_launcher", btn_w * 2, nav_y, btn_w, btn_h),
        ]

        for name, bx, by, bw, bh in buttons:
            btn_issues = self.check_button_tearing(
                bx, by, bw, bh, event_name=name
            )
            issues.extend(btn_issues)
            # Return to an app so nav bar is visible for next test
            apps = self.api.apps()
            if len(apps) > 1:
                self.api.launch(1)
                time.sleep(self.settle_time)

        return issues

    def check_animation_stability(
        self,
        frame_count: int = 8,
        interval: float = 0.4,
    ) -> list:
        """Capture frames over time and verify animations are subtle and tear-free."""
        frames = []
        for i in range(frame_count):
            img = self.api.screenshot_raw()
            if img is not None:
                frames.append(img)
                self._save(img, f"anim_f{i}")
            time.sleep(interval)

        if len(frames) < 2:
            return []

        return self.checker.check_animation_stability(frames)

    def run_all(self, skip_system: bool = False,
                check_tearing: bool = True,
                check_animations: bool = True) -> SuiteResult:
        """Run visual checks on the launcher and every registered app.

        Args:
            skip_system: Skip system apps (Launcher, Settings, etc.)
            check_tearing: Run event tearing checks on nav bar buttons
            check_animations: Run animation stability checks
        """
        suite = SuiteResult(device_host=self.api.host)

        if not self.api.is_reachable():
            return suite

        # Check launcher
        path, report = self.check_launcher()
        suite.launcher_screenshot = path
        suite.launcher_report = report

        # Get app list from device
        apps = self.api.apps()
        for app in apps:
            if skip_system and app.system:
                continue
            result = self.check_app(app.index, app.name)
            suite.app_results.append(result)

        # Event tearing checks on nav bar buttons
        if check_tearing and len(apps) > 1:
            # Launch an app so nav bar is visible
            self.api.launch(1)
            time.sleep(self.settle_time)
            tearing_issues = self.check_nav_button_tearing()
            if tearing_issues and suite.launcher_report:
                suite.launcher_report.issues.extend(tearing_issues)

        # Animation stability — check on launcher (should have alive animations)
        if check_animations:
            self.api.home()
            time.sleep(self.settle_time)
            anim_issues = self.check_animation_stability()
            if anim_issues and suite.launcher_report:
                suite.launcher_report.issues.extend(anim_issues)

        return suite
